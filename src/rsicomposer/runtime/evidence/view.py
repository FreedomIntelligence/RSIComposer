"""Read-only parent evidence with host-enforced sealed and credential boundaries."""
from copy import deepcopy
from dataclasses import asdict, is_dataclass, replace
import json
import re
from urllib.parse import parse_qsl, urlsplit

from rsicomposer.loops.contracts.artifacts import ArtifactRef
from rsicomposer.loops.contracts.controller_version import EvidenceScope
from rsicomposer.loops.contracts.feedback_context import EvidenceEvent
from rsicomposer.runtime.storage.codec import decode


_REF_FIELDS = {'artifact_id', 'kind', 'uri', 'version', 'digest'}
_SECRET_KEY = re.compile(
    r'^(?:(?:.*[_-])?(?:api[_-]?key|access[_-]?key|private[_-]?key|secret|password|passwd|'
    r'credential|credentials|authorization|bearer|cookie|cookies|session(?:[_-]?(?:id|token))?|'
    r'auth[_-]?token|access[_-]?token|refresh[_-]?token|token)|RSICOMPOSER_.*(?:TOKEN|SECRET)_.*)$', re.I)
_SECRET_TEXT = re.compile(
    r'\bBearer\s+\S+|\bsk-[a-z0-9_-]+|-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----|'
    r'\b(?:api[_-]?key|access[_-]?token|password|secret|authorization|cookie|session)'
    r'\s*[:=]\s*[\"\']?[^\s,;\"\'}]+', re.I)
_URL = re.compile(r'\b[a-z][a-z0-9+.-]*://[^\s<>\"\']+', re.I)
_HIDDEN = '[restricted]'
_TOKENIZER_FIELDS = {'bos_token', 'eos_token'}
MIN_EMBEDDED_SECRET_LENGTH = 8


def _secret_url(text):
    for match in _URL.finditer(text):
        try:
            url = urlsplit(match.group())
            if url.username is not None or url.password is not None or any(
                    key.lower() == 'key' or _SECRET_KEY.fullmatch(key)
                    for key, _ in parse_qsl(url.query, keep_blank_values=True)):
                return True
        except ValueError:
            continue
    return False


class ReadOnlyEvidenceView:
    def __init__(self, experience, artifact_authority, scope=EvidenceScope(), *, run_id=None,
                 checkpoints=None, secret_values=()):
        if not isinstance(scope, EvidenceScope):
            raise TypeError('Evidence view scope must be an EvidenceScope')
        self._experience = experience
        self._store = artifact_authority.store
        self.scope, self._run_id = scope, run_id
        self._checkpoints = dict(checkpoints or {})
        self._secrets = tuple(value for value in secret_values if isinstance(value, str) and value)
        self._discovered = {}
        self._artifact_index = None

    def _refs(self):
        if self._artifact_index is None:
            self._artifact_index = {key: ref for ref in self._store.refs() for key in (ref.artifact_id, ref.uri)}
        return self._artifact_index

    def _reference(self, value):
        if isinstance(value, ArtifactRef):
            return value
        if isinstance(value, dict) and set(value) == _REF_FIELDS:
            try:
                return ArtifactRef(**value)
            except (TypeError, ValueError) as error:
                raise PermissionError('Invalid evidence artifact reference') from error
        if isinstance(value, str):
            return self._refs().get(value)
        return None

    def _secret_text(self, value):
        return isinstance(value, str) and (
            _SECRET_TEXT.search(value) is not None or _secret_url(value)
            or any(value == item or (len(item) >= MIN_EMBEDDED_SECRET_LENGTH and item in value)
                   for item in self._secrets))

    def _walk(self, value, *, strict=False, found=None, sealed=False):
        if isinstance(value, dict) and set(value) in ({'$map'}, {'$list'}, {'$tuple'}, {'$set'},
                                                      {'$frozenset'}, {'$type', 'fields'}, {'$enum', 'value'}):
            try:
                value = decode(value)
            except (TypeError, ValueError, KeyError) as error:
                if strict:
                    raise PermissionError('Encoded evidence cannot be checked safely') from error
                return _HIDDEN
        ref = self._reference(value)
        if ref is not None:
            if any(self._secret_text(item) for item in asdict(ref).values()):
                if strict:
                    raise PermissionError('Credential-bearing artifact reference is unavailable')
                return _HIDDEN
            try:
                restricted = self._store.access(ref)['sealed']
            except (KeyError, ValueError) as error:
                raise PermissionError('Evidence artifact is not managed by its host') from error
            if restricted:
                if strict:
                    raise PermissionError('Sealed per-item evidence is unavailable')
                return _HIDDEN
            if found is not None:
                found[ref.artifact_id] = ref
            return deepcopy(value)
        if is_dataclass(value):
            return self._walk(asdict(value), strict=strict, found=found, sealed=sealed)
        if isinstance(value, dict):
            if (value.get('phase') == 'host_finish' or any(
                    isinstance(value.get(key), str) and value[key].startswith('sealed-')
                    for key in ('event_id', 'record_id'))):
                if strict:
                    raise PermissionError('Host final evidence is unavailable')
                return _HIDDEN
            sealed = sealed or value.get('split') == 'sealed' or value.get('sealed') is True
            result = {}
            for key, item in value.items():
                clean_key = self._walk(key, strict=strict)
                if sealed and key in {'items', 'per_item', 'per_item_results', 'scores', 'labels', 'predictions', 'traces'}:
                    if strict and item != _HIDDEN:
                        raise PermissionError('Sealed per-item evidence is unavailable')
                    result[clean_key] = _HIDDEN
                elif isinstance(key, str) and key not in _TOKENIZER_FIELDS and _SECRET_KEY.fullmatch(key):
                    if strict and item != _HIDDEN:
                        raise PermissionError('Credential-bearing evidence is unavailable')
                    result[clean_key] = _HIDDEN
                else:
                    result[clean_key] = self._walk(item, strict=strict, found=found, sealed=sealed)
            return result
        if isinstance(value, (list, tuple)):
            cleaned = [self._walk(item, strict=strict, found=found, sealed=sealed) for item in value]
            return tuple(cleaned) if isinstance(value, tuple) else cleaned
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, (dict, list)):
                cleaned = self._walk(parsed, strict=strict, found=found, sealed=sealed)
                return json.dumps(cleaned, ensure_ascii=False)
            if self._secret_text(value):
                if strict:
                    raise PermissionError('Credential-bearing evidence is unavailable')
                return _HIDDEN
        return deepcopy(value)

    @staticmethod
    def _event(record):
        raw = record.details.get('event')
        if isinstance(raw, dict) and any(key.startswith('$') for key in raw):
            raw = decode(raw)
        if isinstance(raw, EvidenceEvent):
            return deepcopy(raw)
        if isinstance(raw, dict) and 'event_id' in raw and 'source' in raw:
            fields = deepcopy(raw)
            # SQLite stores ordinary event dictionaries as JSON, whose arrays
            # have no tuple representation. Restore the current typed boundary.
            if isinstance(fields.get('source_refs'), list):
                fields['source_refs'] = tuple(fields['source_refs'])
            return EvidenceEvent(**fields)
        operation = record.details.get('operation')
        source = ('controller' if record.kind in {'controller_event', 'memory_observation', 'insight'}
                  else 'operation' if record.kind == 'runtime_feedback' else 'lab')
        return EvidenceEvent(record.record_id, source, kind='receipt' if source != 'controller' else 'interpretation',
            task_id=record.directive_id or '', status=record.details.get('status', ''),
            output=deepcopy(record.details.get('receipt', record.details)), request=deepcopy(operation),
            references=record.source_refs, decision_id=record.details.get('decision_id'),
            identity=deepcopy(record.details.get('identity', {})))

    @staticmethod
    def _components(event):
        values = {event.source}
        for key in ('component', 'lab_id', 'controller_version'):
            if isinstance(event.identity.get(key), str):
                values.add(event.identity[key])
        request = event.request
        name = request.get('name') if isinstance(request, dict) else request
        if isinstance(name, str) and '.' in name:
            values.add(name.split('.', 1)[0])
        return values

    def _selected(self):
        rows = tuple(self._experience.retrieve())
        start = 0
        if self.scope.since is not None:
            since = self._checkpoints.get(self.scope.since, self.scope.since)
            positions = [index for index, row in enumerate(rows)
                         if row.record_id == since or self._event(row).event_id == since]
            if not positions:
                raise ValueError('Unknown evidence event or checkpoint: ' + self.scope.since)
            start = positions[0]
        for record in rows[start:]:
            # Memory delivery/rebuild journals track notification attempts for
            # canonical evidence; they are not new observations of that fact.
            if (record.kind == 'memory_observation' or record.record_id.startswith('sealed-')
                    or record.details.get('phase') == 'host_finish'):
                continue
            if self._run_id is not None and record.run_id not in (None, self._run_id):
                continue
            event = self._event(record)
            if (event.source in self.scope.sources
                    and (self.scope.kinds is None or event.kind in self.scope.kinds)
                    and (self.scope.components is None or self._components(event) & self.scope.components)):
                yield record, event

    def records(self):
        return tuple(replace(row, record_id=self._walk(row.record_id), directive_id=self._walk(row.directive_id),
            run_id=self._walk(row.run_id), created_at=self._walk(row.created_at), summary=self._walk(row.summary),
            source_refs=self._walk(row.source_refs), details=self._walk(row.details))
            for row, _ in self._selected())

    def events(self):
        return tuple(replace(event, event_id=self._walk(event.event_id),
            kind='restricted' if self._secret_text(event.kind) else event.kind,
            output=self._walk(event.output), request=self._walk(event.request),
            references=self._walk(event.references), identity=self._walk(event.identity),
            evidence_packet=self._walk(event.evidence_packet),
            source_refs=tuple(dict.fromkeys(self._walk(event.source_refs))),
            candidate_evidence=self._walk(event.candidate_evidence),
            task_id=self._walk(event.task_id), status=self._walk(event.status),
            decision_id=self._walk(event.decision_id))
            for _, event in self._selected())

    def read(self, ref):
        if isinstance(ref, dict):
            ref = ArtifactRef(**ref)
        if not isinstance(ref, ArtifactRef):
            raise TypeError('Evidence reads require an ArtifactRef')
        reachable = dict(self._discovered)
        for row, event in self._selected():
            self._walk(asdict(row), found=reachable)
            self._walk(asdict(event), found=reachable)
        if reachable.get(ref.artifact_id) != ref:
            raise PermissionError('Artifact is outside the declared evidence scope')
        if self._store.access(ref)['sealed']:
            raise PermissionError('Sealed per-item evidence is unavailable')
        data = self._store.read(ref)
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError as error:
            raise PermissionError('Opaque evidence cannot be checked for restricted content') from error
        discovered = {}
        self._walk(text, strict=True, found=discovered)
        self._discovered.update(discovered)
        return data
