"""Research notebook capabilities using the existing ActionSpace/Tools boundary."""

from copy import deepcopy
from dataclasses import asdict
from functools import partial
from hashlib import sha256

from rsicomposer.harnesses.decision.context import encoded, result_id
from rsicomposer.harnesses.decision.research import (
    KINDS,
    MAX_PATCH_CHANGES,
    ResearchAgenda,
    ResearchInputRejected,
    page_limit,
    page_offset,
)
from rsicomposer.loops.contracts.coordination import OperationResult
from rsicomposer.loops.contracts.decision import OperationCall, OperationSpec
from rsicomposer.loops.contracts.lifecycle import ExecutionStatus
from rsicomposer.loops.decision_making.action_space.capabilities import (
    Capabilities,
    Capability,
)
from rsicomposer.runtime.evidence.retrieval import controller_experience
from rsicomposer.runtime.storage.json_tools import EvidencePathError, evidence_value


class _Definition:
    def __init__(self, spec, validator):
        self.spec, self.validator = spec, validator

    def validate(self, arguments):
        return OperationCall(self.spec.name, self.validator(arguments))


def _read_arguments(arguments):
    if not isinstance(arguments, dict) or set(arguments) - {'view', 'kind', 'limit', 'cursor'}:
        raise ValueError('Read accepts only view, kind, limit and cursor')
    args = dict(view='items', kind=None, limit=4, cursor=None)
    args.update(arguments)
    page_limit(args['limit'])
    if (args['view'] not in ('items', 'revisions') or args['kind'] not in (*KINDS, None)
            or (args['view'] == 'revisions' and args['kind'] is not None)
            or (args['cursor'] is not None and not isinstance(args['cursor'], str))):
        raise ValueError('Invalid research read arguments')
    return deepcopy(args)


def _archive_arguments(arguments, *, experience=True):
    if not isinstance(arguments, dict) or set(arguments) - {'collection', 'record_id', 'limit', 'cursor', 'evidence_path'}:
        raise ValueError('Archive accepts only collection, record_id, limit, cursor and evidence_path')
    args = {'collection': 'history', 'record_id': None, 'limit': 8, 'cursor': None, 'evidence_path': None}
    args.update(arguments)
    page_limit(args['limit'])
    if args['collection'] == 'experience' and not experience:
        raise ValueError('Experience archive provider is not bound')
    if args['collection'] not in ('history', 'memories', 'operation_results', 'decision_graph',
                                  *(['experience'] if experience else []), 'research_items', 'research_revisions'):
        raise ValueError('Invalid archive collection')
    if args['record_id'] is not None and (not isinstance(args['record_id'], str) or not args['record_id']):
        raise ValueError('record_id must be nonempty text or null')
    if args['cursor'] is not None and not isinstance(args['cursor'], str):
        raise ValueError('cursor must be a string or null')
    pointer = args['evidence_path']
    if pointer is not None and (args['record_id'] is None or not isinstance(pointer, str)
            or len(pointer) > 1000 or pointer and not pointer.startswith('/')):
        raise ValueError('evidence_path requires record_id and a JSON pointer of at most 1000 characters')
    return deepcopy(args)


class ResearchTools:
    """Inject the same agenda and memory instances as the decision harness.

    Read/archive are read-only; patch remains subject to existing permissions.
    No automatic installation or external discovery occurs here.
    """

    def __init__(self, agenda: ResearchAgenda, memory, *, experience=None):
        self.agenda, self.memory = agenda, memory
        self.experience = controller_experience(experience)
        self.decision_graph = None  # Bound to the owning harness's live ledger, never a frozen copy.
        text = {'type': 'string', 'minLength': 1, 'maxLength': 128, 'pattern': r'\S'}
        refs = {'type': 'array', 'items': {'type': 'string', 'minLength': 1, 'maxLength': 512,
                                         'pattern': r'\S'}, 'maxItems': 64}
        schema = lambda props, required=(): {'type': 'object', 'properties': props,
                                           'required': list(required), 'additionalProperties': False}
        read_schema = schema({'view': {'enum': ['items', 'revisions']}, 'kind': {'enum': [*KINDS, None]},
            'limit': {'type': 'integer', 'minimum': 1, 'maximum': 32}, 'cursor': {'type': ['string', 'null']}})
        read_schema['oneOf'] = [
            {'properties': {'view': {'enum': ['items']}}},
            {'properties': {'view': {'enum': ['revisions']}, 'kind': {'type': 'null'}}, 'required': ['view']},
        ]
        read = _Definition(OperationSpec('research_read',
            'Read model-owned agenda, todos, stage summaries or revision evidence. Notes are not scientific truth. '
            'Follow next_cursor; restart pagination after a revision changes.',
            read_schema, read_only=True), _read_arguments)
        patch_props = {'expected_revision': {'type': 'integer', 'minimum': 0},
            'upserts': {'type': 'array', 'maxItems': MAX_PATCH_CHANGES, 'items': schema({
                'id': text, 'kind': {'enum': list(KINDS)},
                'text': {'type': 'string', 'minLength': 1, 'maxLength': 16000, 'pattern': r'\S'},
                'status': text, 'evidence_refs': refs}, ('id',))},
            'remove_ids': {'type': 'array', 'items': text, 'maxItems': MAX_PATCH_CHANGES},
            'reason': {'type': 'string', 'minLength': 1, 'maxLength': 4000, 'pattern': r'\S'}, 'evidence_refs': refs}
        patch_schema = schema(patch_props, patch_props)
        patch_schema['anyOf'] = [{'properties': {name: {'minItems': 1}}}
                                for name in ('upserts', 'remove_ids')]
        patch = _Definition(OperationSpec('research_patch',
            'Atomically patch stable IDs at expected_revision. Omitted IDs and fields survive. New IDs require kind/text. '
            'Choose your own statuses and research process. Supply reason and evidence_refs (empty means no evidence). '
            f'Upserts plus removals must total 1..{MAX_PATCH_CHANGES}; IDs must be distinct across both lists. '
            'Read revision before retrying conflicts; summaries are model notes, not validated findings.',
            patch_schema), agenda.validate_patch)
        archive = _Definition(OperationSpec('research_archive',
            'Read exact archived JSON in bounded 1000-character chunks; follow next_cursor. '
            'IDs: history uses record_id, memories uses memory_id, decision_graph uses decision_id; '
            'operation_results uses SHA-256 and /result. evidence_path is a JSON Pointer.',
            schema({'collection': {'enum': ['history', 'memories', 'operation_results', 'decision_graph',
                                            'research_items', 'research_revisions',
                                            *(['experience'] if experience is not None else [])]},
                    'record_id': {'type': ['string', 'null'], 'minLength': 1}, 'cursor': {'type': ['string', 'null']},
                    'evidence_path': {'type': ['string', 'null'], 'maxLength': 1000},
                    'limit': {'type': 'integer', 'minimum': 1, 'maximum': 32}}), read_only=True),
            partial(_archive_arguments, experience=experience is not None))
        self.capabilities = Capabilities((Capability(read, self._read), Capability(patch, self._patch),
                                          Capability(archive, self._archive)))
        self.definitions, self.bindings = self.capabilities.definitions, self.capabilities.bindings

    def describe(self):
        return tuple(self.capabilities)

    def _read(self, arguments):
        args = _read_arguments(arguments)
        try:
            data = self.agenda.read(**args)
        except ResearchInputRejected as exc:
            return OperationResult('Research cursor rejected; restart pagination', (), False,
                ExecutionStatus.FAILED, str(exc), data={
                    'revision': self.agenda.revision,
                    'read_path': {'operation': 'research_read', **args, 'cursor': None}})
        return OperationResult('Model-owned research notes', (), False, data=data)

    def _patch(self, arguments):
        try:
            data = self.agenda.patch(**arguments)
        except ResearchInputRejected as exc:
            return OperationResult('Research patch rejected; read notes and correct the patch', (), False,
                ExecutionStatus.FAILED, str(exc), data={
                    'revision': self.agenda.revision, 'changed_ids': [], 'removed_ids': [],
                    'read_path': {'operation': 'research_read', 'view': 'items', 'cursor': None}})
        return OperationResult('Research patch recorded', (), False, data=data)

    def _archive(self, arguments):
        args = _archive_arguments(arguments, experience=self.experience is not None)
        def reject_cursor(message):
            # A model cursor can become stale or refer to another selection.
            # This read has no external effect: return actionable native feedback,
            # while unexpected provider/storage faults still propagate normally.
            return OperationResult('Archive cursor rejected; restart pagination', (), False,
                ExecutionStatus.FAILED, message, data={
                    'collection': args['collection'], 'record_id': args['record_id'],
                    'evidence_path': args['evidence_path'],
                    'read_path': {'operation': 'research_archive', **args, 'cursor': None}})
        collection = args['collection']
        if collection == 'decision_graph':
            if self.decision_graph is None:
                raise ValueError('Decision ledger archive provider is not bound')
            rows, identity = list(self.decision_graph()), 'decision_id'
        elif collection == 'experience':
            if self.experience is None:
                raise ValueError('Experience archive provider is not bound')
            rows = [asdict(row) for row in self.experience.retrieve(limit=None)]
            identity = 'record_id'
        elif collection == 'operation_results':
            receipts = {}
            for entry in self.memory.history(kind='operation'):
                result = entry.details.get('result')
                if isinstance(result, dict):
                    ident = result_id(result)
                    row = receipts.setdefault(ident, {'record_id': ident, 'result': deepcopy(result),
                                                      'history_refs': []})
                    row['history_refs'].append(f'history:decision_maker:{entry.record_id}')
            rows, identity = list(receipts.values()), 'record_id'
        elif collection.startswith('research_'):
            rows = self.agenda.export()['items' if collection == 'research_items' else 'revisions']
            identity = 'id' if collection == 'research_items' else 'revision'
        else:
            rows = (self.memory.history() if collection == 'history'
                    else self.memory.recall(limit=2**63 - 1))
            rows = [asdict(row) for row in rows]
            identity = 'record_id' if collection == 'history' else 'memory_id'
        if args['record_id'] is not None:
            rows = [row for row in rows if str(row[identity]) == args['record_id']]
        # Pin the original prefix so receipts appended by this read do not
        # invalidate history pagination. Changed records still invalidate it.
        count = len(rows)
        if args['cursor'] is not None:
            parts = args['cursor'].rsplit(':', 3)
            if len(parts) != 4 or not parts[1].isascii() or not parts[1].isdecimal():
                return reject_cursor('Invalid archive cursor; restart pagination')
            try:
                count = int(parts[1])
            except ValueError:
                return reject_cursor('Invalid archive cursor; restart pagination')
            if count > len(rows):
                return reject_cursor('Stale archive cursor; restart pagination')
            rows = rows[:count]
        # Fingerprint only the selected records so an exact-ID read remains stable
        # while the harness appends receipts for the read operations themselves.
        digest = sha256()
        for row in rows:
            digest.update(encoded(row).encode('utf-8'))
            digest.update(b'\n')
        revision = digest.hexdigest()
        pointer = args['evidence_path']
        if pointer is not None:
            revision = result_id({'archive': revision, 'evidence_path': pointer})
        scope = f'{collection}:{args["record_id"] or "all"}:{count}:{revision}'
        try:
            start = page_offset(args['cursor'], scope)
        except ValueError as exc:
            return reject_cursor(str(exc))
        chunks, offset, total = [], 0, 0
        page_full = False
        for row in rows:
            value = row
            if pointer is not None:
                try:
                    value = evidence_value(row, pointer)
                except EvidencePathError as exc:
                    return OperationResult('Archive evidence path rejected', (), False,
                        ExecutionStatus.FAILED, str(exc), data={
                            'collection': collection, 'record_id': args['record_id'],
                            'evidence_path': pointer, 'root_fields': list(row)[:16],
                            'read_path': {'operation': 'research_archive', 'collection': collection,
                                          'record_id': args['record_id'], 'evidence_path': ''}})
            content = encoded(value)
            record_chunks = max(1, (len(content) + 999) // 1000)
            for chunk in range(max(0, start - offset), min(record_chunks, start + args['limit'] - offset)):
                item = {'record_id': str(row[identity]), 'chunk': chunk, 'chunks': record_chunks,
                        'text': content[chunk * 1000:(chunk + 1) * 1000]}
                if page_full or chunks and len(encoded([*chunks, item]).encode('utf-8')) > 6000:
                    page_full = True
                    break
                chunks.append(item)
            offset += record_chunks
            total += record_chunks
        end = start + len(chunks)
        return OperationResult('Raw research archive page', (), False, data={
            'revision': revision, 'collection': args['collection'], 'evidence_path': pointer, 'items': chunks,
            'found': bool(rows),
            'total_chunks': total, 'next_cursor': f'{scope}:{end}' if end < total else None})
