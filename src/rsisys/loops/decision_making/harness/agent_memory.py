"""Explicit memory and task context owned by the acting meta-agent.

No model, extraction worker, embedding service or implicit consolidation lives
here. Mutations are ordinary accountable decisions. Native evidence stays in
the execution archive; memory prose never becomes verified evidence.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import re

from rsisys.loops.contracts.decision import OperationArgumentError, OperationCall, OperationSpec
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.decision_making.action_space import OperationResult
from rsisys.runtime.files import atomic_write, json_bytes
from rsisys.runtime.json_tools import encoded, result_id


def _text(value, name, maximum):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise OperationArgumentError(f'{name} requires nonempty text of at most {maximum} characters')
    return value


def _strings(value, name, maximum):
    if (not isinstance(value, list) or len(value) > maximum
            or any(not isinstance(x, str) or not x.strip() or len(x) > 512 for x in value)
            or len(set(value)) != len(value)):
        raise OperationArgumentError(f'{name} requires at most {maximum} distinct references')
    return value


IDENT = {'type': 'string', 'minLength': 1, 'maxLength': 128}
REVISION = {'type': 'integer', 'minimum': 0}
REFS = {'type': 'array', 'items': {'type': 'string', 'minLength': 1, 'maxLength': 512},
        'maxItems': 16, 'uniqueItems': True}


def _spec(name, description, properties, required, read_only=False):
    return OperationSpec(name, description, {'type': 'object', 'properties': properties,
        'required': required, 'additionalProperties': False}, read_only=read_only)


SPECS = (
    _spec('memory_search', 'Search meta-agent-authored memory by literal text; no model or embedding call. '
        'Results are interpretations, not verified experiment results. Follow next_offset with expected_revision.',
        {'query': {'type': 'string', 'maxLength': 512}, 'offset': REVISION,
         'limit': {'type': 'integer', 'minimum': 1, 'maximum': 8}, 'expected_revision': REVISION}, [], True),
    _spec('memory_read', 'Read one exact memory and provenance, or a historical revision. '
        'Large JSON records use literal UTF-8 pages; follow next_offset with expected_digest.',
        {'memory_id': IDENT, 'revision': {'type': 'integer', 'minimum': 1}, 'offset': REVISION,
         'expected_digest': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'}}, ['memory_id'], True),
    _spec('memory_write', 'Create or replace your own memory, including a concise summary if useful. '
        'Supply current global expected_revision. Cited sources must exist in this run. '
        'status is your assessment; prose is always unverified. This performs no inference.',
        {'expected_revision': REVISION, 'memory_id': IDENT,
         'content': {'type': 'string', 'minLength': 1, 'maxLength': 3000},
         'source_refs': REFS, 'status': {'enum': ['unverified', 'supported', 'refuted']},
         'applicability': {'type': 'string', 'maxLength': 1000}},
        ['expected_revision', 'memory_id', 'content', 'source_refs', 'status']),
    _spec('memory_delete', 'Remove a memory from retrieval. Prior versions remain in the revision archive.',
        {'expected_revision': REVISION, 'memory_id': IDENT}, ['expected_revision', 'memory_id']),
    _spec('research_task', 'Keep native feedback through a debugging/experiment task. '
        'open pins the latest native receipt unless explicit record_ids are supplied; pin adds sources. '
        'Reading files, editing and retesting do not close the task. close releases its resident feedback '
        'without erasing archives or automatically writing memory. At most two tasks and four pins each.',
        {'expected_revision': REVISION, 'task_id': IDENT, 'action': {'enum': ['open', 'pin', 'close']},
         'title': {'type': 'string', 'minLength': 1, 'maxLength': 300},
         'record_ids': {**REFS, 'maxItems': 4}}, ['expected_revision', 'task_id', 'action']),
)
SPEC_BY_NAME = {spec.name: spec for spec in SPECS}


class AgentMemory:
    """Single-writer atomic JSON store plus deterministic bounded retrieval.

    Configuring this provider performs no I/O. The default database is next to
    the feedback journal; explicit paths are useful for custom integrations.
    """
    def __init__(self, path=None):
        self.configured_path = str(Path(path).resolve()) if path is not None else None
        self.path = None
        self.harness = None
        self._state = {'version': 1, 'revision': 0, 'memories': {}, 'tasks': {}, 'revisions': []}

    def configuration(self):
        return {'provider': 'agent_managed', 'path': self.configured_path,
            'default_path': 'agent-memory.json beside feedback journal',
            'retrieval': 'literal_text_and_recency', 'organization_owner': 'acting_meta_agent',
            'model_calls': 0, 'task_context': 'explicit_open_pin_close/v1'}

    def bind(self, harness, journal_path=None):
        if self.harness is not None:
            raise ValueError('Agent memory is already bound to a run')
        self.harness = harness
        self.path = (Path(self.configured_path) if self.configured_path else
                     Path(journal_path).parent / 'agent-memory.json' if journal_path is not None else None)
        if self.path is not None and self.path.exists():
            self._state = self._validated(json.loads(self.path.read_text(encoding='utf-8')))
            if any(not self.validates_source(ref, note['provenance'][ref])
                   for note in self._state['memories'].values() for ref in note['source_refs']):
                raise ValueError('Agent memory belongs to another or incomplete evidence archive; '
                                 'use an explicit continuation adapter')

    @staticmethod
    def _validated(state):
        if (not isinstance(state, dict) or set(state) != {'version', 'revision', 'memories', 'tasks', 'revisions'}
                or type(state['version']) is not int or state['version'] != 1
                or type(state['revision']) is not int or state['revision'] < 0
                or not isinstance(state['memories'], dict) or not isinstance(state['tasks'], dict)
                or not isinstance(state['revisions'], list) or len(state['revisions']) != state['revision']):
            raise ValueError('Invalid agent memory state')
        memories, tasks = {}, {}
        for i, row in enumerate(state['revisions'], 1):
            if (not isinstance(row, dict) or set(row) != {'revision', 'operation', 'arguments', 'memories', 'tasks'}
                    or type(row['revision']) is not int or row['revision'] != i or not isinstance(row['arguments'], dict)
                    or not isinstance(row['memories'], dict) or not isinstance(row['tasks'], dict)):
                raise ValueError('Invalid agent memory revision')
            try:
                AgentMemory().validate(OperationCall(row['operation'], row['arguments']))
            except OperationArgumentError as exc:
                raise ValueError('Invalid agent memory revision arguments') from exc
            args = row['arguments']
            if args.get('expected_revision') != i - 1 or row['operation'] not in ('memory_write', 'memory_delete', 'research_task'):
                raise ValueError('Agent memory revision does not describe a mutation')
            if row['operation'] == 'research_task':
                if row['memories'] or set(row['tasks']) != {args['task_id']}:
                    raise ValueError('Invalid task revision delta')
            elif row['tasks'] or set(row['memories']) != {args['memory_id']}:
                raise ValueError('Invalid memory revision delta')
            for ident, value in row['memories'].items():
                if row['operation'] == 'memory_delete' and value is not None:
                    raise ValueError('Memory deletion revision must remove the record')
                if value is None:
                    if ident not in memories or row['operation'] != 'memory_delete':
                        raise ValueError('Invalid memory deletion revision')
                    del memories[ident]
                    continue
                if (not isinstance(value, dict) or set(value) != {'memory_id', 'content', 'source_refs', 'status',
                        'applicability', 'revision', 'authority', 'provenance'}
                        or value['memory_id'] != ident or type(value['revision']) is not int or value['revision'] != i
                        or value['authority'] != 'meta_agent_assertion; prose_unverified'
                        or not isinstance(value['source_refs'], list)
                        or value['source_refs'] != args['source_refs']
                        or not isinstance(value['provenance'], dict)
                        or set(value['provenance']) != set(value['source_refs'])):
                    raise ValueError('Invalid persisted memory record')
                expected = {key: val for key, val in args.items() if key != 'expected_revision'}
                expected.setdefault('applicability', '')
                if any(value[key] != val for key, val in expected.items()):
                    raise ValueError('Memory record differs from its write arguments')
                if value['status'] != 'unverified' and not value['source_refs']:
                    raise ValueError('Persisted memory assessment requires cited sources')
                for provenance in value['provenance'].values():
                    if (not isinstance(provenance, dict) or set(provenance) != {'history_id', 'sha256'}
                            or not isinstance(provenance['history_id'], str)
                            or not isinstance(provenance['sha256'], str)
                            or re.fullmatch('[a-f0-9]{64}', provenance['sha256']) is None):
                        raise ValueError('Invalid memory source provenance')
                memories.pop(ident, None)
                memories[ident] = value
            for ident, value in row['tasks'].items():
                if (not isinstance(value, dict) or set(value) != {'task_id', 'title', 'status', 'record_ids', 'start_after', 'revision'}
                        or value['task_id'] != ident or type(value['revision']) is not int or value['revision'] != i
                        or value['status'] not in ('open', 'closed')
                        or value['start_after'] is not None and
                           (not isinstance(value['start_after'], str) or not value['start_after'])):
                    raise ValueError('Invalid persisted task record')
                _text(value['title'], 'title', 300)
                _strings(value['record_ids'], 'record_ids', 4)
                action = args['action']
                if action == 'open':
                    if ident in tasks or sum(task['status'] == 'open' for task in tasks.values()) >= 2:
                        raise ValueError('Invalid task opening revision')
                    if (value['status'] != 'open' or value['title'] != args['title']
                            or 'record_ids' in args and value['record_ids'] != args['record_ids']
                            or 'record_ids' not in args and len(value['record_ids']) > 1):
                        raise ValueError('Task opening differs from its arguments')
                else:
                    if ident not in tasks or tasks[ident]['status'] != 'open':
                        raise ValueError('Task mutation requires an open task')
                    expected = {**deepcopy(tasks[ident]), 'revision': i}
                    if action == 'close':
                        expected['status'] = 'closed'
                    else:
                        expected['record_ids'] = list(dict.fromkeys([*expected['record_ids'], *args['record_ids']]))
                    if value != expected:
                        raise ValueError('Task mutation differs from its arguments or prior state')
                tasks[ident] = value
        if state['memories'] != memories or state['tasks'] != tasks:
            raise ValueError('Agent memory indexes differ from their revision archive')
        json_bytes(state)
        return deepcopy(state)

    def export(self):
        return deepcopy(self._state)

    def seed(self, state):
        """Host continuation adapter only; ordinary decisions use revisioned CRUD."""
        if self._state['revision'] or self.harness is None:
            raise ValueError('Memory seeding requires a fresh bound agent memory')
        candidate = self._validated(state)
        sources = self._sources()
        for note in candidate['memories'].values():
            if any(sources.get(ref) != note['provenance'][ref] for ref in note['source_refs']):
                raise ValueError('Inherited memory source is missing or differs')
        if self.path is not None:
            if self.path.exists() and self._validated(json.loads(self.path.read_text())) != self._state:
                from rsisys.runtime.recovery import RecoveryRequired
                raise RecoveryRequired('Memory changed before continuation import')
            atomic_write(self.path, json_bytes(candidate))
        self._state = candidate

    def restore(self, state):
        """Same-directory recovery verifies mutations; it never rolls them back."""
        candidate = self._validated(state)
        if self.path is not None and self.path.exists():
            current = self._validated(json.loads(self.path.read_text(encoding='utf-8')))
            if current != candidate:
                from rsisys.runtime.recovery import RecoveryRequired
                raise RecoveryRequired('Agent memory changed after checkpoint; reconcile the interrupted mutation')
        if self.path is not None:
            atomic_write(self.path, json_bytes(candidate))
        self._state = candidate

    def _history(self):
        return self.harness.memory.history() if self.harness is not None else ()

    def _sources(self):
        sources = {}
        for row in self._history():
            ref = 'history:decision_maker:' + row.record_id
            sources[ref] = {'history_id': row.record_id, 'sha256': result_id(asdict(row))}
            for item in row.source_refs:
                sources[item] = {'history_id': row.record_id, 'sha256': result_id(asdict(row))}
            result = row.details.get('result')
            if isinstance(result, dict):
                sources['result:' + result_id(result)] = {'history_id': row.record_id, 'sha256': result_id(result)}
        return sources

    def validates_source(self, ref, provenance):
        """Resolve the recorded occurrence, even after an identical later read."""
        if not isinstance(provenance, dict):
            return False
        row = next((row for row in self._history() if row.record_id == provenance.get('history_id')), None)
        if row is None:
            return False
        if ref == 'history:decision_maker:' + row.record_id or ref in row.source_refs:
            return provenance.get('sha256') == result_id(asdict(row))
        result = row.details.get('result')
        return (isinstance(result, dict) and ref == 'result:' + result_id(result)
                and provenance.get('sha256') == result_id(result))

    @staticmethod
    def _native(row):
        from .observations import native_observation
        return native_observation(row)

    def validate(self, call):
        if call.name not in SPEC_BY_NAME:
            raise OperationArgumentError('Unknown memory operation')
        args, spec = call.arguments, SPEC_BY_NAME[call.name]
        if (not isinstance(args, dict) or set(args) - set(spec.input_schema['properties'])
                or not set(spec.input_schema['required']) <= set(args)):
            raise OperationArgumentError('Invalid fields for ' + call.name)
        for key in ('expected_revision', 'offset', 'revision'):
            if key in args and (type(args[key]) is not int or args[key] < 0):
                raise OperationArgumentError(key + ' must be a nonnegative integer')
        if 'revision' in args and args['revision'] < 1:
            raise OperationArgumentError('revision must be positive')
        if 'expected_digest' in args and (not isinstance(args['expected_digest'], str)
                or re.fullmatch('[a-f0-9]{64}', args['expected_digest']) is None):
            raise OperationArgumentError('expected_digest must be a SHA-256 digest')
        for key in ('memory_id', 'task_id'):
            if key in args:
                _text(args[key], key, 128)
        for key, length in (('content', 3000), ('title', 300)):
            if key in args:
                _text(args[key], key, length)
        for key, length in (('query', 512), ('applicability', 1000)):
            if key in args and (not isinstance(args[key], str) or len(args[key]) > length):
                raise OperationArgumentError('Invalid ' + key)
        for key, limit in (('source_refs', 16), ('record_ids', 4)):
            if key in args:
                _strings(args[key], key, limit)
        if 'status' in args and args['status'] not in ('unverified', 'supported', 'refuted'):
            raise OperationArgumentError('Unknown memory status')
        if 'limit' in args and (type(args['limit']) is not int or not 1 <= args['limit'] <= 8):
            raise OperationArgumentError('limit must be in 1..8')
        if call.name == 'research_task':
            if args['action'] not in ('open', 'pin', 'close'):
                raise OperationArgumentError('Unknown task action')
            if args['action'] == 'open' and 'title' not in args:
                raise OperationArgumentError('Opening a task requires title')
            if args['action'] == 'pin' and not args.get('record_ids'):
                raise OperationArgumentError('Pin requires explicit record_ids')
            if args['action'] != 'open' and 'title' in args or args['action'] == 'close' and 'record_ids' in args:
                raise OperationArgumentError('Fields do not match the task action')
        return deepcopy(call)

    def search(self, *, query='', offset=0, limit=8, expected_revision=None):
        revision = self._state['revision']
        if (expected_revision is not None and expected_revision != revision) or (offset and expected_revision is None):
            raise OperationArgumentError('Stale or missing memory revision; restart search')
        values = [v for v in sorted(self._state['memories'].values(), key=lambda row: row['revision'], reverse=True)
                  if query.casefold() in (v['content'] + ' ' + v['applicability']).casefold()]
        selected = values[offset:offset + limit]
        return {'revision': revision, 'items': [{**{k: deepcopy(v[k]) for k in
            ('memory_id', 'revision', 'status', 'authority')},
            'source_refs': v['source_refs'][:2], 'source_count': len(v['source_refs']),
            'content_preview': v['content'][:240], 'complete': len(v['content']) <= 240,
            'read_operation': 'memory_read'} for v in selected], 'total': len(values),
            'next_offset': offset + limit if offset + limit < len(values) else None}

    def context(self, query):
        # Literal relevance plus recency, explicitly not semantic retrieval.
        tokens = set(re.findall(r'\w+', query.objective.casefold()))
        values = sorted(self._state['memories'].values(), key=lambda row: row['revision'], reverse=True)
        values.sort(key=lambda row: sum(t in (row['content'] + ' ' + row['applicability']).casefold()
                                        for t in tokens), reverse=True)
        return encoded({'provider': 'agent_managed', 'revision': self._state['revision'],
            'authority': 'meta_agent_interpretations_not_verified_evidence', 'total': len(values),
            'items': [{'memory_id': row['memory_id'], 'status': row['status'],
                       'content_preview': row['content'][:300], 'complete': len(row['content']) <= 300,
                       'source_refs': row['source_refs'][:2], 'source_count': len(row['source_refs']),
                       'read_operation': 'memory_read'} for row in values[:6]],
            'write_operation': 'memory_write', 'delete_operation': 'memory_delete'})

    def execute(self, call):
        call = self.validate(call)
        args = call.arguments
        if call.name == 'memory_search':
            return self.search(**args)
        if call.name == 'memory_read':
            ident = args['memory_id']
            if 'revision' in args:
                if args['revision'] > self._state['revision']:
                    raise OperationArgumentError('Unknown memory revision')
                row = next((r['memories'][ident] for r in reversed(self._state['revisions'][:args['revision']])
                            if ident in r['memories']), None)
            else:
                row = self._state['memories'].get(ident)
            if row is None:
                raise OperationArgumentError('Unknown memory_id')
            raw, digest = encoded(row).encode('utf-8'), result_id(row)
            offset = args.get('offset', 0)
            if (args.get('expected_digest', digest) != digest or offset and 'expected_digest' not in args
                    or offset > len(raw)):
                raise OperationArgumentError('Stale digest or invalid offset; restart memory_read')
            try:
                raw[:offset].decode('utf-8')
            except UnicodeDecodeError:
                raise OperationArgumentError('offset must be a UTF-8 byte boundary') from None
            value = {'revision': self._state['revision'], 'memory_id': ident, 'sha256': digest}
            if len(raw) <= 4096 and offset == 0:
                return {**value, 'memory': deepcopy(row), 'complete': True}
            content = raw[offset:offset + 4096].decode('utf-8', errors='ignore')
            end = offset + len(content.encode('utf-8'))
            return {**value, 'complete': False, 'content': content, 'offset': offset,
                'end_offset': end, 'total_bytes': len(raw), 'eof': end == len(raw),
                'next_offset': end if end < len(raw) else None,
                'read_operation': 'memory_read'}
        if args['expected_revision'] != self._state['revision']:
            raise OperationArgumentError(f"Stale memory revision; current revision is {self._state['revision']}")
        state = deepcopy(self._state)
        revision = state['revision'] + 1
        if call.name == 'memory_write':
            sources = self._sources()
            if any(ref not in sources for ref in args['source_refs']):
                raise OperationArgumentError('Unknown source reference; use exact history:decision_maker:<record_id> or result:<sha256>')
            if args['status'] != 'unverified' and not args['source_refs']:
                raise OperationArgumentError('Supported/refuted assessments require cited sources')
            row = {**deepcopy(args), 'revision': revision, 'applicability': args.get('applicability', ''),
                'authority': 'meta_agent_assertion; prose_unverified',
                'provenance': {ref: sources[ref] for ref in args['source_refs']}}
            row.pop('expected_revision')
            state['memories'].pop(args['memory_id'], None)
            state['memories'][args['memory_id']] = row
        elif call.name == 'memory_delete':
            if state['memories'].pop(args['memory_id'], None) is None:
                raise OperationArgumentError('Unknown memory_id')
        else:
            tasks, ident, action = state['tasks'], args['task_id'], args['action']
            history = self._history()
            native = {row.record_id: row for row in history if self._native(row)}
            if any(ref not in native for ref in args.get('record_ids', [])):
                raise OperationArgumentError('Task sources must identify native operation/feedback history record_ids')
            if action == 'open':
                if ident in tasks:
                    raise OperationArgumentError('Task ID already exists; use a new task_id')
                if sum(row['status'] == 'open' for row in tasks.values()) >= 2:
                    raise OperationArgumentError('Close an existing task before opening a third')
                pins = args.get('record_ids', list(native)[-1:])
                tasks[ident] = {'task_id': ident, 'title': args['title'], 'status': 'open',
                    'record_ids': list(pins), 'start_after': history[-1].record_id if history else None,
                    'revision': revision}
            else:
                if ident not in tasks or tasks[ident]['status'] != 'open':
                    raise OperationArgumentError('Task is not open')
                task = tasks[ident]
                if action == 'close':
                    task['status'] = 'closed'
                else:
                    pins = list(dict.fromkeys([*task['record_ids'], *args['record_ids']]))
                    if len(pins) > 4:
                        raise OperationArgumentError('A task supports at most four pinned receipts')
                    task['record_ids'] = pins
                task['revision'] = revision
        state['revision'] = revision
        state['revisions'].append({'revision': revision, 'operation': call.name, 'arguments': deepcopy(args),
            'memories': ({args['memory_id']: deepcopy(state['memories'].get(args['memory_id']))}
                         if call.name != 'research_task' else {}),
            'tasks': ({args['task_id']: deepcopy(state['tasks'][args['task_id']])}
                      if call.name == 'research_task' else {})})
        if self.path is not None:
            atomic_write(self.path, json_bytes(state))
        self._state = state
        return {'revision': revision, 'operation': call.name,
            **({'task': deepcopy(state['tasks'][args['task_id']])} if call.name == 'research_task' else
               {'memory_id': args['memory_id']}), 'model_calls': 0,
            'authority': 'meta_agent_notes_not_verified_evidence'}

    def project_context(self, context):
        history = self._history()
        indexed = {row.record_id: row for row in history}
        positions = {row.record_id: i for i, row in enumerate(history)}
        tasks, selected = [], []
        for task in self._state['tasks'].values():
            if task['status'] != 'open':
                continue
            if task['start_after'] is not None and task['start_after'] not in indexed:
                raise ValueError('Active task history anchor is missing')
            if any(ref not in indexed for ref in task['record_ids']):
                raise ValueError('Pinned task feedback is missing from the archive')
            start = positions.get(task['start_after'], -1)
            # Auto-retain the latest terminal producer observations in addition
            # to explicit pins. Intermediate code/log reads cannot evict pins.
            produced = [row.record_id for row in history[start + 1:] if self._native(row)
                and ((row.details.get('operation') or {}).get('name') in ('lab_result', 'medical_diagnose')
                     or row.kind == 'feedback')]
            refs = list(dict.fromkeys([*task['record_ids'], *produced[-2:]]))
            tasks.append({**deepcopy(task), 'resident_record_ids': refs,
                'later_producer_receipts': len(produced), 'older_results': 'research_archive'})
            selected.extend(ref for ref in refs if ref not in selected)
        pages = []
        from .research_state import archive_available
        can_read = archive_available(context, 'history')
        per_record = min(16384, 24576 // max(1, len(selected)))
        for ref in selected:
            row = indexed[ref]
            original = row.details.get('result') if row.kind == 'operation' else row.details.get('feedback', row.details)
            raw = encoded(original).encode('utf-8')
            page = {'record_id': ref, 'source_ref': 'history:decision_maker:' + ref,
                'source_sha256': result_id(original), 'operation': deepcopy(row.details.get('operation')),
                'authority': 'raw_observation_not_memory', 'complete': len(raw) <= per_record or not can_read,
                'read_path': {'operation': 'research_archive', 'collection': 'history', 'record_id': ref,
                    'evidence_path': '/details/result' if row.kind == 'operation' else '/details/feedback'}}
            if page['complete']:
                page['result'] = deepcopy(original)
            else:
                content = raw[:per_record].decode('utf-8', errors='ignore')
                page['raw_json_page'] = {'content': content, 'offset': 0,
                    'end_offset': len(content.encode('utf-8')), 'total_bytes': len(raw), 'eof': False}
            pages.append(page)
        agenda = {**context.research_agenda, 'agent_memory': {'revision': self._state['revision'],
            'organization_owner': 'acting_meta_agent', 'background_model_calls': 0,
            'memory_count': len(self._state['memories']),
            'open_task_count': len(tasks),
            'task_operation': 'research_task', 'recent_source_refs': [
                {'record_id': row.record_id, 'source_ref': 'history:decision_maker:' + row.record_id}
                for row in history if self._native(row)][-4:]},
            'active_feedback': {'tasks': tasks, 'observations': pages,
                'lifetime': 'until_explicit_task_close', 'archive_unchanged': True}}
        from .observations import failed_observation
        failures = [row for row in history if failed_observation(row)]
        agenda['feedback_health'] = {
            'authority': 'host_observation_inventory_not_interpretation',
            'recent_failures': [{'record_id': row.record_id,
                'source_ref': 'history:decision_maker:' + row.record_id,
                'source_sha256': result_id(asdict(row)),
                'tracked_by_open_task': row.record_id in selected,
                'read_path': {'operation': 'research_archive', 'collection': 'history',
                              'record_id': row.record_id}}
                for row in failures[-2:]],
            'memory_is_agent_authored': True,
            'instruction': 'Read the original failures. Open/pin a research_task for a continuing '
                           'investigation; write a memory only when useful. No implicit resolution or improvement.'}
        # An old feedback_route hint must not hide native data in this mode.
        from .research_state import previous_feedback_view
        previous = previous_feedback_view(replace(context, history=history), include_memory_route=True,
                                          include_plugin_observations=True)
        agenda.pop('previous_feedback', None)
        if previous is not None:
            resident = next((p for p in pages if p['source_sha256'] == previous['source_sha256']), None)
            agenda['previous_feedback'] = (previous if resident is None else {
                'source_sha256': resident['source_sha256'], 'record_id': resident['record_id'],
                'delivery': 'body_in_active_feedback_observations', 'complete': resident['complete'],
                'read_path': resident['read_path']})
        return replace(context, research_agenda=agenda)


class AgentMemoryState:
    state_id = 'rsisys-agent-memory/v1'

    def __init__(self, memory):
        self.memory = memory
        if memory.path is not None and not memory.path.exists():
            memory.restore(memory.export())

    def configuration(self):
        return self.memory.configuration()

    def capture(self):
        return self.memory.export()

    def restore(self, state):
        self.memory.restore(state)

    def validate_snapshot(self):
        raise TypeError('Agent memory supports same-directory recovery; portable snapshots require explicit file relocation integration')


class MemoryActionSpace:
    def __init__(self, base, memory):
        if set(SPEC_BY_NAME).intersection(spec.name for spec in base.describe()):
            raise ValueError('Agent memory tools are already installed')
        self.base, self.memory = base, memory

    @property
    def permissions(self):
        return self.base.permissions

    def __getattr__(self, name):
        return getattr(self.base, name)

    def describe(self):
        return (*self.base.describe(), *deepcopy(SPECS))

    def validate(self, call):
        return self.memory.validate(call) if call.name in SPEC_BY_NAME else self.base.validate(call)


class MemoryTools:
    def __init__(self, base, memory):
        self.base, self.memory = base, memory

    def __getattr__(self, name):
        return getattr(self.base, name)

    def execute(self, call):
        if call.name not in SPEC_BY_NAME:
            return self.base.execute(call)
        try:
            data = self.memory.execute(call)
        except OperationArgumentError as exc:
            return OperationResult('Memory/task arguments rejected; correct the next decision', (), False,
                                   ExecutionStatus.FAILED, str(exc))
        return OperationResult('Meta-agent memory/task operation completed', (), False, data=data)


def install_agent_memory(loop, memory, *, path=None):
    from .engine import DefaultDecisionHarness
    if not isinstance(loop.harness, DefaultDecisionHarness):
        raise TypeError('Custom harness must supply its own explicit agent-memory tool/context integration')
    if memory.harness is None:
        memory.bind(loop.harness, path)
    elif memory.harness is not loop.harness:
        raise ValueError('Agent memory is bound to another harness')
    space = MemoryActionSpace(loop.action_space, memory)
    loop.harness.tools = MemoryTools(loop.harness.tools, memory)
    parser = getattr(loop.harness.policy, 'parser', loop.harness.policy)
    if hasattr(parser, 'space'):
        parser.space = space
    loop.action_space = space
    loop.harness.agent_memory = memory
    loop.harness.add_context_processor(memory.project_context)
