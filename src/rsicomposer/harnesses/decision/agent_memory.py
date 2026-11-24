"""Explicit memory and task context owned by the acting meta-agent.

No model, extraction worker, embedding service or implicit consolidation lives
here. Mutations are ordinary accountable decisions. Native evidence stays in
the execution archive; memory prose never becomes verified evidence.
"""
import json
import re
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

from rsicomposer.harnesses.decision.task_acceptance import (
    CHECKS,
    EVIDENCE,
    RESOLUTIONS,
    accept,
    validate_checks,
)
from rsicomposer.loops.contracts.decision import (
    OperationArgumentError,
    OperationCall,
    OperationSpec,
)
from rsicomposer.runtime.storage.files import atomic_write, json_bytes
from rsicomposer.runtime.storage.json_tools import encoded, result_id


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
    _spec('memory_review', 'Explicitly disposition original results that do not warrant a new note yet. '
        'Use deferred only with the missing evidence/next action in reason; not_useful needs an explanation; '
        'already_known links existing memory_ids. A review is not a finding or verified improvement.',
        {'expected_revision': REVISION, 'source_refs': {**REFS, 'minItems': 1},
         'disposition': {'enum': ['deferred', 'not_useful', 'already_known']},
         'reason': {'type': 'string', 'minLength': 1, 'maxLength': 2000},
         'memory_ids': {**REFS}}, ['expected_revision', 'source_refs', 'disposition', 'reason']),
    _spec('memory_delete', 'Remove a memory from retrieval. Prior versions remain in the revision archive.',
        {'expected_revision': REVISION, 'memory_id': IDENT}, ['expected_revision', 'memory_id']),
    _spec('research_task', 'Keep native feedback through a debugging/experiment task. '
        'open pins the latest native receipt unless explicit record_ids are supplied; pin adds sources. '
        'Reading files, editing and retesting do not close the task. close releases its resident feedback '
        'without erasing archives or automatically writing memory. Record next_step and acceptance_criteria; '
        'Resolved closure requires acceptance_checks matched against native receipts. Text criteria are not '
        'machine verified. Close incomplete work explicitly as abandoned/inconclusive/interrupted. '
        'At most two tasks and four pins each.',
        {'expected_revision': REVISION, 'task_id': IDENT, 'action': {'enum': ['open', 'update', 'pin', 'close']},
         'title': {'type': 'string', 'minLength': 1, 'maxLength': 300},
         'next_step': {'type': 'string', 'maxLength': 1000},
         'acceptance_criteria': {'type': 'string', 'maxLength': 1000},
         'acceptance_checks': CHECKS,
         'resolution': {'enum': RESOLUTIONS}, 'acceptance_evidence': EVIDENCE,
         'outcome': {'type': 'string', 'maxLength': 2000},
         'record_ids': {**REFS, 'maxItems': 4}}, ['expected_revision', 'task_id', 'action']),
    _spec('research_task_read', 'Read an exact task, including a closed task and its evidence references, '
        'or list tasks by status. Follow next_offset with expected_revision. No inference or mutation.',
        {'task_id': IDENT, 'status': {'enum': ['open', 'closed', 'all']}, 'offset': REVISION,
         'limit': {'type': 'integer', 'minimum': 1, 'maximum': 8}, 'expected_revision': REVISION}, [], True),
)
SPEC_BY_NAME = {spec.name: spec for spec in SPECS}


class AgentMemory:
    """Single-writer atomic JSON store plus selectable bounded retrieval.

    Configuring this provider performs no I/O. The default database is next to
    the feedback journal; explicit paths are useful for custom integrations.
    """
    def __init__(self, path=None, *, require_review=False, retrieval_backend='literal'):
        if type(require_review) is not bool:
            raise TypeError('require_review must be a bool')
        if retrieval_backend not in ('literal', 'reme'):
            raise ValueError('retrieval_backend must be literal or reme')
        self.retrieval_backend = retrieval_backend
        self._retrieval = None
        if retrieval_backend == 'reme':
            from rsicomposer.harnesses.decision.reme_memory import ReMeBM25Retrieval
            self._retrieval = ReMeBM25Retrieval()
        self.require_review = require_review
        self.configured_path = str(Path(path).resolve()) if path is not None else None
        self.path = None
        self.harness = None
        self._history_provider = lambda: ()
        self._state = {'version': 2, 'revision': 0, 'memories': {}, 'tasks': {}, 'revisions': []}

    def configuration(self):
        config = {'provider': 'agent_managed', 'path': self.configured_path,
            'default_path': 'agent-memory.json beside feedback journal',
            'retrieval': 'literal_objective_feedback_tasks_and_recency/v2', 'organization_owner': 'acting_meta_agent',
            'model_calls': 0, 'task_context': 'receipt_checked_acceptance/v3',
            'consolidation': 'producer_inbox_explicit_disposition/v1',
            'require_review': self.require_review}
        if self._retrieval is not None:
            config.update(provider='reme', retrieval='reme_bm25/v1',
                          retrieval_backend=self._retrieval.configuration())
        return config

    def clone(self):
        """Copy configuration, never a prior run's notes, tasks or search cache."""
        return AgentMemory(self.configured_path, require_review=self.require_review,
                           retrieval_backend=self.retrieval_backend)

    def run_lifecycle(self):
        from .memory_integration import AgentMemoryLifecycle
        return AgentMemoryLifecycle(self)

    def prepare(self):
        if self._retrieval is not None:
            self._retrieval.prepare()

    def specifications(self):
        specs = deepcopy(SPECS)
        if self._retrieval is not None:
            specs = tuple(replace(spec, description=(
                'Search your current memory IDs, content and applicability using ReMe BM25 keyword ranking. '
                'Use related keywords rather than an exact sentence; this is lexical, not semantic search. '
                'Empty query lists notes by recency. No model/embedding calls. Results are interpretations, '
                'not verified evidence. Read relevant results with memory_read and use their source references. '
                'Follow next_offset with expected_revision.')) if spec.name == 'memory_search' else spec
                for spec in specs)
        return specs

    def bind(self, harness, journal_path=None):
        if self.harness is not None:
            raise ValueError('Agent memory is already bound to a run')
        from rsicomposer.runtime.memory import memory_host
        host = memory_host(harness)
        self.prepare()
        self.harness = harness
        self._history_provider = host.history
        self.path = (Path(self.configured_path) if self.configured_path else
                     Path(journal_path).parent / 'agent-memory.json' if journal_path is not None else None)
        if self.path is not None and self.path.exists():
            self._state = self._validated(json.loads(self.path.read_text(encoding='utf-8')))
            if any(not self.validates_source(ref, note['provenance'][ref])
                   for note in self._state['memories'].values() for ref in note['source_refs']):
                raise ValueError('Agent memory belongs to another or incomplete evidence archive; '
                                 'use an explicit continuation adapter')
            self._verify_acceptance(self._state)

    def _verify_acceptance(self, state):
        native = {row.record_id: row for row in self._history() if self._native(row)}
        for task in state['tasks'].values():
            if task.get('resolution') == 'resolved':
                receipt = accept(task, task.get('acceptance_evidence', {}), native)
                if receipt != task.get('acceptance_receipt'):
                    raise ValueError('Task acceptance source identity differs from its receipt')

    @staticmethod
    def _validated(state):
        if (not isinstance(state, dict) or set(state) != {'version', 'revision', 'memories', 'tasks', 'revisions'}
                or type(state['version']) is not int or state['version'] != 2
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
            if args.get('expected_revision') != i - 1 or row['operation'] not in ('memory_write', 'memory_delete', 'memory_review', 'research_task'):
                raise ValueError('Agent memory revision does not describe a mutation')
            if row['operation'] == 'memory_review':
                if row['memories'] or row['tasks']:
                    raise ValueError('Memory review cannot synthesize notes or task changes')
                if any(ident not in memories for ident in args.get('memory_ids', [])):
                    raise ValueError('Memory review references unknown memory')
            elif row['operation'] == 'research_task':
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
                required = {'task_id', 'title', 'status', 'record_ids', 'start_after', 'revision'}
                if (not isinstance(value, dict) or not required <= set(value)
                        or set(value) - required - {'next_step', 'acceptance_criteria', 'outcome',
                            'acceptance_checks', 'resolution', 'acceptance_evidence', 'acceptance_receipt'}
                        or value['task_id'] != ident or type(value['revision']) is not int or value['revision'] != i
                        or value['status'] not in ('open', 'closed')
                        or value['start_after'] is not None and
                           (not isinstance(value['start_after'], str) or not value['start_after'])):
                    raise ValueError('Invalid persisted task record')
                _text(value['title'], 'title', 300)
                _strings(value['record_ids'], 'record_ids', 4)
                for key, maximum in (('next_step', 1000), ('acceptance_criteria', 1000), ('outcome', 2000)):
                    if key in value and (not isinstance(value[key], str) or len(value[key]) > maximum):
                        raise ValueError('Invalid persisted task ' + key)
                action = args['action']
                if args.get('resolution') == 'resolved' and 'acceptance_receipt' not in value:
                    raise ValueError('Resolved closure is missing its acceptance receipt')
                if action == 'open':
                    if ident in tasks or sum(task['status'] == 'open' for task in tasks.values()) >= 2:
                        raise ValueError('Invalid task opening revision')
                    if (value['status'] != 'open' or value['title'] != args['title']
                            or 'record_ids' in args and value['record_ids'] != args['record_ids']
                            or 'record_ids' not in args and len(value['record_ids']) > 1
                            or any(value.get(key) != args.get(key) for key in
                                   ('next_step', 'acceptance_criteria', 'acceptance_checks', 'outcome'))):
                        raise ValueError('Task opening differs from its arguments')
                else:
                    if ident not in tasks or tasks[ident]['status'] != 'open':
                        raise ValueError('Task mutation requires an open task')
                    expected = {**deepcopy(tasks[ident]), 'revision': i}
                    if action == 'close':
                        if args.get('resolution') not in RESOLUTIONS or not args.get('outcome', '').strip():
                            raise ValueError('Task closure requires explicit resolution and outcome')
                        expected['status'] = 'closed'
                        if 'acceptance_receipt' in value:
                            receipt = value['acceptance_receipt']
                            evidence = args.get('acceptance_evidence', {})
                            if (args.get('resolution') != 'resolved' or not isinstance(receipt, dict)
                                    or set(receipt) != {c['id'] for c in expected.get('acceptance_checks', [])}
                                    or not receipt or any(not isinstance(v, dict)
                                        or set(v) != {'record_id', 'sha256'} or v['record_id'] != evidence.get(k)
                                        or not isinstance(v['sha256'], str)
                                        or re.fullmatch('[a-f0-9]{64}', v['sha256']) is None
                                        for k, v in receipt.items())):
                                raise ValueError('Invalid acceptance receipt')
                            expected['acceptance_receipt'] = receipt
                    elif action == 'pin':
                        expected['record_ids'] = list(dict.fromkeys([*expected['record_ids'], *args['record_ids']]))
                    for key in ('title', 'next_step', 'acceptance_criteria', 'acceptance_checks',
                                'outcome', 'resolution', 'acceptance_evidence'):
                        if key in args:
                            expected[key] = args[key]
                    if value != expected:
                        raise ValueError('Task mutation differs from its arguments or prior state')
                tasks[ident] = value
        if state['memories'] != memories or state['tasks'] != tasks:
            raise ValueError('Agent memory indexes differ from their revision archive')
        json_bytes(state)
        return deepcopy(state)

    def export(self):
        return deepcopy(self._state)

    def export_version(self):
        """A Controller version carries the evidence needed to verify its Memory."""
        return {'format': 'rsicomposer-agent-memory-version/v1', 'memory': self.export(),
                'history': deepcopy(tuple(self._history()))}

    def seed_version(self, state):
        """Import a host-approved version after the host has seeded its history."""
        if (not isinstance(state, dict) or set(state) != {'format', 'memory', 'history'}
                or state['format'] != 'rsicomposer-agent-memory-version/v1'):
            raise ValueError('AgentMemory versions require their explicit evidence envelope')
        expected = deepcopy(tuple(state['history']))
        available = {entry.record_id: entry for entry in self._history()}
        if any(available.get(entry.record_id) != entry for entry in expected):
            raise ValueError('AgentMemory version history has not been approved and seeded by the host')
        raw = state['memory']
        if self.export() == raw:
            self.restore(raw)  # Exact idempotent continuation; retain strict disk checks.
        else:
            self.seed(raw)

    def seed(self, state):
        """Host continuation adapter only; ordinary decisions use revisioned CRUD."""
        if self._state['revision'] or self.harness is None:
            raise ValueError('Memory seeding requires a fresh bound agent memory')
        candidate = self._validated(state)
        self._verify_acceptance(candidate)
        sources = self._sources()
        for note in candidate['memories'].values():
            if any(sources.get(ref) != note['provenance'][ref] for ref in note['source_refs']):
                raise ValueError('Inherited memory source is missing or differs')
        if self.path is not None:
            if self.path.exists() and self._validated(json.loads(self.path.read_text())) != self._state:
                from rsicomposer.runtime.state.recovery import RecoveryRequired
                raise RecoveryRequired('Memory changed before continuation import')
            atomic_write(self.path, json_bytes(candidate))
        self._state = candidate

    def restore(self, state):
        """Same-directory recovery verifies mutations; it never rolls them back."""
        candidate = self._validated(state)
        self._verify_acceptance(candidate)
        if self.path is not None and self.path.exists():
            current = self._validated(json.loads(self.path.read_text(encoding='utf-8')))
            if current != candidate:
                from rsicomposer.runtime.state.recovery import RecoveryRequired
                raise RecoveryRequired('Agent memory changed after checkpoint; reconcile the interrupted mutation')
        if self.path is not None:
            atomic_write(self.path, json_bytes(candidate))
        self._state = candidate

    def _history(self):
        return self._history_provider()

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
        from rsicomposer.harnesses.decision.observations import native_observation
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
        for key, length in (('query', 512), ('applicability', 1000), ('next_step', 1000),
                            ('acceptance_criteria', 1000), ('outcome', 2000)):
            if key in args and (not isinstance(args[key], str) or len(args[key]) > length):
                raise OperationArgumentError('Invalid ' + key)
        for key, limit in (('source_refs', 16), ('record_ids', 4), ('memory_ids', 16)):
            if key in args:
                _strings(args[key], key, limit)
        if 'status' in args and args['status'] not in spec.input_schema['properties']['status']['enum']:
            raise OperationArgumentError('Unknown memory status')
        if 'limit' in args and (type(args['limit']) is not int or not 1 <= args['limit'] <= 8):
            raise OperationArgumentError('limit must be in 1..8')
        if call.name == 'memory_review':
            _text(args['reason'], 'reason', 2000)
            if not args['source_refs'] or args['disposition'] not in ('deferred', 'not_useful', 'already_known'):
                raise OperationArgumentError('Memory review requires sources and a valid disposition')
            if any(not ref.startswith('history:decision_maker:') for ref in args['source_refs']):
                raise OperationArgumentError('Memory review requires exact history:decision_maker:<record_id> sources')
            if args['disposition'] == 'already_known' and not args.get('memory_ids'):
                raise OperationArgumentError('already_known requires existing memory_ids')
        if call.name == 'research_task':
            if args['action'] not in ('open', 'update', 'pin', 'close'):
                raise OperationArgumentError('Unknown task action')
            if args['action'] == 'open' and 'title' not in args:
                raise OperationArgumentError('Opening a task requires title')
            if args['action'] == 'pin' and not args.get('record_ids'):
                raise OperationArgumentError('Pin requires explicit record_ids')
            if 'acceptance_checks' in args:
                validate_checks(args['acceptance_checks'])
            if 'resolution' in args and args['resolution'] not in RESOLUTIONS:
                raise OperationArgumentError('Unknown task resolution')
            if 'acceptance_evidence' in args and (not isinstance(args['acceptance_evidence'], dict)
                    or len(args['acceptance_evidence']) > 4
                    or any(not isinstance(k, str) or not k or len(k) > 128
                           or not isinstance(v, str) or not v or len(v) > 512
                           for k, v in args['acceptance_evidence'].items())):
                raise OperationArgumentError('Acceptance evidence maps check ids to native record_ids')
            allowed = {'open': {'title', 'record_ids', 'next_step', 'acceptance_criteria', 'acceptance_checks'},
                       'update': {'title', 'next_step', 'acceptance_criteria', 'acceptance_checks'}, 'pin': {'record_ids'},
                       'close': {'outcome', 'resolution', 'acceptance_evidence'}}[args['action']]
            if set(args) - {'expected_revision', 'task_id', 'action'} - allowed:
                raise OperationArgumentError('Fields do not match the task action')
            if args['action'] == 'update' and not set(args).intersection(allowed):
                raise OperationArgumentError('Updating a task requires a progress field')
        if call.name == 'research_task_read' and 'task_id' in args and set(args) - {'task_id', 'expected_revision'}:
            raise OperationArgumentError('Read one task or list tasks, not both')
        return deepcopy(call)

    def search(self, *, query='', offset=0, limit=8, expected_revision=None):
        revision = self._state['revision']
        if (expected_revision is not None and expected_revision != revision) or (offset and expected_revision is None):
            raise OperationArgumentError('Stale or missing memory revision; restart search')
        retrieval = {}
        if self._retrieval is None:
            values = [v for v in sorted(self._state['memories'].values(), key=lambda row: row['revision'], reverse=True)
                      if query.casefold() in (v['content'] + ' ' + v['applicability']).casefold()]
        else:
            values, retrieval = self._rank(query)
        selected = values[offset:offset + limit]
        return {'revision': revision, 'items': [{**{k: deepcopy(v[k]) for k in
            ('memory_id', 'revision', 'status', 'authority')},
            'source_refs': v['source_refs'][:2], 'source_count': len(v['source_refs']),
            'content_preview': v['content'][:240], 'complete': len(v['content']) <= 240,
            'read_operation': 'memory_read'} for v in selected], 'total': len(values),
            'next_offset': offset + limit if offset + limit < len(values) else None,
            **({'retrieval': retrieval} if retrieval else {})}

    def _rank(self, query):
        values = list(self._state['memories'].values())
        scores, digest = self._retrieval.rank(query, values)
        # Empty queries browse all notes. For nonempty queries only actual hits
        # enter the view; recency never masquerades as a successful retrieval.
        if query.strip():
            values = [row for row in values if row['memory_id'] in scores or row['memory_id'] == query]
        values.sort(key=lambda row: (row['memory_id'] != query, -scores.get(row['memory_id'], 0),
                                     -row['revision'], row['memory_id']))
        return values, {'provider': 'reme', 'method': 'bm25' if query.strip() else 'recency',
            'semantic_search': False, 'model_calls': 0, 'embedding_calls': 0,
            'corpus_sha256': digest, 'matched_count': len(values),
            'usage_verified': False}

    def context(self, query):
        # Both providers use current feedback and task context; neither calls a model.
        text = query.objective
        if query.latest_feedback is not None:
            text += ' ' + encoded(query.latest_feedback.output)[:8192]
        text += ' ' + ' '.join(task['title'] + ' ' + task.get('next_step', '')
                              for task in self._state['tasks'].values() if task['status'] == 'open')
        retrieval = {}
        if self._retrieval is None:
            tokens = {token for token in re.findall(r'\w+', text.casefold()) if len(token) >= 3}
            values = sorted(self._state['memories'].values(), key=lambda row: row['revision'], reverse=True)
            values.sort(key=lambda row: len(tokens.intersection(re.findall(r'\w+',
                (row['content'] + ' ' + row['applicability']).casefold()))), reverse=True)
        else:
            values, retrieval = self._rank(text)
        return encoded({'provider': 'reme' if self._retrieval is not None else 'agent_managed',
            'revision': self._state['revision'],
            'authority': 'meta_agent_interpretations_not_verified_evidence', 'total': len(values),
            'items': [{'memory_id': row['memory_id'], 'status': row['status'],
                       'content_preview': row['content'][:300], 'complete': len(row['content']) <= 300,
                       'source_refs': row['source_refs'][:2], 'source_count': len(row['source_refs']),
                       'read_operation': 'memory_read'} for row in values[:6]],
            'write_operation': 'memory_write', 'delete_operation': 'memory_delete',
            **({'retrieval': retrieval} if retrieval else {})})

    def execute(self, call):
        call = self.validate(call)
        args = call.arguments
        if call.name == 'memory_search':
            return self.search(**args)
        if call.name == 'research_task_read':
            revision = self._state['revision']
            offset, limit = args.get('offset', 0), args.get('limit', 8)
            if (args.get('expected_revision', revision) != revision
                    or offset and 'expected_revision' not in args):
                raise OperationArgumentError('Stale or missing task revision; restart task read')
            if 'task_id' in args:
                if args['task_id'] not in self._state['tasks']:
                    raise OperationArgumentError('Unknown task_id')
                return {'revision': revision, 'task': deepcopy(self._state['tasks'][args['task_id']]),
                        'authority': 'meta_agent_task_not_verified_evidence'}
            tasks = sorted(self._state['tasks'].values(), key=lambda task: task['revision'], reverse=True)
            status = args.get('status', 'all')
            tasks = [task for task in tasks if status == 'all' or task['status'] == status]
            return {'revision': revision, 'items': [{
                        **{key: deepcopy(task[key]) for key in ('task_id', 'title', 'status', 'revision', 'record_ids')},
                        **{key: task[key][:240] for key in ('next_step', 'acceptance_criteria', 'outcome') if key in task},
                        'complete': False, 'read_operation': 'research_task_read'}
                        for task in tasks[offset:offset + limit]], 'total': len(tasks),
                    'next_offset': offset + limit if offset + limit < len(tasks) else None,
                    'authority': 'meta_agent_tasks_not_verified_evidence'}
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
        if call.name in ('memory_write', 'memory_review'):
            sources = self._sources()
            if any(ref not in sources for ref in args['source_refs']):
                raise OperationArgumentError('Unknown source reference; use exact history:decision_maker:<record_id> or result:<sha256>')
        if call.name == 'memory_review':
            if any(ident not in state['memories'] for ident in args.get('memory_ids', [])):
                raise OperationArgumentError('Review must link existing memory_ids')
        elif call.name == 'memory_write':
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
                    'revision': revision, **{key: args[key] for key in
                        ('next_step', 'acceptance_criteria', 'acceptance_checks') if key in args}}
            else:
                if ident not in tasks or tasks[ident]['status'] != 'open':
                    raise OperationArgumentError('Task is not open')
                task = tasks[ident]
                if action == 'close':
                    if not args.get('resolution') or not args.get('outcome', '').strip():
                        raise OperationArgumentError('Close requires explicit resolution and outcome; '
                            'use resolved only with receipt-checked acceptance, otherwise inconclusive/abandoned/interrupted')
                    if args['resolution'] == 'resolved':
                        task['acceptance_receipt'] = accept(task, args.get('acceptance_evidence', {}), native)
                    elif args.get('acceptance_evidence'):
                        raise OperationArgumentError('Only resolved closure may claim acceptance_evidence')
                    task['status'] = 'closed'
                elif action == 'pin':
                    pins = list(dict.fromkeys([*task['record_ids'], *args['record_ids']]))
                    if len(pins) > 4:
                        raise OperationArgumentError('A task supports at most four pinned receipts')
                    task['record_ids'] = pins
                for key in ('title', 'next_step', 'acceptance_criteria', 'acceptance_checks',
                            'outcome', 'resolution', 'acceptance_evidence'):
                    if key in args:
                        task[key] = args[key]
                task['revision'] = revision
        state['revision'] = revision
        state['revisions'].append({'revision': revision, 'operation': call.name, 'arguments': deepcopy(args),
            'memories': ({args['memory_id']: deepcopy(state['memories'].get(args['memory_id']))}
                         if call.name in ('memory_write', 'memory_delete') else {}),
            'tasks': ({args['task_id']: deepcopy(state['tasks'][args['task_id']])}
                      if call.name == 'research_task' else {})})
        if self.path is not None:
            atomic_write(self.path, json_bytes(state))
        self._state = state
        return {'revision': revision, 'operation': call.name,
            **({'task': deepcopy(state['tasks'][args['task_id']])} if call.name == 'research_task' else
               {'memory_id': args['memory_id']} if 'memory_id' in args else
               {'disposition': args['disposition'], 'source_refs': deepcopy(args['source_refs'])}), 'model_calls': 0,
            'authority': 'meta_agent_notes_not_verified_evidence'}

    def pending_review(self):
        from rsicomposer.harnesses.decision.memory_review import pending_review
        return pending_review(self)

    def project_context(self, context):
        history = self._history()
        indexed = {row.record_id: row for row in history}
        positions = {row.record_id: i for i, row in enumerate(history)}
        tasks, selected, task_reviews = [], [], []
        available = {spec.name for spec in context.operations}
        recent_closed = sorted((task for task in self._state['tasks'].values() if task['status'] == 'closed'),
                               key=lambda task: task['revision'], reverse=True)[:4]
        for task in self._state['tasks'].values():
            if task['status'] != 'open':
                continue
            if task['start_after'] is not None and task['start_after'] not in indexed:
                raise ValueError('Active task history anchor is missing')
            if any(ref not in indexed for ref in task['record_ids']):
                raise ValueError('Pinned task feedback is missing from the archive')
            start = positions.get(task['start_after'], -1)
            # Auto-retain producer feedback in addition to explicit pins.
            # Intermediate successful code/log reads cannot evict it.
            from rsicomposer.harnesses.decision.observations import task_observation
            produced = [row.record_id for row in history[start + 1:] if task_observation(row)]
            # Locate the agent's last explicit task change in real operation
            # history. This is navigation, not an implicit task mutation. An
            # imported task falls back to its rebound start anchor; imported
            # results alone never manufacture a new review obligation.
            changed_at = start
            for i, row in enumerate(history[start + 1:], start + 1):
                operation = row.details.get('operation') or {}
                arguments = operation.get('arguments', {})
                result = row.details.get('result', {})
                if (row.kind == 'operation' and operation.get('name') == 'research_task'
                        and arguments.get('task_id') == task['task_id']
                        and arguments.get('action') in ('open', 'update')
                        and type(arguments.get('expected_revision')) is int
                        and arguments['expected_revision'] < task['revision']
                        and result.get('status') == 'completed'):
                    changed_at = i
            fresh = [ref for ref in produced if not indexed[ref].details.get('continuation_origin')]
            unreviewed = [ref for ref in fresh if positions[ref] > changed_at]
            if unreviewed and 'research_task' in available:
                args = {'expected_revision': self._state['revision'], 'task_id': task['task_id']}
                actions = [
                    {'operation': 'research_task', 'arguments': {**args, 'action': 'update'},
                     'agent_authored_fields': ['next_step']},
                    {'operation': 'research_task', 'arguments': {**args, 'action': 'close'},
                     'agent_authored_fields': ['outcome', 'resolution', 'acceptance_evidence']}]
                pins = [ref for ref in unreviewed[-2:] if ref not in task['record_ids']]
                if pins and len(task['record_ids']) + len(pins) <= 4:
                    actions.insert(0, {'operation': 'research_task',
                        'arguments': {**args, 'action': 'pin', 'record_ids': pins}, 'agent_authored_fields': []})
                task_reviews.append({'task_id': task['task_id'], 'new_receipt_count': len(unreviewed),
                    'record_ids': unreviewed[-2:], 'action_templates': actions,
                    'instruction': 'Read the new result against your acceptance_criteria. Pin decisive evidence, '
                        'then update next_step or close with your outcome. A successful process alone does not '
                        'establish task resolution or improvement.'})
            refs = list(dict.fromkeys([*task['record_ids'], *produced[-2:]]))
            tasks.append({**deepcopy(task), 'resident_record_ids': refs,
                'later_producer_receipts': len(produced), 'unreviewed_producer_receipts': len(unreviewed),
                'older_results': 'research_archive'})
            selected.extend(ref for ref in refs if ref not in selected)
        pages = []
        from rsicomposer.harnesses.decision.research_state import archive_available
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
            'task_count': len(self._state['tasks']), 'task_read_operation': 'research_task_read',
            'recent_closed_tasks': [{
                'task_id': task['task_id'], 'title': task['title'], 'outcome_preview': task.get('outcome', '')[:240],
                'resolution': task['resolution'],
                'acceptance_predicates_verified': bool(task.get('acceptance_receipt')),
                'record_ids': deepcopy(task['record_ids']), 'read_operation': 'research_task_read'}
                for task in recent_closed],
            'task_operation': 'research_task', 'recent_source_refs': [
                {'record_id': row.record_id, 'source_ref': 'history:decision_maker:' + row.record_id}
                for row in history if self._native(row)][-4:]},
            'active_feedback': {'tasks': tasks, 'observations': pages,
                'lifetime': 'until_explicit_task_close', 'archive_unchanged': True}}
        if task_reviews:
            agenda['agent_memory']['task_reviews'] = task_reviews
        agenda.pop('memory_consolidation', None)
        review = self.pending_review()
        agenda['agent_memory']['memory_health'] = {key: value for key, value in review.items() if key != 'items'}
        if review['pending_count'] and {'memory_write', 'memory_review'} <= available:
            refs = ['history:decision_maker:' + row['record_id'] for row in review['items']]
            agenda['agent_memory']['memory_review'] = {
                'source_refs': refs, 'items': review['items'], 'pending_count': review['pending_count'],
                'write_arguments': {'expected_revision': self._state['revision']},
                'review_arguments': {'expected_revision': self._state['revision'], 'source_refs': refs},
                'agent_authored_fields': ['memory_id', 'content', 'source_refs', 'status', 'applicability'],
                'instruction': 'Read these original receipts and search existing memory. Before starting another '
                    'experiment, write/update source-cited reusable findings, or call memory_review with your '
                    'reason and disposition (already_known, not_useful, deferred). Deferred must explain what '
                    'evidence/next action is needed. Operational procedures, artifact paths, dependency fixes and '
                    'repeated tool mistakes are reusable knowledge too, even without a scientific gain. '
                    'A not_useful reason must consider future operational use. Do not invent notes to fill the store.',
                'authority': 'host_navigation_not_a_finding', 'required': self.require_review}
            if not self.require_review:
                agenda['agent_memory']['memory_review']['instruction'] = (
                    'Consider the original receipts and search existing memory. Write/update useful source-cited '
                    'notes or record a reasoned memory_review disposition; this assembly uses advisory review.')
            if self.require_review:
                agenda['memory_consolidation'] = {'pending_count': review['pending_count'],
                    'owner': 'acting_meta_agent', 'completion': 'source_cited_write_or_reasoned_disposition'}
                # Consolidation is a decision phase, not a separate model. Keep
                # investigation and lifecycle tools, including completion/cancel of
                # already-running jobs. Explicit deferral releases the next action.
                operations = tuple(spec for spec in context.operations if spec.read_only or spec.name in
                    {*SPEC_BY_NAME, 'research_archive', 'research_read', 'research_patch',
                     'lab_read_file', 'lab_read_log', 'lab_poll', 'lab_result', 'lab_cancel', 'reflect_begin', 'reflect_read'})
                context = replace(context, operations=operations)
        from rsicomposer.harnesses.decision.observations import failed_observation
        failures = [row for row in history if failed_observation(row)]
        tracked = {ref for task in self._state['tasks'].values() for ref in task['record_ids']} | set(selected)
        # Historical failures remain evidence, but importing an archive is not
        # a request to reopen or replay the old investigation.
        untracked = [row for row in failures if row.record_id not in tracked
                     and not row.details.get('continuation_origin')]
        agenda['agent_memory']['untracked_failure_count'] = len(untracked)
        if untracked and not task_reviews and len(tasks) < 2 and 'research_task' in available:
            row = untracked[-1]
            task_id = 'investigate-' + result_id(row.record_id)[:16]
            if task_id in self._state['tasks']:
                task_id += '-' + str(self._state['revision'])
            operation = (row.details.get('operation') or {}).get('name', 'feedback')
            agenda['agent_memory']['suggested_task_action'] = {
                'operation': 'research_task', 'arguments': {'expected_revision': self._state['revision'],
                    'task_id': task_id, 'action': 'open', 'title': 'Investigate ' + operation[:240],
                    'record_ids': [row.record_id]},
                'authority': 'host_navigation_not_a_decision',
                'instruction': 'If continuing this investigation, open this task and supply your next_step '
                               'and acceptance_criteria. The agent must choose and execute it; no task was created.'}
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
        from rsicomposer.harnesses.decision.research_state import previous_feedback_view
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
