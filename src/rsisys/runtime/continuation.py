"""Host-approved knowledge/artifact handoff between distinct run identities.

This is not recovery. No decision state, process, action, reservation or budget
is restored. Public clearance is supplied by the host adapter, never inferred
from prose or a model-provided filename. The same campaign account must remain
in force. Hashes establish integrity, not the truth of agent-authored notes.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from types import SimpleNamespace

from .bundles import tree_manifest
from .files import atomic_write, json_bytes
from .json_tools import result_id
from .recovery import RecoveryRequired


SCHEMA = 'rsisys-attempt-continuation/v1'


def validate_identity(value):
    if (not isinstance(value, dict) or set(value) != {'campaign_id', 'attempt_id', 'axes'}
            or any(not isinstance(value[k], str) or not value[k].strip() for k in ('campaign_id', 'attempt_id'))
            or not isinstance(value['axes'], dict) or not value['axes']
            or any(not isinstance(k, str) or not k or not isinstance(v, str)
                   or re.fullmatch('[a-f0-9]{64}', v) is None for k, v in value['axes'].items())):
        raise ValueError('Continuation requires campaign/attempt identity and SHA-256 compatibility axes')
    return deepcopy(value)


def _history(rows):
    from rsisys.loops.decision_making.harness.memory import DecisionMemory
    from rsisys.loops._memory import HistoryEntry
    memory = DecisionMemory()
    for value in rows:
        row = {**value, 'source_refs': tuple(value.get('source_refs', ()))}
        memory.append(HistoryEntry(**row))
    return memory


def freeze_continuation(destination, *, identity, history, agent_state, record_ids=(),
                        memory_ids=(), task_ids=(), artifacts=None, quiescent=False, limits=None):
    """Freeze only explicitly cleared public records, notes, tasks and artifacts.

    Callers must check recovery, terminal receipts and exact ownership first.
    Omitted objects are not discovered automatically. Missing cited evidence is
    an error, never silently dropped or promoted to an unsupported assertion.
    """
    from rsisys.loops.decision_making.harness.agent_memory import AgentMemory
    identity = validate_identity(identity)
    limits = _limits(limits or {})
    if quiescent is not True:
        raise RecoveryRequired('Continuation export requires verified quiescence and reconciled effects')
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError('Continuation destination must be fresh')
    state = AgentMemory._validated(agent_state)
    indexed = {row.record_id: row for row in history}
    selected = set(record_ids)
    if not selected <= indexed.keys():
        raise ValueError('Unknown public history record')
    cleared = AgentMemory()
    cleared.bind(SimpleNamespace(memory=_history([asdict(row) for row in history if row.record_id in selected])))
    notes, tasks = [], []
    for ident in memory_ids:
        note = deepcopy(state['memories'][ident])
        if any(not cleared.validates_source(ref, note['provenance'][ref]) for ref in note['source_refs']):
            raise ValueError('Memory provenance is missing or not cleared for continuation')
        notes.append(note)
    for ident in task_ids:
        task = deepcopy(state['tasks'][ident])
        if (not set(task['record_ids']) <= selected
                or not set(task.get('acceptance_evidence', {}).values()) <= selected
                or task['start_after'] is not None and task['start_after'] not in selected):
            raise ValueError('Task pins/anchor/acceptance require cleared source records')
        cleared._verify_acceptance({'tasks': {ident: task}})
        tasks.append(task)
    artifacts = artifacts or {}
    if any(not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', name) for name in artifacts):
        raise ValueError('Artifact names must be safe single path components')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.continuation-', dir=destination.parent))
    try:
        manifest = {}
        for name, source in artifacts.items():
            source = Path(source)
            if destination.resolve().is_relative_to(source.resolve()):
                raise ValueError('Continuation destination cannot be inside a source artifact')
            before = tree_manifest(source)
            target = stage / 'artifacts' / name
            target.parent.mkdir(exist_ok=True)
            shutil.copytree(source, target)
            if tree_manifest(source) != before or tree_manifest(target) != before:
                raise ValueError('Artifact changed while freezing continuation')
            manifest[name] = before
        payload = {'schema': SCHEMA, 'identity': identity,
            'history': [asdict(row) for row in history if row.record_id in selected],
            'memories': notes, 'tasks': tasks, 'artifacts': manifest,
            'limits': limits,
            'authority': 'host_cleared_inventory; agent_notes_unverified',
            'replays_actions': False, 'transfers_accounting': False}
        atomic_write(stage / 'handoff.json', json_bytes({'payload': payload, 'sha256': result_id(payload)}))
        if destination.exists():
            raise ValueError('Continuation destination already exists')
        stage.rename(destination)
        return {'path': str(destination), 'sha256': result_id(payload), 'identity': identity}
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def read_continuation(path):
    path = Path(path)
    tree_manifest(path)  # Reject symlinks and special files before reading.
    envelope = json.loads((path / 'handoff.json').read_text())
    payload = envelope['payload']
    if (set(envelope) != {'payload', 'sha256'} or envelope['sha256'] != result_id(payload)
            or payload.get('schema') != SCHEMA or payload.get('replays_actions') is not False
            or payload.get('transfers_accounting') is not False):
        raise ValueError('Invalid continuation envelope')
    validate_identity(payload['identity'])
    _limits(payload['limits'])
    expected = {'handoff.json'}
    for name, manifest in payload['artifacts'].items():
        if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', name):
            raise ValueError('Invalid artifact name')
        if tree_manifest(path / 'artifacts' / name) != manifest:
            raise ValueError('Continuation artifact digest mismatch')
        expected.update('artifacts/' + name + '/' + entry for entry in manifest)
    if set(tree_manifest(path)) != expected:
        raise ValueError('Unexpected files in continuation bundle')
    _history(payload['history'])
    return deepcopy(payload), envelope['sha256']


def _limits(value):
    if not isinstance(value, dict) or set(value) - {'deadline_epoch', 'remaining_model_calls', 'source_model_calls'}:
        raise ValueError('Invalid continuation host limits')
    if 'deadline_epoch' in value and (type(value['deadline_epoch']) not in (int, float)
            or not math.isfinite(value['deadline_epoch']) or value['deadline_epoch'] <= 0):
        raise ValueError('Invalid inherited deadline')
    for key in ('remaining_model_calls', 'source_model_calls'):
        if key in value and (type(value[key]) is not int or value[key] < 0):
            raise ValueError('Invalid inherited model call accounting')
    return deepcopy(value)


def install_continuation(path, *, identity, harness, workspace, receipt_path, resume=False):
    """Default-harness adapter, installed before the first durable checkpoint."""
    from rsisys.loops.decision_making.harness.agent_memory import AgentMemory
    from rsisys.loops.contracts.decision import OperationCall
    identity = validate_identity(identity)
    payload, sha = read_continuation(path)
    source = payload['identity']
    if source['campaign_id'] != identity['campaign_id'] or source['attempt_id'] == identity['attempt_id']:
        raise ValueError('Continuation requires another attempt in the same campaign')
    if source['axes'].get('evaluation_account') != identity['axes'].get('evaluation_account'):
        raise ValueError('Continuation must retain the shared evaluation account')
    receipt_path = Path(receipt_path)
    marker = receipt_path.with_suffix('.applying')
    if marker.exists():
        raise RecoveryRequired('Interrupted continuation import requires reconciliation; no automatic replay')
    prefix = 'prior-' + sha[:16] + ':'
    rows = []
    for row in _history(payload['history']).history():
        details = deepcopy(row.details)
        if details.get('decision_id'):
            details['decision_id'] = prefix + details['decision_id']
        details['continuation_origin'] = {'identity': source, 'record_id': row.record_id,
                                          'sha256': result_id(asdict(row))}
        rows.append(replace(row, record_id=prefix + row.record_id, task_id=None,
            source_refs=tuple(prefix + ref for ref in row.source_refs), details=details))
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['handoff_sha256'] != sha or receipt['target_identity'] != identity:
            raise RecoveryRequired('Continuation receipt identity conflict')
        current = {row.record_id: row for row in harness.memory.history()}
        if any(current.get(row.record_id) != row for row in rows):
            raise RecoveryRequired('Inherited evidence changed or disappeared')
    else:
        if resume:
            raise RecoveryRequired('Missing continuation receipt on recovery')
        workspace = Path(workspace)
        if harness.memory.history() or harness.agent_memory.export()['revision']:
            raise ValueError('Continuation import requires a fresh default harness')
        inherited = workspace / 'inherited'
        if inherited.exists() or inherited.is_symlink():
            raise ValueError('Inherited artifact destination is not fresh')
        memory = _history([asdict(row) for row in rows])
        notes = AgentMemory()
        notes.bind(SimpleNamespace(memory=memory))
        original_sources = AgentMemory()
        original_sources.bind(SimpleNamespace(memory=_history(payload['history'])))
        def mutate(name, args):
            return notes.execute(OperationCall(name, {'expected_revision': notes.export()['revision'], **args}))
        note_origins = {}
        for note in payload['memories']:
            if any(not original_sources.validates_source(ref, note['provenance'][ref])
                   for ref in note['source_refs']):
                raise ValueError('Inherited note provenance mismatch')
            mutate('memory_write', {k: note[k] for k in ('memory_id', 'content', 'status', 'applicability')} | {
                'source_refs': list(dict.fromkeys('history:decision_maker:' + prefix + note['provenance'][ref]['history_id']
                                for ref in note['source_refs']))})
            note_origins[note['memory_id']] = {'revision': note['revision'], 'sha256': result_id(note)}
        for task in payload['tasks']:
            mutate('research_task', {'task_id': task['task_id'], 'action': 'open', 'title': task['title'],
                'record_ids': [prefix + ref for ref in task['record_ids']],
                **{key: task[key] for key in ('next_step', 'acceptance_criteria', 'acceptance_checks') if key in task}})
            anchor = prefix + task['start_after'] if task['start_after'] is not None else None
            if anchor is not None and anchor not in {row.record_id for row in rows}:
                raise ValueError('Inherited task anchor is missing')
            notes._state['tasks'][task['task_id']]['start_after'] = anchor
            notes._state['revisions'][-1]['tasks'][task['task_id']]['start_after'] = anchor
            if task['status'] == 'closed':
                mutate('research_task', {'task_id': task['task_id'], 'action': 'close',
                    'resolution': task.get('resolution', 'inconclusive'),
                    'outcome': task.get('outcome') or 'Historical closure without verified acceptance.',
                    **({'acceptance_evidence': {k: prefix + ref for k, ref in task['acceptance_evidence'].items()}}
                       if task.get('resolution') == 'resolved' else {})})
        AgentMemory._validated(notes.export())
        changed = sorted(key for key in source['axes'].keys() | identity['axes'].keys()
                         if source['axes'].get(key) != identity['axes'].get(key))
        from rsisys.loops.decision_making.harness.research_state import factual_fields
        from rsisys.loops.decision_making.harness.observations import failed_observation
        outcomes = {}
        for row in rows:
            result = row.details.get('result', row.details.get('feedback', {}))
            facts = factual_fields(result)
            measured = any(key.rsplit('/', 1)[-1].endswith('metrics') and value
                           for key, value in facts.items())
            if measured or failed_observation(row):
                key = result_id(result)
                outcomes.pop(key, None)
                outcomes[key] = {'record_id': row.record_id, 'sha256': result_id(asdict(row)),
                    'measured': bool(measured), 'failed': failed_observation(row), 'facts': facts,
                    'read_path': {'operation': 'research_archive', 'collection': 'history',
                                  'record_id': row.record_id}}
        receipt = {'schema': SCHEMA, 'handoff_sha256': sha, 'source_identity': source,
            'target_identity': identity, 'changed_axes': changed,
            'source_comparison_axes_match': not any(key != 'source' for key in changed),
            'historical_scores_require_per_record_comparison': True,
            'history_record_ids': [row.record_id for row in rows], 'memory_origins': note_origins,
            'inherited_limits': deepcopy(payload['limits']),
            'outcomes': list(outcomes.values()),
            'task_ids': [task['task_id'] for task in payload['tasks']],
            'artifacts': {name: {'workspace_path': 'inherited/' + name, 'sha256': result_id(manifest)}
                          for name, manifest in payload['artifacts'].items()},
            'actions_replayed': 0, 'accounting_reset': False, 'adoption_verified': False}
        # Everything is validated before committing. A partial commit deliberately
        # leaves this marker, so another start cannot apply it as a fresh import.
        atomic_write(marker, json_bytes({'handoff_sha256': sha, 'target_identity': identity}))
        if payload['artifacts']:
            shutil.copytree(Path(path) / 'artifacts', inherited)
            for name, manifest in payload['artifacts'].items():
                if tree_manifest(inherited / name) != manifest:
                    raise ValueError('Inherited artifact copy differs')
        harness.memory.seed(rows)
        harness.agent_memory.seed(notes.export())
        atomic_write(receipt_path, json_bytes(receipt))
        marker.unlink()
    return ContinuationState(receipt_path, receipt, harness)


class ContinuationState:
    state_id = SCHEMA

    def __init__(self, path, receipt, harness):
        self.path, self.receipt = Path(path), deepcopy(receipt)
        harness.add_context_processor(self.project_context)

    def project_context(self, context):
        card = {key: deepcopy(value) for key, value in self.receipt.items()
                if key not in ('history_record_ids', 'memory_origins', 'outcomes')}
        outcomes = self.receipt['outcomes']
        keep = {row['record_id'] for row in [r for r in outcomes if r['measured']][-4:]}
        keep.update(row['record_id'] for row in [r for r in outcomes if r['failed']][-2:])
        cards = []
        for row in outcomes:
            if row['record_id'] not in keep:
                continue
            copy = deepcopy(row)
            if len(json_bytes(copy)) > 4096:
                copy.pop('facts')
                copy['facts_omitted'] = True
            cards.append(copy)
        card.update(history_count=len(self.receipt['history_record_ids']),
            memory_count=len(self.receipt['memory_origins']),
            prior_outcomes=cards,
            recent_history_record_ids=self.receipt['history_record_ids'][-8:],
            archive_read_path={'operation': 'research_archive', 'collection': 'history'},
            instruction='Inherited records are historical evidence, not live jobs. Read originals '
                'with research_archive and memory_read. Verify copied artifacts before use. '
                'Changed data/protocol/runtime requires new scores; adoption requires fresh evidence.')
        return replace(context, research_agenda={**context.research_agenda, 'continuation': card})

    def configuration(self):
        return {'receipt_sha256': result_id(self.receipt)}

    def capture(self):
        if json.loads(self.path.read_text()) != self.receipt:
            raise RecoveryRequired('Continuation receipt changed')
        return deepcopy(self.receipt)

    def restore(self, state):
        if state != self.capture():
            raise RecoveryRequired('Continuation state identity mismatch')
