"""Operator handoff adapter for stopped Open Lab attempts; no model calls."""
import argparse
import json
import math
from pathlib import Path
import subprocess

from rsisys.runtime.continuation import freeze_continuation
from rsisys.runtime.bundles import workspace_path
from rsisys.runtime.json_tools import result_id
from rsisys.runtime.recovery import RecoveryRequired, DurableRecovery
from .checkpoint_evaluation import digest
from rsisys.loops.decision_making.harness.memory import DecisionMemory
from rsisys.loops.decision_making.harness.agent_memory import AgentMemory
from rsisys.adapters.local_lab_state import process_identity, _read
from .open_lab_recovery import inspect_recovery, controller_lock


def continuation_limits(handoff, handoff_sha256, *, campaign_id, attempt_id,
                        deadline_epoch, max_calls, authorization=None, now=None):
    """Keep the account; an expired window needs a new explicit host mandate.

    Renewal applies only to a fresh attempt and never grants additional calls.
    The launcher freezes the authorization alongside its durable launch state.
    """
    if handoff['identity']['campaign_id'] != campaign_id:
        raise ValueError('Continuation must retain the original campaign')
    limits = handoff['limits']
    if not {'deadline_epoch', 'remaining_model_calls'} <= limits.keys():
        raise ValueError('Medical continuation requires inherited deadline and remaining call budget')
    inherited_deadline = limits['deadline_epoch']
    if authorization is not None:
        required = {'schema', 'handoff_sha256', 'campaign_id', 'target_attempt_id',
                    'previous_deadline_epoch', 'deadline_epoch', 'authorized_at', 'reason'}
        if (not isinstance(authorization, dict) or set(authorization) != required
                or authorization['schema'] != 'rsisys-host-continuation-renewal/v1'
                or authorization['handoff_sha256'] != handoff_sha256
                or authorization['campaign_id'] != campaign_id
                or authorization['target_attempt_id'] != attempt_id
                or authorization['previous_deadline_epoch'] != inherited_deadline
                or not isinstance(authorization['reason'], str) or not authorization['reason'].strip()):
            raise ValueError('Invalid explicit host continuation authorization')
        for key in ('previous_deadline_epoch', 'deadline_epoch', 'authorized_at'):
            value = authorization[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('Invalid host authorization time')
        if (now is None or not inherited_deadline < authorization['authorized_at']
                <= now < authorization['deadline_epoch']):
            raise ValueError('Renewal requires an expired window and a current host authorization')
        inherited_deadline = authorization['deadline_epoch']
    calls = min(max_calls, limits['remaining_model_calls'])
    if calls < 1:
        raise ValueError('The inherited campaign model-call budget is exhausted')
    return min(deadline_epoch, inherited_deadline), calls


def continuation_identity(*, campaign_id, run_id, source, groups, protocol_sha256, runtime_sha256):
    return {'campaign_id': campaign_id, 'attempt_id': run_id, 'axes': {
        'source': result_id(source), 'data': result_id({g['benchmark_id']: g['sha256'] for g in groups}),
        'protocol': protocol_sha256, 'runtime': runtime_sha256}}


def export_attempt(root, destination, *, artifacts=None):
    with controller_lock(Path(root), resume=True):
        return _export_attempt(root, destination, artifacts=artifacts)


def _export_attempt(root, destination, *, artifacts=None):
    """Trust only this adapter's public-operation interface, not arbitrary logs.

    The closed boundary exposes aggregate scores/status only. Arbitrary host
    files, raw closed gateway records, model request logs and live identities
    are never included. Artifact directories must be explicitly selected from
    the retained Lab workspace by the operator.
    """
    root = Path(root).resolve()
    inspected = inspect_recovery(root)
    if inspected['intervention'] or inspected['uncommitted_mutation']:
        raise RecoveryRequired('Reconcile the source attempt before exporting knowledge')
    cleanup = json.loads((root / 'controller-cleanup.json').read_text())
    application = root / 'application'
    if cleanup.get('owned_lab_closed') is not True:
        raise RecoveryRequired('Source Lab has not confirmed cleanup')
    for row in inspected['owned_processes']:
        if not row['terminal_receipt_present']:
            raise RecoveryRequired('Source process lacks a terminal receipt')
        receipt = _read(application / 'allocated-workspaces/process-receipts' / (row['job_id'] + '.json'))
        if (receipt['state']['status'] not in ('succeeded', 'failed', 'cancelled')
                or any(receipt['identity'].get(key) != value for key, value in row.items()
                       if key not in ('job_id', 'in_checkpoint', 'terminal_receipt_present'))):
            raise RecoveryRequired('Source terminal receipt identity/status mismatch')
        try:
            actual = process_identity(row['pid'])
        except (OSError, RuntimeError, subprocess.CalledProcessError):
            continue
        if all(actual.get(key) == row.get(key) for key in actual):
            raise RecoveryRequired('A source owned process is still alive')
    launch = json.loads((root / 'recovery-launch.json').read_text())
    memory = DecisionMemory(application / 'decision_memory.json')
    agent = AgentMemory._validated(json.loads((application / 'agent-memory.json').read_text()))
    # Only the Lab facade's already-public data/receipts cross this boundary.
    # Include terminal observations plus the exact closure of memory/task refs.
    allowed = {'lab_result', 'lab_poll', 'medical_evaluate', 'medical_preflight', 'medical_package'}
    records = []
    for row in memory.history():
        operation = row.details.get('operation') or {}
        result = row.details.get('result', {})
        if row.kind != 'operation' or operation.get('name') not in allowed:
            continue
        # The adapter contract must keep closed receipt fields allowlisted;
        # reject rather than guess when an unrecognized private field appears.
        fixed = result.get('data', {}).get('fixed_evaluation', {})
        if fixed.get('split') == 'closed' and set(fixed) - {
                'protocol', 'evaluation_protocol', 'protocol_sha256', 'candidate_id', 'split', 'metrics',
                'dataset_sha256', 'submission_sha256', 'record_sha256', 'adopted', 'evaluator_sha256',
                'verified', 'runtime_sha256'}:
            raise ValueError('Closed receipt contains unexpected fields')
        records.append(row.record_id)
    sources = set(records)
    referenced = AgentMemory()
    from types import SimpleNamespace
    referenced.bind(SimpleNamespace(memory=memory))
    needed = set()
    for note in agent['memories'].values():
        for ref in note['source_refs']:
            if not referenced.validates_source(ref, note['provenance'][ref]):
                raise ValueError('A memory source disappeared or changed')
            needed.add(note['provenance'][ref]['history_id'])
    history = memory.history()
    positions = {row.record_id: i for i, row in enumerate(history)}
    for task in agent['tasks'].values():
        needed.update(task['record_ids'])
        if task['start_after'] is not None:
            needed.add(task['start_after'])
        if task['status'] == 'open':
            start = positions.get(task['start_after'], -1)
            needed.update(row.record_id for row in history[start + 1:] if AgentMemory._native(row))
    if not needed <= positions.keys():
        raise ValueError('Task provenance disappeared')
    sources |= needed
    records = [row.record_id for row in history if row.record_id in sources]
    notes = [ident for ident, note in agent['memories'].items()
             if all(note['provenance'][ref]['history_id'] in sources for ref in note['source_refs'])]
    tasks = [ident for ident, task in agent['tasks'].items() if set(task['record_ids']) <= sources
             and (task['start_after'] is None or task['start_after'] in sources)]
    if len(notes) != len(agent['memories']) or len(tasks) != len(agent['tasks']):
        raise ValueError('Some memory/task sources require another explicit host adapter')
    state = DurableRecovery.read(application)
    extension = state['components']['lab_execution']['configuration']['extension']
    formal = extension['formal_evaluation']
    identity = continuation_identity(campaign_id=launch['arguments']['evaluation_campaign_id'],
        run_id=inspected['run_id'], source=launch['source'],
        groups=extension['groups'],
        protocol_sha256=digest({'protocol': formal['evaluation_protocol'],
            'source': formal['source_sha256'], 'runtime_sha256': formal['installed_runtime_sha256'],
            'closed_runtime_sha256': (formal['closed_gateway'] or {}).get('runtime_sha256'),
            'execution_budget': formal['execution_budget']}), runtime_sha256=formal['installed_runtime_sha256'])
    identity['axes']['evaluation_account'] = result_id(formal['ledger_path'])
    workspace = Path(json.loads((root / 'retained-workspace.json').read_text())['path'])
    if not workspace.resolve().is_relative_to(root):
        raise ValueError('Retained workspace is outside the attempt')
    selected = {name: workspace_path(workspace, relative) for name, relative in (artifacts or {}).items()}
    calls = json.loads((root / 'model_calls.json').read_text())['attempted']
    limits = {'source_model_calls': calls, 'remaining_model_calls': launch['arguments']['max_calls'] - calls,
              'deadline_epoch': launch['arguments']['deadline_epoch']}
    return freeze_continuation(destination, identity=identity, history=memory.history(), agent_state=agent,
        record_ids=records, memory_ids=notes, task_ids=tasks, artifacts=selected, quiescent=True, limits=limits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--artifact', action='append', default=[], metavar='NAME=WORKSPACE_RELATIVE_DIRECTORY')
    args = parser.parse_args()
    artifacts = dict(item.split('=', 1) for item in args.artifact)
    print(json.dumps(export_attempt(args.root, args.destination, artifacts=artifacts), ensure_ascii=False))


if __name__ == '__main__':
    main()
