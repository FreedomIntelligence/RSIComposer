"""Periodic reflection inside the existing meta-agent, with durable receipts.

The host schedules a review; the same decision loop reads its own raw archive,
builds an optional harness candidate and reports a change or a reasoned no-op.
This component never calls another model or trains meta-agent weights.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.decision import OperationSpec
from rsisys.loops.contracts.execution import LabToolError
from rsisys.runtime.files import atomic_write, json_bytes


class PeriodicReflection:
    state_id = 'medical-periodic-reflection/v1'

    def __init__(self, path, interval_seconds, *, clock=time.time):
        if type(interval_seconds) is not int or interval_seconds < 60:
            raise ValueError('Reflection interval must be an integer of at least 60 seconds')
        self.path, self.interval, self.clock = Path(path), interval_seconds, clock
        self.harness = self.registry = None
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state['interval_seconds'] != self.interval:
                raise ValueError('Cannot change the recovered reflection interval')
        else:
            now = clock()
            self.state = {'interval_seconds': self.interval, 'started_at': now,
                          'next_due_at': now + self.interval, 'reports': []}
            self._save(self.state)

    def _save(self, value):
        atomic_write(self.path, json_bytes(value))
        self.state = value

    def bind(self, harness, registry):
        self.harness, self.registry = harness, registry

    def configuration(self):
        return {'interval_seconds': self.interval, 'path': str(self.path),
                'scope': ['meta.prompts', 'meta.harness'], 'meta_weight_updates': False}

    def capture(self):
        if json.loads(self.path.read_text()) != self.state:
            raise RuntimeError('Reflection journal changed outside its host component')
        return deepcopy(self.state)

    def restore(self, state):
        if state != self.capture():
            from rsisys.runtime.recovery import RecoveryRequired
            raise RecoveryRequired('Reflection receipt advanced after checkpoint; reconcile, never replay')

    def snapshot(self):
        due = self.clock() >= self.state['next_due_at']
        records = self.harness.state.decision_records if self.harness is not None else {}
        return {'due': due, 'window_id': str(len(self.state['reports']) + 1),
                'next_due_at': self.state['next_due_at'], 'interval_seconds': self.interval,
                'completed_reviews': len(self.state['reports']),
                'recent_decision_ids': list(records)[-16:],
                'archive': {'operation': 'research_archive', 'collection': 'decision_graph', 'limit': 16},
                'instruction': ('Review existing decision trajectories and raw failures using research_archive. '
                    'Prioritize this review when due. Build/register/activate a meta.prompts or meta.harness '
                    'candidate only when evidence warrants it, then call reflect_report. '
                    'A no_change report is valid. Preserve tasks and explicit agent_memory. '
                    'Do not train or replace meta-agent weights. Research on Lab agent weights remains allowed.'),
                'last_report': deepcopy(self.state['reports'][-1]) if self.state['reports'] else None}

    @staticmethod
    def specifications():
        text = {'type': 'string', 'minLength': 1}
        return (OperationSpec('reflect_report',
            'Finish a scheduled trajectory review. Cite existing decision IDs; report no_change or an '
            'actually activated harness system version. This is a self-report, not proof of improvement.',
            {'type': 'object', 'properties': {'window_id': text, 'outcome': {'enum': ['no_change', 'changed']},
                'analysis': text, 'decision_ids': {'type': 'array', 'items': text, 'minItems': 1,
                    'maxItems': 32, 'uniqueItems': True}, 'system_version': text},
             'required': ['window_id', 'outcome', 'analysis', 'decision_ids', 'system_version'],
             'additionalProperties': False}),)

    def report(self, args):
        required = {'window_id', 'outcome', 'analysis', 'decision_ids', 'system_version'}
        if not isinstance(args, dict) or set(args) != required:
            raise LabToolError('Invalid reflection report fields')
        # Exact replay after a confirmed receipt is harmless; a changed report
        # with the same window is an identity conflict, not another reflection.
        previous = next((r for r in self.state['reports'] if r['window_id'] == args['window_id']), None)
        if previous is not None:
            if any(previous[key] != args[key] for key in required):
                raise LabToolError('Reflection window already has a different receipt')
            return OperationResult('Existing reflection receipt', (), False, data=deepcopy(previous))
        status = self.snapshot()
        if not status['due'] or args['window_id'] != status['window_id']:
            raise LabToolError('Reflection window is not due or has a stale identity')
        ids = args['decision_ids']
        if (not isinstance(ids, list) or not 1 <= len(ids) <= 32
                or any(not isinstance(v, str) or v not in self.harness.state.decision_records for v in ids)
                or len(set(ids)) != len(ids)):
            raise LabToolError('Cite 1..32 distinct existing decision IDs')
        if not isinstance(args['analysis'], str) or not args['analysis'].strip() or len(args['analysis']) > 16000:
            raise LabToolError('Reflection analysis must be nonempty and at most 16000 characters')
        current = self.registry.current
        if args['system_version'] != current.version or args['outcome'] not in ('no_change', 'changed'):
            raise LabToolError('Report the current actual system version and a valid outcome')
        baseline = self.state['reports'][-1]['components'] if self.state['reports'] else {}
        components = {k: asdict(v) for k, v in current.components.items()}
        changed = components != baseline
        if changed != (args['outcome'] == 'changed'):
            raise LabToolError('Outcome disagrees with host-verified component bindings')
        from rsisys.snapshot.codec import encode
        evidence = {ident: hashlib.sha256(json_bytes(encode(self.harness.state.decision_records[ident]))).hexdigest()
                    for ident in ids}
        now = self.clock()
        receipt = {**deepcopy(args), 'at': now, 'components': components,
                   'decision_record_sha256': evidence, 'authority': 'meta_agent_self_report',
                   'activation_verified': changed, 'improvement_verified': False}
        state = deepcopy(self.state)
        state['reports'].append(receipt)
        # Preserve the original cadence; missed periods never create a burst of
        # catch-up model calls or extend the research deadline after recovery.
        slot = max(1, int((now - state['started_at']) // self.interval) + 1)
        state['next_due_at'] = state['started_at'] + slot * self.interval
        self._save(state)
        return OperationResult('Reflection review recorded', (), False, data=deepcopy(receipt))
