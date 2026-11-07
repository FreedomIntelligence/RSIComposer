"""Periodic reflection inside the existing meta-agent, with durable receipts.

The host schedules a review; the same decision loop reads its own raw archive,
builds an optional harness candidate and reports a change or a reasoned no-op.
This component never calls another model or trains meta-agent weights.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time

from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.decision import OperationSpec
from rsisys.loops.contracts.execution import LabToolError
from rsisys.runtime.files import atomic_write, json_bytes


class PeriodicReflection:
    state_id = 'rsisys-periodic-reflection/v2'

    def __init__(self, path, interval_seconds, *, clock=time.time, started_at=None,
                 max_review_operations=16):
        if type(interval_seconds) is not int or interval_seconds < 60:
            raise ValueError('Reflection interval must be an integer of at least 60 seconds')
        self.path, self.interval, self.clock = Path(path), interval_seconds, clock
        if type(max_review_operations) is not int or max_review_operations < 2:
            raise ValueError('Reflection operation budget must be an integer >=2')
        self.max_review_operations = max_review_operations
        if started_at is not None and (type(started_at) not in (int, float)
                or not math.isfinite(started_at) or not 0 <= started_at <= clock()):
            raise ValueError('Reflection start must be a finite past host epoch')
        self.harness = self.registry = None
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state['interval_seconds'] != self.interval:
                raise ValueError('Cannot change the recovered reflection interval')
            if started_at is not None and self.state['started_at'] != started_at:
                raise ValueError('Cannot change the recovered reflection start')
        else:
            now = clock() if started_at is None else started_at
            self.state = {'interval_seconds': self.interval, 'started_at': now,
                          'next_due_at': now + self.interval, 'reports': []}
            self._save(self.state)

    def _save(self, value):
        atomic_write(self.path, json_bytes(value))
        self.state = value

    def bind(self, harness, registry):
        self.harness, self.registry = harness, registry
        if 'initial_components' not in self.state:
            self._save({**self.state, 'initial_components':
                        {k: asdict(v) for k, v in registry.current.components.items()}})
        harness.add_context_processor(self.project_context)

    def configuration(self):
        return {'interval_seconds': self.interval, 'path': str(self.path),
                'scope': ['meta.prompts', 'meta.harness'], 'meta_weight_updates': False,
                'scheduling': 'recent_archive_then_report/v2',
                'max_review_operations_per_interval': self.max_review_operations}

    def _review_prefix(self):
        return f"Scheduled harness reflection window {len(self.state['reports']) + 1}."

    def _ordinary_rows(self):
        from .observations import native_observation
        start = self.state['reports'][-1].get('history_end', 0) if self.state['reports'] else 0
        return [row for row in self._history()[start:] if native_observation(row)
            and row.kind == 'operation'
            and row.details.get('decision_id') in self.harness.state.decision_records
            and not self.harness.state.decision_records[row.details['decision_id']].get(
                'objective', '').startswith('Scheduled harness reflection window ')]

    def _required_ids(self):
        from .observations import failed_observation
        rows = self._ordinary_rows()
        failures = [row for row in rows if failed_observation(row)]
        return list(dict.fromkeys(row.details['decision_id'] for row in
                                  [*rows[-1:], *failures[-1:]]))

    def _used(self):
        slot = max(1, int((self.clock() - self.state['started_at']) // self.interval))
        marker = f' Review slot {slot}.'
        return sum(row.kind == 'operation' and (
            objective := self.harness.state.decision_records.get(
                row.details.get('decision_id'), {}).get('objective', '')).startswith(self._review_prefix())
            and (' Review slot ' not in objective or marker in objective)
            for row in self._history()), slot

    def _history(self):
        return self.harness.memory.history() if self.harness is not None else ()

    def _review_evidence(self):
        """Reassemble only real, complete raw operation pages from this window."""
        from rsisys.loops.decision_making.harness.context import encoded
        history = self._history()
        start = self.state['reports'][-1].get('history_end', 0) if self.state['reports'] else 0
        originals = {row.record_id: row for row in history if row.kind == 'operation'}
        pages = {}
        for row in history[start:]:
            if row.kind != 'operation':
                continue
            operation = row.details.get('operation') or {}
            result = row.details.get('result', {})
            args, data = operation.get('arguments', {}), result.get('data', {})
            if (operation.get('name') != 'research_archive' or result.get('status') != 'completed'
                    or args.get('collection') != 'history' or args.get('evidence_path') not in (None, '')):
                continue
            for item in data.get('items', []):
                key = (data.get('revision'), item['record_id'], item['chunks'])
                pages.setdefault(key, {})[item['chunk']] = item['text']
        evidence = {}
        for (_, ident, chunks), parts in pages.items():
            original = originals.get(ident)
            if original is None or set(parts) != set(range(chunks)):
                continue
            raw = ''.join(parts[n] for n in range(chunks))
            if raw != encoded(asdict(original)):
                continue
            operation = original.details.get('operation') or {}
            decision_id = original.details.get('decision_id')
            if (operation.get('name') not in ('research_archive', 'reflect_report')
                    and decision_id in self.harness.state.decision_records):
                evidence[decision_id] = {'record_id': ident, 'sha256': hashlib.sha256(raw.encode()).hexdigest()}
        return evidence

    def project_context(self, context):
        """A due review owns decision turns; running Lab jobs are left running."""
        status = self.snapshot()
        if not status['due'] or not status['recent_decision_ids']:
            return context
        used, slot = self._used()
        if used >= self.max_review_operations:
            return replace(context, research_agenda={**context.research_agenda,
                'reflection_attention': {'status': 'incomplete_budget_exhausted',
                    'window_id': status['window_id'], 'review_slot': slot,
                    'operations_used': used, 'improvement_verified': False,
                    'retry_at': self.state['started_at'] + (slot + 1) * self.interval}})
        evidence = self._review_evidence()
        prefix = self._review_prefix() + f' Review slot {slot}.'
        allowed = {'research_archive'}
        if evidence:
            allowed |= {'reflect_report', 'research_read', 'research_patch', 'research_task', 'research_task_read',
                'memory_search', 'memory_read', 'memory_write', 'memory_delete',
                'lab_read_file', 'lab_read_log', 'lab_write_file', 'system_status',
                'system_register', 'system_activate'}
            if used >= 8:
                # Archive access is also the compactor's guarantee that old
                # receipts remain recoverable. Keep it while requiring a report;
                # hiding it makes every historical failure irreducibly resident
                # and prevents recovery from an invalid report citation.
                allowed = {'research_archive', 'reflect_report'}
        operations = []
        for spec in context.operations:
            if spec.name not in allowed:
                continue
            if spec.name == 'reflect_report':
                schema = deepcopy(spec.input_schema)
                schema['properties']['window_id'] = {'const': status['window_id']}
                schema['properties']['system_version'] = {'const': self.registry.current.version}
                schema['properties']['decision_ids']['items'] = {'enum': list(evidence)}
                spec = replace(spec, input_schema=schema)
            operations.append(spec)
        if not operations:
            raise RuntimeError('Scheduled reflection requires authorized research_archive/reflect_report operations')
        agenda = {**context.research_agenda, 'scheduled_reflection': {
            'window_id': status['window_id'], 'raw_feedback_read': evidence,
            'required_decision_ids': self._required_ids(),
            'required_raw_record_ids': status['required_raw_record_ids'],
            'operations_remaining': self.max_review_operations - used,
            'ordinary_research_resumes_after_report': True}}
        return replace(context, objective=prefix + ' Read complete raw operation history using research_archive, '
            'following next_cursor until every chunk is read. Analyze the real trajectory and failures; '
            'optionally change meta.prompts/meta.harness, then submit reflect_report citing read decision IDs. '
            'Cite the required recent decision and latest failure, not just early preparation. '
            'Use distinct decision IDs. A reasoned no_change is valid. '
            'Existing Lab jobs continue independently. Research objective after this review: ' + context.objective,
            operations=tuple(operations), directive_modes=(), research_agenda=agenda)

    @staticmethod
    def validate_decision(decision, context):
        if context.research_agenda.get('scheduled_reflection') and (decision.kind != 'invoke'
                or decision.operation.name not in {spec.name for spec in context.operations}):
            raise ValueError('Finish the scheduled reflection with an available review operation before ordinary research')

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
        required = self._required_ids() if self.harness is not None else []
        return {'due': due, 'window_id': str(len(self.state['reports']) + 1),
                'next_due_at': self.state['next_due_at'], 'interval_seconds': self.interval,
                'completed_reviews': len(self.state['reports']),
                'recent_decision_ids': list(records)[-16:],
                'required_decision_ids': required,
                'required_raw_record_ids': [row.record_id for row in self._ordinary_rows()
                    if row.details['decision_id'] in required] if self.harness is not None else [],
                'archive': {'operation': 'research_archive', 'collection': 'history', 'limit': 8},
                'raw_feedback_record_ids': [row.record_id for row in self._history() if row.kind == 'operation'
                    and (row.details.get('operation') or {}).get('name') not in ('research_archive', 'reflect_report')][-8:],
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
            raise LabToolError('Cite 1..32 distinct existing decision IDs',
                data={'reason': 'invalid_decision_citations', 'required_decision_ids': self._required_ids(),
                      'read_decision_ids': list(self._review_evidence()), 'duplicates_allowed': False})
        archive_evidence = self._review_evidence()
        if any(ident not in archive_evidence for ident in ids):
            raise LabToolError('Read complete raw operation history pages for every cited decision before reporting')
        missing = [ident for ident in self._required_ids() if ident not in ids]
        if missing:
            raise LabToolError('Reflection must cover the recent trajectory and latest failure',
                data={'reason': 'recent_evidence_missing', 'required_decision_ids': self._required_ids(),
                      'missing_decision_ids': missing})
        if not isinstance(args['analysis'], str) or not args['analysis'].strip() or len(args['analysis']) > 16000:
            raise LabToolError('Reflection analysis must be nonempty and at most 16000 characters')
        current = self.registry.current
        if args['system_version'] != current.version or args['outcome'] not in ('no_change', 'changed'):
            raise LabToolError('Report the current actual system version and a valid outcome')
        baseline = (self.state['reports'][-1]['components'] if self.state['reports']
                    else self.state['initial_components'])
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
                   'activation_verified': changed, 'improvement_verified': False,
                   'archive_evidence': {ident: archive_evidence[ident] for ident in ids},
                   'required_decision_ids': self._required_ids(),
                   'coverage': 'latest_ordinary_operation_and_latest_failure_since_previous_report',
                   'history_end': len(self._history())}
        state = deepcopy(self.state)
        state['reports'].append(receipt)
        # Preserve the original cadence; missed periods never create a burst of
        # catch-up model calls or extend the research deadline after recovery.
        slot = max(1, int((now - state['started_at']) // self.interval) + 1)
        state['next_due_at'] = state['started_at'] + slot * self.interval
        self._save(state)
        return OperationResult('Reflection review recorded', (), False, data=deepcopy(receipt))
