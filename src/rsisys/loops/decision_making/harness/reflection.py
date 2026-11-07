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
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.runtime.files import atomic_write, json_bytes


class PeriodicReflection:
    state_id = 'rsisys-periodic-reflection/v3'

    def __init__(self, path, interval_seconds, *, clock=time.time, started_at=None,
                 max_review_seconds=3600, deadline_epoch=None):
        if type(interval_seconds) is not int or interval_seconds < 60:
            raise ValueError('Reflection interval must be an integer of at least 60 seconds')
        self.path, self.interval, self.clock = Path(path), interval_seconds, clock
        if type(max_review_seconds) is not int or not 1 <= max_review_seconds <= 3600:
            raise ValueError('Reflection time limit must be an integer in 1..3600 seconds')
        if deadline_epoch is not None and (type(deadline_epoch) not in (int, float)
                or not math.isfinite(deadline_epoch) or deadline_epoch < 0):
            raise ValueError('Reflection host deadline must be a finite epoch')
        self.max_review_seconds, self.deadline_epoch = max_review_seconds, deadline_epoch
        if started_at is not None and (type(started_at) not in (int, float)
                or not math.isfinite(started_at) or not 0 <= started_at <= clock()):
            raise ValueError('Reflection start must be a finite past host epoch')
        self.harness = self.registry = None
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if (self.state.get('max_review_seconds') != max_review_seconds
                    or self.state.get('deadline_epoch') != deadline_epoch):
                raise ValueError('Cannot change recovered reflection time limits; use explicit continuation')
            if self.state['interval_seconds'] != self.interval:
                raise ValueError('Cannot change the recovered reflection interval')
            if started_at is not None and self.state['started_at'] != started_at:
                raise ValueError('Cannot change the recovered reflection start')
        else:
            now = clock() if started_at is None else started_at
            self.state = {'interval_seconds': self.interval, 'started_at': now,
                          'next_due_at': now + self.interval, 'reports': [],
                          'max_review_seconds': max_review_seconds, 'deadline_epoch': deadline_epoch,
                          'active_review': None, 'incomplete_reviews': []}
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
        if hasattr(harness, 'tools'):
            harness.tools = ReflectionTools(harness.tools, self)

    def configuration(self):
        return {'interval_seconds': self.interval, 'path': str(self.path),
                'scope': ['meta.prompts', 'meta.harness'], 'meta_weight_updates': False,
                'scheduling': 'durable_timed_review/v4',
                'max_review_seconds': self.max_review_seconds, 'deadline_epoch': self.deadline_epoch,
                'max_review_operations_per_interval': None}

    def _review_prefix(self):
        return f"Scheduled harness reflection window {len(self.state['reports']) + 1}."

    def _ordinary_rows(self, *, snapshot=True):
        from .observations import native_observation
        start = self.state['reports'][-1].get('history_end', 0) if self.state['reports'] else 0
        end = (self.state.get('active_review') or {}).get('history_end') if snapshot else None
        return [row for row in self._history()[start:end] if native_observation(row)
            and row.kind == 'operation'
            and row.details.get('decision_id') in self.harness.state.decision_records]

    def _required_ids(self, *, snapshot=True):
        if snapshot and self.state.get('active_review'):
            return self.state['active_review']['required_decision_ids']
        from .observations import failed_observation
        rows = self._ordinary_rows(snapshot=snapshot)
        failures = [row for row in rows if failed_observation(row)]
        return list(dict.fromkeys(row.details['decision_id'] for row in
                                  [*rows[-1:], *failures]))

    def begin(self, args):
        """Explicit journalled tool: context projection never starts/resets a timer."""
        status = self.snapshot()
        if not isinstance(args, dict) or args != {'window_id': status['window_id']}:
            raise LabToolError('Begin the current reflection window')
        active = self.state.get('active_review')
        if active and not status['timed_out']:
            return OperationResult('Existing reflection session', (), False, data=deepcopy(active))
        if not status['due']:
            raise LabToolError('Reflection is not due or host deadline has passed')
        state = deepcopy(self.state)
        if active:
            state['incomplete_reviews'].append({**active, 'status': 'timed_out',
                'improvement_verified': False})
            state['active_review'] = None
        # Include unfinished backlog from the last completed review. Never
        # advance history_end on timeout or turn unfinished work into no_change.
        required = self._required_ids(snapshot=False)
        if not required:
            raise LabToolError('No unreviewed native decisions are available')
        now = self.clock()
        deadline = min(now + self.max_review_seconds, self.deadline_epoch or float('inf'))
        state['active_review'] = {'window_id': status['window_id'], 'started_at': now,
            'deadline_at': deadline, 'history_end': len(self._history()),
            'required_decision_ids': required}
        self._save(state)
        return OperationResult('Reflection session started', (), False, data=deepcopy(state['active_review']))

    def _history(self):
        return self.harness.memory.history() if self.harness is not None else ()

    def _review_evidence(self):
        return self._review_progress()['read_decision_ids']

    def _review_progress(self):
        """Reassemble only real, complete raw operation pages from this window."""
        from rsisys.loops.decision_making.harness.context import encoded
        history = self._history()
        start = self.state['reports'][-1].get('history_end', 0) if self.state['reports'] else 0
        originals = {row.record_id: row for row in history if row.kind == 'operation'}
        pages, continuations = {}, {}
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
                if args.get('record_id') == item['record_id'] and data.get('next_cursor'):
                    end = max(v['chunk'] for v in data['items']) + 1
                    continuations.setdefault(key, {})[end] = {
                        'cursor': data['next_cursor'], 'evidence_path': args.get('evidence_path')}
        evidence, progress = {}, {}
        for key, parts in pages.items():
            _, ident, chunks = key
            original = originals.get(ident)
            if original is None:
                continue
            raw = encoded(asdict(original))
            if chunks != max(1, (len(raw) + 999) // 1000):
                continue
            # A cursor is guidance only. Never count stale or partial-field
            # content as a read of the full immutable operation record.
            valid = {n for n, text in parts.items()
                     if 0 <= n < chunks and text == raw[n * 1000:(n + 1) * 1000]}
            prefix = 0
            while prefix in valid:
                prefix += 1
            continuation = continuations.get(key, {}).get(prefix)
            if continuation and prefix > progress.get(ident, {}).get('chunks_read', 0):
                progress[ident] = {'chunks_read': prefix, **continuation}
            if valid != set(range(chunks)):
                continue
            operation = original.details.get('operation') or {}
            decision_id = original.details.get('decision_id')
            if (operation.get('name') not in ('research_archive', 'reflect_report')
                    and decision_id in self.harness.state.decision_records):
                evidence[decision_id] = {'record_id': ident, 'sha256': hashlib.sha256(raw.encode()).hexdigest()}
        required = self._required_ids()
        required_rows = [row for row in self._ordinary_rows() if row.details['decision_id'] in required]
        reads = []
        for row in required_rows:
            ident = row.details['decision_id']
            if ident in evidence:
                continue
            pending = progress.get(row.record_id, {})
            reads.append({'decision_id': ident, 'record_id': row.record_id,
                'chunks_read': pending.get('chunks_read', 0),
                'operation': 'research_archive', 'arguments': {
                    'collection': 'history', 'record_id': row.record_id,
                    'evidence_path': pending.get('evidence_path'),
                    'cursor': pending.get('cursor'), 'limit': 8}})
        return {'read_decision_ids': evidence, 'required_decision_ids': required,
            'missing_decision_ids': [ident for ident in required if ident not in evidence],
            'next_reads': reads, 'report_ready': bool(evidence) and all(ident in evidence for ident in required),
            'instruction': 'Read the entire operation record with evidence_path=null, not /details/result. '
                'Copy the next read arguments and follow the returned cursor. '
                'Partial-field reads remain useful feedback but do not satisfy reflection evidence.'}

    def project_context(self, context):
        """A due review owns decision turns; running Lab jobs are left running."""
        status = self.snapshot()
        # Keep the authorized archive while awaiting begin. Removing it makes
        # recoverable historical evidence irreducible to the context compactor.
        if status['timed_out'] and status['due']:
            return replace(context, objective=self._review_prefix() + ' Resume the unfinished backlog with reflect_begin. ' + context.objective,
                operations=tuple(spec for spec in context.operations if spec.name == 'reflect_begin'
                    or spec.name == 'research_archive' and spec.read_only),
                directive_modes=(), research_agenda={**context.research_agenda, 'scheduled_reflection': {
                    'window_id': status['window_id'], 'begin_arguments': {'window_id': status['window_id']}}})
        if (not status['due'] and not status['timed_out']) or not status['required_decision_ids']:
            return context
        active = self.state.get('active_review')
        if status['timed_out']:
            return replace(context, operations=tuple(spec for spec in context.operations
                if spec.name not in ('reflect_begin', 'reflect_report')),
                research_agenda={**context.research_agenda, 'reflection_attention': {
                    'status': 'incomplete_time_limit', 'window_id': status['window_id'],
                    'deadline_at': active['deadline_at'], 'retry_at': status['retry_at'],
                    'improvement_verified': False}})
        prefix = self._review_prefix()
        if active is None:
            return replace(context, objective=prefix + ' Start the durable review with reflect_begin. ' + context.objective,
                operations=tuple(spec for spec in context.operations if spec.name == 'reflect_begin'
                    or spec.name == 'research_archive' and spec.read_only),
                directive_modes=(), research_agenda={**context.research_agenda, 'scheduled_reflection': {
                    'window_id': status['window_id'], 'begin_arguments': {'window_id': status['window_id']},
                    'max_review_seconds': self.max_review_seconds}})
        progress = self._review_progress()
        evidence = progress['read_decision_ids']
        # Preserve all capabilities already authorized/projected by the host.
        # Reading, building, executing and testing can interleave; only the
        # completion report requires full evidence coverage.
        allowed = {spec.name for spec in context.operations} - {'reflect_begin', 'reflect_report'}
        if progress['report_ready']:
            allowed.add('reflect_report')
        operations = []
        for spec in context.operations:
            if spec.name not in allowed:
                continue
            if spec.name == 'reflect_report':
                schema = deepcopy(spec.input_schema)
                schema['properties']['window_id'] = {'const': status['window_id']}
                schema['properties']['system_version'] = {'const': self.registry.current.version}
                schema['properties']['decision_ids']['items'] = {'enum': list(evidence)}
                schema['properties']['decision_ids']['minItems'] = max(1, len(progress['required_decision_ids']))
                schema['properties']['decision_ids']['maxItems'] = len(evidence)
                spec = replace(spec, input_schema=schema)
            operations.append(spec)
        if not operations:
            raise RuntimeError('Scheduled reflection requires authorized research_archive/reflect_report operations')
        agenda = {**context.research_agenda, 'scheduled_reflection': {
            'window_id': status['window_id'], 'raw_feedback_read': evidence,
            'required_decision_ids': self._required_ids(),
            'required_raw_record_ids': status['required_raw_record_ids'],
            'evidence_progress': progress,
            'next_read': progress['next_reads'][0] if progress['next_reads'] else None,
            'report_arguments': {'window_id': status['window_id'],
                'decision_ids': progress['required_decision_ids'] or list(evidence)[-1:],
                'system_version': self.registry.current.version},
            'agent_authored_report_fields': ['analysis', 'outcome'],
            'deadline_at': active['deadline_at'],
            'seconds_remaining': max(0, active['deadline_at'] - self.clock()),
            'operations_remaining': None,
            'ordinary_research_resumes_after_report': True}}
        return replace(context, objective=prefix + ' Read complete raw operation history using research_archive, '
            'with evidence_path=null, following the supplied next_read and next_cursor until every chunk is read. '
            'A /details/result read is not a full operation record. Analyze the real trajectory and failures; '
            'optionally change meta.prompts/meta.harness, then submit reflect_report citing read decision IDs. '
            'Cite the required recent decision and latest failure, not just early preparation. '
            'Use distinct decision IDs. A reasoned no_change is valid. '
            'Existing Lab jobs continue independently. Research objective after this review: ' + context.objective,
            operations=tuple(operations), directive_modes=(), research_agenda=agenda)

    def validate_decision(self, decision, context):
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
        now = self.clock()
        active = self.state.get('active_review')
        timed_out = bool(active and now >= active['deadline_at'])
        retry_at = (self.state['started_at'] + (int((active['deadline_at'] - self.state['started_at'])
                    // self.interval) + 1) * self.interval) if timed_out else None
        due = now >= (retry_at if timed_out else self.state['next_due_at'])
        if self.deadline_epoch is not None and now >= self.deadline_epoch:
            due = False
        records = self.harness.state.decision_records if self.harness is not None else {}
        required = self._required_ids() if self.harness is not None else []
        return {'due': due, 'timed_out': timed_out, 'retry_at': retry_at,
                'active_review': deepcopy(active), 'window_id': str(len(self.state['reports']) + 1),
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
        return (OperationSpec('reflect_begin', 'Start or resume a durable reflection session. No operation-count cap; '
            'the host fixes a wall deadline and a snapshot of the unreviewed evidence. Retries do not reset time.',
            {'type': 'object', 'properties': {'window_id': text}, 'required': ['window_id'],
             'additionalProperties': False}), OperationSpec('reflect_report',
            'Finish a scheduled trajectory review. Cite existing decision IDs; report no_change or an '
            'actually activated harness system version. This is a self-report, not proof of improvement.',
            {'type': 'object', 'properties': {'window_id': text, 'outcome': {'enum': ['no_change', 'changed']},
                'analysis': text, 'decision_ids': {'type': 'array', 'items': text, 'minItems': 1,
                    'uniqueItems': True}, 'system_version': text},
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
        active = self.state.get('active_review')
        if active is None or status['timed_out']:
            raise LabToolError('Start reflection with reflect_begin; expired sessions cannot report success')
        memory = getattr(self.harness, 'agent_memory', None)
        if memory is not None and memory.require_review and memory.pending_review()['pending_count']:
            raise LabToolError('Review pending Memory evidence before completing reflection')
        progress = self._review_progress()
        ids = args['decision_ids']
        if (not isinstance(ids, list) or not ids
                or any(not isinstance(v, str) or v not in self.harness.state.decision_records for v in ids)
                or len(set(ids)) != len(ids)):
            raise LabToolError('Cite distinct existing decision IDs',
                data={'reason': 'invalid_decision_citations', 'required_decision_ids': self._required_ids(),
                      'read_decision_ids': list(progress['read_decision_ids']), 'duplicates_allowed': False,
                      'suggested_decision_ids': progress['required_decision_ids'],
                      'next_reads': progress['next_reads']})
        archive_evidence = progress['read_decision_ids']
        if any(ident not in archive_evidence for ident in ids):
            raise LabToolError('Read complete raw operation history pages for every cited decision before reporting',
                data={'reason': 'raw_evidence_incomplete', **progress})
        missing = [ident for ident in self._required_ids() if ident not in ids]
        if missing:
            raise LabToolError('Reflection must cover the recent trajectory and latest failure',
                data={'reason': 'recent_evidence_missing', 'required_decision_ids': self._required_ids(),
                      'missing_decision_ids': missing, 'next_reads': progress['next_reads'],
                      'suggested_decision_ids': progress['required_decision_ids']})
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
                   'coverage': 'latest_ordinary_operation_and_all_failures_in_review_snapshot',
                   'review_started_at': active['started_at'], 'review_deadline_at': active['deadline_at'],
                   'history_end': active['history_end']}
        state = deepcopy(self.state)
        state['reports'].append(receipt)
        state['active_review'] = None
        # Preserve the original cadence; missed periods never create a burst of
        # catch-up model calls or extend the research deadline after recovery.
        slot = max(1, int((now - state['started_at']) // self.interval) + 1)
        state['next_due_at'] = state['started_at'] + slot * self.interval
        self._save(state)
        return OperationResult('Reflection review recorded', (), False, data=deepcopy(receipt))


class ReflectionTools:
    """Reject a late review action as a normal recorded result, before effects.

Raising a parser error on every expired response would instead exhaust schema
retries and fail the whole run. The original decision remains in the journal;
the next decision gets the ordinary context. No synthetic agent action is made.
"""
    def __init__(self, base, reflection):
        self.base, self.reflection = base, reflection

    def __getattr__(self, name):
        return getattr(self.base, name)

    def execute(self, call):
        reflection = self.reflection
        state = reflection.harness.state
        ident = getattr(state, 'pending_operation_decision_id', None)
        decision = state.decision_records.get(ident, {})
        active = reflection.state.get('active_review')
        if (active and reflection.clock() >= active['deadline_at']
                and decision.get('objective', '').startswith('Scheduled harness reflection window ')
                and not (call.name == 'reflect_begin' and reflection.snapshot()['due'])):
            return OperationResult('Reflection time limit reached; action was not executed', (), False,
                ExecutionStatus.CANCELLED, 'reflection_time_limit', data={
                    'deadline_at': active['deadline_at'], 'executed': False,
                    'improvement_verified': False, 'next_action': 'continue_with_fresh_host_context'})
        return self.base.execute(call)
