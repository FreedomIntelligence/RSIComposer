"""Single-writer feedback journal, job scheduling and product delivery.

Workers return proposals only. The owning thread journals receipts and publishes
memory; model output never authorizes training, edits, or candidate activation.
"""
import json
from concurrent.futures import ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from threading import get_ident

from rsicomposer.harnesses.decision.feedback.archive import (
    ArchiveBinding,
    captured_archive_event,
    validate_archive,
)
from rsicomposer.harnesses.decision.feedback.attempts import (
    apply_attempt_updates,
    attempt_view,
)
from rsicomposer.harnesses.decision.feedback.components import JournalTrajectoryArchive
from rsicomposer.harnesses.decision.feedback.context_memory import (
    ContextMemoryBinding,
    feedback_context_event,
    validate_context_memory,
)
from rsicomposer.harnesses.decision.feedback.delivery import producer_delivery
from rsicomposer.harnesses.decision.feedback.evidence_packet import evidence_packet
from rsicomposer.harnesses.decision.feedback.model_gateway import SerializedLearningBackend
from rsicomposer.harnesses.decision.feedback.proposal_validation import checked_processor_result, validate_publication
from rsicomposer.harnesses.decision.feedback.scheduling import (
    FifoFeedbackScheduler,
    validate_scheduler,
)
from rsicomposer.harnesses.decision.feedback.stages import (
    FeedbackStageBinding,
    component_configuration,
)
from rsicomposer.loops.contracts.feedback_context import ContextQuery
from rsicomposer.loops.contracts.feedback_modules import (
    FeedbackFailurePolicy,
    PendingFeedback,
)
from rsicomposer.loops.contracts.model_service import ModelRef
from rsicomposer.loops.contracts.receipts import ExecutionReceipt
from rsicomposer.runtime.storage.files import atomic_write, json_bytes
from rsicomposer.runtime.storage.json_tools import (
    EvidencePathError,
    result_id,
)

from .reader import FeedbackReader

_DEFAULT_COMPONENT = object()


def validate_journal_captures(journal):
    rows = journal['jobs']
    if any(not isinstance(row, dict) or row.get('job_id') != ident
            or type(row.get('capture_index')) is not int
            or not isinstance(row.get('raw_input'), dict)
            or type(row.get('capture_archive_only')) is not bool
            for ident, row in rows.items()) or sorted(row['capture_index'] for row in rows.values()) != list(range(len(rows))):
        raise ValueError('Invalid learning journal capture fields')


def bind_feedback_processor(processor):
    # Pipeline is host composition, not a second external processor protocol.
    from rsicomposer.harnesses.decision.feedback.pipeline import FeedbackPipeline
    return processor if isinstance(processor, FeedbackPipeline) else FeedbackStageBinding(processor)


class FeedbackLearning(FeedbackReader):
    """Single-writer journal plus one background inference worker.

    Reload preserves completed outputs and marks uncertain in-flight work
    interrupted; it never repeats that request automatically. Queued jobs have not
    called the model and may be dispatched after reopening. This journal is not a
    replacement for the Runner's own snapshot/recovery contract.
    """
    def __init__(self, backend: SerializedLearningBackend | None, *, path=None, max_calls=64,
                 max_pending=16, max_input_tokens=16000, max_output_tokens=4096,
                 max_input_bytes=64000, executor=None,
                 context_memory, trajectory_archive=_DEFAULT_COMPONENT,
                 processor, assembly_configuration=None, scheduler=None,
                 max_concurrent_jobs=1, failure_policy=None, admission=None):
        if processor is None:
            raise TypeError('Feedback learning requires a processor, e.g. a FeedbackPipeline')
        binding = bind_feedback_processor(processor)
        if backend is None and binding.max_calls > 0:
            raise ValueError('Feedback inference requires a model backend')
        if backend is not None and not isinstance(backend, SerializedLearningBackend):
            raise TypeError('Use a shared SerializedLearningBackend for decision and learning calls')
        if trajectory_archive is _DEFAULT_COMPONENT:
            trajectory_archive = JournalTrajectoryArchive()
        validate_context_memory(context_memory)
        validate_archive(trajectory_archive)
        self._context_memory, self._trajectory_archive = context_memory, trajectory_archive
        self.context_observer = None  # host write-ahead delivery hook for rebuildable Memory
        self._processor_binding = binding
        self.processor = processor
        processor_config = self._processor_binding.configuration()
        self.scheduler = FifoFeedbackScheduler() if scheduler is None else scheduler
        validate_scheduler(self.scheduler)
        if admission is not None and not callable(getattr(admission, 'admit', None)):
            raise TypeError('Feedback admission requires admit(event, pending)')
        self.admission = admission
        if type(max_concurrent_jobs) is not int or max_concurrent_jobs < 1:
            raise ValueError('Host concurrency must be a positive integer')
        self.max_concurrent_jobs = max_concurrent_jobs
        self.failure_policy = FeedbackFailurePolicy() if failure_policy is None else failure_policy
        if not isinstance(self.failure_policy, FeedbackFailurePolicy):
            raise TypeError('failure_policy must be a FeedbackFailurePolicy')
        runtime_config = {'scheduler': component_configuration(self.scheduler),
                          'max_concurrent_jobs': max_concurrent_jobs,
                          'failure_policy': asdict(self.failure_policy)}
        if admission is not None:
            runtime_config['admission'] = component_configuration(admission)
        self.backend, self.path = backend, Path(path) if path is not None else None
        self.limits = dict(max_calls=max_calls, max_pending=max_pending, max_input_tokens=max_input_tokens,
                           max_output_tokens=max_output_tokens, max_input_bytes=max_input_bytes)
        if any(type(v) is not int or v < 1 for v in self.limits.values()):
            raise ValueError('Learning limits must be positive integers')
        self._owner, self._closed = get_ident(), False
        self._executor = executor
        self._owns_executor = executor is None
        self._futures = {}
        self.state = {'version': 5, 'limits': self.limits, 'attempted': 0, 'jobs': {}}
        self.state['runtime'] = runtime_config
        if assembly_configuration is not None:
            json_bytes(assembly_configuration)
            self.state['assembly'] = deepcopy(assembly_configuration)
        if processor_config is not None:
            self.state['processor'] = processor_config
        if self.path is not None and self.path.exists():
            loaded = json.loads(self.path.read_text())
            if (loaded.get('version') != 5 or loaded.get('limits') != self.limits
                    or loaded.get('processor') != processor_config
                    or loaded.get('runtime') != runtime_config
                    or loaded.get('assembly') != assembly_configuration
                    or not isinstance(loaded.get('jobs'), dict)
                    or type(loaded.get('attempted')) is not int or loaded['attempted'] < 0):
                raise ValueError('Invalid learning journal or changed limits')
            validate_journal_captures(loaded)
            self.state = loaded
            recovered = deepcopy(loaded)
            for row in sorted(recovered['jobs'].values(), key=lambda r: r['capture_index']):
                for name in ('context_delivery', 'archive_delivery', 'publication_delivery'):
                    delivery = row.get(name, {})
                    if delivery.get('status') == 'running':
                        row[name] = {**delivery, 'status': 'interrupted',
                            'error': 'Delivery outcome unknown; no automatic replay'}
                if row['status'] == 'running':
                    try:
                        result = self._processor_binding.recover_job(row, path=self._exchange_path(row))
                    except Exception as exc:
                        row.update(status='interrupted', error=f'Recovery failed: {type(exc).__name__}: {exc}; '
                                   'reservation retained; no automatic retry')
                        continue
                    if result is None:
                        row.update(status='interrupted', error='Model request outcome unknown; no automatic retry')
                    else:
                        self._settle_processor(recovered, row, result)
                        self._finalize_job(recovered, row)
            if recovered != loaded:
                self._commit(recovered)

    def _assert_owner(self):
        if get_ident() != self._owner:
            raise RuntimeError('Only the owning thread may mutate the learning journal')

    def _commit(self, state):
        self._assert_owner()
        if self.path is not None:
            atomic_write(self.path, json_bytes(state))
        self.state = state

    def capture_receipt(self, observation, model, history=(), *, evaluations=(), research_context=None,
                        grouping_context=None, archive_only=False):
        self._assert_owner()
        if self._closed:
            raise RuntimeError('Feedback learning is closed')
        if not isinstance(model, ModelRef) or not isinstance(observation, ExecutionReceipt):
            raise ValueError('Learning requires a decision model and execution receipt')
        if type(archive_only) is not bool:
            raise TypeError('archive_only must be a bool')
        observation.__post_init__()
        receipt = observation.payload()
        digest = result_id(receipt)
        prefix = {'operation': 'learning-operation-', 'lab': 'learning-lab-'}[observation.source]
        # Explicit Lab/run identities prevent different environments from
        # sharing a journal key.
        identity = ([observation.lab_id, observation.run_id, observation.task_id]
                    if observation.lab_id or observation.run_id else observation.task_id)
        ident = prefix + result_id(identity)
        previous = self.state['jobs'].get(ident)
        if previous is not None:
            if previous['receipt_digest'] != digest:
                raise ValueError('Learning receipt changed for an existing task')
            if previous['capture_archive_only'] != archive_only:
                raise ValueError('Existing receipt processing mode cannot change')
            return ident
        producer_event, duplicate_of = producer_delivery(observation, self.state['jobs'].values(),
                                                         archive_only=archive_only)
        # Preserve raw evidence independently of the bounded model projection.
        scope = deepcopy(research_context or {})
        if observation.decision_id is not None:
            if scope.get('decision_id', observation.decision_id) != observation.decision_id:
                raise ValueError('Receipt decision identity differs from supplied research context')
            scope['decision_id'] = observation.decision_id
        if observation.issuing_model is not None:
            issuer = asdict(observation.issuing_model)
            if scope.get('issuing_decision_model') not in (None, issuer):
                raise ValueError('Receipt issuing model differs from supplied research context')
            scope['issuing_decision_model'] = issuer
        if observation.decision_id is not None or observation.issuing_model is not None:
            scope['learning_decision_model'] = asdict(model)
        members = scope.get('member_decisions', [])
        decisions = {item['decision_id'] for item in members}
        task_ids = {call_id for item in members for call_id in item.get('call_ids', ())}
        task_ids.add(observation.task_id)
        if observation.decision_id:
            decisions.add(observation.decision_id)
        selected = [asdict(row) for row in history if row.task_id in task_ids
                    or row.record_id in decisions or row.details.get('decision_id') in decisions]
        packet = {'receipt': receipt, 'evaluations': [asdict(row) for row in evaluations],
                  'history': selected, 'research_context': scope}
        if grouping_context is not None:
            packet['grouping_context'] = deepcopy(grouping_context)
        row = {'job_id': ident, 'capture_index': len(self.state['jobs']),
               'capture_archive_only': archive_only,
               'receipt_digest': digest, 'model': asdict(model), 'raw_input': packet,
               'status': 'queued', 'proposals': [], 'error': None,
               'simulated': bool(observation.request.get('simulated', False) or observation.result['simulated']
                   or any(item.get('simulated', False) for item in packet['history'])
                   or any(item.get('simulated', False) for item in packet['evaluations'])
                   or any(item.get('simulated', False) for item in (grouping_context or {}).get('observations', []))
                   or any(item.get('proposal', {}).get('simulated', False)
                          for item in (grouping_context or {}).get('decisions', []))), 'response': None}
        if producer_event is not None:
            row.update(producer_event=producer_event, capture_archive_only=archive_only)
        if duplicate_of is not None:
            row['duplicate_of'] = duplicate_of
        admitted = self._bound_input(row)
        state = deepcopy(self.state)
        admission = self.admission
        if admission is not None:
            if not archive_only and duplicate_of is None:
                from rsicomposer.loops.contracts.feedback_modules import FeedbackAdmission
                pending = tuple(PendingFeedback(feedback_context_event(item), item['capture_index'])
                    for item in state['jobs'].values() if item['status'] == 'queued')
                choice = admission.admit(feedback_context_event(row), pending)
                eligible = {item.event.event_id for item in pending}
                if (not isinstance(choice, FeedbackAdmission) or type(choice.organize) is not bool
                        or not isinstance(choice.coalesce, tuple) or not isinstance(choice.reason, str)
                        or len(set(choice.coalesce)) != len(choice.coalesce)
                        or not set(choice.coalesce) <= eligible or (choice.coalesce and not choice.organize)):
                    raise ValueError('Invalid feedback admission or nonqueued coalescing target')
                row['admission'] = asdict(choice)
                if not choice.organize:
                    row.update(status='archived', organization_skipped=True, published=True)
                elif admitted:
                    for prior in choice.coalesce:
                        state['jobs'][prior].update(status='archived', organization_skipped=True,
                            coalesced_into=ident, published=True)
        if archive_only or duplicate_of is not None:
            row.update(status='archived', archive_only=True, published=True)
        elif row.get('organization_skipped'):
            pass
        elif not admitted:
            row.update(status='input_too_large', error='Indivisible learning evidence exceeds byte budget')
        elif sum(item['status'] in ('queued', 'running') for item in state['jobs'].values()) >= self.limits['max_pending']:
            row.update(status='queue_full', error='Learning queue limit reached; raw feedback remains available')
        state['jobs'][ident] = row
        self._commit(state)
        return ident

    def _bound_input(self, row, *, tokens=False):
        # Token admission includes the system instructions AND generated schema.
        # Profiles are deterministic, contain no generated summaries, and never
        # enlarge a budget. Small evidence remains verbatim.
        for count, text_limit, preview_feedback in (
                (32, 512, False), (32, 512, True), (16, 256, True), (8, 192, True),
                (4, 128, True), (2, 96, True), (1, 64, True), (0, 0, True)):
            row['input'], row['coverage'] = evidence_packet(
                row, count, text_limit, preview_feedback=preview_feedback)
            if len(json_bytes(row['input'])) > self.limits['max_input_bytes']:
                continue
            return True
        return False

    def _exchange_path(self, row):
        return (self.path.parent / (self.path.stem + '-exchanges') / row['job_id'] / 'product.json'
                if self.path is not None else None)

    def _work(self, row):
        path = self._exchange_path(row)
        return self._processor_binding.run(row, self.backend,
            {**self.limits, 'max_calls': row['reserved_calls'],
             'exchange_directory': str(path.parent) if path is not None else None}, path=path)

    @staticmethod
    def _settle_processor(state, row, result):
        reserved = row['reserved_calls']
        result = checked_processor_result(result, reserved)
        state['attempted'] -= reserved - result['charged_calls']
        simulated = bool(row['simulated'] or result.get('simulated', False))
        row.update(result)
        row['simulated'] = simulated

    @staticmethod
    def _finalize_job(state, row):
        """Identical validation/revision transition for live completion and recovery."""
        if row['status'] != 'completed':
            return
        try:
            validate_publication(row)
        except (ValueError, TypeError, KeyError) as exc:
            row.update(status='failed', proposals=[], attempt_updates=[], attempt_revisions=[],
                       error='Evidence validation failed: ' + str(exc))
        else:
            try:
                current = attempt_view(r for ident, r in state['jobs'].items() if ident != row['job_id'])
                row['attempt_revisions'] = apply_attempt_updates(row, current['items'])
            except ValueError as exc:
                row.update(attempt_revisions=[], grouping_error=str(exc))

    def ordered_jobs(self):
        """Read the explicit capture order persisted by this journal."""
        return sorted(self.state['jobs'].values(), key=lambda row: row['capture_index'])

    def poll(self, *, dispatch=True):
        """Commit ready results without waiting, then admit at most one request."""
        self._assert_owner()
        for ident, future in list(self._futures.items()):
            if not future.done():
                continue
            try:
                result = future.result()
            except Exception as exc:
                result = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
            state = deepcopy(self.state)
            row = state['jobs'][ident]
            self._settle_processor(state, row, result)
            self._finalize_job(state, row)
            self._commit(state)
            del self._futures[ident]
        if not dispatch or self._closed or len(self._futures) >= self.max_concurrent_jobs:
            return
        pending = [r for r in self.ordered_jobs() if r['status'] == 'queued']
        if not pending:
            return
        try:
            selected = self.scheduler.select(tuple(PendingFeedback(feedback_context_event(row),
                row['capture_index']) for row in pending), tuple(self._futures))
            if selected is None:
                return
            if not isinstance(selected, str) or selected not in {r['job_id'] for r in pending}:
                raise ValueError('Scheduler selected an event that is not pending')
        except Exception as exc:
            state = deepcopy(self.state)
            state['scheduler_error'] = f'{type(exc).__name__}: {exc}'
            self._commit(state)
            return
        queued = next(row for row in pending if row['job_id'] == selected)
        state = deepcopy(self.state)
        state.pop('scheduler_error', None)
        row = state['jobs'][queued['job_id']]
        # Refresh only the interpretation catalog at dispatch. Execution evidence
        # remains the immutable capture, and model provenance hashes the actual input.
        grouping = row['raw_input'].get('grouping_context')
        if grouping is not None:
            supplied = {r['decision_id'] for r in grouping['decisions']}
            catalog = self.attempts()['items']
            row['dispatch_attempts'] = [r for r in catalog
                                   if set(r['member_decision_ids']).intersection(supplied) or r in catalog[-8:]]
        if (state['attempted'] >= self.limits['max_calls']
                and self._processor_binding.max_calls > 0):
            row.update(status='budget_exhausted', error='Learning request budget exhausted')
            self._commit(state)
            return
        try:
            admitted = self._bound_input(row, tokens=True)
        except Exception as exc:
            # Tokenizer/transport failures are learning failures, not exceptions
            # in the execution owner's feedback delivery path.
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            self._commit(state)
            return
        if not admitted:
            row.update(status='input_too_large', error='Indivisible learning evidence exceeds request budget')
            self._commit(state)
            return
        # Persist the exact feedback prompt/schema before inference. Recovery
        # reads this exchange rather than reconstructing it from newer code.
        row['reserved_calls'] = min(self._processor_binding.max_calls,
            self.limits['max_calls'] - state['attempted'])
        row['request'] = {'processor': deepcopy(state['processor']),
                          'input': deepcopy(row['input']),
                          'actual_exchanges_read_path': self._processor_binding.actual_exchanges_read_path}
        state['attempted'] += row['reserved_calls']
        row['status'] = 'running'
        self._commit(state)
        try:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rsi-feedback')
            self._futures[row['job_id']] = self._executor.submit(self._work, deepcopy(row))
        except Exception as exc:
            state = deepcopy(self.state)
            state['jobs'][row['job_id']].update(status='failed', error=f'Submission failed: {exc}')
            self._commit(state)

    def attempts(self, decision_ids=()):
        return attempt_view(self.state['jobs'].values(), decision_ids)

    @property
    def context_memory(self):
        return self._context_memory

    @property
    def trajectory_archive(self):
        return self._trajectory_archive

    @property
    def context_binding(self):
        return ContextMemoryBinding(self.context_memory)

    @property
    def archive_binding(self):
        return ArchiveBinding(self.trajectory_archive)

    def notify_archive(self):
        """Persist intent before external writes; no replay of unknown effects."""
        self._assert_owner()
        binding = self.archive_binding
        if self._closed or not binding.requires_delivery:
            return
        for row in self.ordered_jobs():
            if row.get('archive_delivery'):
                continue
            ident = row['job_id']
            event = captured_archive_event(row)
            delivery = {'status': 'running', 'event_digest': result_id(asdict(event))}
            state = deepcopy(self.state)
            state['jobs'][ident]['archive_delivery'] = delivery
            self._commit(state)
            try:
                reference = binding.write(event)
            except Exception as exc:
                delivery = {**delivery, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
            else:
                delivery = {**delivery, 'status': 'delivered', 'reference': reference}
            state = deepcopy(self.state)
            state['jobs'][ident]['archive_delivery'] = delivery
            self._commit(state)

    def acknowledge_context_delivery(self, job_id):
        """Record the Harness's successful canonical receipt delivery.

        Host receipts reach Memory through ControllerHarness.observe before their
        feedback job is captured. The journal must retain that acknowledgement
        rather than offer the same execution again under a feedback job ID.
        Producer captures without a canonical Harness delivery use notify_context.
        """
        self._assert_owner()
        row = self.state['jobs'][job_id]
        if row.get('context_delivery'):
            return
        state = deepcopy(self.state)
        state['jobs'][job_id]['context_delivery'] = {'status': 'delivered'}
        self._commit(state)

    def notify_context(self):
        """Offer raw feedback on a later tick, independently of interpretation.

        Notification does not acknowledge/re-execute a Lab task. Persist intent
        before a provider hook; successful, failed or uncertain side effects are
        not automatically repeated. Providers can deduplicate by event_id.
        """
        self._assert_owner()
        if self._closed:
            return
        observer = self.context_observer or self.context_binding.observer
        if observer is None:
            return
        for captured in self.ordered_jobs():
            if captured.get('archive_only') or captured.get('context_delivery'):
                continue
            ident = captured['job_id']
            event = feedback_context_event(captured)
            state = deepcopy(self.state)
            state['jobs'][ident]['context_delivery'] = {'status': 'running'}
            self._commit(state)
            try:
                observer(event)
            except Exception as exc:
                delivery = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
            else:
                delivery = {'status': 'delivered'}
            state = deepcopy(self.state)
            state['jobs'][ident]['context_delivery'] = delivery
            self._commit(state)

    def context(self, query=None):
        """Return the current text view of the Controller's memory."""
        query = ContextQuery() if query is None else query
        if not isinstance(query, ContextQuery):
            raise TypeError('Context query must be a ContextQuery')
        binding = self.context_binding
        jobs = self.ordered_jobs()
        if query.latest_feedback is None:
            latest = next((row for row in reversed(jobs) if not row.get('archive_only')), None)
            if latest is not None:
                query = replace(query, latest_feedback=feedback_context_event(latest))
        try:
            value = binding.context(query)
        except Exception as exc:
            previous = self.state.get('context_cache', {})
            reuse = (self.failure_policy.context == 'last_valid' and
                     previous.get('objective') == query.objective and 'text' in previous)
            state = deepcopy(self.state)
            state['context_error'] = {'error': f'{type(exc).__name__}: {exc}', 'used_previous': reuse,
                                      'objective': query.objective}
            self._commit(state)
            if reuse:
                return previous['text']
            raise
        if self.failure_policy.context == 'last_valid' or 'context_error' in self.state:
            state = deepcopy(self.state)
            state.pop('context_error', None)
            if self.failure_policy.context == 'last_valid':
                state['context_cache'] = {'objective': query.objective, 'text': value}
            self._commit(state)
        return value

    def context_record(self, job_id):
        return self.context_binding.record(self.state['jobs'][job_id])

    def trajectories(self, *, research_attempt_id=None):
        if self.trajectory_archive is None:
            return []
        return self._trajectory_records(self.ordered_jobs(), research_attempt_id=research_attempt_id)

    def _trajectory_records(self, jobs, *, research_attempt_id=None):
        return self.archive_binding.records(jobs, research_attempt_id=research_attempt_id)

    def trajectory_snapshot(self, job_id):
        if self.trajectory_archive is None:
            raise EvidencePathError('Trajectory archival is disabled')
        return self.archive_binding.snapshot(self.state['jobs'][job_id])


    def close(self, *, timeout=30, preserve_queued=False):
        """Stop admissions; preserve uncertain calls, never claim cancellation.

        Python cannot kill a running backend call. A timed-out close reports failure;
        the finite transport timeout must ultimately release the worker.
        """
        self._assert_owner()
        self._closed = True
        # A host pause retains unsubmitted work for the next process. Repeated
        # resource cleanup must not convert those reservations to cancellation.
        self._preserve_queued = getattr(self, '_preserve_queued', False) or preserve_queued
        self.poll(dispatch=False)
        if self._futures:
            _, pending = wait(tuple(self._futures.values()), timeout=timeout)
            self.poll(dispatch=False)
            if pending:
                raise TimeoutError('Feedback inference still running; transport cancellation not confirmed')
        state = deepcopy(self.state)
        for row in state['jobs'].values():
            if row['status'] == 'queued' and not self._preserve_queued:
                row.update(status='cancelled', error='Closed before model submission')
        self._commit(state)
        if self._executor is not None and self._owns_executor:
            self._executor.shutdown(wait=True)
