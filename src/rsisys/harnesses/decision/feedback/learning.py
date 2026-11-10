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

from rsisys.harnesses.decision.feedback.archive import (
    ArchiveBinding,
    captured_archive_event,
    validate_archive,
)
from rsisys.harnesses.decision.feedback.attempts import (
    apply_attempt_updates,
    attempt_view,
)
from rsisys.harnesses.decision.feedback.components import (
    JournalContextMemory,
    JournalTrajectoryArchive,
)
from rsisys.harnesses.decision.feedback.context_memory import (
    ContextMemoryBinding,
    feedback_context_event,
    validate_context_memory,
)
from rsisys.harnesses.decision.feedback.delivery import producer_delivery
from rsisys.harnesses.decision.feedback.knowledge import (
    interpretation_page,
    proposal_records,
)
from rsisys.harnesses.decision.feedback.legacy import LegacyFeedbackInterpreter, interpret_feedback
from rsisys.harnesses.decision.feedback.evidence_packet import evidence_packet
from rsisys.harnesses.decision.feedback.model_gateway import SerializedLearningBackend
from rsisys.harnesses.decision.feedback.proposal_validation import checked_processor_result, validate_publication
from rsisys.harnesses.decision.feedback.scheduling import (
    FifoFeedbackScheduler,
    validate_scheduler,
)
from rsisys.harnesses.decision.feedback.stages import (
    FeedbackStageBinding,
    component_configuration,
)
from rsisys.loops.contracts.feedback_context import ContextQuery
from rsisys.loops.contracts.feedback_modules import (
    FeedbackFailurePolicy,
    PendingFeedback,
)
from rsisys.loops.contracts.model_service import GenerationRequest, ModelRef
from rsisys.loops.contracts.receipts import ExecutionReceipt
from rsisys.runtime.storage.files import atomic_write, json_bytes
from rsisys.runtime.storage.json_tools import (
    EvidencePathError,
    encoded,
    evidence_value,
    result_id,
)

_EVIDENCE_PAGE_BYTES = 6000
_DEFAULT_COMPONENT = object()


class FeedbackLearning:
    """Single-writer journal plus one background inference worker.

    Reload preserves completed outputs and marks uncertain in-flight work
    interrupted; it never repeats that request automatically. Queued jobs have not
    called the model and may be dispatched after reopening. This journal is not a
    replacement for the Runner's own snapshot/recovery contract.
    """
    def __init__(self, backend: SerializedLearningBackend | None, *, path=None, max_calls=64,
                 max_pending=16, max_input_tokens=16000, max_output_tokens=4096,
                 max_input_bytes=64000, executor=None,
                 context_memory=_DEFAULT_COMPONENT, trajectory_archive=_DEFAULT_COMPONENT,
                 processor=None, assembly_configuration=None, scheduler=None,
                 max_concurrent_jobs=1, failure_policy=None, legacy_interpreter=None, admission=None):
        binding = FeedbackStageBinding(processor)
        if backend is None and (processor is None or binding.max_calls > 0):
            raise ValueError('Feedback inference requires a model backend')
        if backend is not None and not isinstance(backend, SerializedLearningBackend):
            raise TypeError('Use a shared SerializedLearningBackend for decision and learning calls')
        if context_memory is _DEFAULT_COMPONENT:
            context_memory = JournalContextMemory()
        if trajectory_archive is _DEFAULT_COMPONENT:
            trajectory_archive = JournalTrajectoryArchive()
        validate_context_memory(context_memory)
        validate_archive(trajectory_archive)
        self._context_memory, self._trajectory_archive = context_memory, trajectory_archive
        self._processor_binding = binding
        self.processor = processor
        self.legacy_interpreter = ((LegacyFeedbackInterpreter() if legacy_interpreter is None else legacy_interpreter)
                                   if processor is None else None)
        if processor is None and not callable(getattr(self.legacy_interpreter, 'request', None)):
            raise TypeError('Legacy feedback interpreter requires request(row, backend, limits)')
        processor_config = self._processor_binding.configuration() if processor is not None else None
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
                          'failure_policy': asdict(self.failure_policy),
                          'legacy_interpreter': component_configuration(self.legacy_interpreter)
                                                if processor is None else None}
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
        self.state = {'version': 1, 'limits': self.limits, 'attempted': 0, 'jobs': {}}
        self.state['runtime'] = runtime_config
        if assembly_configuration is not None:
            json_bytes(assembly_configuration)
            self.state['assembly'] = deepcopy(assembly_configuration)
        if processor_config is not None:
            self.state['processor'] = processor_config
        if self.path is not None and self.path.exists():
            loaded = json.loads(self.path.read_text())
            if (loaded.get('version') != 1 or loaded.get('limits') != self.limits
                    or loaded.get('processor') != processor_config
                    or loaded.get('runtime') != runtime_config
                    or loaded.get('assembly') != assembly_configuration
                    or not isinstance(loaded.get('jobs'), dict)
                    or type(loaded.get('attempted')) is not int or loaded['attempted'] < 0):
                raise ValueError('Invalid learning journal or changed limits')
            self.state = loaded
            recovered = deepcopy(loaded)
            for row in sorted(recovered['jobs'].values(), key=lambda r: r.get('capture_index', -1)):
                for name in ('context_delivery', 'archive_delivery', 'publication_delivery'):
                    delivery = row.get(name, {})
                    if delivery.get('status') == 'running':
                        row[name] = {**delivery, 'status': 'interrupted',
                            'error': 'Delivery outcome unknown; no automatic replay'}
                if row['status'] == 'running':
                    try:
                        result = self._processor_binding.recover(row) if processor is not None else None
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

    def capture(self, directive, feedback, model, history=(), *, evaluations=(), research_context=None,
                grouping_context=None):
        receipt = ExecutionReceipt.from_feedback(directive, feedback)
        return self.capture_receipt(receipt, model, history, evaluations=evaluations,
            research_context=research_context, grouping_context=grouping_context)

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
        prefix = {'operation': 'learning-operation-', 'lab': 'learning-lab-',
                  'exploration': 'learning-'}[observation.source]
        # Explicit Lab/run identities prevent different environments from
        # sharing a journal key. Legacy captures retain their historical key.
        identity = ([observation.lab_id, observation.run_id, observation.task_id]
                    if observation.lab_id or observation.run_id else observation.task_id)
        ident = prefix + result_id(identity)
        previous = self.state['jobs'].get(ident)
        if previous is not None:
            if previous['receipt_digest'] != digest:
                raise ValueError('Learning receipt changed for an existing task')
            if previous.get('capture_archive_only', previous.get('archive_only', False)) != archive_only:
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
        directives = {item['directive_id'] for item in members if item.get('directive_id')}
        directives.add(observation.task_id)
        if observation.decision_id:
            decisions.add(observation.decision_id)
        selected = [asdict(row) for row in history if row.task_id in directives
                    or row.record_id in decisions or row.details.get('decision_id') in decisions
                    or (isinstance(row.details.get('directive'), dict)
                        and row.details['directive'].get('directive_id') in directives)]
        packet = {'receipt': receipt, 'evaluations': [asdict(row) for row in evaluations],
                  'history': selected, 'research_context': scope}
        if grouping_context is not None:
            packet['grouping_context'] = deepcopy(grouping_context)
        row = {'job_id': ident, 'capture_index': len(self.state['jobs']),
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
                from rsisys.loops.contracts.feedback_modules import FeedbackAdmission
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
            if (not tokens or self.processor is not None
                    or self.backend.count_tokens(self._request(row)) <= self.limits['max_input_tokens']):
                return True
        return False

    def _request(self, row):
        if row.get('request') is not None:
            request = deepcopy(row['request'])
            request['model'] = ModelRef(**request['model'])
            return GenerationRequest(**request)
        return self.legacy_interpreter.request(row, self.backend, self.limits)

    def _work(self, row):
        if self.processor is not None:
            directory = (self.path.parent / (self.path.stem + '-exchanges') / row['job_id']
                         if self.path is not None else None)
            return self._processor_binding.process(row, self.backend,
                {**self.limits, 'max_calls': row['reserved_calls'],
                 'exchange_directory': str(directory) if directory is not None else None},
                path=directory / 'product.json' if directory is not None else None)
        return interpret_feedback(row, self.backend, self.limits, self._request(row))

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
        """Use persisted capture order; legacy rows retain their stored order.

        Pre-index journals cannot reconstruct capture chronology after JSON key
        sorting. Do not infer it from hashes or replay an uncertain model call.
        """
        rows = list(self.state['jobs'].values())
        return [row for _, row in sorted(enumerate(rows),
            key=lambda pair: (pair[1].get('capture_index', -1), pair[0]))]

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
            if self.processor is not None:
                self._settle_processor(state, row, result)
            else:
                row.update(result)
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
                row.get('capture_index', -1)) for row in pending), tuple(self._futures))
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
        # Legacy queued journals contain only input; adopt it as the raw snapshot
        # before projecting. Completed historical requests are never rewritten.
        row.setdefault('raw_input', deepcopy(row['input']))
        grouping = row['raw_input'].get('grouping_context')
        if grouping is not None:
            supplied = {r['decision_id'] for r in grouping['decisions']}
            catalog = self.attempts()['items']
            row['dispatch_attempts'] = [r for r in catalog
                                   if set(r['member_decision_ids']).intersection(supplied) or r in catalog[-8:]]
        if (state['attempted'] >= self.limits['max_calls']
                and (self.processor is None or self._processor_binding.max_calls > 0)):
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
        if self.processor is None:
            row['request'] = asdict(self._request(row))
            state['attempted'] += 1  # Conservative reservation includes rejected requests.
        else:
            row['reserved_calls'] = min(self._processor_binding.max_calls,
                self.limits['max_calls'] - state['attempted'])
            row['request'] = {'processor': deepcopy(state['processor']),
                              'input': deepcopy(row['input']),
                              'actual_exchanges_read_path': getattr(self.processor,
                                  'actual_exchanges_read_path', '/context_memory/model_exchanges')}
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

    def notify_context(self):
        """Offer raw feedback on a later tick, independently of interpretation.

        Notification does not acknowledge/re-execute a Lab task. Persist intent
        before a provider hook; successful, failed or uncertain side effects are
        not automatically repeated. Providers can deduplicate by event_id.
        """
        self._assert_owner()
        if self._closed:
            return
        observer = self.context_binding.observer
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

    def context(self, query=None, *, active_decision_ids=(), policy=None):
        """Return the current text view; no journal types reach a new provider."""
        query = ContextQuery() if query is None else query
        if not isinstance(query, ContextQuery):
            raise TypeError('Context query must be a ContextQuery')
        binding = self.context_binding
        jobs = self.ordered_jobs()
        if not binding.legacy and query.latest_feedback is None:
            latest = next((row for row in reversed(jobs) if not row.get('archive_only')), None)
            if latest is not None:
                query = replace(query, latest_feedback=feedback_context_event(latest))
        try:
            value = (binding.context(query) if not binding.legacy else
                binding.context(query, jobs=jobs, attempts=self.attempts()['items'],
                    active_decision_ids=active_decision_ids, policy=policy))
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

    def working_memory(self, *, active_decision_ids=(), policy=None):
        """Compatibility inspection; harnesses consume context() for new views."""
        binding = self.context_binding
        if binding.legacy:
            return binding.working_memory(self.ordered_jobs(), self.attempts()['items'],
                active_decision_ids=active_decision_ids, policy=policy)
        return {'context': self.context()}

    def trajectories(self, *, research_attempt_id=None):
        if self.trajectory_archive is None:
            return []
        return self._trajectory_records(self.ordered_jobs(), research_attempt_id=research_attempt_id)

    def extract_trajectory(self, job_id):
        if self.trajectory_archive is None:
            raise EvidencePathError('Trajectory archival is disabled')
        return self.archive_binding.extract(self.state['jobs'][job_id])

    def _trajectory_records(self, jobs, *, research_attempt_id=None):
        return self.archive_binding.records(jobs, research_attempt_id=research_attempt_id)

    def trajectory_snapshot(self, job_id):
        if self.trajectory_archive is None:
            raise EvidencePathError('Trajectory archival is disabled')
        return self.archive_binding.snapshot(self.state['jobs'][job_id])

    def read(self, job_id=None, *, offset=0, limit=8, research_attempt_id=None, evidence_path=None,
             view='jobs', target=None, expected_digest=None):
        """Read proposals, or a bounded JSON-text page of original evidence.

        evidence_path is a JSON pointer into raw_input (empty for the root).
        /dispatch_attempts addresses the separately refreshed catalog. Evidence
        offsets count characters, with at most 128 characters per limit unit.
        Pages also fit 6000 JSON-encoded UTF-8 bytes including metadata (except
        irreducible metadata plus one character). Assemble all pages before
        parsing their JSON. Reads never expand model coverage.
        """
        self._assert_owner()
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 16:
            raise ValueError('Invalid learning page')
        if view not in ('jobs', 'attempts', 'proposals', 'trajectories'):
            raise ValueError('Unknown research feedback view')
        if expected_digest is not None and (not isinstance(expected_digest, str) or len(expected_digest) != 64):
            raise ValueError('expected_digest must be a SHA-256 digest')
        if view != 'jobs':
            if job_id is not None or evidence_path is not None:
                raise ValueError('Knowledge views cannot be combined with a job evidence read')
            if research_attempt_id is not None and (not isinstance(research_attempt_id, str)
                    or not research_attempt_id.strip() or len(research_attempt_id) > 200):
                raise ValueError('Invalid research attempt ID')
            if view == 'trajectories':
                if target is not None:
                    raise ValueError('target applies only to proposal views')
                value = self.trajectories(research_attempt_id=research_attempt_id)
            elif view == 'attempts':
                if target is not None:
                    raise ValueError('target applies only to proposal views')
                value = self.attempts()['items']
                if research_attempt_id is not None:
                    value = [a for a in value if a['attempt_id'] == research_attempt_id]
            else:
                value = proposal_records(self.ordered_jobs(), target=target,
                                         research_attempt_id=research_attempt_id)
            metadata = {'view': view, 'total': len(value),
                        'target': target, 'research_attempt_id': research_attempt_id}
            if view == 'trajectories':
                metadata['enabled'] = self.trajectory_archive is not None
            return interpretation_page(value, offset=offset, limit=limit,
                expected_digest=expected_digest, metadata=metadata,
                authority='archived_public_trajectory' if view == 'trajectories' else 'unverified_interpretations')
        if target is not None:
            raise ValueError('target applies only to proposal views')
        if research_attempt_id is not None and (not isinstance(research_attempt_id, str)
                or not research_attempt_id.strip() or len(research_attempt_id) > 200 or job_id is not None):
            raise ValueError('An attempt filter requires a bounded ID and a job listing')
        if evidence_path is not None:
            if (job_id is None or not isinstance(evidence_path, str) or len(evidence_path) > 1000
                    or (evidence_path and not evidence_path.startswith('/'))):
                raise ValueError('Evidence reading requires a job ID and a JSON pointer of at most 1000 characters')
            if job_id not in self.state['jobs']:
                raise EvidencePathError('Evidence job ID is not recorded; list feedback jobs before reading evidence')
            row = self.state['jobs'][job_id]
            if evidence_path.split('/')[1:2] in (['context_memory'], ['trajectory'], ['trajectory_segment'], ['trajectory_snapshot']):
                namespace = evidence_path.split('/')[1]
                if namespace == 'context_memory':
                    value = self.context_record(job_id)
                elif namespace == 'trajectory_segment':
                    value = self.extract_trajectory(job_id)
                elif namespace == 'trajectory_snapshot':
                    value = self.trajectory_snapshot(job_id)
                else:
                    if self.trajectory_archive is None:
                        raise EvidencePathError('Trajectory archival is disabled')
                    records = self._trajectory_records([row])
                    if len(records) != 1:
                        raise EvidencePathError('No unique trajectory segment for this job')
                    value = records[0]
                value = evidence_value({namespace: value}, evidence_path)
                return interpretation_page(value, offset=offset, limit=limit,
                    expected_digest=expected_digest, authority=(
                        'organized_context_memory' if namespace == 'context_memory' else 'archived_public_trajectory'),
                    metadata={'job_id': job_id, 'evidence_path': evidence_path, 'simulated': row['simulated']})
            if evidence_path.split('/')[1:2] in (['proposals'], ['attempt_revisions'], ['outputs'], ['output']):
                # Interpretation output has its own namespace. It is never
                # presented as captured execution evidence or admitted coverage.
                value = evidence_value(row, evidence_path)
                return interpretation_page(value, offset=offset, limit=limit,
                    expected_digest=expected_digest, metadata={'job_id': job_id,
                        'evidence_path': evidence_path, 'simulated': row['simulated']})
            value = row.get('raw_input', row['input'])
            root = '/raw_input' if 'raw_input' in row else '/input'
            pointer = evidence_path
            if pointer == '/dispatch_attempts' or pointer.startswith('/dispatch_attempts/'):
                value = row['dispatch_attempts']
                root = '/dispatch_attempts'
                pointer = pointer[len('/dispatch_attempts'):]
            for namespace, key in (('/feedback_request', 'request'), ('/feedback_response', 'response')):
                if pointer == namespace or pointer.startswith(namespace + '/'):
                    if row.get(key) is None:
                        raise EvidencePathError('Feedback exchange was not recorded for this job')
                    value, root, pointer = row[key], '/' + key, pointer[len(namespace):]
                    break
            value = evidence_value(value, pointer)
            if expected_digest is not None and expected_digest != result_id(value):
                raise EvidencePathError('Feedback evidence changed; restart pagination')
            text = encoded(value)
            interpretation = evidence_path.startswith(('/dispatch_attempts', '/grouping_context/attempts'))
            metadata = {'job_id': job_id, 'evidence_path': evidence_path,
                    'raw_ref': 'journal:' + job_id + root + pointer, 'sha256': result_id(value),
                    'authority': 'untrusted_evidence',
                    'content_authority': 'model_interpretation' if interpretation else 'untrusted_evidence',
                    'encoding': 'json', 'offset': offset, 'total_characters': len(text),
                    'simulated': row['simulated'], 'training_ready': False, 'applied': False}
            chunk = text[offset:offset + limit * 128]

            def page(size):
                end = offset + size
                return {**metadata, 'text': chunk[:size],
                        'next_offset': end if end < len(text) else None}

            complete = page(len(chunk))
            if len(json_bytes(complete)) <= _EVIDENCE_PAGE_BYTES:
                return complete
            # Price the outer JSON string too: quotes, backslashes and control
            # escapes can grow again when JSON evidence is nested inside a page.
            # Keep at least one source character, never a byte fragment or a
            # non-advancing cursor, even if metadata alone exhausts the soft cap.
            lo, hi = min(1, len(chunk)), len(chunk)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if len(json_bytes(page(mid))) <= _EVIDENCE_PAGE_BYTES:
                    lo = mid
                else:
                    hi = mid - 1
            return page(lo)
        if job_id is not None:
            if expected_digest is not None:
                raise ValueError('expected_digest requires an evidence or knowledge page')
            row = self.state['jobs'][job_id]
            return deepcopy({'job_id': job_id, 'status': row['status'], 'model': row['model'],
                'receipt_digest': row['receipt_digest'], 'proposals': row['proposals'],
                'products': row.get('products', {}),
                'research_context': row['input'].get('research_context', {}),
                'attempt_revisions': deepcopy(row.get('attempt_revisions', [])),
                'grouping_error': row.get('grouping_error'),
                'publication_error': row.get('publication_error'),
                'source_refs': row['input']['source_refs'], 'error': row['error'],
                'evidence_packaging': row['input'].get('evidence_packaging'),
                'context_memory': {'enabled': True,
                                  'read_path': {'operation': 'feedback_learning_read', 'job_id': job_id,
                                                'evidence_path': '/context_memory'}},
                'trajectory': ({'enabled': True,
                               'read_path': {'operation': 'feedback_learning_read', 'job_id': job_id,
                                             'evidence_path': '/trajectory'}}
                               if self.trajectory_archive is not None else {'enabled': False}),
                'simulated': row['simulated'], 'authority': 'unverified_proposals',
                'training_ready': False, 'applied': False})
        rows = self.ordered_jobs()
        if expected_digest is not None:
            raise ValueError('expected_digest requires an evidence or knowledge page')
        if research_attempt_id is not None:
            rows = [row for row in rows if row['input'].get('research_context', {}).get('research_attempt_id') == research_attempt_id
                    or any(r['attempt_id'] == research_attempt_id for r in row.get('attempt_revisions', []))]
        return {'attempted': self.state['attempted'], 'limit': self.limits['max_calls'], 'total': len(rows),
                'items': [{'job_id': r['job_id'], 'status': r['status'],
                           'products': deepcopy(r.get('products', {})),
                           **{key: r['input'].get('research_context', {}).get(key)
                              for key in ('decision_id', 'research_attempt_id', 'research_attempt_title')},
                           'interpreted_attempt_ids': [a['attempt_id'] for a in r.get('attempt_revisions', [])],
                           'grouping_error': r.get('grouping_error'),
                           'kinds': [p['kind'] for p in r['proposals']]} for r in rows[offset:offset+limit]],
                'next_offset': offset+limit if offset+limit < len(rows) else None}

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
