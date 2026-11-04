import json
from contextlib import contextmanager
from dataclasses import asdict, replace
from copy import deepcopy
from uuid import uuid4
from time import perf_counter

from rsisys.loops._memory import MemoryStore, HistoryEntry
from .memory import DecisionMemory
from .context import BasicContextCompactor
from .research import ResearchAgenda

from rsisys.experience.records import ExperienceRecord
from rsisys.experience.retrieval import ExperienceSystem, RecentFeedbackSelector
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.jobs import JobRef, JobOutcome, JobStatus
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.metrics import InvalidMetrics, validated_metrics
from rsisys.loops.contracts.training import TrainingResult, ModelActivator
from rsisys.runtime.model import prepare_model_activation
from rsisys.loops.contracts.model_service import ModelRef, ModelUpdatable
from rsisys.runtime.jobs import JobBook

from rsisys.loops.contracts.coordination import LoopStepResult, ExecutionView, ExplorationProgress, ReviewDecision
from rsisys.loops.contracts.requests import EvaluationSpec
from ..action_space import DecisionActionSpace, OperationCall, OperationResult
from ..action_space.access import authorized_space
from .learner import DecisionLearner, DecisionTrainingRequest
from .policy import DecisionPolicy, ModelDecisionPolicy
from .decision import Decision
from .attempts import attempt_details, requested_attempt, AttemptContract
from .tools import DecisionTools, execute_operation, record_operation

from .state import (DecisionContext, DecisionState, _PendingOperationRecord, _PendingReviewRecord, _ReviewSession,
                    DirectiveProvenance, CandidateEvaluation)


def _json_details(value):
    """Match MemoryBuffer's JSON load representation before the first write."""
    return json.loads(json.dumps(value))


class DefaultDecisionHarness:
    def __init__(self, policy: DecisionPolicy, tools: DecisionTools,
                 experience: ExperienceSystem, learner: DecisionLearner | None = None,
                 *, experience_limit: int = 32, memory: MemoryStore | None = None, memory_limit: int = 32,
                 model_activator: ModelActivator | None = None, review_tool_limit: int = 4,
                 max_outstanding: int = 1, feedback_selector=None,
                 context_compactor=None, research_agenda=None, attempt_contract=None):
        self._feedback_listeners, self._context_processors, self._cleanup_callbacks = [], [], []
        self.feedback_proposal_resolver = None
        self._receipt_listeners = []
        self.attempt_contract = attempt_contract if attempt_contract is not None else AttemptContract()
        if not isinstance(self.attempt_contract, AttemptContract):
            raise TypeError('attempt_contract must be an AttemptContract')
        self.policy, self.tools, self.experience, self.learner = policy, tools, experience, learner
        if type(experience_limit) is not int or experience_limit < 1:
            raise ValueError("experience_limit must be a positive integer")
        if type(memory_limit) is not int or memory_limit < 1:
            raise ValueError("memory_limit must be a positive integer")
        if type(review_tool_limit) is not int or review_tool_limit < 0:
            raise ValueError('review_tool_limit must be a nonnegative integer')
        if type(max_outstanding) is not int or max_outstanding < 1:
            raise ValueError('max_outstanding must be a positive integer')
        self.max_outstanding = max_outstanding
        self.feedback_selector = feedback_selector if feedback_selector is not None else RecentFeedbackSelector()
        self.review_tool_limit = review_tool_limit
        self.memory = memory if memory is not None else DecisionMemory()
        self.context_compactor = context_compactor if context_compactor is not None else BasicContextCompactor()
        if not callable(getattr(self.context_compactor, 'compact', None)):
            raise TypeError('context_compactor requires compact(context, recent_limit=...)')
        self.research_agenda = research_agenda if research_agenda is not None else ResearchAgenda()
        self.memory_limit = memory_limit
        self.experience_limit = experience_limit
        self.model_activator = model_activator
        self.state = DecisionState()
        from .tools.research import ResearchTools
        research_tools = getattr(tools, 'research_tools', None)
        if isinstance(research_tools, ResearchTools):
            research_tools.decision_graph = self._decision_graph
            research_tools.ledger_providers.update({name: (lambda name=name:
                [asdict(row) for row in getattr(self.state, name).values()])
                for name in ('candidates', 'candidate_evaluations', 'directive_provenance')})
            research_tools.ledger_providers['candidate_origins'] = lambda: [
                {'candidate_id': key, 'decision_id': value}
                for key, value in self.state.candidate_origins.items()]
        self.training = JobBook[DecisionTrainingRequest, TrainingResult](learner)

    def add_feedback_listener(self, listener):
        self._feedback_listeners.append(listener)

    def add_receipt_listener(self, listener):
        """Observe recorded local completions; subscribers never execute tools."""
        self._receipt_listeners.append(listener)

    def add_context_processor(self, processor):
        self._context_processors.append(processor)

    def add_cleanup(self, callback):
        self._cleanup_callbacks.append(callback)

    def close(self):
        callbacks, self._cleanup_callbacks = self._cleanup_callbacks, []
        failed, errors = [], []
        for callback in reversed(callbacks):
            try:
                callback()
            except BaseException as error:
                failed.append(callback)
                errors.append(error)
        # A timed-out feedback worker may need another close. Retry failed
        # callbacks only; successful resource releases must not run twice.
        self._cleanup_callbacks.extend(reversed(failed))
        if errors:
            for error in errors[1:]:
                errors[0].add_note(f'Additional harness cleanup failure: {type(error).__name__}: {error}')
            raise errors[0]

    def _project_context(self, context):
        from .research_state import bounded_evidence_view
        context = bounded_evidence_view(context)
        for processor in self._context_processors:
            context = processor(context)
        return context

    @staticmethod
    def _copy_run_control(context: dict[str, object]) -> dict[str, object]:
        if not isinstance(context, dict) or any(not isinstance(key, str) for key in context):
            raise TypeError('Run control context must be a dict with string keys')
        return deepcopy(context)

    def update_run_control(self, context: dict[str, object]) -> None:
        """Replace external running hints without changing research state or objectives."""
        self.state.run_control = self._copy_run_control(context)

    def pending_research_job(self) -> JobRef | None:
        """Report work awaiting external resume, without polling or cancelling it."""
        return self.state.pending_job

    def pending_records(self) -> bool:
        """Include unfinished reviews even when their tool receipts are recorded."""
        return bool(self.state.pending_operation_record is not None
                    or self.state.pending_review_record is not None
                    or self.state.review_session is not None
                    or self.state.review_records or self.state.review_sessions
                    or any(row.get('review_completion_record') is not None for row in self.state.decision_records.values()))

    def flush_pending_records(self) -> None:
        """Retry existing receipts only; never decide, execute tools or finish a review."""
        for row in self.state.decision_records.values():
            if row.get('review_completion_record') is not None:
                self._flush_review_completion(row)
        if self.state.pending_operation_record is not None:
            self._flush_operation_record()
        self._flush_review_query()
        if self.state.pending_review_record is not None:
            self._flush_review_record()
        for ident in tuple(dict.fromkeys((*self.state.review_sessions, *self.state.review_records))):
            with self._review_scope(ident):
                self._flush_review_query()
                if self.state.pending_review_record is not None:
                    self._flush_review_record()

    def restart_research(self, context: dict[str, object]) -> None:
        """Reopen a finished, drained cycle; retries of accepted rounds are no-ops."""
        context = self._copy_run_control(context)
        research_round = context.get('research_round')
        if type(research_round) is not int or research_round < 1:
            raise ValueError('research_round must be a positive integer')
        previous = self.state.research_cycles.get(research_round)
        if previous is not None:
            if previous != context:
                raise ValueError('Research round context differs from its original request')
            return
        if not self.state.finished:
            raise RuntimeError('Research can restart only from a finished state')
        if (self.state.pending_directive_id is not None
                or self.state.pending_directive is not None
                or self.state.outstanding
                or self.state.feedback_record_id is not None
                or self.state.feedback_record_ids
                or self.state.pending_job is not None
                or self.state.pending_operation is not None
                or self.state.pending_operation_record is not None
                or self.state.pending_review_record is not None
                or self.state.review_session is not None
                or self.state.review_sessions
                or self.state.review_records):
            raise RuntimeError('Cannot restart research with pending directives, jobs, operations or reviews')
        run_control = deepcopy(context)
        self.state.research_cycles[research_round] = context
        self.state.run_control = run_control
        self.state.finished = False

    def _record_operation(self, call: OperationCall | None, result: OperationResult,
                          *, elapsed_seconds: float | None = None,
                          resume_result: OperationResult | None = None) -> None:
        if self.state.pending_operation_record is not None:
            raise RuntimeError('Record the pending operation before starting another')
        ident = self.state.pending_operation_decision_id
        suffix = 'authorization' if self.state.decision_records.get(ident, {}).get('authorization_rejected') else 'operation'
        self.state.pending_operation_record = _PendingOperationRecord(
            f'{ident}:{suffix}' if ident else str(uuid4()), deepcopy(call), deepcopy(result), elapsed_seconds,
            self.state.pending_job, deepcopy(resume_result),
            decision_id=ident,
        )
        self.state.pending_decision = None
        self._flush_operation_record()

    def _decision_graph(self):
        graph = []
        for row in self.state.decision_records.values():
            item = {key: row[key] for key in ('decision_id', 'status')}
            item.update(attempt_details(row))
            directive = row['proposal'].get('directive')
            if directive:
                item['mode'] = directive['mode']
            for key in ('operation', 'depends_on_decision_ids', 'source_decision_ids', 'feedback_proposal_ids', 'adopted_candidate_ids',
                        'adopted_decision_ids', 'unresolved_adopted_candidate_ids'):
                if row.get(key):
                    item[key] = row[key]
            graph.append(item)
        return deepcopy(tuple(graph))

    def _begin_decision(self, decision, context, *, record_history=True):
        if not isinstance(decision, Decision):
            raise TypeError('Policy must return a Decision')
        decision.__post_init__()
        ident = decision.decision_id
        row = self.state.decision_records.get(ident)
        if row is not None:
            if {'research_attempt_id': None, 'research_attempt_title': '',
                    'feedback_proposal_ids': (), **row['proposal']} != asdict(decision):
                raise ValueError('Decision ID has different content')
            if row['status'] == 'executing':
                raise RuntimeError('Decision execution outcome is unknown; refusing to execute it again')
            if (row['status'] in ('blocked', 'failed') and not row.get('authorization_rejected') and ident not in self.state.decision_results
                    and self.state.pending_decision is None):
                raise ValueError(row['error'])
        else:
            row = dict(decision_id=ident, kind=decision.kind, status='pending', reason=decision.reason,
                directive_id=decision.directive.directive_id if decision.directive else None,
                operation=decision.operation.name if decision.operation else None,
                depends_on_decision_ids=decision.depends_on_decision_ids,
                source_decision_ids=decision.source_decision_ids,
                feedback_proposal_ids=decision.feedback_proposal_ids,
                adopted_candidate_ids=(), adopted_decision_ids=(), unresolved_adopted_candidate_ids=(),
                confirmed_edges=(), error=None, validation_error=None,
                research_attempt_id=None, research_attempt_title='',
                proposal=deepcopy(asdict(decision)),
                objective=context.objective,
                operation_read_only=next((spec.read_only for spec in context.operations
                    if decision.operation is not None and spec.name == decision.operation.name), False),
                decision_model=asdict(context.decision_model) if context.decision_model else None,
                execution_model=asdict(context.execution.model) if context.execution.model else None)
            for name in ('depends_on_decision_ids', 'source_decision_ids'):
                for ref in getattr(decision, name):
                    known = self.state.decision_records.get(ref)
                    if known is None or (name == 'depends_on_decision_ids' and known['status'] != 'completed'):
                        message = f'Unknown, forward or incomplete decision dependency/reference: {ref}'
                        row.update(status='failed', error=message, validation_error=message)
                        break
                if row['validation_error']:
                    break
            if not row['validation_error']:
                try:
                    if decision.feedback_proposal_ids:
                        if not callable(self.feedback_proposal_resolver):
                            raise ValueError('No feedback proposal provider is installed')
                        row['feedback_proposals'] = self.feedback_proposal_resolver(decision.feedback_proposal_ids)
                    self.attempt_contract.validate(decision.kind,
                        decision.operation.name if decision.operation else None,
                        decision.research_attempt_id, decision.research_attempt_title)
                    title = requested_attempt(decision.research_attempt_id, decision.research_attempt_title,
                                              self.state.decision_records.values())
                    row.update(research_attempt_id=(decision.research_attempt_id or str(uuid4())) if title else None,
                               research_attempt_title=title)
                except ValueError as error:
                    row.update(status='failed', error=str(error), validation_error=str(error),
                               research_attempt_id=None, research_attempt_title='')
            if not row['validation_error']:
                row['confirmed_edges'] = tuple({'from': ref, 'to': ident, 'kind': kind, 'confirmed': True}
                    for kind, refs in (('depends_on', decision.depends_on_decision_ids), ('source', decision.source_decision_ids))
                    for ref in refs)
            self.state.decision_records[ident] = row
        self.state.pending_decision = deepcopy(decision)
        if record_history or row['validation_error']:
            self.memory.append(HistoryEntry(ident, 'decision', None, decision.kind, decision.simulated,
                details=_json_details({**deepcopy(row['proposal']), 'decision_model': row['decision_model'],
                         **attempt_details(row),
                         'execution_model': row['execution_model'],
                         'status': 'failed' if row['validation_error'] else 'pending',
                         'validation_error': row['validation_error']})))
        if row['validation_error']:
            self.state.pending_decision = None
            raise ValueError(row['error'])
        return row

    def _adoption(self, ident, receipt):
        receipt.__post_init__()
        adopted = receipt.adopted_candidate_ids
        origins, unresolved = [], []
        for candidate_id in adopted:
            source = self.state.candidate_origins.get(candidate_id)
            if (ident is None or source is None or source == ident
                    or self.state.decision_records.get(source, {}).get('status') != 'completed'):
                unresolved.append(candidate_id)
                continue
            if source not in origins:
                origins.append(source)
        row = self.state.decision_records.get(ident, {})
        edges = tuple(edge for edge in row.get('confirmed_edges', ()) if edge['kind'] != 'adopted')
        edges += tuple({'from': source, 'to': ident, 'kind': 'adopted', 'confirmed': True} for source in origins)
        return dict(decision_id=ident, **attempt_details(row), adopted_candidate_ids=adopted,
                    adopted_decision_ids=tuple(origins), unresolved_adopted_candidate_ids=tuple(unresolved), confirmed_edges=edges)

    def _write_operation_record(self, pending, *, experience=True):
        metadata = self._adoption(pending.decision_id, pending.result)
        if pending.event is not None and pending.job is not None and pending.resume_result is None:
            metadata['status'] = 'pending'
        if pending.decision_id is None:
            record_operation(pending, self.memory, self.experience if experience else None)
            return metadata
        result, call = pending.result, pending.call
        details = {'operation': asdict(call) if call else None, 'result': asdict(result), **metadata}
        if pending.event is not None:
            details.update(event=pending.event, elapsed_seconds=pending.elapsed_seconds)
        self.memory.append(HistoryEntry(pending.record_id, 'operation', pending.task_id, result.summary,
            result.simulated, source_refs=tuple(ref.artifact_id for ref in result.artifacts), details=_json_details(details)))
        if experience:
            self.experience.record(ExperienceRecord(pending.record_id, 'runtime_feedback', 'decision_maker',
                pending.task_id, f'Preparation operation {result.status.value}', result.simulated,
                source_refs=(f'history:decision_maker:{pending.record_id}',),
                details=_json_details({'operation': call.name if call else None, 'status': result.status.value,
                         'error': result.error, 'elapsed_seconds': pending.elapsed_seconds,
                         'history_ref': pending.record_id, 'history_owner': 'decision_maker', **metadata})))
        return metadata

    def _complete_operation(self, pending, metadata):
        if pending.decision_id is not None:
            if pending.event is not None and pending.job is not None and pending.resume_result is None:
                self.state.decision_records[pending.decision_id].update(metadata, status='pending',
                    review_job=pending.job, review_submission=deepcopy(pending))
                self.state.decision_results[pending.decision_id] = LoopStepResult('waiting',
                    job=pending.job, decision_id=pending.decision_id)
                return
            self.state.decision_records[pending.decision_id].update(metadata,
                status=pending.result.status.value, error=pending.result.error,
                operation_elapsed_seconds=pending.elapsed_seconds)
            if not self.state.decision_records[pending.decision_id].get('authorization_rejected'):
                self.state.decision_results[pending.decision_id] = LoopStepResult('progressed',
                    operation_result=deepcopy(pending.result), decision_id=pending.decision_id)
            if self._receipt_listeners:
                from rsisys.loops.contracts.receipts import ExecutionReceipt
                row = self.state.decision_records[pending.decision_id]
                objective = row.get('objective')
                if objective is not None:
                    receipt = ExecutionReceipt.from_operation(pending.record_id, objective,
                        pending.call, pending.result, decision_id=pending.decision_id,
                        read_only=row.get('operation_read_only', False), job=pending.job,
                        issuing_model=ModelRef(**row['decision_model']) if row.get('decision_model') else None)
                    # If observation persistence fails, the existing pending
                    # record retries the same receipt, without executing again.
                    for listener in self._receipt_listeners:
                        listener(receipt)

    def _flush_operation_record(self) -> OperationResult:
        pending = self.state.pending_operation_record
        if pending is None:
            raise RuntimeError('No operation recording is pending')
        result = pending.result
        metadata = self._write_operation_record(pending)
        self._complete_operation(pending, metadata)
        self.state.operation_results.append(deepcopy(result))
        if pending.job is not None:
            self.state.operation_receipts[(pending.job, pending.decision_id)] = deepcopy(pending.resume_result or result)
            self.state.pending_job = None
            self.state.pending_operation = None
            self.state.pending_simulated = False
        self.state.pending_operation_decision_id = None
        self.state.pending_operation_record = None
        return deepcopy(result)

    def update_execution(self, view: ExecutionView) -> None:
        if not isinstance(view, ExecutionView):
            raise TypeError('Expected an ExecutionView')
        self.state.execution = deepcopy(view)
        if view.progress is not None:
            self._bind_execution(view.progress.directive_id, view)

    def _bind_execution(self, ident, view):
        self.state.executions[ident] = deepcopy(view)
        provenance = self.state.directive_provenance.get(ident)
        if provenance is not None and provenance.execution_model is None and view.model is not None:
            self.state.directive_provenance[ident] = replace(provenance, execution_model=view.model)

    def _context(self, objective: str, action_space: DecisionActionSpace | None = None) -> DecisionContext:
        model = getattr(self.policy, 'model', None)
        context = DecisionContext(
            objective, deepcopy(tuple(self.feedback_selector.select(
                self.experience, objective, limit=self.experience_limit))),
            action_space.describe() if action_space is not None else (),
            deepcopy(tuple(self.state.operation_results)), self.memory.recall(limit=self.memory_limit),
            deepcopy(self.state.last_feedback), self.memory.history(),
            action_space.directive_modes() if action_space is not None and len(self.pending_all()) < self.max_outstanding else (),
            candidates=deepcopy(tuple(self.state.candidates.values())),
            execution=deepcopy(self.state.execution), decision_model=model if isinstance(model, ModelRef) else None,
            directive_provenance=deepcopy(tuple(self.state.directive_provenance.values())),
            candidate_evaluations=deepcopy(tuple(self.state.candidate_evaluations.values())),
            outstanding=self.pending_all(), executions=deepcopy(self.state.executions),
            run_control=deepcopy(self.state.run_control),
            research_agenda=self.research_agenda.context(),
            decision_graph=self._decision_graph(), candidate_origins=deepcopy(self.state.candidate_origins),
        )
        return self._project_context(context)

    def review(self, objective: str, progress: ExplorationProgress) -> ReviewDecision:
        return self._review(objective, progress)

    def review_with_tools(self, objective: str, progress: ExplorationProgress,
                          action_space: DecisionActionSpace | None = None) -> ReviewDecision:
        # Existing subclasses overriding the two-argument hook retain control.
        # An explicit review_with_tools override opts into the new capability.
        if getattr(self.review, '__func__', None) is not DefaultDecisionHarness.review:
            return self.review(objective, progress)
        return self._review(objective, progress, action_space)

    @contextmanager
    def _review_scope(self, directive_id):
        if self.max_outstanding == 1:
            yield
            return
        old_session, old_record = self.state.review_session, self.state.pending_review_record
        self.state.review_session = self.state.review_sessions.get(directive_id)
        self.state.pending_review_record = self.state.review_records.get(directive_id)
        try:
            yield
        finally:
            for mapping, value in ((self.state.review_sessions, self.state.review_session),
                                   (self.state.review_records, self.state.pending_review_record)):
                if value is None:
                    mapping.pop(directive_id, None)
                else:
                    mapping[directive_id] = value
            self.state.review_session, self.state.pending_review_record = old_session, old_record

    def _review(self, objective: str, progress: ExplorationProgress,
                action_space: DecisionActionSpace | None = None) -> ReviewDecision:
        if not isinstance(progress, ExplorationProgress):
            raise TypeError('Expected an ExplorationProgress')
        with self._review_scope(progress.directive_id):
            return self._review_one(objective, progress, action_space)

    def _review_one(self, objective: str, progress: ExplorationProgress,
                    action_space: DecisionActionSpace | None = None) -> ReviewDecision:
        if action_space is None:
            from ..action_space import ModularDecisionActionSpace
            action_space = ModularDecisionActionSpace((), directive_modes=())
        action_space = authorized_space(action_space, 'review')
        if not isinstance(progress, ExplorationProgress):
            raise TypeError('Expected an ExplorationProgress')
        if progress.directive != self.state.outstanding.get(progress.directive_id, self.state.pending_directive):
            raise ValueError('Review does not match the pending directive')
        pending = self.state.pending_review_record
        if pending is not None:
            if pending.objective != objective or pending.directive_id != progress.directive_id:
                raise ValueError('Review differs from the pending review recording')
            result = self._flush_review_record()
            action_space.validate_review(result)
            return result
        if progress.terminal:
            raise ValueError('Cannot review a terminal execution')
        self.update_execution(replace(self.state.execution, progress=progress))
        session = self.state.review_session
        if session is None:
            context = self._context(objective, action_space)
            context = replace(context, directive_modes=(), review_actions=action_space.review_actions())
            session = _ReviewSession(objective, progress.directive_id, context, self.review_tool_limit)
            self.state.review_session = session
        elif session.objective != objective or session.directive_id != progress.directive_id:
            raise ValueError('Review differs from the pending research session')
        session.context = replace(session.context, execution=deepcopy(self.state.execution),
                                  run_control=deepcopy(self.state.run_control))
        self._flush_review_query()
        review = getattr(self.policy, 'review', None)
        while True:
            session.context = replace(session.context,
                operations=action_space.describe() if session.queries_used < session.tool_limit else (),
                review_actions=action_space.review_actions())
            context = session.context
            result = self.state.pending_decision or (self._call_policy(review, context) if callable(review) else ReviewDecision())
            if not isinstance(result, (ReviewDecision, Decision)):
                raise TypeError('Policy review must return ReviewDecision or an operation invocation')
            if isinstance(result, ReviewDecision):
                result.__post_init__()
                action_space.validate_review(result)
                result = replace(result, simulated=result.simulated or session.simulated)
                break
            if (result.kind != 'invoke' or result.operation.name not in
                    {spec.name for spec in context.operations}):
                raise ValueError('Review may only invoke available operations within its budget')
            operation_spec = next(spec for spec in context.operations if spec.name == result.operation.name)
            existing = self.state.decision_records.get(result.decision_id)
            if existing is not None and existing.get('review_directive_id') not in (None, progress.directive_id):
                # Legacy custom policies may reuse one Decision object across
                # independent reviews. Allocate a system identity for this scope.
                result = replace(result, decision_id=str(uuid4()))
            row = self._begin_decision(result, context, record_history=False)
            row['review_directive_id'] = progress.directive_id
            prior = self.state.decision_results.get(result.decision_id)
            if prior is None:
                row['status'] = 'executing'
                execution = execute_operation(self.tools, action_space, result.operation,
                                              simulated=result.simulated, clock=perf_counter)
                observed = execution.observation()
                try:
                    self._adoption(result.decision_id, observed)
                except (TypeError, ValueError) as invalid:
                    observed = OperationResult('Invalid executor adoption receipt', (), observed.simulated,
                        ExecutionStatus.FAILED, str(invalid))
                call, elapsed = execution.call, execution.elapsed_seconds
                job = execution.outcome if isinstance(execution.outcome, JobRef) else None
            else:
                submission = row.get('review_submission') if prior.kind == 'waiting' else None
                observed = submission.result if submission else prior.operation_result
                call = submission.call if submission else result.operation
                elapsed = submission.elapsed_seconds if submission else row.get('operation_elapsed_seconds')
                job = submission.job if submission else None
            session.simulated = session.simulated or observed.simulated
            session.queries_used += 1
            session.pending_record = _PendingOperationRecord(
                f'{result.decision_id}:submission' if job else f'{result.decision_id}:operation',
                deepcopy(call), deepcopy(observed), elapsed, job,
                task_id=progress.directive_id,
                event='review_query' if operation_spec.read_only else 'review_operation', decision_id=result.decision_id)
            self.state.pending_decision = None
            self._flush_review_query()
        if not isinstance(result, ReviewDecision):
            raise TypeError('Policy review must return a ReviewDecision')
        result.__post_init__()
        self.state.pending_review_record = _PendingReviewRecord(
            str(uuid4()), objective, progress.directive_id, deepcopy(result),
            {"event": "review", "review": asdict(result), "execution": asdict(context.execution),
             "query_refs": list(session.query_refs),
             "decision_model": asdict(context.decision_model) if context.decision_model else None,
             "provenance": (asdict(self.state.directive_provenance[progress.directive_id])
                            if progress.directive_id in self.state.directive_provenance else None)},
        )
        self.state.review_session = None
        result = self._flush_review_record()
        action_space.validate_review(result)
        return result

    def _flush_review_query(self):
        session = self.state.review_session
        if session is None or session.pending_record is None:
            return
        pending = session.pending_record
        metadata = self._write_operation_record(pending, experience=False)
        self._complete_operation(pending, metadata)
        session.query_refs.append(f'history:decision_maker:{pending.record_id}')
        session.context = replace(session.context,
            operation_results=(*session.context.operation_results, deepcopy(pending.result)))
        session.pending_record = None

    def _refresh_context(self, context):
        """Refresh canonical observations; extensions may add derived work views."""
        return replace(context, history=self.memory.history(),
            research_agenda=self.research_agenda.context(), decision_graph=self._decision_graph(),
            candidate_origins=deepcopy(self.state.candidate_origins))

    def _call_policy(self, method, context):
        # Reviews can append new tool results to a saved session. Re-project them
        # before every model call; full receipts remain in memory and session state.
        context = self._project_context(self._refresh_context(context))
        # Model policies budget their final request after adding protocol/schema
        # overhead. Other policies receive the same protected context projection.
        if isinstance(self.policy, ModelDecisionPolicy):
            self.policy.context_compactor = self.context_compactor
        else:
            context = self.context_compactor.compact(context, recent_limit=self.memory_limit)
        received = 0

        def record(usage):
            nonlocal received
            self._record_usage(usage)
            received += 1

        if isinstance(self.policy, ModelDecisionPolicy):
            with self.policy.track_usage(record):
                result = method(context)
        else:
            result = method(context)
        if not received and isinstance(result, (Decision, ReviewDecision)):
            self._record_usage(result.usage)
        return result

    def _flush_review_record(self) -> ReviewDecision:
        pending = self.state.pending_review_record
        if pending is None:
            raise RuntimeError('No review recording is pending')
        result = pending.decision
        self.memory.append(HistoryEntry(
            pending.record_id, 'decision', pending.directive_id, f'Review: {result.action}',
            result.simulated, details=_json_details(pending.details),
        ))
        self.experience.record(ExperienceRecord(
            pending.record_id, 'runtime_feedback', 'decision_maker', pending.directive_id,
            f'Execution review: {result.action}', result.simulated,
            source_refs=(f'history:decision_maker:{pending.record_id}',), details=deepcopy(pending.details),
        ))
        self.state.pending_review_record = None
        return deepcopy(result)

    def can_step(self) -> bool:
        """Optional scheduler readiness; the serial feedback guard remains intact."""
        return self.max_outstanding > 1 or self.state.pending_directive_id is None

    def step(self, objective: str, action_space: DecisionActionSpace) -> LoopStepResult:
        action_space = authorized_space(action_space, 'decision')
        if self.state.pending_review_record is not None or self.state.review_session is not None:
            raise RuntimeError('Record the pending review before another decision')
        if self.state.pending_operation_record is not None:
            ident = self.state.pending_operation_record.decision_id
            return LoopStepResult("progressed", operation_result=self._flush_operation_record(), decision_id=ident)
        if self.state.finished:
            return LoopStepResult("finished")
        if self.max_outstanding == 1 and self.state.pending_directive_id is not None:
            raise RuntimeError("Consume pending exploration feedback before another decision.")
        if self.state.pending_job is not None:
            return LoopStepResult("waiting", job=self.state.pending_job,
                                  decision_id=self.state.pending_operation_decision_id)
        context = self._context(objective, action_space)
        decision = self.state.pending_decision or self._call_policy(self.policy.decide, context)
        row = self._begin_decision(decision, context)
        ident = decision.decision_id
        previous = self.state.decision_results.get(ident)
        if previous is not None:
            self.state.pending_decision = None
            return deepcopy(previous)
        try:
            if decision.kind == "invoke":
                # Denial occurs before any executor call. Existing callers may
                # retry after their external access policy changes.
                try:
                    action_space.validate(decision.operation)
                except PermissionError as denied:
                    row['authorization_rejected'] = True
                    self.state.pending_operation_decision_id = ident
                    self._record_operation(decision.operation, OperationResult('Decision operation failed', (),
                        decision.simulated, ExecutionStatus.FAILED, f'PermissionError: {denied}'))
                    raise
                except (TypeError, ValueError):
                    pass  # Shared executor records invalid calls without running tools.
                row.pop('authorization_rejected', None)
                # Mark before calling the executor. If a checkpoint catches this
                # boundary, an unknown outcome is never blindly executed again.
                row['status'] = 'executing'
                self.state.pending_operation_decision_id = ident
                execution = execute_operation(self.tools, action_space, decision.operation,
                                              simulated=decision.simulated, clock=perf_counter)
                if isinstance(execution.outcome, JobRef):
                    self.state.pending_job = execution.outcome
                    self.state.pending_operation = execution.call
                    self.state.pending_simulated = execution.simulated
                    row['status'] = 'pending'
                    result = LoopStepResult("waiting", job=execution.outcome, decision_id=ident)
                else:
                    observed, error = execution.observation(), execution.error
                    try:
                        self._adoption(ident, observed)
                    except (TypeError, ValueError) as invalid:
                        error = invalid
                        observed = OperationResult('Invalid executor adoption receipt', (), observed.simulated,
                            ExecutionStatus.FAILED, str(invalid))
                    try:
                        self._record_operation(execution.call, observed, elapsed_seconds=execution.elapsed_seconds)
                    except Exception as record_error:
                        if error is None:
                            raise
                        error.add_note(f"Could not record operation failure: {record_error}")
                    if error is not None:
                        raise error
                    return LoopStepResult("progressed", operation_result=observed, decision_id=ident)
            elif decision.kind == "explore":
                if len(self.pending_all()) >= self.max_outstanding:
                    raise RuntimeError("Outstanding experiment capacity exhausted; research or control existing experiments")
                directive = replace(decision.directive, simulated=decision.directive.simulated or decision.simulated)
                action_space.validate_directive(directive)
                if directive.directive_id in self.state.issued_directive_ids:
                    raise ValueError('A new decision must use a fresh directive ID')
                self.state.issued_directive_ids.add(directive.directive_id)
                self.state.issued += 1
                self.state.pending_directive_id = directive.directive_id
                self.state.pending_directive = deepcopy(directive)
                model = ModelRef(**row['decision_model']) if row['decision_model'] else None
                execution_model = ModelRef(**row['execution_model']) if row['execution_model'] else None
                self.state.directive_provenance[directive.directive_id] = DirectiveProvenance(
                    directive.directive_id, model, execution_model if self.max_outstanding == 1 else None)
                self.state.feedback_record_id = str(uuid4())
                self.state.outstanding[directive.directive_id] = deepcopy(directive)
                self.state.feedback_record_ids[directive.directive_id] = self.state.feedback_record_id
                self.state.directive_decisions[directive.directive_id] = ident
                result = LoopStepResult("dispatch", directive=directive, decision_id=ident)
            elif decision.kind == "wait":
                self.state.pending_job = decision.job
                self.state.pending_simulated = decision.simulated
                self.state.pending_operation_decision_id = ident if decision.job else None
                row['status'] = 'pending' if decision.job else 'completed'
                result = LoopStepResult("waiting", job=decision.job, decision_id=ident)
            else:
                self.state.finished = True
                row['status'] = 'completed'
                result = LoopStepResult("finished", reason=decision.reason, decision_id=ident)
        except Exception as error:
            if self.state.pending_operation_record is None:
                row.update(status='failed', error=f'{type(error).__name__}: {error}')
                self.state.pending_decision = None
                self.state.pending_operation_decision_id = None
            raise
        self.state.decision_results[ident] = deepcopy(result)
        self.state.pending_decision = None
        return result

    def _flush_review_completion(self, row):
        pending = row['review_completion_record']
        metadata = self._write_operation_record(pending)
        self._complete_operation(pending, metadata)
        self.state.operation_results.append(deepcopy(pending.result))
        self.state.operation_receipts[(pending.job, pending.decision_id)] = deepcopy(pending.resume_result)
        row.pop('review_completion_record')

    def resume(self, job: JobRef, result: OperationResult, *, decision_id: str | None = None) -> None:
        """Idempotent explicit completion handoff, correlated to its original decision."""
        if not isinstance(result, OperationResult):
            raise TypeError('Expected an OperationResult')
        result.__post_init__()
        if self.state.pending_job != job or (decision_id is not None and decision_id != self.state.pending_operation_decision_id):
            # Review jobs are independent observations, never the main wait slot.
            matches = [row for row in self.state.decision_records.values()
                       if row.get('review_job') == job and row['status'] == 'pending'
                       and (decision_id is None or row['decision_id'] == decision_id)]
            if len(matches) > 1:
                raise ValueError('Ambiguous review job; supply decision_id explicitly')
            if matches:
                row = matches[0]
                pending = row.get('review_completion_record')
                if pending is None:
                    submission = row['review_submission']
                    normalized = replace(result, simulated=result.simulated or submission.result.simulated)
                    self._adoption(row['decision_id'], normalized)
                    row['review_completion_record'] = _PendingOperationRecord(
                        f"{row['decision_id']}:operation", deepcopy(submission.call), deepcopy(normalized), None,
                        job, deepcopy(result), task_id=submission.task_id, event='review_completion',
                        decision_id=row['decision_id'])
                elif pending.resume_result != result:
                    raise ValueError('Result differs from the pending review completion')
                self._flush_review_completion(row)
                return
            if any(key[0] == job for key in self.state.operation_receipts):
                raise ValueError('Operation completion was already delivered')
            raise ValueError("Job does not match the pending operation")
        pending = self.state.pending_operation_record
        if pending is not None:
            if pending.job != job or pending.resume_result != result:
                raise ValueError('Result differs from the pending operation completion')
            self._flush_operation_record()
            return
        normalized = replace(result, simulated=result.simulated or self.state.pending_simulated)
        self._adoption(self.state.pending_operation_decision_id, normalized)
        self._record_operation(self.state.pending_operation, normalized, resume_result=result)

    def receive_feedback(self, directive: Directive, feedback: Feedback) -> None:
        with self._review_scope(directive.directive_id):
            self._receive_feedback(directive, feedback)
        for listener in self._feedback_listeners:
            listener(directive, feedback)

    def _receive_feedback(self, directive: Directive, feedback: Feedback) -> None:
        """Record the outcome and advance control state; retry the latest delivery safely."""
        if not isinstance(feedback, Feedback):
            raise TypeError('Expected a Feedback')
        feedback.__post_init__()
        previous = self.state.delivered_feedback.get(directive.directive_id)
        if previous is not None:
            if previous != (directive, feedback):
                raise ValueError('Delivered feedback differs from its original receipt')
            return
        if (not self.state.outstanding and self.state.pending_directive_id is None
                and directive == self.state.last_directive and feedback == self.state.last_feedback):
            return
        pending = self.state.outstanding.get(directive.directive_id, self.state.pending_directive)
        if feedback.directive_id != directive.directive_id or pending is None or pending.directive_id != directive.directive_id:
            raise ValueError('Feedback does not match the pending directive')
        if pending != directive:
            raise ValueError('Directive content differs from the dispatched request')
        record_id = self.state.feedback_record_ids.get(directive.directive_id, self.state.feedback_record_id)
        if (self.state.review_session is not None
                and self.state.review_session.directive_id == directive.directive_id):
            self._flush_review_query()
            self.state.review_session = None
        if (self.state.pending_review_record is not None
                and self.state.pending_review_record.directive_id == directive.directive_id):
            # Completion can overtake a review whose bookkeeping failed. Preserve
            # its evidence without returning (or executing) an obsolete cancel.
            self._flush_review_record()
        directive, feedback = deepcopy(directive), deepcopy(feedback)
        decision_id = self.state.directive_decisions.get(directive.directive_id)
        metadata = self._adoption(decision_id, feedback)
        provenance = self.state.directive_provenance.get(directive.directive_id)
        provenance_details = asdict(provenance) if provenance is not None else None
        candidates = dict(self.state.candidates)
        bases = dict(self.state.candidate_bases)
        origins = dict(self.state.candidate_origins)
        # Validate the entire batch before any write, including duplicate IDs
        # within this feedback. Publish the index only after both records succeed.
        for candidate in deepcopy(feedback.candidates):
            if candidate.candidate_id in candidates and candidates[candidate.candidate_id] != candidate:
                raise ValueError(f'Candidate ID has different content: {candidate.candidate_id}')
            candidates[candidate.candidate_id] = candidate
            if (feedback.status == ExecutionStatus.COMPLETED and directive.mode not in ('evaluate', 'apply_candidate')
                    and decision_id is not None and candidate.candidate_id not in self.state.candidates):
                origins.setdefault(candidate.candidate_id, decision_id)
            if directive.mode in ('train', 'sft'):
                origin = getattr(directive.payload, 'origin', None)
                base = (getattr(directive.payload, 'base', None) or
                        (origin.initial_model if origin else None) or
                        (provenance.execution_model if provenance else None))
                if isinstance(base, ModelRef):
                    if base.revision != candidate.base_version:
                        raise ValueError('Candidate base revision differs from the dispatched model')
                    if candidate.candidate_id in bases and bases[candidate.candidate_id] != base:
                        raise ValueError('Candidate has conflicting full model base identities')
                    bases[candidate.candidate_id] = base
        evaluations = dict(self.state.candidate_evaluations)
        evaluation = None
        if directive.mode == 'evaluate' and isinstance(directive.payload, EvaluationSpec):
            candidate = directive.payload.candidate
            if candidate.candidate_id in candidates and candidates[candidate.candidate_id] != candidate:
                raise ValueError(f'Candidate ID has different content: {candidate.candidate_id}')
            candidates[candidate.candidate_id] = deepcopy(candidate)
            evaluation = CandidateEvaluation(
                candidate.candidate_id, directive.directive_id, directive.payload.protocol_ref,
                directive.input_refs, feedback.status, deepcopy(feedback.quality_metrics),
                directive.simulated or feedback.simulated, feedback.evidence_refs,
            )
            evaluations[candidate.candidate_id] = evaluation
        self.memory.append(HistoryEntry(
            record_id=record_id, kind="feedback", task_id=directive.directive_id,
            summary=feedback.summary, simulated=feedback.simulated, source_refs=feedback.evidence_refs,
            details=_json_details({"directive": asdict(directive), "feedback": asdict(feedback),
                     "provenance": provenance_details, **metadata}),
        ))
        self.experience.record(ExperienceRecord(
            record_id=record_id, kind="improvement", scope="decision_maker",
            directive_id=directive.directive_id, summary=f"Strategy outcome: {feedback.status.value}",
            simulated=feedback.simulated, source_refs=feedback.evidence_refs,
            details={"strategy": directive.strategy, "mode": directive.mode,
                     "status": feedback.status.value, "error": feedback.error,
                     "quality_metrics": feedback.quality_metrics, "training_metrics": feedback.training_metrics,
                     "execution_metrics": feedback.execution_metrics,
                     "candidates": [asdict(c) for c in feedback.candidates],
                     "evaluation": asdict(evaluation) if evaluation else None,
                     "provenance": provenance_details, **metadata},
        ))
        self.state.candidates = candidates
        self.state.candidate_bases = bases
        self.state.candidate_evaluations = evaluations
        if decision_id is not None:
            self.state.decision_records[decision_id].update(metadata, status=feedback.status.value, error=feedback.error)
        self.state.candidate_origins = origins
        self.state.last_feedback = feedback
        self.state.last_directive = directive
        self.state.delivered_feedback[directive.directive_id] = (deepcopy(directive), deepcopy(feedback))
        self.state.outstanding.pop(directive.directive_id, None)
        self.state.feedback_record_ids.pop(directive.directive_id, None)
        self.state.executions.pop(directive.directive_id, None)
        oldest = next(iter(self.state.outstanding.values()), None)
        self.state.pending_directive = deepcopy(oldest)
        self.state.pending_directive_id = oldest.directive_id if oldest else None
        self.state.feedback_record_id = self.state.feedback_record_ids.get(self.state.pending_directive_id)
        if self.max_outstanding == 1:
            self.state.operation_results.clear()
        self.state.completed += 1

    def pending(self) -> Directive | None:
        if len(self.state.outstanding) > 1:
            raise RuntimeError('Multiple directives: use pending_all()')
        return deepcopy(self.state.pending_directive)

    def pending_all(self) -> tuple[Directive, ...]:
        if self.state.outstanding:
            return deepcopy(tuple(self.state.outstanding.values()))
        return () if self.state.pending_directive is None else (deepcopy(self.state.pending_directive),)

    def update_executions(self, views: dict[str, ExecutionView]) -> None:
        if not isinstance(views, dict) or any(not isinstance(view, ExecutionView) or
                (view.progress is not None and view.progress.directive_id != ident)
                for ident, view in views.items()):
            raise ValueError('Execution views must be indexed by their directive IDs')
        self.state.executions = deepcopy(views)
        for ident, view in views.items():
            self._bind_execution(ident, view)

    def compare_candidates(self, candidate_ids, *, metric: str, higher_is_better: bool = True):
        """Rank completed evaluations only when base, protocol and inputs agree."""
        import math
        ids = tuple(candidate_ids)
        if len(ids) < 2 or len(set(ids)) != len(ids):
            raise ValueError('Comparison requires at least two distinct candidates')
        candidates = [self.state.candidates[ident] for ident in ids]
        rows = [self.state.candidate_evaluations[ident] for ident in ids]
        first = rows[0]
        parents = [candidate.parent_system for candidate in candidates]
        if any(parent is not None for parent in parents):
            if any(parent is None or parent != parents[0] for parent in parents):
                raise ValueError('System comparison requires the same explicit parent system')
            # A prompt/data candidate need not produce new weights. Same-parent
            # ranking compares systems, not an equal-compute training experiment.
        else:
            bases = [self.state.candidate_bases.get(ident) for ident in ids]
            if any(base is None for base in bases):
                raise ValueError('Comparison requires known full training base identities')
            if any(base != bases[0] for base in bases[1:]):
                raise ValueError('Cannot compare candidates from different full model bases')
            if any(candidate.base_version != candidates[0].base_version for candidate in candidates):
                raise ValueError('Cannot compare candidates from different bases')
        for candidate, row in zip(candidates, rows):
            if (row.protocol_ref, row.input_refs, row.simulated) != (first.protocol_ref, first.input_refs, first.simulated):
                raise ValueError('Cannot compare different evaluation protocols, inputs or simulation modes')
            if row.status != ExecutionStatus.COMPLETED or metric not in row.quality_metrics:
                raise ValueError('Comparison requires completed evaluations with the requested metric')
            value = row.quality_metrics[metric]
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError('Comparison metric must be finite')
        return deepcopy(tuple(sorted(rows, key=lambda row: row.quality_metrics[metric], reverse=higher_is_better)))

    def submit_learning(self, request: DecisionTrainingRequest, *, idempotency_key: str) -> JobRef:
        """Submit local learning explicitly; no cross-loop scheduling is implied."""
        if not isinstance(request, DecisionTrainingRequest):
            raise TypeError("Expected DecisionTrainingRequest")
        available = {r.record_id: r for r in self.experience.retrieve()}
        if any(ref not in available for ref in request.experience_refs):
            raise ValueError("Training references must identify visible experience records")
        return self.training.submit(request, idempotency_key=idempotency_key)

    def poll_learning(self, job: JobRef) -> JobOutcome[TrainingResult]:
        outcome = self.training.poll(job, normalize=self._normalize_training_metrics)
        return self._record_learning(outcome)

    def cancel_learning(self, job: JobRef) -> JobOutcome[TrainingResult]:
        return self._record_learning(self.training.cancel(job, normalize=self._normalize_training_metrics))

    def _normalize_training_metrics(self, outcome):
        result = outcome.result
        if outcome.status != JobStatus.SUCCEEDED or not isinstance(result, TrainingResult):
            return outcome
        request = self.training.request(outcome.job)
        if result.base != request.base or 'decision_maker' not in result.candidate.affected_roles:
            return outcome  # Preserve the strict identity/type failure path below.
        try:
            validated_metrics(result.metrics, 'training metrics')
        except InvalidMetrics:
            return replace(outcome, status=JobStatus.FAILED, result=None,
                simulated=outcome.simulated or result.simulated,
                error='Invalid training metrics: expected nonempty names and finite numeric values')
        return outcome

    def _record_learning(self, outcome: JobOutcome[TrainingResult]) -> JobOutcome[TrainingResult]:
        request = self.training.request(outcome.job)
        if outcome.status == JobStatus.SUCCEEDED:
            result = outcome.result
            if not isinstance(result, TrainingResult) or result.base != request.base:
                raise ValueError("Training result does not match the submitted base")
            if "decision_maker" not in result.candidate.affected_roles:
                raise ValueError("Training result does not target this loop's role")
        if outcome.terminal and outcome.job not in self.state.recorded_training:
            self.experience.record(ExperienceRecord(
                record_id=self.state.training_record_ids.setdefault(outcome.job, str(uuid4())),
                kind="training_run", scope="decision_maker", directive_id=None,
                summary=f"decision_maker learning {outcome.status}",
                simulated=outcome.simulated or bool(outcome.result and outcome.result.simulated),
                source_refs=request.experience_refs + tuple(ref.artifact_id for ref in request.data_refs)
                + (tuple(ref.artifact_id for ref in outcome.result.candidate.changes.values())
                   if outcome.result else ()),
                details={"request": asdict(request), "outcome": asdict(outcome)},
            ))
            self.state.recorded_training.add(outcome.job)
        return outcome

    def apply_learning(self, job: JobRef) -> ModelRef:
        """Explicitly bind a learned version to this policy only."""
        outcome = self.poll_learning(job)
        if outcome.status != JobStatus.SUCCEEDED:
            raise RuntimeError(f"Training has no usable result: {outcome.status}")
        result = outcome.result
        if job in self.state.applied_training:
            if getattr(self.policy, 'model', None) != result.model:
                raise ValueError('Applied training result is stale relative to this policy')
            return result.model
        if not isinstance(self.policy, ModelUpdatable):
            raise TypeError("This policy does not support model updates")
        if self.policy.model != result.base:
            raise ValueError("Training base is stale relative to this policy")
        if self.training.busy:
            raise RuntimeError("Cannot replace the model while another training job is active")
        ready = prepare_model_activation(result, self.model_activator,
            idempotency_key=json.dumps(('decision_making', job.backend, job.job_id)))
        self.policy.update_model(ready)
        self.state.applied_training.add(job)
        return result.model

    def _record_usage(self, usage) -> None:
        if usage is None or usage.input_tokens is None or usage.output_tokens is None:
            self.state.calls_with_missing_usage += 1
        if usage is not None:
            self.state.measured_input_tokens += usage.input_tokens or 0
            self.state.measured_output_tokens += usage.output_tokens or 0
