import json
from contextlib import contextmanager
from dataclasses import asdict, replace
from copy import deepcopy
from uuid import uuid4
from time import perf_counter

from rsisys.loops._memory import MemoryStore, HistoryEntry
from .memory import DecisionMemory

from rsisys.experience.records import ExperienceRecord
from rsisys.experience.retrieval import ExperienceSystem, RecentFeedbackSelector
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.jobs import JobRef, JobOutcome, JobStatus
from rsisys.loops.contracts.lifecycle import ExecutionStatus
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
from .tools import DecisionTools, execute_operation, record_operation

from .state import (DecisionContext, DecisionState, _PendingOperationRecord, _PendingReviewRecord, _ReviewSession,
                    DirectiveProvenance, CandidateEvaluation)


class DefaultDecisionHarness:
    def __init__(self, policy: DecisionPolicy, tools: DecisionTools,
                 experience: ExperienceSystem, learner: DecisionLearner | None = None,
                 *, experience_limit: int = 32, memory: MemoryStore | None = None, memory_limit: int = 32,
                 model_activator: ModelActivator | None = None, review_tool_limit: int = 4,
                 max_outstanding: int = 1, feedback_selector=None):
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
        self.memory_limit = memory_limit
        self.experience_limit = experience_limit
        self.model_activator = model_activator
        self.state = DecisionState()
        self.training = JobBook[DecisionTrainingRequest, TrainingResult](learner)

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
                    or self.state.review_records or self.state.review_sessions)

    def flush_pending_records(self) -> None:
        """Retry existing receipts only; never decide, execute tools or finish a review."""
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
        self.state.pending_operation_record = _PendingOperationRecord(
            str(uuid4()), deepcopy(call), deepcopy(result), elapsed_seconds,
            self.state.pending_job, deepcopy(resume_result),
        )
        self._flush_operation_record()

    def _flush_operation_record(self) -> OperationResult:
        pending = self.state.pending_operation_record
        if pending is None:
            raise RuntimeError('No operation recording is pending')
        result = pending.result
        record_operation(pending, self.memory, self.experience)
        self.state.operation_results.append(deepcopy(result))
        if pending.job is not None:
            self.state.pending_job = None
            self.state.pending_operation = None
            self.state.pending_simulated = False
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
        return DecisionContext(
            objective, deepcopy(tuple(self.feedback_selector.select(
                self.experience, objective, limit=self.experience_limit))),
            action_space.describe() if action_space is not None else (),
            deepcopy(tuple(self.state.operation_results)), self.memory.recall(limit=self.memory_limit),
            deepcopy(self.state.last_feedback), self.memory.history(limit=self.memory_limit),
            action_space.directive_modes() if action_space is not None and len(self.pending_all()) < self.max_outstanding else (),
            candidates=deepcopy(tuple(self.state.candidates.values())),
            execution=deepcopy(self.state.execution), decision_model=model if isinstance(model, ModelRef) else None,
            directive_provenance=deepcopy(tuple(self.state.directive_provenance.values())),
            candidate_evaluations=deepcopy(tuple(self.state.candidate_evaluations.values())),
            outstanding=self.pending_all(), executions=deepcopy(self.state.executions),
            run_control=deepcopy(self.state.run_control),
        )

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
            result = self._call_policy(review, context) if callable(review) else ReviewDecision()
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
            execution = execute_operation(self.tools, action_space, result.operation,
                                          simulated=result.simulated, clock=perf_counter)
            observed = execution.observation()
            session.simulated = session.simulated or observed.simulated
            session.queries_used += 1
            session.pending_record = _PendingOperationRecord(
                str(uuid4()), execution.call, observed, execution.elapsed_seconds, None,
                task_id=progress.directive_id,
                event='review_query' if operation_spec.read_only else 'review_operation')
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
        record_operation(pending, self.memory)
        session.query_refs.append(f'history:decision_maker:{pending.record_id}')
        session.context = replace(session.context,
            operation_results=(*session.context.operation_results, deepcopy(pending.result)))
        session.pending_record = None

    def _call_policy(self, method, context):
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
            result.simulated, details=deepcopy(pending.details),
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
            return LoopStepResult("progressed", operation_result=self._flush_operation_record())
        if self.state.finished:
            return LoopStepResult("finished")
        if self.max_outstanding == 1 and self.state.pending_directive_id is not None:
            raise RuntimeError("Consume pending exploration feedback before another decision.")
        if self.state.pending_job is not None:
            return LoopStepResult("waiting", job=self.state.pending_job)
        context = self._context(objective, action_space)
        decision = self._call_policy(self.policy.decide, context)
        self.memory.append(HistoryEntry(str(uuid4()), "decision", None, decision.kind,
                                        decision.simulated, details={**asdict(decision),
                                            "decision_model": asdict(context.decision_model) if context.decision_model else None,
                                            "execution_model": asdict(context.execution.model) if context.execution.model else None}))
        if decision.kind == "invoke":
            execution = execute_operation(self.tools, action_space, decision.operation,
                                          simulated=decision.simulated, clock=perf_counter)
            if isinstance(execution.outcome, JobRef):
                self.state.pending_job = execution.outcome
                self.state.pending_operation = execution.call
                self.state.pending_simulated = execution.simulated
                return LoopStepResult("waiting", job=execution.outcome)
            result = execution.observation()
            try:
                self._record_operation(execution.call, result, elapsed_seconds=execution.elapsed_seconds)
            except Exception as record_error:
                if execution.error is None:
                    raise
                execution.error.add_note(f"Could not record operation failure: {record_error}")
            if execution.error is not None:
                raise execution.error
            return LoopStepResult("progressed", operation_result=result)
        if decision.kind == "explore":
            if len(self.pending_all()) >= self.max_outstanding:
                raise RuntimeError("Outstanding experiment capacity exhausted; research or control existing experiments")
            directive = replace(decision.directive,
                                simulated=decision.directive.simulated or decision.simulated)
            action_space.validate_directive(directive)
            if directive.directive_id in self.state.issued_directive_ids:
                raise ValueError('A new decision must use a fresh directive ID')
            self.state.issued_directive_ids.add(directive.directive_id)
            self.state.issued += 1
            self.state.pending_directive_id = directive.directive_id
            self.state.pending_directive = deepcopy(directive)
            self.state.directive_provenance[directive.directive_id] = DirectiveProvenance(
                directive.directive_id, context.decision_model,
                context.execution.model if self.max_outstanding == 1 else None)
            self.state.feedback_record_id = str(uuid4())
            self.state.outstanding[directive.directive_id] = deepcopy(directive)
            self.state.feedback_record_ids[directive.directive_id] = self.state.feedback_record_id
            return LoopStepResult("dispatch", directive=directive)
        if decision.kind == "wait":
            self.state.pending_job = decision.job
            self.state.pending_simulated = decision.simulated
            return LoopStepResult("waiting", job=decision.job)
        self.state.finished = True
        return LoopStepResult("finished", reason=decision.reason)

    def resume(self, job: JobRef, result: OperationResult) -> None:
        """Explicit completion handoff; automatic polling remains future work."""
        if self.state.pending_job != job:
            raise ValueError("Job does not match the pending operation")
        if not isinstance(result, OperationResult):
            raise TypeError('Expected an OperationResult')
        pending = self.state.pending_operation_record
        if pending is not None:
            if pending.job != job or pending.resume_result != result:
                raise ValueError('Result differs from the pending operation completion')
            self._flush_operation_record()
            return
        normalized = replace(result, simulated=result.simulated or self.state.pending_simulated)
        self._record_operation(self.state.pending_operation, normalized, resume_result=result)

    def receive_feedback(self, directive: Directive, feedback: Feedback) -> None:
        with self._review_scope(directive.directive_id):
            self._receive_feedback(directive, feedback)

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
        provenance = self.state.directive_provenance.get(directive.directive_id)
        provenance_details = asdict(provenance) if provenance is not None else None
        candidates = dict(self.state.candidates)
        bases = dict(self.state.candidate_bases)
        # Validate the entire batch before any write, including duplicate IDs
        # within this feedback. Publish the index only after both records succeed.
        for candidate in deepcopy(feedback.candidates):
            if candidate.candidate_id in candidates and candidates[candidate.candidate_id] != candidate:
                raise ValueError(f'Candidate ID has different content: {candidate.candidate_id}')
            candidates[candidate.candidate_id] = candidate
            if directive.mode in ('train', 'sft'):
                base = getattr(directive.payload, 'base', None) or (provenance.execution_model if provenance else None)
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
            details={"directive": asdict(directive), "feedback": asdict(feedback),
                     "provenance": provenance_details},
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
                     "provenance": provenance_details},
        ))
        self.state.candidates = candidates
        self.state.candidate_bases = bases
        self.state.candidate_evaluations = evaluations
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
        bases = [self.state.candidate_bases.get(ident) for ident in ids]
        if any(base is None for base in bases):
            raise ValueError('Comparison requires known full training base identities')
        if any(base != bases[0] for base in bases[1:]):
            raise ValueError('Cannot compare candidates from different full model bases')
        for candidate, row in zip(candidates, rows):
            if candidate.base_version != candidates[0].base_version:
                raise ValueError('Cannot compare candidates from different bases')
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
        outcome = self.training.poll(job)
        return self._record_learning(outcome)

    def cancel_learning(self, job: JobRef) -> JobOutcome[TrainingResult]:
        return self._record_learning(self.training.cancel(job))

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
