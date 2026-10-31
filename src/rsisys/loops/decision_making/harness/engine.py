from dataclasses import asdict, replace
from copy import deepcopy
from uuid import uuid4
from time import perf_counter

from rsisys.loops._memory import MemoryStore, HistoryEntry
from .memory import DecisionMemory

from rsisys.experience.records import ExperienceRecord
from rsisys.experience.retrieval import ExperienceSystem
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.jobs import JobRef, JobOutcome, JobStatus
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.training import TrainingResult
from rsisys.loops.contracts.model_service import ModelRef, ModelUpdatable
from rsisys.runtime.jobs import JobBook

from rsisys.loops.contracts.coordination import LoopStepResult
from ..action_space import DecisionActionSpace, OperationCall, OperationResult
from .learner import DecisionLearner, DecisionTrainingRequest
from .policy import DecisionPolicy
from .tools import DecisionTools

from .state import DecisionContext, DecisionState, _PendingOperationRecord


class DefaultDecisionHarness:
    def __init__(self, policy: DecisionPolicy, tools: DecisionTools,
                 experience: ExperienceSystem, learner: DecisionLearner | None = None,
                 *, experience_limit: int = 32, memory: MemoryStore | None = None, memory_limit: int = 32):
        self.policy, self.tools, self.experience, self.learner = policy, tools, experience, learner
        if type(experience_limit) is not int or experience_limit < 1:
            raise ValueError("experience_limit must be a positive integer")
        if type(memory_limit) is not int or memory_limit < 1:
            raise ValueError("memory_limit must be a positive integer")
        self.memory = memory if memory is not None else DecisionMemory()
        self.memory_limit = memory_limit
        self.experience_limit = experience_limit
        self.state = DecisionState()
        self.training = JobBook[DecisionTrainingRequest, TrainingResult](learner)

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
        ident, call, result = pending.record_id, pending.call, pending.result
        self.memory.append(HistoryEntry(
            record_id=ident, kind="operation", task_id=None, summary=result.summary,
            simulated=result.simulated, source_refs=tuple(ref.artifact_id for ref in result.artifacts),
            details={"operation": asdict(call) if call else None, "result": asdict(result)},
        ))
        self.experience.record(ExperienceRecord(
            record_id=ident, kind="runtime_feedback", scope="decision_maker", directive_id=None,
            summary=f"Preparation operation {result.status.value}", simulated=result.simulated,
            source_refs=(f"history:decision_maker:{ident}",),
            details={"operation": call.name if call else None, "status": result.status.value,
                     "error": result.error, "elapsed_seconds": pending.elapsed_seconds,
                     "history_ref": ident, "history_owner": "decision_maker"},
        ))
        self.state.operation_results.append(deepcopy(result))
        if pending.job is not None:
            self.state.pending_job = None
            self.state.pending_operation = None
            self.state.pending_simulated = False
        self.state.pending_operation_record = None
        return deepcopy(result)

    def step(self, objective: str, action_space: DecisionActionSpace) -> LoopStepResult:
        if self.state.pending_operation_record is not None:
            return LoopStepResult("progressed", operation_result=self._flush_operation_record())
        if self.state.finished:
            return LoopStepResult("finished")
        if self.state.pending_directive_id is not None:
            raise RuntimeError("Consume pending exploration feedback before another decision.")
        if self.state.pending_job is not None:
            return LoopStepResult("waiting", job=self.state.pending_job)
        context = DecisionContext(objective, self.experience.retrieve(limit=self.experience_limit),
                                  action_space.describe(), tuple(self.state.operation_results),
                                  self.memory.recall(limit=self.memory_limit),
                                  deepcopy(self.state.last_feedback),
                                  self.memory.history(limit=self.memory_limit), action_space.directive_modes())
        decision = self.policy.decide(context)
        self._record_usage(decision.usage)
        self.memory.append(HistoryEntry(str(uuid4()), "decision", None, decision.kind,
                                        decision.simulated, details=asdict(decision)))
        if decision.kind == "invoke":
            started = perf_counter()
            call = decision.operation
            try:
                call = action_space.validate(call)
                result = self.tools.execute(call)
            except Exception as exc:
                failure = OperationResult("Decision operation failed", (), decision.simulated,
                                          ExecutionStatus.FAILED, f"{type(exc).__name__}: {exc}")
                try:
                    self._record_operation(call, failure, elapsed_seconds=perf_counter() - started)
                except Exception as record_error:
                    exc.add_note(f"Could not record operation failure: {record_error}")
                raise
            if isinstance(result, JobRef):
                self.state.pending_job = result
                self.state.pending_operation = deepcopy(call)
                self.state.pending_simulated = decision.simulated
                return LoopStepResult("waiting", job=result)
            result = replace(result, simulated=result.simulated or decision.simulated)
            self._record_operation(call, result, elapsed_seconds=perf_counter() - started)
            return LoopStepResult("progressed", operation_result=result)
        if decision.kind == "explore":
            directive = replace(decision.directive,
                                simulated=decision.directive.simulated or decision.simulated)
            action_space.validate_directive(directive)
            if directive.directive_id in self.state.issued_directive_ids:
                raise ValueError('A new decision must use a fresh directive ID')
            self.state.issued_directive_ids.add(directive.directive_id)
            self.state.issued += 1
            self.state.pending_directive_id = directive.directive_id
            self.state.pending_directive = deepcopy(directive)
            self.state.feedback_record_id = str(uuid4())
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
        """Record the outcome and advance control state; retry the latest delivery safely."""
        if (self.state.pending_directive_id is None
                and directive == self.state.last_directive and feedback == self.state.last_feedback):
            return
        if not (feedback.directive_id == directive.directive_id == self.state.pending_directive_id):
            raise ValueError("Feedback does not match the pending directive.")
        if self.state.pending_directive != directive:
            raise ValueError('Directive content differs from the dispatched request')
        self.memory.append(HistoryEntry(
            record_id=self.state.feedback_record_id, kind="feedback", task_id=directive.directive_id,
            summary=feedback.summary, simulated=feedback.simulated, source_refs=feedback.evidence_refs,
            details={"directive": asdict(directive), "feedback": asdict(feedback)},
        ))
        self.experience.record(ExperienceRecord(
            record_id=self.state.feedback_record_id, kind="improvement", scope="decision_maker",
            directive_id=directive.directive_id, summary=f"Strategy outcome: {feedback.status.value}",
            simulated=feedback.simulated, source_refs=feedback.evidence_refs,
            details={"strategy": directive.strategy, "mode": directive.mode,
                     "status": feedback.status.value, "error": feedback.error,
                     "quality_metrics": feedback.quality_metrics, "training_metrics": feedback.training_metrics,
                     "execution_metrics": feedback.execution_metrics,
                     "candidates": [asdict(c) for c in feedback.candidates]},
        ))
        self.state.last_feedback = deepcopy(feedback)
        self.state.last_directive = deepcopy(directive)
        self.state.pending_directive_id = None
        self.state.pending_directive = None
        self.state.feedback_record_id = None
        self.state.operation_results.clear()
        self.state.completed += 1

    def pending(self) -> Directive | None:
        return deepcopy(self.state.pending_directive)

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
                source_refs=request.experience_refs + tuple(
                    ref.artifact_id for ref in outcome.result.candidate.changes.values()
                ) if outcome.result else request.experience_refs,
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
            return result.model
        if not isinstance(self.policy, ModelUpdatable):
            raise TypeError("This policy does not support model updates")
        if self.policy.model != result.base:
            raise ValueError("Training base is stale relative to this policy")
        if self.training.busy:
            raise RuntimeError("Cannot replace the model while another training job is active")
        self.policy.update_model(result.model)
        self.state.applied_training.add(job)
        return result.model

    def _record_usage(self, usage) -> None:
        if usage is None or usage.input_tokens is None or usage.output_tokens is None:
            self.state.calls_with_missing_usage += 1
        if usage is not None:
            self.state.measured_input_tokens += usage.input_tokens or 0
            self.state.measured_output_tokens += usage.output_tokens or 0
