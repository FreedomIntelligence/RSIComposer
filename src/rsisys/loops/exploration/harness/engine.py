from dataclasses import asdict, replace
from copy import deepcopy
from uuid import uuid4
from time import perf_counter

from rsisys.loops._memory import MemoryStore, HistoryEntry
from .memory import ExplorationMemory

from rsisys.experience.records import ExperienceRecord
from rsisys.experience.retrieval import ExperienceSystem
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.jobs import JobRef, JobOutcome, JobStatus

from ..action_space import ActionSpace
from .evaluation import Evaluator, EvaluationRequest, EvaluationResult
from .rl import RLPlan, RLState
from .environment import Environment
from .learner import ActorLearner, ActorTrainingRequest
from .policy import ActorContext, ActorPolicy
from rsisys.loops.contracts.training import TrainingResult
from rsisys.loops.contracts.model_service import ModelRef, ModelUpdatable
from rsisys.runtime.jobs import JobBook

from .state import ExplorationState, PendingEpisode
from .commands import ExplorationCommands


class DefaultExplorationHarness:
    """Owns environment interaction, local learning jobs, evaluation and RL updates."""

    def __init__(self, policy: ActorPolicy, environment: Environment,
                 experience: ExperienceSystem, learner: ActorLearner | None = None,
                 evaluator: Evaluator | None = None, *, memory: MemoryStore | None = None, memory_limit: int = 32):
        self.policy, self.environment = policy, environment
        self.experience, self.learner = experience, learner
        if type(memory_limit) is not int or memory_limit < 1:
            raise ValueError("memory_limit must be a positive integer")
        self.memory = memory if memory is not None else ExplorationMemory()
        self.memory_limit = memory_limit
        self.state = ExplorationState()
        self.training = JobBook[ActorTrainingRequest, TrainingResult](learner)
        self.evaluation = JobBook[EvaluationRequest, EvaluationResult](evaluator)
        self.commands = ExplorationCommands(self)

    @property
    def command_busy(self) -> bool:
        return self.commands.busy

    def accept(self, directive):
        return self.commands.accept(directive)

    def advance(self, directive_id, action_space: ActionSpace):
        return self.commands.advance(directive_id, action_space)

    def inspect(self, directive_id):
        return self.commands.inspect(directive_id)

    def cancel(self, directive_id):
        return self.commands.cancel(directive_id)

    def acknowledge(self, directive_id):
        self.commands.acknowledge(directive_id)

    def _require_local_control(self):
        if self.command_busy:
            raise RuntimeError('A directive owns exploration; advance or cancel it through its lifecycle')

    def run(self, directive: Directive, action_space: ActionSpace) -> Feedback:
        self._require_local_control()
        if self.state.rl is not None and not self.state.rl.terminal:
            raise RuntimeError("An RL cycle is active; advance it with step_rl")
        if self.training.busy:
            raise RuntimeError("A training job is still active")
        return self._run_episode(directive, action_space)

    def _flush_episode(self) -> Feedback:
        pending = self.state.pending_episode
        self.experience.record(pending.record)
        self.state.last_feedback = deepcopy(pending.feedback)
        if pending.count_completion:
            self.state.completed += 1
        self.state.pending_episode = None
        return deepcopy(self.state.last_feedback)

    def _run_episode(self, directive: Directive, action_space: ActionSpace) -> Feedback:
        if self.state.pending_episode is not None:
            if directive != self.state.pending_episode.directive:
                raise RuntimeError("Retry the pending episode feedback before new execution")
            return self._flush_episode()
        if directive.mode != "execute":
            raise NotImplementedError(f"Exploration mode {directive.mode!r} is not implemented.")
        if self.state.active_directive_id is not None:
            raise RuntimeError("Exploration already has an active directive.")
        self.state.active_directive_id = directive.directive_id
        self.state.last_feedback = None
        evidence, artifacts = [], []
        steps_executed = 0
        episode_id = str(uuid4())
        started = perf_counter()
        input_before, output_before = self.state.measured_input_tokens, self.state.measured_output_tokens
        unknown_before = self.state.calls_with_missing_usage
        simulated = directive.simulated
        status, error = ExecutionStatus.PARTIAL, None
        observation = ""
        failure: BaseException | None = None
        try:
            observation = self.environment.reset(directive)
            for _ in range(directive.max_steps):
                context = ActorContext(observation, directive, self.memory.history(task_id=directive.directive_id, limit=self.memory_limit),
                                       action_space.describe(),
                                       self.memory.recall(limit=self.memory_limit),
                                       instructions=self.state.instructions)
                proposal = self.policy.act(context)
                self._record_usage(proposal.usage)
                simulated = simulated or proposal.simulated
                action = action_space.validate(proposal, directive.allowed_actions)
                transition = self.environment.step(action)
                steps_executed += 1
                simulated = simulated or proposal.simulated or transition.simulated
                record = HistoryEntry(
                    record_id=str(uuid4()), kind="trajectory",
                    task_id=directive.directive_id, summary=transition.observation,
                    simulated=simulated, source_refs=tuple(ref.artifact_id for ref in transition.artifacts),
                    details={"observation": observation, "action": asdict(action),
                             "transition": asdict(transition),
                             "model": asdict(proposal.model) if proposal.model else None,
                             "usage": asdict(proposal.usage) if proposal.usage else None},
                )
                artifacts.extend(transition.artifacts)
                observation = transition.observation
                self.memory.append(record)
                evidence.append(record.record_id)
                if transition.error:
                    status, error = ExecutionStatus.FAILED, transition.error
                    break
                if transition.terminated:
                    status = ExecutionStatus.COMPLETED
                    break
        except BaseException as exc:
            # Preserve the exception for the caller and the outcome for the other loop.
            failure = exc
            status = ExecutionStatus.FAILED if isinstance(exc, Exception) else ExecutionStatus.CANCELLED
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                self.environment.close()
            except BaseException as exc:
                cleanup_error = f"Environment cleanup failed: {type(exc).__name__}: {exc}"
                error = f"{error}; {cleanup_error}" if error else cleanup_error
                if failure is None:
                    failure = exc
                    status = ExecutionStatus.FAILED if isinstance(exc, Exception) else ExecutionStatus.CANCELLED
                    raise
                failure.add_note(cleanup_error)
            finally:
                self.state.active_directive_id = None
                metrics = {
                    "elapsed_seconds": perf_counter() - started,
                    "steps": steps_executed,
                    "measured_input_tokens": self.state.measured_input_tokens - input_before,
                    "measured_output_tokens": self.state.measured_output_tokens - output_before,
                    "calls_with_missing_usage": self.state.calls_with_missing_usage - unknown_before,
                }
                self.state.last_feedback = Feedback(
                    directive_id=directive.directive_id, status=status,
                    summary=observation or "Exploration ended before an observation was available.",
                    simulated=simulated, artifact_refs=tuple(artifacts),
                    evidence_refs=(episode_id,), execution_metrics=metrics, error=error,
                )
                record = ExperienceRecord(
                    episode_id, "runtime_feedback", "actor", directive.directive_id,
                    f"Exploration execution {status.value}", simulated,
                    source_refs=tuple(f"history:actor:{ref}" for ref in evidence) + tuple(a.artifact_id for a in artifacts),
                    details={"status": status.value, "error": error, "execution_metrics": metrics,
                             "trajectory_refs": tuple(evidence), "history_owner": "actor",
                             "model": asdict(self.policy.model) if isinstance(getattr(self.policy, "model", None), ModelRef) else None},
                )
                self.state.pending_episode = PendingEpisode(
                    deepcopy(directive), record, deepcopy(self.state.last_feedback), failure is None,
                )
                try:
                    self._flush_episode()
                except Exception as record_error:
                    if failure is not None:
                        failure.add_note(f"Could not record system feedback: {record_error}")
                    else:
                        raise
        return self.state.last_feedback

    def submit_learning(self, request: ActorTrainingRequest, *, idempotency_key: str) -> JobRef:
        """Submit local learning explicitly; no cross-loop scheduling is implied."""
        self._require_local_control()
        if self.state.rl is not None and not self.state.rl.terminal:
            raise RuntimeError("An RL cycle owns the learner; advance it with step_rl")
        return self._submit_learning(request, idempotency_key=idempotency_key)

    def _submit_learning(self, request: ActorTrainingRequest, *, idempotency_key: str) -> JobRef:
        if not isinstance(request, ActorTrainingRequest):
            raise TypeError("Expected ActorTrainingRequest")
        if self.state.active_directive_id is not None or self.state.pending_episode is not None:
            raise RuntimeError("Cannot start training during environment interaction")
        available = {r.record_id for r in self.memory.history(kind="trajectory")}
        if any(ref not in available for ref in request.trajectory_refs):
            raise ValueError("Training references must identify this loop's trajectory history")
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
            if "actor" not in result.candidate.affected_roles:
                raise ValueError("Training result does not target this loop's role")
        if outcome.terminal and outcome.job not in self.state.recorded_training:
            self.experience.record(ExperienceRecord(
                record_id=self.state.training_record_ids.setdefault(outcome.job, str(uuid4())),
                kind="training_run", scope="actor", directive_id=None,
                summary=f"actor learning {outcome.status}",
                simulated=outcome.simulated or bool(outcome.result and outcome.result.simulated),
                source_refs=tuple(f"history:actor:{ref}" for ref in request.trajectory_refs) + tuple(ref.artifact_id for ref in request.data_refs)
                + (tuple(ref.artifact_id for ref in outcome.result.candidate.changes.values())
                   if outcome.result else ()),
                details={"request": asdict(request), "outcome": asdict(outcome)},
            ))
            self.state.recorded_training.add(outcome.job)
        return outcome

    def apply_learning(self, job: JobRef) -> ModelRef:
        """Explicitly bind a learned version to this policy only."""
        self._require_local_control()
        if self.state.rl is not None and not self.state.rl.terminal:
            raise RuntimeError("An RL cycle owns model updates; advance it with step_rl")
        return self._apply_learning(job)

    def _apply_learning(self, job: JobRef) -> ModelRef:
        if self.state.active_directive_id is not None or self.state.pending_episode is not None:
            raise RuntimeError("Cannot replace the actor during environment interaction")
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

    def submit_evaluation(self, request: EvaluationRequest, *, idempotency_key: str) -> JobRef:
        self._require_local_control()
        return self._submit_evaluation(request, idempotency_key=idempotency_key)

    def _submit_evaluation(self, request: EvaluationRequest, *, idempotency_key: str) -> JobRef:
        if not isinstance(request, EvaluationRequest):
            raise TypeError("Expected EvaluationRequest")
        return self.evaluation.submit(request, idempotency_key=idempotency_key)

    def poll_evaluation(self, job: JobRef) -> JobOutcome[EvaluationResult]:
        return self._record_evaluation(self.evaluation.poll(job))

    def cancel_evaluation(self, job: JobRef) -> JobOutcome[EvaluationResult]:
        return self._record_evaluation(self.evaluation.cancel(job))

    def _record_evaluation(self, outcome: JobOutcome[EvaluationResult]) -> JobOutcome[EvaluationResult]:
        request = self.evaluation.request(outcome.job)
        if outcome.status == JobStatus.SUCCEEDED:
            result = outcome.result
            if (not isinstance(result, EvaluationResult)
                    or result.candidate_id != request.candidate.candidate_id
                    or result.protocol_ref != request.protocol_ref):
                raise ValueError("Evaluation result does not match the candidate and protocol")
        if outcome.terminal and outcome.job not in self.state.recorded_evaluation:
            self.experience.record(ExperienceRecord(
                record_id=self.state.evaluation_record_ids.setdefault(outcome.job, str(uuid4())),
                kind="evaluation", scope="actor", directive_id=None,
                summary=f"Candidate evaluation {outcome.status}",
                simulated=outcome.simulated or bool(outcome.result and outcome.result.simulated),
                source_refs=outcome.result.evidence_refs if outcome.result else (),
                details={"request": asdict(request), "outcome": asdict(outcome)},
            ))
            self.state.recorded_evaluation.add(outcome.job)
        return outcome

    def start_rl(self, plan: RLPlan) -> RLState:
        self._require_local_control()
        return self._start_rl(plan)

    def _start_rl(self, plan: RLPlan) -> RLState:
        if not isinstance(plan, RLPlan):
            raise TypeError("Expected RLPlan")
        if self.state.rl is not None and not self.state.rl.terminal:
            raise RuntimeError("An RL cycle is already active")
        if self.state.active_directive_id is not None or self.state.pending_episode is not None or self.training.busy:
            raise RuntimeError("Actor is busy")
        if self.learner is None:
            raise NotImplementedError("RL requires a learner")
        if not isinstance(self.policy, ModelUpdatable):
            raise TypeError("RL requires a policy that can bind updated model versions")
        self.state.rl = RLState(deepcopy(plan))
        return deepcopy(self.state.rl)

    def step_rl(self, action_space: ActionSpace) -> RLState:
        self._require_local_control()
        return self._step_rl(action_space)

    def _step_rl(self, action_space: ActionSpace) -> RLState:
        """Advance one local phase; waiting jobs are polled once, never busy-waited."""
        state = self.state.rl
        if state is None:
            raise RuntimeError("No RL cycle has been started")
        if state.terminal:
            return deepcopy(state)
        if state.phase == "rollout":
            directive = replace(state.plan.rollout,
                                directive_id=f"{state.run_id}:{state.updates_completed}")
            try:
                feedback = self._run_episode(directive, action_space)
            except BaseException as exc:
                if self.state.pending_episode is not None:
                    raise  # Retry evidence persistence before advancing or repeating this rollout.
                state.phase = "failed" if isinstance(exc, Exception) else "cancelled"
                state.error = f"{type(exc).__name__}: {exc}"
                if self.state.last_feedback is not None:
                    state.feedback.append(self.state.last_feedback)
                raise
            state.feedback.append(feedback)
            if state.cancel_requested:
                state.phase = "cancelled"
            elif feedback.status in (ExecutionStatus.FAILED, ExecutionStatus.CANCELLED):
                state.phase = "failed" if feedback.status == ExecutionStatus.FAILED else "cancelled"
                state.error = feedback.error
            else:
                state.request = ActorTrainingRequest(self.policy.model, (), state.plan.recipe_ref,
                                                     "rl", tuple(r.record_id for r in self.memory.history(
                                                         task_id=directive.directive_id, kind="trajectory")))
                state.phase = "submitting"
        elif state.phase == "submitting":
            # Keep request and key stable if submission raises after a backend accepted it.
            state.submission_attempted = True
            state.job = self._submit_learning(state.request,
                idempotency_key=f"{state.run_id}:{state.updates_completed}")
            if state.job not in state.jobs:
                state.jobs.append(state.job)
            state.phase = "training"
        elif state.phase == "training":
            outcome = (self.cancel_learning(state.job) if state.cancel_requested
                       else self.poll_learning(state.job))
            state.simulated = state.simulated or outcome.simulated
            if state.cancel_requested:
                if outcome.terminal:
                    state.phase = "cancelled"
                return deepcopy(state)
            if outcome.status == JobStatus.SUCCEEDED:
                self._apply_learning(state.job)
                state.results.append(outcome.result)
                state.updates_completed += 1
                state.phase = "completed" if state.updates_completed == state.plan.updates else "rollout"
                state.job = None
                state.request = None
                state.submission_attempted = False
            elif outcome.status in (JobStatus.FAILED, JobStatus.CANCELLED):
                state.phase = "failed" if outcome.status == JobStatus.FAILED else "cancelled"
                state.error = outcome.error
        return deepcopy(state)

    def cancel_rl(self) -> RLState:
        self._require_local_control()
        return self._cancel_rl()

    def _cancel_rl(self) -> RLState:
        state = self.state.rl
        if state is None:
            raise RuntimeError("No RL cycle has been started")
        if state.terminal:
            return deepcopy(state)
        state.cancel_requested = True
        if self.state.pending_episode is not None:
            state.feedback.append(self._flush_episode())
        if state.phase == "submitting" and state.submission_attempted:
            # A lost submit response may hide a real job. Resolve the same key before cancelling.
            return deepcopy(state)
        if state.job is not None:
            outcome = self.cancel_learning(state.job)
            state.simulated = state.simulated or outcome.simulated
            if not outcome.terminal:
                return deepcopy(state)
        state.phase = "cancelled"
        return deepcopy(state)

    def _record_usage(self, usage) -> None:
        if usage is None or usage.input_tokens is None or usage.output_tokens is None:
            self.state.calls_with_missing_usage += 1
        if usage is not None:
            self.state.measured_input_tokens += usage.input_tokens or 0
            self.state.measured_output_tokens += usage.output_tokens or 0
