from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Protocol
from uuid import uuid4

from rsisys.harnesses.actor.policy import ActorContext, ActorPolicy
from rsisys.harnesses.actor.state import ExplorationState, PendingEpisode
from rsisys.harnesses.memory import HistoryEntry, MemoryStore
from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.jobs import JobStatus
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.exploration.action_space import Action, ActionSpace
from rsisys.runtime.evidence.records import ExperienceRecord
from rsisys.runtime.evidence.retrieval import ExperienceSystem
from rsisys.runtime.execution.sandbox import (
    ExecRequest,
    SandboxBackend,
    SandboxRef,
    SandboxSpec,
)
from rsisys.runtime.storage.artifacts import ArtifactStore


@dataclass(frozen=True)
class Transition:
    observation: str
    terminated: bool
    simulated: bool
    reward: float | None = None
    artifacts: tuple[ArtifactRef, ...] = ()
    error: str | None = None


class Environment(Protocol):
    def reset(self, directive: Directive) -> str: ...

    def step(self, action: Action) -> Transition: ...

    def close(self) -> None: ...


class SandboxEchoEnvironment:
    """Demo task semantics on a sandbox backend that completes exec immediately."""

    def __init__(self, sandbox: SandboxBackend, artifacts: ArtifactStore):
        self.sandbox, self.artifacts = sandbox, artifacts
        self._session: SandboxRef | None = None

    def reset(self, directive: Directive) -> str:
        self.close()
        inputs = [self.artifacts.get(ref).decode() for ref in directive.input_refs]
        self._session = self.sandbox.create(SandboxSpec())
        return f"Task: {directive.objective}; prepared inputs: {inputs}"

    def step(self, action: Action) -> Transition:
        if self._session is None:
            raise RuntimeError("Reset the environment before acting")
        if action.name != "echo":
            raise ValueError("Environment does not support this action")
        process = self.sandbox.exec(self._session, ExecRequest(("echo", action.text)))
        result = self.sandbox.poll(process)
        if result.status not in (JobStatus.SUCCEEDED, JobStatus.FAILED):
            raise NotImplementedError("Demo environment requires immediate command completion")
        ref = self.artifacts.put((result.stdout + result.stderr).encode(), kind="execution_output")
        error = result.stderr or f"exit={result.exit_code}" if result.status == JobStatus.FAILED else None
        return Transition(result.stdout, True, result.simulated, artifacts=(ref,), error=error)

    def close(self) -> None:
        if self._session is not None:
            self.sandbox.close(self._session)
            self._session = None


class EpisodeExecution:
    """Run and persist one bounded episode using explicit loop dependencies."""

    def __init__(self, policy: ActorPolicy, environment: Environment, memory: MemoryStore,
                 experience: ExperienceSystem, state: ExplorationState, *, memory_limit: int = 32):
        self.policy, self.environment, self.memory = policy, environment, memory
        self.experience, self.state, self.memory_limit = experience, state, memory_limit

    def flush(self) -> Feedback:
        pending = self.state.pending_episode
        self.experience.record(pending.record)
        self.state.last_feedback = deepcopy(pending.feedback)
        if pending.count_completion:
            self.state.completed += 1
        self.state.pending_episode = None
        return deepcopy(self.state.last_feedback)

    def run(self, directive: Directive, action_space: ActionSpace) -> Feedback:
        if self.state.pending_episode is not None:
            if directive != self.state.pending_episode.directive:
                raise RuntimeError("Retry the pending episode feedback before new execution")
            return self.flush()
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
                    self.flush()
                except Exception as record_error:
                    if failure is not None:
                        failure.add_note(f"Could not record system feedback: {record_error}")
                    else:
                        raise
        return self.state.last_feedback

    def _record_usage(self, usage) -> None:
        if usage is None or usage.input_tokens is None or usage.output_tokens is None:
            self.state.calls_with_missing_usage += 1
        if usage is not None:
            self.state.measured_input_tokens += usage.input_tokens or 0
            self.state.measured_output_tokens += usage.output_tokens or 0
