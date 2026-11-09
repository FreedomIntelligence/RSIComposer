from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

from rsisys.harnesses.actor.policy import ActorPolicy
from rsisys.harnesses.memory import MemoryStore
from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.jobs import JobOutcome, JobRef, JobStatus
from rsisys.loops.contracts.metrics import InvalidMetrics, validated_metrics
from rsisys.loops.contracts.model_service import ModelRef, ModelUpdatable
from rsisys.loops.contracts.training import (
    ModelActivator,
    TrainingOrigin,
    TrainingResult,
)
from rsisys.runtime.evidence.records import ExperienceRecord
from rsisys.runtime.evidence.retrieval import ExperienceSystem
from rsisys.runtime.models.activation import prepare_model_activation
from rsisys.runtime.orchestration.jobs import InvalidJobResult, JobBook

if TYPE_CHECKING:
    from rsisys.harnesses.actor.state import ExplorationState


@dataclass(frozen=True)
class ActorTrainingRequest:
    base: ModelRef
    data_refs: tuple[ArtifactRef, ...]
    recipe_ref: str
    method: str = "sft"
    trajectory_refs: tuple[str, ...] = ()
    origin: TrainingOrigin | None = None

    def __post_init__(self):
        if not isinstance(self.base, ModelRef):
            raise ValueError("Training base must be a ModelRef")
        if self.origin is not None:
            if not isinstance(self.origin, TrainingOrigin):
                raise TypeError('origin must be TrainingOrigin')
            if self.base != self.origin.initial_model:
                raise ValueError('Training base differs from origin.initial_model')
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("Training method must be a nonempty string")
        if not isinstance(self.recipe_ref, str) or not self.recipe_ref.strip():
            raise ValueError("Training recipe must be a nonempty string")
        if not isinstance(self.data_refs, tuple) or not all(isinstance(ref, ArtifactRef) for ref in self.data_refs):
            raise ValueError("data_refs must be a tuple of ArtifactRef objects")
        if not isinstance(self.trajectory_refs, tuple) or not all(isinstance(ref, str) for ref in self.trajectory_refs):
            raise ValueError("trajectory_refs must be a tuple of strings")


class ActorLearner(Protocol):
    def submit(self, request: ActorTrainingRequest, *, idempotency_key: str) -> JobRef: ...

    def poll(self, job: JobRef) -> JobOutcome[TrainingResult]: ...

    def cancel(self, job: JobRef) -> None: ...

    def result(self, job: JobRef) -> TrainingResult: ...


class UnimplementedActorLearner:
    def submit(self, request: ActorTrainingRequest, *, idempotency_key: str = "") -> JobRef:
        raise NotImplementedError("Actor training is not implemented.")

    def result(self, job: JobRef) -> TrainingResult:
        raise NotImplementedError("Actor training is not implemented.")

    def poll(self, job: JobRef) -> JobOutcome[TrainingResult]:
        raise NotImplementedError("Training backend is not implemented.")

    def cancel(self, job: JobRef) -> None:
        raise NotImplementedError("Training backend is not implemented.")


class LearningExecution:
    """Own training validation, evidence retries, retained candidates and activation."""

    def __init__(self, policy: ActorPolicy, memory: MemoryStore, experience: ExperienceSystem,
                 state: ExplorationState, jobs: JobBook[ActorTrainingRequest, TrainingResult],
                 model_activator: ModelActivator | None = None):
        self.policy, self.memory, self.experience = policy, memory, experience
        self.state, self.jobs, self.model_activator = state, jobs, model_activator

    def validate_payload(self, payload, expected, *, apply: bool) -> None:
        if self.jobs.backend is None:
            raise NotImplementedError('Training directives require a learner')
        if not isinstance(payload, expected):
            raise ValueError(f'Training requires {expected.__name__}')
        if apply and not isinstance(self.policy, ModelUpdatable):
            raise TypeError('This command requires a model-updatable policy')

    def submit(self, request: ActorTrainingRequest, *, idempotency_key: str) -> JobRef:
        if not isinstance(request, ActorTrainingRequest):
            raise TypeError("Expected ActorTrainingRequest")
        if self.state.active_directive_id is not None or self.state.pending_episode is not None:
            raise RuntimeError("Cannot start training during environment interaction")
        available = {r.record_id for r in self.memory.history(kind="trajectory")}
        if any(ref not in available for ref in request.trajectory_refs):
            raise ValueError("Training references must identify this loop's trajectory history")
        from rsisys.loops.contracts.backend import (
            BackendCapabilityProvider,
            validate_training_capabilities,
        )
        if isinstance(self.jobs.backend, BackendCapabilityProvider):
            validate_training_capabilities(self.jobs.backend.capabilities(),
                                           method=request.method, origin=request.origin)
        return self.jobs.submit(request, idempotency_key=idempotency_key)

    def poll(self, job: JobRef) -> JobOutcome[TrainingResult]:
        outcome = self.jobs.poll(job, normalize=self._validate_metrics)
        return self._record(outcome)

    def cancel(self, job: JobRef) -> JobOutcome[TrainingResult]:
        return self._record(self.jobs.cancel(job, normalize=self._validate_metrics))

    def _validate_metrics(self, outcome):
        result = outcome.result
        request = self.jobs.request(outcome.job)
        if (outcome.status != JobStatus.SUCCEEDED or not isinstance(result, TrainingResult)
                or result.base != request.base or 'actor' not in result.candidate.affected_roles
                or (request.origin is not None
                    and result.candidate.parent_system != request.origin.parent_system)):
            return outcome  # Identity/type violations retain the InvalidJobResult path.
        try:
            validated_metrics(result.metrics, 'training metrics')
        except InvalidMetrics:
            return replace(outcome, status=JobStatus.FAILED, result=None,
                simulated=outcome.simulated or result.simulated,
                error='Invalid training metrics: expected nonempty names and finite numeric values')
        return outcome

    def _record(self, outcome: JobOutcome[TrainingResult]) -> JobOutcome[TrainingResult]:
        request = self.jobs.request(outcome.job)
        if outcome.status == JobStatus.SUCCEEDED:
            result = outcome.result
            if not isinstance(result, TrainingResult) or result.base != request.base:
                raise InvalidJobResult("Training result does not match the submitted base", outcome)
            if "actor" not in result.candidate.affected_roles:
                raise InvalidJobResult("Training result does not target this loop's role", outcome)
            if request.origin is not None and result.candidate.parent_system != request.origin.parent_system:
                raise InvalidJobResult('Candidate experiment parent differs from request', outcome)
            candidate_id = result.candidate.candidate_id
            if (candidate_id in self.state.candidate_results
                    and self.state.candidate_results[candidate_id] != result):
                raise InvalidJobResult('Candidate ID reused with different training content', outcome)
            self.state.candidate_results[candidate_id] = deepcopy(result)
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

    def apply(self, job: JobRef) -> ModelRef:
        if self.state.active_directive_id is not None or self.state.pending_episode is not None:
            raise RuntimeError("Cannot replace the actor during environment interaction")
        outcome = self.poll(job)
        if outcome.status != JobStatus.SUCCEEDED:
            raise RuntimeError(f"Training has no usable result: {outcome.status}")
        result = outcome.result
        model = self.apply_result(result)
        self.state.applied_training.add(job)
        return model

    def candidate_result(self, candidate_id: str) -> TrainingResult:
        """Read a retained result without requiring its training job or backend."""
        if candidate_id not in self.state.candidate_results:
            raise ValueError(f'Unknown candidate: {candidate_id}')
        return deepcopy(self.state.candidate_results[candidate_id])

    def apply_candidate(self, candidate_id: str) -> ModelRef:
        if self.state.rl is not None and not self.state.rl.terminal:
            raise RuntimeError('An RL cycle owns model updates; advance it with step_rl')
        return self.apply_result(self.candidate_result(candidate_id))

    def apply_result(self, result: TrainingResult) -> ModelRef:
        if self.state.active_directive_id is not None or self.state.pending_episode is not None:
            raise RuntimeError('Cannot replace the actor during environment interaction')
        if self.jobs.busy:
            raise RuntimeError('Cannot replace the model while another training job is active')
        if 'actor' not in result.candidate.affected_roles:
            raise ValueError("Training result does not target this loop's role")
        if not isinstance(self.policy, ModelUpdatable):
            raise TypeError("This policy does not support model updates")
        candidate_id = result.candidate.candidate_id
        if candidate_id in self.state.applied_candidates:
            if self.policy.model != result.model:
                raise ValueError('Previously applied candidate is stale relative to this policy')
            return result.model
        if self.policy.model != result.base:
            raise ValueError("Training base is stale relative to this policy")
        ready = prepare_model_activation(result, self.model_activator,
            idempotency_key=json.dumps(('exploration', 'candidate', candidate_id)))
        self.policy.update_model(ready)
        self.state.applied_candidates.add(candidate_id)
        return result.model
