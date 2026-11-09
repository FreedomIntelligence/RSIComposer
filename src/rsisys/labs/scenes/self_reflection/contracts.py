"""Draft reference contracts for persistent meta-agent self-improvement.

References identify immutable artifacts, not live mutable objects or file copies.
Storage, serialization, training and validation execution are not implemented.
No record or successful invocation alone establishes that a candidate is better.
"""
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class MetaAgentVersion:
    version_id: str
    harness_ref: str
    model_ref: str
    memory_ref: str | None = None
    training_data_ref: str | None = None


@dataclass(frozen=True)
class ResearchEpisode:
    """A multi-decision trajectory; an absent outcome is still pending.

    Later outcomes should be new immutable revisions linked to the same episode
    identity. Revisions must not be counted as independent research episodes.
    """
    episode_id: str
    task_ref: str
    agent_version: MetaAgentVersion
    trajectory_ref: str
    cost_ref: str
    outcome_ref: str | None = None


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    version: MetaAgentVersion
    parent_version_ref: str
    construction_episode_refs: tuple[str, ...]


@dataclass(frozen=True)
class ValidationRequest:
    """Pin candidate, baseline, tasks and method-owned evaluation policy.

    policy_ref defines the budget, horizon, metrics and evidence separation.
    Validation tasks must not silently become candidate-construction experience.
    """
    candidate_ref: str
    baseline_version_ref: str
    task_suite_ref: str
    policy_ref: str


@dataclass(frozen=True)
class ValidationEvidence:
    """Execution progress is distinct from a decision to adopt the candidate.

    A completed validation may still have insufficient evidence. Interpretation
    and adoption belong to the method; this record has no automatic winner flag.
    """
    validation_id: str
    request: ValidationRequest
    state: Literal['pending', 'running', 'completed', 'failed', 'cancelled']
    evidence_refs: tuple[str, ...] = ()


class SelfReflectionBackend(Protocol):
    """Future backend port; no implementation or resource ownership is implied.

    Implementations must persist version-linked episodes and job identities,
    reconcile interrupted work, and manage their own workers and cancellation.
    Controller switching is outside this port and requires explicit adoption.
    """
    def record_episode(self, episode_ref: str) -> str: ...
    def build_candidate(self, baseline_version_ref: str,
                        experience_set_ref: str, proposal_ref: str) -> str: ...
    def start(self, request: ValidationRequest) -> str: ...
    def poll(self, validation_id: str) -> ValidationEvidence: ...
    def result(self, validation_id: str) -> ValidationEvidence: ...
    def cancel(self, validation_id: str) -> ValidationEvidence: ...
