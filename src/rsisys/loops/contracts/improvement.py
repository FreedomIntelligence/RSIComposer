"""Lab-side decision-policy data construction; feedback only archives sources.

Recipes own filtering, labels and data format. A count or an archived receipt is
not a training label. Builders must retain source/recipe identity in their report.
Training consumes the resulting artifacts through DecisionTrainingRequest.
"""
from dataclasses import dataclass
import re
from typing import Protocol

from .artifacts import ArtifactRef
from .jobs import JobRef, JobOutcome


@dataclass(frozen=True)
class TrajectoryRef:
    archive_id: str
    trajectory_id: str
    digest: str

    def __post_init__(self):
        for name in ('archive_id', 'trajectory_id'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f'Trajectory {name} must be nonempty')
        if not isinstance(self.digest, str) or not re.fullmatch('[0-9a-f]{64}', self.digest):
            raise ValueError('Trajectory digest must be SHA256')


class TrajectorySource(Protocol):
    """Read a pinned immutable capture; reject an unknown archive or digest."""
    def read(self, ref: TrajectoryRef) -> dict: ...


@dataclass(frozen=True)
class DecisionDatasetBuildRequest:
    trajectory_refs: tuple[TrajectoryRef, ...]
    recipe_ref: str

    def __post_init__(self):
        if (not isinstance(self.trajectory_refs, tuple) or not self.trajectory_refs
                or any(not isinstance(ref, TrajectoryRef) for ref in self.trajectory_refs)):
            raise ValueError('Dataset construction requires explicit trajectory references')
        for ref in self.trajectory_refs:
            ref.__post_init__()
        if len({(ref.archive_id, ref.trajectory_id) for ref in self.trajectory_refs}) != len(self.trajectory_refs):
            raise ValueError('Dataset trajectory references must be distinct')
        if not isinstance(self.recipe_ref, str) or not self.recipe_ref.strip():
            raise ValueError('Dataset construction requires a recipe reference')


@dataclass(frozen=True)
class DecisionDatasetBuildResult:
    data_refs: tuple[ArtifactRef, ...]
    report_ref: ArtifactRef
    trajectory_refs: tuple[TrajectoryRef, ...]
    recipe_ref: str
    example_count: int
    simulated: bool = False

    def __post_init__(self):
        DecisionDatasetBuildRequest(self.trajectory_refs, self.recipe_ref)
        if (not isinstance(self.data_refs, tuple) or not self.data_refs
                or any(not isinstance(ref, ArtifactRef) for ref in self.data_refs)):
            raise ValueError('Dataset result requires data artifacts')
        if not isinstance(self.report_ref, ArtifactRef):
            raise TypeError('Dataset result requires a construction report artifact')
        for ref in (*self.data_refs, self.report_ref):
            ref.__post_init__()
            if not ref.version or not ref.digest:
                raise ValueError('Dataset and report artifacts must be versioned and fingerprinted')
        if type(self.example_count) is not int or self.example_count < 0:
            raise ValueError('Dataset example_count must be a nonnegative integer')
        if type(self.simulated) is not bool:
            raise TypeError('Dataset simulated must be a bool')


class DecisionDatasetBuilder(Protocol):
    """Replaceable Lab capability. Submit keys bind to exact immutable requests.

    poll/cancel never invent completion. An unknown response must remain unknown;
    the owning host supplies admission, stopping and recovery integration.
    """
    def submit(self, request: DecisionDatasetBuildRequest, *, idempotency_key: str) -> JobRef: ...
    def poll(self, job: JobRef) -> JobOutcome[DecisionDatasetBuildResult]: ...
    def cancel(self, job: JobRef) -> object: ...
