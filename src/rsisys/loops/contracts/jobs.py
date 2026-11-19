from dataclasses import dataclass, field
from enum import StrEnum
import math
from typing import Generic, TypeVar, Literal

from .artifacts import ArtifactRef


@dataclass(frozen=True)
class JobRef:
    job_id: str
    backend: str


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


T = TypeVar("T")
TERMINAL_STATUSES = frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED})


@dataclass(frozen=True)
class JobProgress:
    """Latest backend observation, not an estimate of the enclosing workflow.

    Missing values are unknown. Metrics are the latest reported numeric values,
    not necessarily logged at ``step``. ``observed_at`` is the backend's sample
    time in Unix seconds, never a controller poll timestamp.
    """

    step: int | None = None
    total_steps: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float | None = None
    observed_at: float | None = None

    def __post_init__(self):
        for name, minimum in (('step', 0), ('total_steps', 1)):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < minimum):
                raise ValueError(f'{name} must be an integer >= {minimum} or None')
        if self.step is not None and self.total_steps is not None and self.step > self.total_steps:
            raise ValueError('step cannot exceed total_steps')
        for name in ('elapsed_seconds', 'observed_at'):
            value = getattr(self, name)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError(f'{name} must be finite and nonnegative or None')
        if not isinstance(self.metrics, dict) or any(
                not isinstance(key, str) or not key.strip()
                or type(value) not in (int, float) or not math.isfinite(value)
                for key, value in self.metrics.items()):
            raise ValueError('Progress metrics must have nonempty names and finite numeric values')
        object.__setattr__(self, 'metrics', dict(self.metrics))


@dataclass(frozen=True)
class JobOutcome(Generic[T]):
    job: JobRef
    status: JobStatus
    result: T | None = None
    error: str | None = None
    simulated: bool = False
    progress: JobProgress | None = None

    def __post_init__(self):
        if self.progress is not None:
            if not isinstance(self.progress, JobProgress):
                raise TypeError('progress must be JobProgress or None')
            self.progress.__post_init__()
        if not isinstance(self.status, JobStatus):
            raise TypeError("status must be a JobStatus")
        if (self.status == JobStatus.SUCCEEDED) != (self.result is not None):
            raise ValueError("Only a succeeded job carries a result")
        if self.status == JobStatus.SUCCEEDED and self.error is not None:
            raise ValueError("A succeeded job cannot carry an error")
        if self.status == JobStatus.FAILED and not self.error:
            raise ValueError("A failed job requires an error")

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


@dataclass(frozen=True)
class OperationProgress:
    """A Lab job observation; checkpoint events require a host control response."""
    call_id: str
    job: JobRef
    sequence: int
    kind: Literal['progress', 'checkpoint']
    progress: JobProgress
    checkpoint: ArtifactRef | None = None

    def __post_init__(self):
        if not isinstance(self.call_id, str) or not self.call_id.strip():
            raise ValueError('Operation progress needs a call_id')
        if not isinstance(self.job, JobRef) or not isinstance(self.progress, JobProgress):
            raise TypeError('Operation progress requires JobRef and JobProgress')
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError('Operation progress sequence must be positive')
        if self.kind not in ('progress', 'checkpoint'):
            raise ValueError('Unknown operation progress kind')
        if ((self.kind == 'checkpoint' and not isinstance(self.checkpoint, ArtifactRef))
                or (self.kind == 'progress' and self.checkpoint is not None)):
            raise ValueError('Only checkpoint events must carry a checkpoint artifact')


@dataclass(frozen=True)
class JobControl:
    """Idempotent response to a particular checkpoint event of one call."""
    call_id: str
    sequence: int
    action: Literal['continue', 'stop']

    def __post_init__(self):
        if not isinstance(self.call_id, str) or not self.call_id.strip():
            raise ValueError('Job control needs a call_id')
        if type(self.sequence) is not int or self.sequence < 1 or self.action not in ('continue', 'stop'):
            raise ValueError('Job control needs a checkpoint sequence and continue/stop action')
