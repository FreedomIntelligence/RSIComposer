from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar


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
class JobOutcome(Generic[T]):
    job: JobRef
    status: JobStatus
    result: T | None = None
    error: str | None = None
    simulated: bool = False

    def __post_init__(self):
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
