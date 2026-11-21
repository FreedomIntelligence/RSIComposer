"""Product windows. Algorithms, SDKs and journal layouts are not protocols.

Payloads retain their producer's shape. Persisted payloads/references must be
finite JSON data; that transport requirement conveys no execution authority.
"""
from dataclasses import dataclass, field
from typing import Protocol

from .feedback_context import EvidenceEvent
from .model_service import GenerationRequest, GenerationResponse, ModelRef


@dataclass(frozen=True)
class ArchiveEvent:
    event_id: str
    kind: str
    payload: object
    identity: dict = field(default_factory=dict)


class TrajectoryArchive(Protocol):
    """Save native events and read them back; storage and indexing are private.

    write may return any finite JSON reference. The host records delivery and
    checks read-back identity/content. Unknown writes are not automatically
    replayed. Implementations should use event_id for idempotency.
    """
    def write(self, event: ArchiveEvent) -> object: ...
    def read(self, reference: object) -> ArchiveEvent: ...


class FeedbackModel(Protocol):
    """Host-accounted inference with explicit output and transport capabilities.

    A processor may choose native tools or response schemas from these flags;
    none of these declarations grants execution tools or budget ownership.
    """
    model: ModelRef
    max_output_tokens: int
    supports_tool_calling: bool
    supports_response_schema: bool
    def generate(self, request: GenerationRequest) -> GenerationResponse: ...


class FeedbackProcessor(Protocol):
    """Return a product, without host status, accounting or proposal schemas.

    Model access is supplied separately. Pure implementations can ignore it.
    An optional recover(event) reads a previously committed product; it must not
    repeat unknown effects. No recovery method is required for a pure function.
    """
    def process(self, event: EvidenceEvent, model: FeedbackModel) -> object: ...


@dataclass(frozen=True)
class PendingFeedback:
    event: EvidenceEvent
    capture_index: int


@dataclass(frozen=True)
class FeedbackAdmission:
    """Organization decision only; original events/context are never discarded."""
    organize: bool = True
    coalesce: tuple[str, ...] = ()
    reason: str = ''


class FeedbackAdmissionPolicy(Protocol):
    """Optional pre-queue policy, independent of the memory implementation.

    coalesce names only queued events represented by this new organization job.
    No running job, original receipt, archive or model allowance can be changed.
    """
    def admit(self, event: EvidenceEvent, pending: tuple[PendingFeedback, ...]) -> FeedbackAdmission: ...


class FeedbackScheduler(Protocol):
    """Choose one pending event, or None to defer until a later host refresh.

    Selection does not grant calls, parallelism, replay or execution authority.
    Host limits and lifecycle remain in force with every scheduling policy.
    """
    def select(self, pending: tuple[PendingFeedback, ...], running: tuple[str, ...]) -> str | None: ...


@dataclass(frozen=True)
class FeedbackFailurePolicy:
    """Host behavior is explicit; it is not selected by a product payload."""
    context: str = 'raise'
    publication: str = 'isolate'

    def __post_init__(self):
        if self.context not in ('raise', 'last_valid') or self.publication not in ('raise', 'isolate'):
            raise ValueError('Invalid feedback failure policy')
