"""Product windows. Algorithms, SDKs and journal layouts are not protocols.

Payloads retain their producer's shape. Persisted payloads/references must be
finite JSON data; that transport requirement conveys no execution authority.
"""
from dataclasses import dataclass, field
from typing import Protocol

from .feedback_context import FeedbackContextEvent
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
    """Optional host-accounted inference; never a source of execution tools."""
    model: ModelRef
    def generate(self, request: GenerationRequest) -> GenerationResponse: ...


class FeedbackProcessor(Protocol):
    """Return a product, without host status, accounting or proposal schemas.

    Model access is supplied separately. Pure implementations can ignore it.
    An optional recover(event) reads a previously committed product; it must not
    repeat unknown effects. No recovery method is required for a pure function.
    """
    def process(self, event: FeedbackContextEvent, model: FeedbackModel) -> object: ...


class FeedbackProposer(Protocol):
    """Suggestions only. A later consumer/Lab defines executable requirements."""
    def propose(self, event: FeedbackContextEvent, model: FeedbackModel) -> object: ...


@dataclass(frozen=True)
class PendingFeedback:
    event: FeedbackContextEvent
    capture_index: int


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
