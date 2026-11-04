"""Optional producer identity for feedback; native result data stays opaque."""
from dataclasses import dataclass


@dataclass(frozen=True)
class FeedbackDelivery:
    """One immutable producer result, independent of collection decisions.

    The executor supplies a stable, namespaced event_id. Equal output from
    different experiments must have different IDs. reprocess requests another
    feedback pass, never replay of the underlying program. Host limits apply.
    Providers without this metadata retain per-receipt processing.
    """
    event_id: str
    reprocess: bool = False

    def __post_init__(self):
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError('Feedback producer event_id must be nonempty')
        if type(self.reprocess) is not bool:
            raise TypeError('Feedback reprocess must be a bool')
