"""Small context window shared by memory providers and consuming harnesses.

Providers own organization, retrieval and storage. These inputs contain public
observations, never a feedback worker's private job/accounting representation.
"""
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class FeedbackContextEvent:
    """A host-associated observation; output and references retain native shape.

    event_id is stable across delivery attempts. Providers with external side
    effects may use it for deduplication. The host supplies all metadata; Labs do
    not need to construct this type or normalize their business output.
    """
    event_id: str
    task_id: str
    status: str
    output: object
    request: object = None
    references: object = ()
    source: str = ''
    identity: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ContextQuery:
    objective: str = ''
    latest_feedback: FeedbackContextEvent | None = None


class ContextMemory(Protocol):
    """The sole required provider method; an empty string is a valid view.

    An optional observe(FeedbackContextEvent) hook receives new feedback on a
    later host tick. A provider can instead use query.latest_feedback, or adapt
    its own update/retrieval API. Internal model use goes through the deployment's
    accounted dependencies; this product contract does not manage model budgets.
    """
    def context(self, query: ContextQuery) -> str: ...
