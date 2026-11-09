"""Controller-side selection of shared evidence for decision context."""
from typing import Protocol

from rsisys.runtime.evidence.records import ExperienceRecord
from rsisys.runtime.evidence.retrieval import ExperienceSystem


class FeedbackSelector(Protocol):
    """Project system evidence into decision input without recording or rewriting it.

    Return at most ``limit`` records (a positive integer). Preserve raw record
    identities and provenance; derived records must cite their inputs in
    ``source_refs`` and remain transient. Selection may depend on the current
    objective. This boundary defines no learning or summarization strategy.
    """

    def select(self, experience: ExperienceSystem, objective: str, *,
               limit: int) -> tuple[ExperienceRecord, ...]: ...


class RecentFeedbackSelector:
    """Raw recent records from all sources, in the store's insertion order."""

    def select(self, experience: ExperienceSystem, objective: str, *,
               limit: int) -> tuple[ExperienceRecord, ...]:
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        return experience.retrieve(limit=limit)
