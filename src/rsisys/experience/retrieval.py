from typing import Protocol

from .records import ExperienceRecord, Scope
from .storage import ExperienceStore


class ExperienceSystem(Protocol):
    def record(self, experience: ExperienceRecord) -> None: ...

    def retrieve(self, scope: Scope | None = None, *, limit: int | None = None,
                 kinds: tuple[str, ...] = (), text: str | None = None) -> tuple[ExperienceRecord, ...]: ...


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


class BasicExperienceSystem:
    """Shared system feedback. Scope identifies a source, not an access boundary."""

    def __init__(self, store: ExperienceStore, *, run_id: str | None = None):
        self.store = store
        self.run_id = run_id

    def record(self, experience: ExperienceRecord) -> None:
        from dataclasses import replace
        if self.run_id is not None:
            if experience.run_id not in (None, self.run_id):
                raise ValueError("Experience belongs to a different run")
            experience = replace(experience, run_id=self.run_id)
        self.store.append(experience)

    def retrieve(self, scope: Scope | None = None, *, limit: int | None = None,
                 kinds: tuple[str, ...] = (), text: str | None = None) -> tuple[ExperienceRecord, ...]:
        return self.store.query(scope, limit=limit, kinds=kinds, text=text)
