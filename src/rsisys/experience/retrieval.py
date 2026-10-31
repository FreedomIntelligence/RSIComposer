from typing import Protocol

from .records import ExperienceRecord, Scope
from .memory import Memory
from .storage import ExperienceStore


class ExperienceSystem(Protocol):
    def record(self, experience: ExperienceRecord) -> None: ...

    def retrieve(self, scope: Scope, *, limit: int | None = None,
                 kinds: tuple[str, ...] = (), text: str | None = None) -> tuple[ExperienceRecord, ...]: ...

    def remember(self, memory: Memory) -> None: ...

    def memories(self, scope: Scope, *, limit: int = 32, text: str | None = None) -> tuple[Memory, ...]: ...


class BasicExperienceSystem:
    """Scoped experience retrieval and explicit, evidence-linked memory."""

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

    def retrieve(self, scope: Scope, *, limit: int | None = None,
                 kinds: tuple[str, ...] = (), text: str | None = None) -> tuple[ExperienceRecord, ...]:
        return self.store.query(scope, limit=limit, kinds=kinds, text=text)

    def remember(self, memory: Memory) -> None:
        self.store.put_memory(memory)

    def memories(self, scope: Scope, *, limit: int = 32, text: str | None = None) -> tuple[Memory, ...]:
        return self.store.memories(scope, limit=limit, text=text)
