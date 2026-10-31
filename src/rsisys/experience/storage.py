from copy import deepcopy
from typing import Protocol

from .records import ExperienceRecord, Scope
from .memory import Memory


class ExperienceStore(Protocol):
    def append(self, record: ExperienceRecord) -> None: ...

    def records(self) -> tuple[ExperienceRecord, ...]: ...

    def query(self, scope: Scope | None = None, *, limit: int | None = None,
              kinds: tuple[str, ...] = (), text: str | None = None) -> tuple[ExperienceRecord, ...]: ...

    def put_memory(self, memory: Memory) -> None: ...

    def memories(self, scope: Scope | None = None, *, limit: int = 32,
                 text: str | None = None) -> tuple[Memory, ...]: ...


class InMemoryExperienceStore:
    """Demo store; no durability or cross-process consistency guarantees."""

    def __init__(self):
        self._records: dict[str, ExperienceRecord] = {}
        self._memories: dict[str, Memory] = {}

    def append(self, record: ExperienceRecord) -> None:
        if record.record_id in self._records:
            if self._records[record.record_id] == record:
                return
            raise ValueError(f"Duplicate experience ID: {record.record_id}")
        self._records[record.record_id] = deepcopy(record)

    def records(self) -> tuple[ExperienceRecord, ...]:
        return tuple(deepcopy(record) for record in self._records.values())

    def query(self, scope: Scope | None = None, *, limit: int | None = None,
              kinds: tuple[str, ...] = (), text: str | None = None) -> tuple[ExperienceRecord, ...]:
        from .sqlite import validate_query
        validate_query(scope, limit)
        records = tuple(r for r in self.records()
                        if (scope is None or r.scope in (scope, "shared"))
                        and (not kinds or r.kind in kinds)
                        and (not text or text.casefold() in r.summary.casefold()))
        return records[-limit:] if limit else records

    def put_memory(self, memory: Memory) -> None:
        if any(ref not in self._records for ref in memory.source_record_ids):
            raise ValueError("Memory evidence does not exist")
        self._memories.pop(memory.memory_id, None)
        self._memories[memory.memory_id] = deepcopy(memory)

    def memories(self, scope: Scope | None = None, *, limit: int = 32,
                 text: str | None = None) -> tuple[Memory, ...]:
        from .sqlite import validate_query
        validate_query(scope, limit)
        items = [m for m in reversed(tuple(self._memories.values()))
                 if (scope is None or m.scope in (scope, "shared"))
                 and (not text or text.casefold() in (m.content + " " + m.applicability).casefold())]
        return tuple(deepcopy(items[:limit]))
