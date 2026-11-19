from copy import deepcopy
from typing import Protocol

from .records import ExperienceRecord, Scope


class ExperienceStore(Protocol):
    def append(self, record: ExperienceRecord) -> None: ...

    def records(self) -> tuple[ExperienceRecord, ...]: ...

    def query(self, scope: Scope | None = None, *, limit: int | None = None,
              kinds: tuple[str, ...] = (), text: str | None = None) -> tuple[ExperienceRecord, ...]: ...



class InMemoryExperienceStore:
    """Demo store; no durability or cross-process consistency guarantees."""

    def __init__(self):
        self._records: dict[str, ExperienceRecord] = {}

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
