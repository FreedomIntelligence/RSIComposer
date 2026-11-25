"""Shared evidence recording and basic retrieval, independent of memory policies."""
from typing import Protocol

from .records import ExperienceRecord, Scope, is_host_final_record
from .storage import ExperienceStore


class EvidenceRepository(Protocol):
    def record(self, experience: ExperienceRecord) -> None: ...

    def retrieve(self, scope: Scope | None = None, *, limit: int | None = None,
                 kinds: tuple[str, ...] = (), text: str | None = None) -> tuple[ExperienceRecord, ...]: ...


class BasicEvidenceRepository:
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


class _ControllerEvidenceRepository:
    """The Controller/Memory read port never returns Host final receipts."""

    def __init__(self, repository):
        self._repository = repository

    def record(self, experience):
        if is_host_final_record(experience):
            raise PermissionError('Host final evidence is unavailable to the Controller')
        self._repository.record(experience)

    def retrieve(self, *args, limit=None, **filters):
        rows = self._repository.retrieve(*args, limit=limit, **filters)
        visible = tuple(row for row in rows if not is_host_final_record(row))
        if limit is not None and len(rows) == limit and len(visible) < limit:
            # Private recent records must not displace visible records from a
            # bounded window. Only widen the underlying read when necessary.
            rows = self._repository.retrieve(*args, limit=None, **filters)
            visible = tuple(row for row in rows if not is_host_final_record(row))[-limit:]
        return visible


def controller_experience(repository):
    if repository is None or isinstance(repository, _ControllerEvidenceRepository):
        return repository
    return _ControllerEvidenceRepository(repository)
