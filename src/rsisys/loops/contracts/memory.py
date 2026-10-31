"""Loop-local history and explicit memory, independent of Experience."""

from dataclasses import dataclass, field
from typing import Literal, Protocol


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')


def _source_refs(value: tuple[str, ...]) -> None:
    if not isinstance(value, tuple):
        raise ValueError('source_refs must be a tuple')
    for ref in value:
        _identifier(ref, 'source reference')


@dataclass(frozen=True)
class Memory:
    memory_id: str
    kind: Literal['episodic', 'semantic', 'procedural']
    content: str
    source_refs: tuple[str, ...] = ()
    applicability: str = ''
    status: Literal['unverified', 'supported', 'refuted'] = 'unverified'

    def __post_init__(self):
        _identifier(self.memory_id, 'memory_id')
        _identifier(self.content, 'content')
        if self.kind not in ('episodic', 'semantic', 'procedural'):
            raise ValueError('Unknown memory kind')
        if self.status not in ('unverified', 'supported', 'refuted'):
            raise ValueError('Unknown memory status')
        if not isinstance(self.applicability, str):
            raise ValueError('applicability must be a string')
        _source_refs(self.source_refs)


@dataclass(frozen=True)
class HistoryEntry:
    record_id: str
    kind: str
    task_id: str | None
    summary: str
    simulated: bool
    source_refs: tuple[str, ...] = ()
    details: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        _identifier(self.record_id, 'record_id')
        _identifier(self.kind, 'kind')
        if self.task_id is not None:
            _identifier(self.task_id, 'task_id')
        if not isinstance(self.summary, str):
            raise ValueError('summary must be a string')
        if not isinstance(self.simulated, bool):
            raise ValueError('simulated must be a bool')
        if not isinstance(self.details, dict):
            raise ValueError('details must be a dict')
        _source_refs(self.source_refs)


class LoopMemory(Protocol):
    def append(self, entry: HistoryEntry) -> None: ...

    def history(self, *, task_id: str | None = None, kind: str | None = None,
                limit: int | None = None) -> tuple[HistoryEntry, ...]: ...

    def remember(self, memory: Memory) -> None: ...

    def recall(self, *, limit: int = 32, text: str | None = None) -> tuple[Memory, ...]: ...
