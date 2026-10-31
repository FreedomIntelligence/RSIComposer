"""Reusable single-writer memory buffer with optional atomic JSON persistence."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import ClassVar

from rsisys.runtime.files import atomic_write, json_bytes

from .contracts.memory import HistoryEntry, Memory


def _validate_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError('limit must be a positive integer')


class MemoryBuffer:
    """Owns insertion-ordered history and memories; updates make memories newest.

    A path belongs to one buffer writer. Separate instances do not coordinate
    concurrent writes. With no path, construction and mutations perform no I/O.
    """

    owner: ClassVar[str] = 'local'
    history_kinds: ClassVar[tuple[str, ...]] = ('decision', 'operation', 'feedback', 'trajectory')
    version: ClassVar[int] = 1

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else None
        self._history: dict[str, HistoryEntry] = {}
        self._memories: dict[str, Memory] = {}
        if self.path is not None and self.path.exists():
            self._load(json.loads(self.path.read_text(encoding='utf-8')))

    def _validate_entry(self, entry: HistoryEntry) -> None:
        if not isinstance(entry, HistoryEntry):
            raise TypeError('entry must be a HistoryEntry')
        if entry.kind not in self.history_kinds:
            raise ValueError(f'Unsupported history kind for {self.owner}: {entry.kind}')

    def append(self, entry: HistoryEntry) -> None:
        self._validate_entry(entry)
        previous = self._history.get(entry.record_id)
        if previous is not None:
            if previous == entry:
                return
            raise ValueError(f'Duplicate history ID: {entry.record_id}')
        history = self._history.copy()
        history[entry.record_id] = deepcopy(entry)
        self._commit(history, self._memories)

    def history(self, *, task_id: str | None = None, kind: str | None = None,
                limit: int | None = None) -> tuple[HistoryEntry, ...]:
        """Return the latest matching entries, ordered oldest to newest."""
        if limit is not None:
            _validate_limit(limit)
        entries = tuple(entry for entry in self._history.values()
                        if (task_id is None or entry.task_id == task_id)
                        and (kind is None or entry.kind == kind))
        return deepcopy(entries[-limit:] if limit is not None else entries)

    def remember(self, memory: Memory) -> None:
        if not isinstance(memory, Memory):
            raise TypeError('memory must be a Memory')
        memories = self._memories.copy()
        memories.pop(memory.memory_id, None)
        memories[memory.memory_id] = deepcopy(memory)
        self._commit(self._history, memories)

    def recall(self, *, limit: int = 32, text: str | None = None) -> tuple[Memory, ...]:
        """Return newest memories first, matching content or applicability."""
        _validate_limit(limit)
        query = text.casefold() if text else ''
        memories = tuple(memory for memory in reversed(tuple(self._memories.values()))
                         if query in (memory.content + ' ' + memory.applicability).casefold())
        return deepcopy(memories[:limit])

    def _payload(self, history: dict[str, HistoryEntry], memories: dict[str, Memory]) -> dict:
        return {
            'version': self.version,
            'owner': self.owner,
            'history': [asdict(entry) for entry in history.values()],
            'memories': [asdict(memory) for memory in memories.values()],
        }

    def _commit(self, history: dict[str, HistoryEntry], memories: dict[str, Memory]) -> None:
        if self.path is not None:
            atomic_write(self.path, json_bytes(self._payload(history, memories)))
        self._history, self._memories = history, memories

    def export(self) -> dict:
        """Return a detached, complete snapshot in persistence order."""
        return self._payload(self._history, self._memories)

    def backup(self, path: Path) -> None:
        """Atomically write a complete snapshot without rebinding this buffer."""
        atomic_write(Path(path), json_bytes(self.export()))

    def _load(self, payload: object) -> None:
        if not isinstance(payload, dict):
            raise ValueError('Memory file must contain an object')
        if type(payload.get('version')) is not int or payload['version'] != self.version:
            raise ValueError('Unsupported memory version')
        if payload.get('owner') != self.owner:
            raise ValueError(f'Memory owner does not match {self.owner}')
        history: dict[str, HistoryEntry] = {}
        memories: dict[str, Memory] = {}
        for key, record_type, destination, identity in (
            ('history', HistoryEntry, history, 'record_id'),
            ('memories', Memory, memories, 'memory_id'),
        ):
            rows = payload.get(key)
            if not isinstance(rows, list):
                raise ValueError(f'Memory {key} must be a list')
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError(f'Invalid {key} record')
                data = row.copy()
                refs = data.get('source_refs', [])
                if not isinstance(refs, list):
                    raise ValueError('Persisted source_refs must be a list')
                data['source_refs'] = tuple(refs)
                try:
                    record = record_type(**data)
                except TypeError as error:
                    raise ValueError(f'Invalid {key} record') from error
                if isinstance(record, HistoryEntry):
                    self._validate_entry(record)
                record_id = getattr(record, identity)
                if record_id in destination:
                    raise ValueError(f'Duplicate {key} ID: {record_id}')
                destination[record_id] = record
        self._history, self._memories = history, memories
