from typing import Protocol

from .manifest import SnapshotManifest


class SnapshotStore(Protocol):
    def save(self, manifest: SnapshotManifest) -> None: ...

    def load(self, snapshot_id: str) -> SnapshotManifest: ...
