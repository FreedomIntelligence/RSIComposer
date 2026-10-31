from typing import Protocol

from .manifest import SnapshotManifest


class SnapshotRestore(Protocol):
    def restore(self, manifest: SnapshotManifest) -> None: ...


class UnimplementedRestore:
    def restore(self, manifest: SnapshotManifest) -> None:
        raise NotImplementedError("System restoration is not implemented.")
