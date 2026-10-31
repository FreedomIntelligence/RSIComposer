from typing import Protocol

from .manifest import SnapshotManifest


class SnapshotCapture(Protocol):
    def capture(self, label: str) -> SnapshotManifest: ...


class UnimplementedCapture:
    def capture(self, label: str) -> SnapshotManifest:
        raise NotImplementedError("Consistent system snapshot capture is not implemented.")
