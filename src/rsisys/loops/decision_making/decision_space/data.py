from typing import Protocol

from rsisys.loops.contracts.artifacts import ArtifactRef


class DataPreparation(Protocol):
    def prepare(self, source_refs: tuple[str, ...], recipe_ref: str) -> ArtifactRef: ...
