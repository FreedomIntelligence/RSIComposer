from dataclasses import dataclass
from typing import Literal

from .records import Scope


@dataclass(frozen=True)
class Memory:
    memory_id: str
    kind: Literal["episodic", "semantic", "procedural"]
    scope: Scope
    content: str
    source_record_ids: tuple[str, ...]
    applicability: str
    status: Literal["unverified", "supported", "refuted"] = "unverified"

    def __post_init__(self):
        if self.kind not in ("episodic", "semantic", "procedural"):
            raise ValueError("Unknown memory kind")
        if self.scope not in ("actor", "decision_maker", "shared"):
            raise ValueError("Unknown memory scope")
        if self.status not in ("unverified", "supported", "refuted"):
            raise ValueError("Unknown memory status")
        if not self.memory_id or not self.content or not self.source_record_ids:
            raise ValueError("Memory requires identity, content, and source evidence")
