from dataclasses import dataclass


@dataclass(frozen=True)
class LineageLink:
    source_id: str
    relation: str
    target_id: str
