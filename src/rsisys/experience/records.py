from dataclasses import dataclass, field
from typing import Literal

Scope = Literal["decision_maker", "actor", "shared"]


@dataclass(frozen=True)
class ExperienceRecord:
    record_id: str
    kind: Literal["runtime_feedback", "training_run", "evaluation", "improvement", "insight"]
    scope: Scope
    directive_id: str | None
    summary: str
    simulated: bool
    source_refs: tuple[str, ...] = ()
    details: dict[str, object] = field(default_factory=dict)
    run_id: str | None = None
    created_at: str | None = None

    def __post_init__(self):
        if self.kind not in ("runtime_feedback", "training_run", "evaluation", "improvement", "insight"):
            raise ValueError("Unknown system feedback kind")
        if self.scope not in ("decision_maker", "actor", "shared"):
            raise ValueError("Unknown feedback source")
        if not self.record_id:
            raise ValueError("Feedback record identity is required")
