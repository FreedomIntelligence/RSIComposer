from dataclasses import dataclass, field
from typing import Literal

Scope = Literal["decision_maker", "actor", "shared"]


@dataclass(frozen=True)
class ExperienceRecord:
    record_id: str
    kind: Literal["decision_episode", "actor_trajectory", "training_run", "evaluation"]
    scope: Scope
    directive_id: str | None
    summary: str
    simulated: bool
    source_refs: tuple[str, ...] = ()
    details: dict[str, object] = field(default_factory=dict)
    run_id: str | None = None
    created_at: str | None = None
