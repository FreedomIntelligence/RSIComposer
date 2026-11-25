from dataclasses import dataclass, field
from typing import Literal

Scope = Literal["decision_maker", "actor", "shared"]


@dataclass(frozen=True)
class ExperienceRecord:
    record_id: str
    kind: Literal["runtime_feedback", "training_run", "evaluation", "improvement", "insight", "controller_event", "memory_observation", "progress"]
    scope: Scope
    directive_id: str | None
    summary: str
    simulated: bool
    source_refs: tuple[str, ...] = ()
    details: dict[str, object] = field(default_factory=dict)
    run_id: str | None = None
    created_at: str | None = None

    def __post_init__(self):
        if self.kind not in ("runtime_feedback", "training_run", "evaluation", "improvement", "insight", "controller_event", "memory_observation", "progress"):
            raise ValueError("Unknown system feedback kind")
        if self.scope not in ("decision_maker", "actor", "shared"):
            raise ValueError("Unknown feedback source")
        if not self.record_id:
            raise ValueError("Feedback record identity is required")


def is_host_final_payload(payload):
    """Recognize the Host final phase and its reserved durable identities."""
    return isinstance(payload, dict) and (payload.get('phase') == 'host_finish' or any(
        isinstance(payload.get(key), str) and payload[key].startswith('sealed-')
        for key in ('record_id', 'event_id')))


def is_host_final_record(record):
    return (record.record_id.startswith('sealed-') or is_host_final_payload(record.details)
            or is_host_final_payload(record.details.get('event')))


def is_host_final_event(event):
    return (event.event_id.startswith('sealed-') or event.kind == 'host_finish'
            or is_host_final_payload(event.identity) or is_host_final_payload(event.output))
