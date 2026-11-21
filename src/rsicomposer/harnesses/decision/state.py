from dataclasses import dataclass, field

from rsicomposer.harnesses.memory import HistoryEntry, Memory
from rsicomposer.loops.contracts.coordination import OperationResult
from rsicomposer.loops.contracts.decision import OperationSpec
from rsicomposer.loops.contracts.jobs import OperationProgress
from rsicomposer.loops.contracts.model_service import ModelRef
from rsicomposer.runtime.evidence.records import ExperienceRecord




@dataclass(frozen=True)
class DecisionContext:
    objective: str
    experience: tuple[ExperienceRecord, ...]
    operations: tuple[OperationSpec, ...]
    operation_results: tuple[OperationResult, ...]
    memories: tuple[Memory, ...] = ()
    history: tuple[HistoryEntry, ...] = ()
    decision_model: ModelRef | None = None
    run_control: dict[str, object] = field(default_factory=dict)
    context_summary: dict[str, object] = field(default_factory=dict)
    research_agenda: dict[str, object] = field(default_factory=dict)
    decision_graph: tuple[dict, ...] = ()
    research_state: dict[str, object] = field(default_factory=dict)
    # Current provider view, replaced each assembly; it is not an append-only log.
    memory_context: str = ''
    pending: tuple[str, ...] = ()
    feedbacks: tuple[OperationResult, ...] = ()
    progresses: tuple[OperationProgress, ...] = ()



@dataclass
class ControllerHarnessState:
    observed_event_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.observed_event_ids, tuple) or any(not isinstance(x, str) for x in self.observed_event_ids):
            raise ValueError("Invalid Controller Harness observation identities")
