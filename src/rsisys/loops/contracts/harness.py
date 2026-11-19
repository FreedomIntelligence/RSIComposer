"""Loop endpoint ports. Concrete policy, memory and tool choices belong to harnesses."""
from dataclasses import dataclass
from typing import Literal, Protocol

from .coordination import LoopStepResult, OperationResult
from .decision import DecisionInput, OperationSpec, OperationCall
from .feedback_context import EvidenceEvent



class ControllerHarness(Protocol):
    """Four required ports; optional extensions are explicitly discovered.

    Durable runs additionally require state()/restore(state). A Harness may
    declare bind_execution(read), where read() supplies a detached execution
    window, and freeze_decision_metadata(policy, decision_id),
    admit_decision_metadata(decision, requested, records), and
    validate_execution_records(records) to own domain metadata semantics.
    HarnessAssembly owns Memory and feedback integration independently; those
    services never require extra attributes on a four-port Harness.
    """
    def window(self, objective: str, capabilities: tuple[dict, ...],
               feedback: tuple['OperationResult', ...]) -> 'DecisionInput': ...
    def observe(self, event: 'EvidenceEvent') -> None: ...
    def tools(self) -> tuple['OperationSpec', ...]: ...
    def execute_tool(self, call: 'OperationCall') -> 'OperationResult': ...


@dataclass(frozen=True)
class DecisionRunResult:
    status: Literal['dispatch', 'waiting', 'finished', 'step_limit']
    steps: tuple[LoopStepResult, ...]



class NullHarness:
    """Pass receipts through without organizing additional Controller context."""
    def window(self, objective, capabilities, feedback):
        return DecisionInput(objective, capabilities=capabilities, feedbacks=feedback)

    def observe(self, event):
        pass

    def tools(self):
        return ()

    def execute_tool(self, call):
        raise ValueError('NullHarness declares no local tools')

    def state(self):
        return {}

    def restore(self, state):
        if not isinstance(state, dict) or state:
            raise ValueError('NullHarness requires empty state')
