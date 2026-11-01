"""System-command execution boundary; does not require an Actor policy or memory."""
from typing import Protocol

from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.coordination import ExplorationProgress
from ..action_space import ActionSpace


class ExplorationHarness(Protocol):
    def accept(self, directive: Directive) -> ExplorationProgress: ...
    def advance(self, directive_id: str, action_space: ActionSpace) -> ExplorationProgress: ...
    def inspect(self, directive_id: str) -> ExplorationProgress: ...
    def cancel(self, directive_id: str) -> ExplorationProgress: ...
    def acknowledge(self, directive_id: str) -> None: ...

# Import after the protocol to avoid an initialization cycle.
from .concurrent import ConcurrentExplorationHarness
