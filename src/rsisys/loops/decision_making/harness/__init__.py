"""Replaceable decision execution boundary; memory and tools are implementation details."""
from typing import Protocol

from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from ..action_space import DecisionActionSpace
from .decision import LoopStepResult


class DecisionHarness(Protocol):
    def step(self, objective: str, action_space: DecisionActionSpace) -> LoopStepResult: ...
    def pending(self) -> Directive | None: ...
    def receive_feedback(self, directive: Directive, feedback: Feedback) -> None: ...
