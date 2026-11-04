"""Compatibility imports; action contracts live outside the default harness."""
from dataclasses import dataclass
from typing import Literal
from rsisys.loops.contracts.decision import Decision
from rsisys.loops.contracts.coordination import LoopStepResult


@dataclass(frozen=True)
class DecisionRunResult:
    status: Literal["dispatch", "waiting", "finished", "step_limit"]
    steps: tuple[LoopStepResult, ...]
