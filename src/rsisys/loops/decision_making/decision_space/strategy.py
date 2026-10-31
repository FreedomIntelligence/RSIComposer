from typing import Protocol

from rsisys.loops.contracts.directive import Directive


class StrategyDesigner(Protocol):
    def design(self, objective: str, evidence_refs: tuple[str, ...]) -> Directive: ...
