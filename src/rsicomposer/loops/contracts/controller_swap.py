"""Method-owned choices for operations outstanding at a Controller boundary."""
from collections.abc import Mapping
from typing import Literal, Protocol

from .decision import OperationCall


class SwapPlanner(Protocol):
    def plan_swap(self, pending: tuple[OperationCall, ...]) -> Mapping[str, Literal['drain', 'cancel', 'handover']]: ...
