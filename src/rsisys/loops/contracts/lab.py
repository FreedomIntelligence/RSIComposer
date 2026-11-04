"""A fixed-objective research environment, independent of a meta-agent harness."""
from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol

from .candidate import SystemVersion
from .coordination import ExplorationProgress
from .directive import Directive
from .receipts import ExecutionReceipt
from .model_service import ModelRef


@dataclass(frozen=True)
class LabSpec:
    lab_id: str
    objective: str
    capabilities: tuple[str, ...]
    system: SystemVersion | None = None

    def __post_init__(self):
        for value in (self.lab_id, self.objective):
            if not isinstance(value, str) or not value.strip():
                raise ValueError('Lab identity and objective must be nonempty')
        if (not isinstance(self.capabilities, tuple)
                or any(not isinstance(name, str) or not name.strip() for name in self.capabilities)
                or len(set(self.capabilities)) != len(self.capabilities)):
            raise ValueError('Lab capabilities must be distinct nonempty names')
        if self.system is not None and not isinstance(self.system, SystemVersion):
            raise TypeError('Lab system must be SystemVersion or None')
        object.__setattr__(self, 'system', deepcopy(self.system))


@dataclass(frozen=True)
class LabTask:
    task_id: str
    decision_id: str
    lab_id: str
    idempotency_key: str
    directive: Directive
    run_id: str | None = None
    agent_id: str | None = None
    issuing_model: ModelRef | None = None

    def __post_init__(self):
        for name in ('task_id', 'decision_id', 'lab_id', 'idempotency_key'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f'Lab task {name} must be nonempty')
        for name in ('run_id', 'agent_id'):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f'Lab task {name} must be nonempty or None')
        if self.issuing_model is not None and not isinstance(self.issuing_model, ModelRef):
            raise TypeError('Lab task issuing_model must be ModelRef or None')
        if not isinstance(self.directive, Directive) or self.directive.directive_id != self.task_id:
            raise ValueError('Lab task must retain the directive task identity')
        self.directive.__post_init__()
        object.__setattr__(self, 'directive', deepcopy(self.directive))


class Lab(Protocol):
    def describe(self) -> LabSpec: ...
    def submit(self, task: LabTask) -> ExplorationProgress: ...
    def advance(self, task_id: str) -> ExplorationProgress: ...
    def inspect(self, task_id: str) -> ExplorationProgress: ...
    def cancel(self, task_id: str) -> ExplorationProgress: ...
    def result(self, task_id: str) -> ExecutionReceipt: ...
    def acknowledge(self, task_id: str) -> None: ...
