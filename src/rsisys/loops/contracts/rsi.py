"""Optional Controller capabilities used by explicit method assembly."""
from typing import Protocol


class ControllerOperationRequirements(Protocol):
    def required_operations(self) -> tuple[str, ...]: ...


class ControllerReportProvider(Protocol):
    def report(self) -> dict: ...


__all__ = ['ControllerOperationRequirements', 'ControllerReportProvider']
