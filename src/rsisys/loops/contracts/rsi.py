"""The two public RSI module boundaries, using the existing wire contracts.

Aliases deliberately preserve serialization, validation and runtime behavior.
LabFeedback carries status/data/artifact references; raw logs are optional evidence.
"""
from typing import Protocol

from .decision import DecisionPolicy, OperationCall
from .coordination import OperationResult
from .execution import LabExecution

RSIController = DecisionPolicy
Lab = LabExecution
LabRequest = OperationCall
LabFeedback = OperationResult


class ControllerOperationRequirements(Protocol):
    """Optional setup check; successful registration alone is not compatibility."""
    def required_operations(self) -> tuple[str, ...]: ...


class ControllerReportProvider(Protocol):
    """Optional method-owned JSON progress, separate from host lifecycle status."""
    def report(self) -> dict: ...

__all__ = ['RSIController', 'Lab', 'LabRequest', 'LabFeedback',
           'ControllerOperationRequirements', 'ControllerReportProvider']
