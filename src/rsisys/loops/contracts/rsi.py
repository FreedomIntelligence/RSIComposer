"""The two public RSI module boundaries, using the existing wire contracts.

Aliases deliberately preserve serialization, validation and runtime behavior.
LabFeedback carries status/data/artifact references; raw logs are optional evidence.
"""
from .decision import DecisionPolicy, OperationCall
from .coordination import OperationResult
from .execution import LabExecution

RSIController = DecisionPolicy
Lab = LabExecution
LabRequest = OperationCall
LabFeedback = OperationResult

__all__ = ['RSIController', 'Lab', 'LabRequest', 'LabFeedback']
