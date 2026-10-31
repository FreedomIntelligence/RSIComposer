"""Execution bindings for validated decision operations."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Protocol

from rsisys.loops.contracts.jobs import JobRef, JobStatus
from rsisys.runtime.artifacts import ArtifactStore
from rsisys.runtime.sandbox import ExecRequest, SandboxBackend, SandboxSpec
from ...action_space import OperationCall, OperationResult


class DecisionTools(Protocol):
    def execute(self, call: OperationCall) -> OperationResult | JobRef: ...


class BoundDecisionTools:
    """Bind each name to a handler receiving its validated argument mapping."""

    def __init__(self, bindings: Mapping[
        str, Callable[[dict[str, object]], OperationResult | JobRef]
    ]):
        self._bindings: dict[str, Callable[[dict[str, object]], OperationResult | JobRef]] = {}
        for name, handler in bindings.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Tool binding names must be nonempty strings")
            if name in self._bindings:
                raise ValueError("Tool binding names must be unique")
            if not callable(handler):
                raise TypeError(f"Tool binding must be callable: {name}")
            self._bindings[name] = handler

    def execute(self, call: OperationCall) -> OperationResult | JobRef:
        if not isinstance(call, OperationCall) or not isinstance(call.arguments, dict):
            raise TypeError("Expected an OperationCall with an argument mapping")
        if not isinstance(call.name, str) or call.name not in self._bindings:
            raise ValueError(f"Unbound decision operation: {call.name}")
        result = self._bindings[call.name](deepcopy(call.arguments))
        if not isinstance(result, (OperationResult, JobRef)):
            raise TypeError("Tool must return OperationResult or JobRef")
        return result


class DemoDecisionTools(BoundDecisionTools):
    """Prepare demonstration data through injected sandbox and artifact APIs."""

    def __init__(self, sandbox: SandboxBackend, artifacts: ArtifactStore):
        self.sandbox, self.artifacts = sandbox, artifacts
        super().__init__({"prepare_data": self._prepare_data})

    def _prepare_data(self, arguments: dict[str, object]) -> OperationResult:
        sandbox = self.sandbox.create(SandboxSpec())
        try:
            process = self.sandbox.exec(sandbox, ExecRequest(("echo", arguments["text"])))
            result = self.sandbox.poll(process)
            if result.status != JobStatus.SUCCEEDED:
                raise RuntimeError(f"Demo preparation did not succeed: {result.status}: {result.stderr}")
            ref = self.artifacts.put(result.stdout.encode(), kind="demo_input")
            return OperationResult("Prepared input artifact", (ref,), result.simulated)
        finally:
            self.sandbox.close(sandbox)
