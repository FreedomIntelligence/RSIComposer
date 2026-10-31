from dataclasses import dataclass
from typing import Protocol

from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.jobs import JobRef, JobStatus
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.runtime.artifacts import ArtifactStore
from rsisys.runtime.sandbox import ExecRequest, SandboxBackend, SandboxSpec


@dataclass(frozen=True)
class OperationSpec:
    name: str
    description: str
    input_schema: dict[str, object]


@dataclass(frozen=True)
class OperationCall:
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class OperationResult:
    summary: str
    artifacts: tuple[ArtifactRef, ...]
    simulated: bool
    status: ExecutionStatus = ExecutionStatus.COMPLETED
    error: str | None = None

    def __post_init__(self):
        if self.status not in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED):
            raise ValueError("An operation result must be terminal")
        if self.status == ExecutionStatus.FAILED and not self.error:
            raise ValueError("A failed operation requires an error")
        if self.status == ExecutionStatus.COMPLETED and self.error:
            raise ValueError("A completed operation cannot carry an error")


class DecisionSpace(Protocol):
    def describe(self) -> tuple[OperationSpec, ...]: ...

    def execute(self, call: OperationCall) -> OperationResult | JobRef: ...


class DemoDecisionSpace:
    """One data-preparation fixture through injected sandbox and artifact APIs."""

    def __init__(self, sandbox: SandboxBackend, artifacts: ArtifactStore):
        self.sandbox, self.artifacts = sandbox, artifacts

    def describe(self) -> tuple[OperationSpec, ...]:
        return (OperationSpec("prepare_data", "Prepare a demonstration input artifact", {
            "type": "object", "properties": {"text": {"type": "string"}},
            "required": ["text"], "additionalProperties": False,
        }),)

    def execute(self, call: OperationCall) -> OperationResult:
        if call.name != "prepare_data":
            raise ValueError(f"Unsupported decision operation: {call.name}")
        if set(call.arguments) != {"text"} or not isinstance(call.arguments["text"], str):
            raise ValueError("prepare_data requires exactly one string argument: text")
        sandbox = self.sandbox.create(SandboxSpec())
        try:
            process = self.sandbox.exec(sandbox, ExecRequest(("echo", call.arguments["text"])))
            result = self.sandbox.poll(process)
            if result.status != JobStatus.SUCCEEDED:
                raise RuntimeError(f"Demo preparation did not succeed: {result.status}: {result.stderr}")
            ref = self.artifacts.put(result.stdout.encode(), kind="demo_input")
            return OperationResult("Prepared input artifact", (ref,), result.simulated)
        finally:
            self.sandbox.close(sandbox)


class DecisionOperation(Protocol):
    """Each operation owns its argument validation and concrete implementation."""
    spec: OperationSpec
    def execute(self, arguments: dict[str, object]) -> OperationResult | JobRef: ...


class ModularDecisionSpace:
    def __init__(self, operations: tuple[DecisionOperation, ...]):
        self._operations = {operation.spec.name: operation for operation in operations}
        if len(self._operations) != len(operations):
            raise ValueError('Decision operation names must be unique')

    def describe(self) -> tuple[OperationSpec, ...]:
        from copy import deepcopy
        return tuple(deepcopy(operation.spec) for operation in self._operations.values())

    def execute(self, call: OperationCall) -> OperationResult | JobRef:
        if not isinstance(call, OperationCall) or not isinstance(call.arguments, dict):
            raise TypeError('Expected an OperationCall with an argument mapping')
        if call.name not in self._operations:
            raise ValueError(f'Unknown decision operation: {call.name}')
        result = self._operations[call.name].execute(call.arguments)
        if not isinstance(result, (OperationResult, JobRef)):
            raise TypeError('Operation must return OperationResult or JobRef')
        return result
