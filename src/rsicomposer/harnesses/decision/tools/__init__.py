"""Shared operation execution and receipts; callers decide whether to wait or continue."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Protocol

from rsicomposer.loops.contracts.coordination import OperationResult
from rsicomposer.loops.contracts.decision import OperationArgumentError, OperationCall
from rsicomposer.loops.contracts.execution import LabInfrastructureError
from rsicomposer.loops.contracts.jobs import JobRef, JobStatus
from rsicomposer.loops.contracts.lifecycle import ExecutionStatus
from rsicomposer.runtime.execution.sandbox import ExecRequest, SandboxBackend, SandboxSpec
from rsicomposer.runtime.storage.artifacts import ArtifactStore


class DecisionTools(Protocol):
    def execute(self, call: OperationCall) -> OperationResult | JobRef: ...


@dataclass(frozen=True)
class OperationExecution:
    call: OperationCall
    outcome: OperationResult | JobRef
    elapsed_seconds: float
    simulated: bool
    error: Exception | None = None

    def observation(self) -> OperationResult:
        """A job reference acknowledges submission, never backend completion."""
        if isinstance(self.outcome, JobRef):
            return OperationResult('Operation submitted; backend job is not yet complete', (),
                self.simulated, data={'job': asdict(self.outcome), 'status': 'submitted'}, call_id=self.call.call_id)
        return replace(deepcopy(self.outcome), call_id=self.call.call_id)


def execute_operation(tools, action_space, call: OperationCall, *, simulated=False,
                      clock=perf_counter, retry=False, retry_sequence=None) -> OperationExecution:
    """Validate/authorize and execute once, without choosing a scheduling policy."""
    started = clock()
    call = deepcopy(call)
    error = None
    try:
        try:
            # Validators rebuild the call from its arguments; the request identity stays.
            call = replace(deepcopy(action_space.validate(call)), call_id=call.call_id)
        except OperationArgumentError as exc:
            # Only an explicit pre-execution argument rejection is recoverable.
            # The same exception from a tool is still an unexpected host fault.
            outcome = OperationResult('Decision arguments rejected; correct the next call', (),
                simulated, ExecutionStatus.FAILED, f'{type(exc).__name__}: {exc}')
            return OperationExecution(call, replace(outcome, call_id=call.call_id), clock() - started, simulated)
        if retry:
            apply_retry = getattr(tools, 'retry', None)
            if not callable(apply_retry):
                raise ValueError('Tools do not provide an explicit infrastructure retry contract')
            outcome = apply_retry(deepcopy(call), after_sequence=retry_sequence)
        else:
            outcome = tools.execute(deepcopy(call))
        if not isinstance(outcome, (OperationResult, JobRef)):
            raise TypeError('Tool must return OperationResult or JobRef')
        if isinstance(outcome, OperationResult):
            outcome.__post_init__()
            outcome = replace(outcome, simulated=outcome.simulated or simulated)
        outcome = deepcopy(outcome)
    except Exception as exc:
        error = exc
        outcome = (replace(exc.result, simulated=exc.result.simulated or simulated)
                   if isinstance(exc, LabInfrastructureError) else
                   OperationResult('Decision operation failed', (), simulated,
                       ExecutionStatus.FAILED, f'{type(exc).__name__}: {exc}'))
    if isinstance(outcome, OperationResult):
        outcome = replace(outcome, call_id=call.call_id)
    return OperationExecution(call, outcome, clock() - started, simulated, error)




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

    def __init__(self, sandbox: SandboxBackend, artifacts: ArtifactStore, *, extra_bindings=None):
        self.sandbox, self.artifacts = sandbox, artifacts
        bindings = {"prepare_data": self._prepare_data}
        if set(bindings).intersection(extra_bindings or {}):
            raise ValueError('Extra tool bindings must not replace built-in operations')
        super().__init__({**bindings, **(extra_bindings or {})})
        self.query_tools = None

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
