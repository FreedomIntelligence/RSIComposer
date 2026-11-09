"""Shared operation execution and receipts; callers decide whether to wait or continue."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Protocol

from rsisys.harnesses.memory import HistoryEntry
from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.decision import OperationArgumentError, OperationCall
from rsisys.loops.contracts.execution import LabInfrastructureError
from rsisys.loops.contracts.jobs import JobRef, JobStatus
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.runtime.evidence.records import ExperienceRecord
from rsisys.runtime.execution.sandbox import ExecRequest, SandboxBackend, SandboxSpec
from rsisys.runtime.storage.artifacts import ArtifactStore


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
                self.simulated, data={'job': asdict(self.outcome), 'status': 'submitted'})
        return deepcopy(self.outcome)


def execute_operation(tools, action_space, call: OperationCall, *, simulated=False,
                      clock=perf_counter) -> OperationExecution:
    """Validate/authorize and execute once, without choosing a scheduling policy."""
    started = clock()
    call = deepcopy(call)
    error = None
    try:
        try:
            call = deepcopy(action_space.validate(call))
        except OperationArgumentError as exc:
            # Only an explicit pre-execution argument rejection is recoverable.
            # The same exception from a tool is still an unexpected host fault.
            outcome = OperationResult('Decision arguments rejected; correct the next call', (),
                simulated, ExecutionStatus.FAILED, f'{type(exc).__name__}: {exc}')
            return OperationExecution(call, outcome, clock() - started, simulated)
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
    return OperationExecution(call, outcome, clock() - started, simulated, error)


def record_operation(pending, memory, experience=None):
    """Write the same frozen receipt on retry; the caller commits its own state afterwards."""
    ident, call, result = pending.record_id, pending.call, pending.result
    details = {'operation': asdict(call) if call else None, 'result': result.payload()}
    if pending.event is not None:
        details.update(event=pending.event, elapsed_seconds=pending.elapsed_seconds)
    memory.append(HistoryEntry(ident, 'operation', pending.task_id, result.summary, result.simulated,
        source_refs=tuple(ref.artifact_id for ref in result.artifacts), details=details))
    if experience is not None:
        experience.record(ExperienceRecord(
            record_id=ident, kind='runtime_feedback', scope='decision_maker', directive_id=pending.task_id,
            summary=f'Preparation operation {result.status.value}', simulated=result.simulated,
            source_refs=(f'history:decision_maker:{ident}',),
            details={'operation': call.name if call else None, 'status': result.status.value,
                     'error': result.error, 'elapsed_seconds': pending.elapsed_seconds,
                     'history_ref': ident, 'history_owner': 'decision_maker'}))


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
