"""A Lab's callable surface, independent of research algorithms and loop state."""
from copy import deepcopy
from typing import Literal, Protocol

from .coordination import OperationResult
from .decision import OperationCall, OperationSpec
from .jobs import JobRef


class LabToolError(ValueError):
    """Provider-declared, model-correctable failure of one tool invocation.

    The adapter returns a failed OperationResult and lets decision continue.
    This does not request retry or prove that no effects occurred. Permissions,
    identity conflicts, accounting and infrastructure failures must not use it.
    """


class LabInfrastructureError(RuntimeError):
    """Host failure carrying an archived execution result, not a correctable tool error."""

    def __init__(self, result: OperationResult):
        self.result = deepcopy(result)
        super().__init__(result.error or result.summary)


class LabExecution(Protocol):
    """Describe installed tools and execute a call after host authorization.

    The Lab owns argument validation and native output under result.data. Tool
    schemas describe arguments; they do not grant permissions. No SFT/RL/eval
    categories, Actor policy, internal state or validator classes are required.

    A JobRef means the host must arrange completion/resume. A nonblocking Lab
    can instead expose start/poll/cancel tools returning ordinary invocation
    receipts with native job state in data. Submission never proves completion.
    Expected tool failures may return a failed result or raise LabToolError;
    other exceptions remain host failures. FeedbackDelivery is optional metadata
    for immutable producer results; data is never interpreted as routing hints.
    """
    def describe(self) -> tuple[OperationSpec, ...]: ...
    def execute(self, call: OperationCall) -> OperationResult | JobRef: ...


class ExecutionLifecycle(Protocol):
    """Optional host lifecycle for Labs owning background work.

    shutdown is bounded, idempotent and stops admission. True confirms that all
    owned work has exited; a cancellation acknowledgement alone returns False.
    The host repeats it on later ticks. close releases owned resources, also on
    assembly failures. close must be idempotent and retryable after failure;
    the owner retains it and its shared dependencies until it succeeds.
    Synchronous providers need neither method. Providers must bound their own
    synchronous calls; a host deadline cannot preempt a blocked Python method.
    """
    def shutdown(self, mode: Literal['drain', 'cancel']) -> bool: ...
    def close(self) -> None: ...
