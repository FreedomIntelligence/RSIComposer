"""Decision inputs and actions, independent of a harness state or prompt format."""
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import Literal, Protocol, ContextManager, Callable
from uuid import uuid4

from .jobs import JobRef, JobControl, OperationProgress
from .model_service import ModelBackend, TokenUsage
from .coordination import OperationResult


_host_decision_id = ContextVar('rsicomposer-host-decision-id', default=None)


def _new_decision_id():
    return _host_decision_id.get() or str(uuid4())


@dataclass(frozen=True)
class OperationSpec:
    """One operation a Lab offers.

    idempotent=True declares that executing the same OperationCall.call_id again,
    after an interruption, is safe: the Lab returns the earlier outcome or redoes
    the work without duplicating its effects. The host then resumes an interrupted
    durable run automatically instead of requiring manual reconciliation.
    """
    name: str
    description: str
    input_schema: dict[str, object]
    read_only: bool = False
    idempotent: bool = False

    def __post_init__(self):
        if type(self.read_only) is not bool or type(self.idempotent) is not bool:
            raise TypeError('read_only and idempotent must be bools')


@dataclass(frozen=True)
class OperationCall:
    """A request for one operation. call_id identifies this request across retries.

    The host keeps call_id unchanged from the decision to the Lab, so a Lab can use
    it as an idempotency key. It does not take part in equality: two calls with the
    same name and arguments compare equal.
    """
    name: str
    arguments: dict[str, object]
    call_id: str = field(default_factory=lambda: str(uuid4()), compare=False)

    def __post_init__(self):
        if not isinstance(self.call_id, str) or not self.call_id.strip() or len(self.call_id) > 200:
            raise ValueError('call_id must be a nonempty string of at most 200 characters')


class OperationArgumentError(ValueError):
    """A validator's explicit rejection of model-correctable call arguments.

    The default executor records this as failed feedback before calling a tool.
    It never retries the call. Permission, identity and internal provider faults
    must not use this signal; ordinary exceptions keep their host semantics.
    """


@dataclass(frozen=True)
class Decision:
    kind: Literal['invoke', 'wait', 'finish', 'control']
    operations: tuple[OperationCall, ...] = ()
    job: JobRef | None = None
    control: JobControl | None = None
    reason: str = ''
    simulated: bool = False
    usage: TokenUsage | None = None
    decision_id: str = field(default_factory=_new_decision_id)
    depends_on_decision_ids: tuple[str, ...] = ()
    source_decision_ids: tuple[str, ...] = ()

    @classmethod
    def invoke(cls, call, **metadata):
        return cls('invoke', operations=(call,), **metadata)

    def __post_init__(self):
        if (not isinstance(self.operations, tuple) or any(not isinstance(call, OperationCall) for call in self.operations)
                or len({call.call_id for call in self.operations}) != len(self.operations)):
            raise TypeError('Decision operations must be distinct OperationCalls in a tuple')
        if self.control is not None and not isinstance(self.control, JobControl):
            raise TypeError('Decision control must be a JobControl')
        if not isinstance(self.decision_id, str) or not self.decision_id.strip():
            raise ValueError('decision_id must be a nonempty string')
        for name in ('depends_on_decision_ids', 'source_decision_ids'):
            refs = getattr(self, name)
            if (not isinstance(refs, tuple) or any(not isinstance(ref, str) or not ref.strip() for ref in refs)
                    or len(set(refs)) != len(refs) or self.decision_id in refs):
                raise ValueError(f'{name} must contain distinct nonempty IDs, excluding self')
        expected = {'invoke': 'operations', 'wait': 'job', 'finish': None, 'control': 'control'}
        if self.kind not in expected:
            raise ValueError(f'Unknown decision kind: {self.kind}')
        present = {key for key in ('job', 'control') if getattr(self, key) is not None}
        if self.operations:
            present.add('operations')
        wanted = set() if expected[self.kind] is None or (self.kind == 'wait' and self.job is None) else {expected[self.kind]}
        if present != wanted:
            raise ValueError(f'Decision {self.kind} requires only {expected[self.kind]}')


@dataclass(frozen=True)
class DecisionInput:
    """Current input window; context organization and reference formats are opaque.

    Capability descriptions are discovery data, never permission grants. Host
    validation still applies to every returned action. Inputs are copied before
    delivery so a policy cannot mutate the host's objects through this window.
    """
    objective: str
    context: str = ''
    capabilities: tuple[dict, ...] = ()
    references: object = ()
    feedbacks: tuple[OperationResult, ...] = ()
    pending: tuple[str, ...] = ()
    progresses: tuple[OperationProgress, ...] = ()
    budget: dict[str, float | int] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.objective, str) or not isinstance(self.context, str):
            raise TypeError('Decision objective and context must be text')
        if not isinstance(self.capabilities, tuple) or any(not isinstance(row, dict) for row in self.capabilities):
            raise TypeError('Decision capabilities must be a tuple of JSON descriptions')
        import math
        if (not isinstance(self.budget, dict) or any(not isinstance(k, str) or not k
                or type(v) not in (int, float) or not math.isfinite(v) or v < 0 for k, v in self.budget.items())):
            raise TypeError('Decision budget must map limit names to nonnegative finite remaining amounts')
        json.dumps(self.budget, allow_nan=False)
        object.__setattr__(self, 'budget', deepcopy(self.budget))
        json.dumps([self.capabilities, self.references], allow_nan=False)
        object.__setattr__(self, 'capabilities', deepcopy(self.capabilities))
        object.__setattr__(self, 'references', deepcopy(self.references))
        if (not isinstance(self.feedbacks, tuple) or any(not isinstance(result, OperationResult)
                or result.call_id is None for result in self.feedbacks)
                or len({result.call_id for result in self.feedbacks}) != len(self.feedbacks)):
            raise TypeError('Decision feedbacks must be distinct receipts with call_id')
        if (not isinstance(self.pending, tuple) or any(not isinstance(ident, str) or not ident for ident in self.pending)
                or len(set(self.pending)) != len(self.pending)):
            raise TypeError('Decision pending must be distinct call IDs')
        if {result.call_id for result in self.feedbacks}.intersection(self.pending):
            raise ValueError('Completed feedback cannot also be pending')
        if not isinstance(self.progresses, tuple) or any(not isinstance(row, OperationProgress) for row in self.progresses):
            raise TypeError('Decision progresses must contain OperationProgress observations')
        object.__setattr__(self, 'progresses', deepcopy(self.progresses))
        object.__setattr__(self, 'feedbacks', deepcopy(self.feedbacks))



class DecisionPolicy(Protocol):
    def decide(self, context: DecisionInput) -> Decision: ...


class UsageObservedPolicy(Protocol):
    """Optional per-response observation, including rejected/summary responses."""
    def track_usage(self, record: Callable[[TokenUsage | None], None]) -> ContextManager: ...


class RequestContextPolicy(Protocol):
    """Optional final-request ownership; do not pre-compact this policy's input.

    compactor is a host default capability. The policy may select an explicitly
    configured implementation or own its native session. It must still use the
    deployment's complete-request admission and accounted model access.
    """
    def context_scope(self, compactor, *, recent_limit: int) -> ContextManager: ...


class SharedBackendPolicy(Protocol):
    """Optional explicit sharing of a stateless, accounted model dependency."""
    def share_backend(self, wrap: Callable[[ModelBackend], ModelBackend]) -> ModelBackend: ...
