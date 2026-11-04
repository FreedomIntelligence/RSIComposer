"""Decision inputs and actions, independent of a harness state or prompt format."""
from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import Literal, Protocol, ContextManager, Callable
from uuid import uuid4

from .directive import Directive
from .jobs import JobRef
from .model_service import ModelBackend, TokenUsage


@dataclass(frozen=True)
class OperationSpec:
    name: str
    description: str
    input_schema: dict[str, object]
    read_only: bool = False

    def __post_init__(self):
        if type(self.read_only) is not bool:
            raise TypeError('read_only must be a bool')


@dataclass(frozen=True)
class OperationCall:
    name: str
    arguments: dict[str, object]


class OperationArgumentError(ValueError):
    """A validator's explicit rejection of model-correctable call arguments.

    The default executor records this as failed feedback before calling a tool.
    It never retries the call. Permission, identity and internal provider faults
    must not use this signal; ordinary exceptions keep their host semantics.
    """


def validate_attempt_fields(ident, title):
    """Compatibility metadata validation; membership itself is optional."""
    if ident is not None and (not isinstance(ident, str) or not ident.strip() or len(ident) > 200):
        raise ValueError('research_attempt_id must be a nonempty string of at most 200 characters or None')
    if not isinstance(title, str) or (title and not title.strip()) or len(title) > 500:
        raise ValueError('research_attempt_title must be a string of at most 500 characters, nonblank when supplied')


@dataclass(frozen=True)
class Decision:
    kind: Literal['invoke', 'explore', 'wait', 'finish']
    operation: OperationCall | None = None
    directive: Directive | None = None
    job: JobRef | None = None
    reason: str = ''
    simulated: bool = False
    usage: TokenUsage | None = None
    decision_id: str = field(default_factory=lambda: str(uuid4()))
    depends_on_decision_ids: tuple[str, ...] = ()
    source_decision_ids: tuple[str, ...] = ()
    research_attempt_id: str | None = None
    research_attempt_title: str = ''
    feedback_proposal_ids: tuple[str, ...] = ()

    def __post_init__(self):
        validate_attempt_fields(self.research_attempt_id, self.research_attempt_title)
        if not isinstance(self.decision_id, str) or not self.decision_id.strip():
            raise ValueError('decision_id must be a nonempty string')
        refs = self.feedback_proposal_ids
        if (not isinstance(refs, tuple) or len(refs) > 16
                or any(not isinstance(ref, str) or not ref.strip() or len(ref) > 512 for ref in refs)
                or len(set(refs)) != len(refs)):
            raise ValueError('feedback_proposal_ids must contain at most 16 distinct proposal IDs')
        for name in ('depends_on_decision_ids', 'source_decision_ids'):
            refs = getattr(self, name)
            if (not isinstance(refs, tuple) or any(not isinstance(ref, str) or not ref.strip() for ref in refs)
                    or len(set(refs)) != len(refs) or self.decision_id in refs):
                raise ValueError(f'{name} must contain distinct nonempty IDs, excluding self')
        expected = {'invoke': 'operation', 'explore': 'directive', 'wait': 'job', 'finish': None}
        if self.kind not in expected:
            raise ValueError(f'Unknown decision kind: {self.kind}')
        present = {key for key in ('operation', 'directive', 'job') if getattr(self, key) is not None}
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

    def __post_init__(self):
        if not isinstance(self.objective, str) or not isinstance(self.context, str):
            raise TypeError('Decision objective and context must be text')
        if not isinstance(self.capabilities, tuple) or any(not isinstance(row, dict) for row in self.capabilities):
            raise TypeError('Decision capabilities must be a tuple of JSON descriptions')
        json.dumps([self.capabilities, self.references], allow_nan=False)
        object.__setattr__(self, 'capabilities', deepcopy(self.capabilities))
        object.__setattr__(self, 'references', deepcopy(self.references))


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
