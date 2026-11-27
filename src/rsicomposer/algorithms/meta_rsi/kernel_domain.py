"""Domain-independent execution of the public MetaRSI v2 loop kernel.

A domain supplies fixed code for compiling observations, evaluating candidates,
validating proposals, applying actions and selecting their results. Only
Diagnose and Propose use the mutable policy; they receive a typed request and
the compiled signal, never the evaluator, trainer or environment. The kernel
runs the supplied order as-is. It does not plan programs or decide releases.

Payloads, state, signals and receipts use JSON values (opaque external artifacts
can be represented by references). The domain owns their schema and semantics.
"""
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from typing import ClassVar, Literal, TypeAlias

from rsicomposer.runtime.storage.files import json_bytes

from .kernel_evidence import (
    Action, Operator, StateIdentity, build_kernel_execution, build_state_identity,
    complete_kernel_evidence, operator_card, validate_transition,
)


JsonValue: TypeAlias = str | int | float | bool | None | list['JsonValue'] | dict[str, 'JsonValue']
Cost: TypeAlias = Mapping[str, int | float]


@dataclass(frozen=True, slots=True)
class KernelState:
    candidate_id: str
    data: JsonValue
    harness: JsonValue
    model: JsonValue

    def identity(self, protected: Mapping[str, JsonValue]) -> StateIdentity:
        return build_state_identity(self.candidate_id, {'data': self.data, 'harness': self.harness, 'model': self.model},
                                    protected=protected)


@dataclass(frozen=True, slots=True)
class DataRequest:
    request_id: str
    operator: ClassVar[Literal['data']] = 'data'


@dataclass(frozen=True, slots=True)
class HarnessRequest:
    request_id: str
    operator: ClassVar[Literal['harness']] = 'harness'


@dataclass(frozen=True, slots=True)
class ModelRequest:
    request_id: str
    operator: ClassVar[Literal['model']] = 'model'


KernelRequest: TypeAlias = DataRequest | HarnessRequest | ModelRequest


@dataclass(frozen=True, slots=True)
class DataProposal:
    payload: JsonValue
    action: ClassVar[Literal['synthesize_tasks']] = 'synthesize_tasks'


@dataclass(frozen=True, slots=True)
class HarnessProposal:
    payload: JsonValue
    action: ClassVar[Literal['patch_harness']] = 'patch_harness'


@dataclass(frozen=True, slots=True)
class ModelProposal:
    payload: JsonValue
    action: ClassVar[Literal['fit_adapter']] = 'fit_adapter'


@dataclass(frozen=True, slots=True)
class NoOpProposal:
    action: ClassVar[Literal['no_op']] = 'no_op'


KernelProposal: TypeAlias = DataProposal | HarnessProposal | ModelProposal | NoOpProposal


@dataclass(frozen=True, slots=True)
class ProposalPolicy:
    diagnose: Callable[[KernelRequest, JsonValue], JsonValue]
    propose: Callable[[KernelRequest, JsonValue, JsonValue], KernelProposal]


@dataclass(frozen=True, slots=True)
class Observation:
    state_sha256: str
    signal: JsonValue


@dataclass(frozen=True, slots=True)
class ValidationContext:
    request: KernelRequest
    state: KernelState
    observation: Observation
    diagnosis: JsonValue


@dataclass(frozen=True, slots=True)
class Approved:
    """Action input constructed or explicitly approved by fixed domain code."""

    input: JsonValue


@dataclass(frozen=True, slots=True)
class Rejected:
    reason: JsonValue = None


Validator: TypeAlias = Callable[[ValidationContext, JsonValue], Approved | Rejected]


@dataclass(frozen=True, slots=True)
class ActionResult:
    """The replacement for exactly the declared component, not a whole state."""

    component: JsonValue
    cost: Cost | None = None


@dataclass(frozen=True, slots=True)
class Comparison:
    request: KernelRequest
    incumbent: KernelState
    candidate: KernelState
    incumbent_outcome: JsonValue
    candidate_outcome: JsonValue
    cost: Cost | None


@dataclass(frozen=True, slots=True)
class Selection:
    decision: Literal['retain', 'promote']
    evidence: JsonValue = None


Selector: TypeAlias = Callable[[Comparison], Selection]


@dataclass(frozen=True, slots=True)
class OperatorAuthority:
    """Domain authority for a Data or Harness action."""

    validate: Validator
    execute: Callable[[KernelState, JsonValue], ActionResult]
    select: Selector


@dataclass(frozen=True, slots=True)
class ModelAuthority:
    """Only fit_input may supply training inputs; validate approves recipe choices.

    fit_input resolves eligible verified input from the current selected state
    plus any fixed domain bindings, or returns Rejected if there is none. Fit
    receives that input and a domain-approved recipe, not a policy payload,
    observation, diagnosis, or mutable state. The domain's fit implementation
    is responsible for consuming training data only from the first argument.
    """

    validate: Validator
    fit_input: Callable[[KernelState], JsonValue | Rejected]
    fit: Callable[[JsonValue, JsonValue], ActionResult]
    select: Selector


@dataclass(frozen=True, slots=True)
class DomainAuthority:
    """Trusted domain ports and the protected bindings for this run.

    compile reevaluates the supplied selected state on adaptation inputs and
    returns the domain's policy-facing projection. oracle independently
    measures incumbent and candidate for selection on the request's fixed
    comparison inputs. Neither authority makes its raw inputs available to
    the proposal policy by default.
    """

    compile: Callable[[KernelState], JsonValue]
    oracle: Callable[[KernelRequest, KernelState], JsonValue]
    data: OperatorAuthority
    harness: OperatorAuthority
    model: ModelAuthority
    protected: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class KernelStep:
    request: KernelRequest
    before: KernelState
    candidate: KernelState
    selected: KernelState
    evidence: dict

    @property
    def artifact(self) -> JsonValue:
        """Materialized selected component described by the typed Export stage."""
        return deepcopy(getattr(self.selected, self.request.operator))


@dataclass(frozen=True, slots=True)
class KernelRun:
    selected: KernelState
    observations: tuple[Observation, ...]
    steps: tuple[KernelStep, ...]


_REQUEST_TYPES = (DataRequest, HarnessRequest, ModelRequest)
_PROPOSAL_TYPES = (DataProposal, HarnessProposal, ModelProposal)


def _snapshot(value: JsonValue) -> JsonValue:
    # Keep a canonical, detached JSON value across authority boundaries.
    return json.loads(json_bytes(value))


def _digest(value: JsonValue) -> str:
    return sha256(json_bytes(value)).hexdigest()


def _observe(domain: DomainAuthority, state: KernelState, identity: StateIdentity) -> Observation:
    return Observation(identity['state_sha256'], _snapshot(domain.compile(deepcopy(state))))


def _validate(
    authority: OperatorAuthority | ModelAuthority, context: ValidationContext, proposal: KernelProposal,
) -> tuple[Action, JsonValue, Approved | Rejected, JsonValue]:
    card = operator_card(context.request.operator)
    if type(proposal) is NoOpProposal:
        return 'no_op', {'action': 'no_op'}, Approved(None), None
    if type(proposal) not in _PROPOSAL_TYPES:
        return card.action, None, Rejected('untyped_proposal'), None
    record = {'action': proposal.action, 'payload': _snapshot(proposal.payload)}
    if proposal.action != card.action:
        return card.action, record, Rejected('wrong_operator'), None
    decision = authority.validate(deepcopy(context), deepcopy(record['payload']))
    if type(decision) is Rejected:
        return card.action, record, Rejected(_snapshot(decision.reason)), None
    if type(decision) is not Approved:
        raise TypeError('MetaRSI validator must return Approved or Rejected')
    approval = Approved(_snapshot(decision.input))
    if isinstance(authority, ModelAuthority):
        source = authority.fit_input(deepcopy(context.state))
        if type(source) is Rejected:
            return card.action, record, Rejected(_snapshot(source.reason)), None
        return card.action, record, approval, _snapshot(source)
    return card.action, record, approval, None


def _apply(state: KernelState, identity: StateIdentity, request: KernelRequest, component: JsonValue,
           protected: Mapping[str, JsonValue]) -> tuple[KernelState, StateIdentity]:
    surface: Operator = request.operator
    if json_bytes(component) == json_bytes(getattr(state, surface)):
        return state, identity
    candidate_id = 'candidate-' + _digest([identity['state_sha256'], request.request_id, surface, component])
    candidate = replace(state, candidate_id=candidate_id, **{surface: component})
    candidate_identity = candidate.identity(protected)
    validate_transition(surface, identity, candidate_identity)
    return candidate, candidate_identity


def _select(domain: DomainAuthority, selector: Selector, request: KernelRequest, before: KernelState,
            candidate: KernelState, cost: Cost | None) -> tuple[bool, JsonValue]:
    left = _snapshot(domain.oracle(request, deepcopy(before)))
    right = _snapshot(domain.oracle(request, deepcopy(candidate)))
    comparison = Comparison(request, deepcopy(before), deepcopy(candidate), deepcopy(left), deepcopy(right),
                            deepcopy(cost))
    choice = selector(comparison)
    if type(choice) is not Selection or choice.decision not in ('retain', 'promote'):
        raise TypeError('MetaRSI selector must return Selection with retain or promote')
    return choice.decision == 'promote', {'incumbent': left, 'candidate': right, 'selector': _snapshot(choice.evidence)}


def run_domain_kernel(initial: KernelState, requests: Iterable[KernelRequest], *, domain: DomainAuthority,
                      policy: ProposalPolicy) -> KernelRun:
    """Run Observe→Diagnose→Propose→Validate→Execute→Select→Export in caller order.

    A rejected, unchanged, or explicitly skipped action retains and exports
    the incumbent surface without a competing candidate. A selected change
    triggers a new observation immediately, including at the final step.
    Domain exceptions propagate; the kernel never substitutes or retries work.
    """
    ordered = tuple(requests)
    if any(type(request) not in _REQUEST_TYPES or type(request.request_id) is not str or not request.request_id
           for request in ordered):
        raise TypeError('MetaRSI requests must be typed DataRequest, HarnessRequest or ModelRequest with an ID')
    if (not isinstance(domain.data, OperatorAuthority) or not isinstance(domain.harness, OperatorAuthority)
            or not isinstance(domain.model, ModelAuthority)):
        raise TypeError('MetaRSI requires Data and Harness authorities and a dedicated Model fit authority')
    protected = _snapshot(dict(domain.protected))
    state = KernelState(initial.candidate_id, _snapshot(initial.data), _snapshot(initial.harness), _snapshot(initial.model))
    identity = state.identity(protected)
    observations = [_observe(domain, state, identity)]
    steps = []
    for request in ordered:
        authority = getattr(domain, request.operator)
        observation = observations[-1]
        diagnosis = _snapshot(policy.diagnose(request, deepcopy(observation.signal)))
        proposal = policy.propose(request, deepcopy(observation.signal), deepcopy(diagnosis))
        context = ValidationContext(request, state, observation, diagnosis)
        action, proposal_record, decision, trusted_input = _validate(authority, context, proposal)

        candidate, candidate_identity, cost = state, identity, None
        fit_input_sha256 = None
        if type(decision) is Approved and action != 'no_op':
            if isinstance(authority, ModelAuthority):
                fit_input_sha256 = _digest(trusted_input)
                result = authority.fit(trusted_input, decision.input)
            else:
                result = authority.execute(deepcopy(state), decision.input)
            if type(result) is not ActionResult:
                raise TypeError('MetaRSI action must return an ActionResult')
            candidate, candidate_identity = _apply(state, identity, request, _snapshot(result.component), protected)
            cost = None if result.cost is None else dict(result.cost)

        rejected = type(decision) is Rejected
        execution = build_kernel_execution(
            request.operator, before=identity, after=candidate_identity, action=action,
            signal=_digest(observation.signal), diagnosis=_digest(diagnosis), proposal=proposal_record,
            validation='rejected' if rejected else 'accepted', allow_no_op=candidate_identity == identity,
        )
        promoted, selection = False, None
        if candidate_identity != identity:
            promoted, selection = _select(domain, authority.select, request, state, candidate, cost)
        selected, selected_identity = (candidate, candidate_identity) if promoted else (state, identity)
        evidence = complete_kernel_evidence(execution, selected=selected_identity, cost=cost, selection={
            'request_id': request.request_id, 'rejection': decision.reason if rejected else None,
            'fit_input_sha256': fit_input_sha256, 'comparison': selection,
        })
        steps.append(KernelStep(request, deepcopy(state), deepcopy(candidate), deepcopy(selected), evidence))
        state, identity = selected, selected_identity
        if promoted:
            observations.append(_observe(domain, state, identity))
    return KernelRun(state, tuple(observations), tuple(steps))
