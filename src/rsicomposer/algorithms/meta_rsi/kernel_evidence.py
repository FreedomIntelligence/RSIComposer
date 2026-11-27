"""Replayable evidence for the public MetaRSI §4.1 loop kernel.

This is an evidence boundary, not an executor or a promotion rule. Callers
bind trusted component and protected inputs, run the operator's own proposal
validator, and supply the actual result of deterministic selection. The wire
format contains technical IDs, digests and numeric cost, not model messages,
raw trajectories, proposal text, or evaluator answers.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import math
import re
from types import MappingProxyType
from typing import Literal, TypedDict

from rsicomposer.runtime.storage.files import json_bytes


KERNEL_PROTOCOL = 'meta-rsi/kernel-evidence/v1'
KERNEL_STAGES = ('observe', 'diagnose', 'propose', 'validate', 'execute', 'select', 'export')
EXECUTION_STAGES = KERNEL_STAGES[:5]
WRITE_SURFACES = ('data', 'harness', 'model')
PROTECTED_SURFACES = ('base_model', 'signal_compiler', 'verifier', 'evaluator', 'promotion_rule',
                      'sealed_measurement', 'release_rule', 'ledger', 'runtime_permissions')

Operator = Literal['data', 'harness', 'model']
WriteAction = Literal['synthesize_tasks', 'patch_harness', 'fit_adapter']
Action = Literal['synthesize_tasks', 'patch_harness', 'fit_adapter', 'no_op']
ArtifactKind = Literal['verified_dataset', 'harness_patch', 'adapter_checkpoint']
Validation = Literal['accepted', 'rejected']

_SHA256 = re.compile(r'[0-9a-f]{64}', re.ASCII)
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,127}', re.ASCII)
_COST_KEY = re.compile(r'[a-z][a-z0-9_]{0,63}', re.ASCII)
_COST_COUNTERS = {'model_calls', 'optimizer_steps', 'rollout_samples', 'train_samples', 'dev_items_scored'}


class StateIdentity(TypedDict):
    candidate_id: str
    component_sha256: dict[Operator, str]
    protected_sha256: dict[str, str]
    state_sha256: str


@dataclass(frozen=True, slots=True)
class OperatorCard:
    """The fixed interface around a mutable proposal policy."""

    operator: Operator
    action: WriteAction
    write_surface: Operator
    read_surfaces: tuple[Operator, ...]
    protected_surfaces: tuple[str, ...]
    artifact_kind: ArtifactKind

    def wire(self) -> dict:
        return {'operator': self.operator, 'action': self.action, 'write_surface': self.write_surface,
                'read_surfaces': list(self.read_surfaces), 'protected_surfaces': list(self.protected_surfaces),
                'artifact_kind': self.artifact_kind}


def _card(operator: Operator, action: WriteAction, artifact_kind: ArtifactKind) -> OperatorCard:
    peers = tuple(name for name in WRITE_SURFACES if name != operator)
    return OperatorCard(operator, action, operator, WRITE_SURFACES, peers + PROTECTED_SURFACES, artifact_kind)


OPERATOR_CARDS: Mapping[Operator, OperatorCard] = MappingProxyType({
    'data': _card('data', 'synthesize_tasks', 'verified_dataset'),
    'harness': _card('harness', 'patch_harness', 'harness_patch'),
    'model': _card('model', 'fit_adapter', 'adapter_checkpoint'),
})


def operator_card(operator: Operator) -> OperatorCard:
    if type(operator) is not str or operator not in OPERATOR_CARDS:
        raise ValueError('Unknown MetaRSI kernel operator')
    return OPERATOR_CARDS[operator]


def validate_action(operator: Operator, action: Action, *, allow_no_op: bool = False) -> None:
    """Check the fixed action vocabulary before a domain-specific validator runs."""
    card = operator_card(operator)
    if type(allow_no_op) is not bool or type(action) is not str or not (
            action == card.action or (allow_no_op and action == 'no_op')):
        raise ValueError('Invalid MetaRSI kernel action for operator')


def _digest(value: object) -> str:
    try:
        return sha256(json_bytes(value)).hexdigest()
    except (TypeError, ValueError, OverflowError, UnicodeError, RecursionError):
        raise ValueError('MetaRSI kernel evidence needs canonical JSON input') from None


def _hash(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _id(value: object) -> bool:
    return type(value) is str and _ID.fullmatch(value) is not None


def _fields(value: object, fields) -> bool:
    return isinstance(value, Mapping) and value.keys() == set(fields)


def _reference(value: object) -> str | None:
    # A producer may pass its precomputed SHA-256 instead of resupplying data.
    return value if value is None or _hash(value) else _digest(value)


def build_state_identity(candidate_id: str, components: Mapping[Operator, object], *,
                         protected: Mapping[str, object]) -> StateIdentity:
    """Freeze exact state/protected bindings, including explicit disabled surfaces.

    Supply the real protected inputs or existing trusted fingerprints; pass
    None explicitly for a surface that is absent in the run. Values are
    fingerprinted even when they are already hashes, so the shape is uniform.
    """
    if not _id(candidate_id) or not _fields(components, WRITE_SURFACES) or not _fields(protected, PROTECTED_SURFACES):
        raise ValueError('Invalid MetaRSI kernel state surfaces or candidate ID')
    identity = {'candidate_id': candidate_id,
                'component_sha256': {key: _digest(components[key]) for key in WRITE_SURFACES},
                'protected_sha256': {key: _digest(protected[key]) for key in PROTECTED_SURFACES}}
    return {**identity, 'state_sha256': _digest(identity)}


def validate_state_identity(state: StateIdentity) -> None:
    if (type(state) is not dict or not _fields(state, ('candidate_id', 'component_sha256', 'protected_sha256',
                                                       'state_sha256'))
            or not _id(state['candidate_id']) or type(state['component_sha256']) is not dict
            or not _fields(state['component_sha256'], WRITE_SURFACES) or type(state['protected_sha256']) is not dict
            or not _fields(state['protected_sha256'], PROTECTED_SURFACES)
            or not all(_hash(v) for group in ('component_sha256', 'protected_sha256') for v in state[group].values())
            or not _hash(state['state_sha256'])
            or state['state_sha256'] != _digest({key: state[key] for key in state if key != 'state_sha256'})):
        raise ValueError('Invalid MetaRSI kernel state identity')


def validate_transition(operator: Operator, before: StateIdentity, after: StateIdentity, *,
                        allow_no_op: bool = False) -> tuple[Operator, ...]:
    """Require exactly one changed surface, or an explicitly allowed no-op."""
    card = operator_card(operator)
    validate_state_identity(before)
    validate_state_identity(after)
    if type(allow_no_op) is not bool:
        raise ValueError('Invalid MetaRSI kernel no-op policy')
    if before['protected_sha256'] != after['protected_sha256']:
        raise ValueError('MetaRSI kernel transition changed a protected surface')
    changes = tuple(key for key in WRITE_SURFACES if before['component_sha256'][key] != after['component_sha256'][key])
    if changes != (card.write_surface,) and not (allow_no_op and not changes):
        raise ValueError('MetaRSI kernel transition must change exactly the declared write surface')
    if bool(changes) != (before['candidate_id'] != after['candidate_id']):
        raise ValueError('MetaRSI kernel candidate identity does not match the state change')
    return changes


def _stage(name: str, **fields) -> dict:
    authority = 'proposal_policy' if name in ('diagnose', 'propose') else 'fixed_code'
    return {'stage': name, 'authority': authority, **fields}


def _envelope(card: OperatorCard, stages: list[dict]) -> dict:
    body = {'protocol': KERNEL_PROTOCOL, 'operator': card.operator, 'card': card.wire(), 'stages': stages}
    return {**body, 'evidence_sha256': _digest(body)}


def build_kernel_execution(operator: Operator, *, before: StateIdentity, after: StateIdentity | None = None,
                           action: Action | None = None, signal: object = None, diagnosis: object = None,
                           proposal: object = None, validation: Validation = 'accepted',
                           allow_no_op: bool = False) -> dict:
    """Build Observe through Execute without anticipating selection or release.

    Inputs to the mutable arc are recorded only by digest; a SHA-256 supplied
    by the caller is preserved. Rejection or an explicit no-op action can
    never write. Missing after-state means an unchanged state, not success.
    """
    card = operator_card(operator)
    action = card.action if action is None else action
    validate_action(operator, action, allow_no_op=allow_no_op)
    if type(validation) is not str or validation not in ('accepted', 'rejected'):
        raise ValueError('Invalid MetaRSI kernel validation decision')
    after = before if after is None else after
    changes = validate_transition(operator, before, after, allow_no_op=allow_no_op)
    if changes and (validation == 'rejected' or action == 'no_op'):
        raise ValueError('A rejected or no-op MetaRSI action cannot write state')
    signal_ref, diagnosis_ref, proposal_ref = (_reference(v) for v in (signal, diagnosis, proposal))
    state = before['state_sha256']
    stages = [
        _stage('observe', state=deepcopy(before), signal_sha256=signal_ref),
        _stage('diagnose', state_sha256=state, signal_sha256=signal_ref, diagnosis_sha256=diagnosis_ref),
        _stage('propose', state_sha256=state, diagnosis_sha256=diagnosis_ref, action=action,
               proposal_sha256=proposal_ref),
        _stage('validate', state_sha256=state, action=action, proposal_sha256=proposal_ref,
               decision=validation, write_surface=card.write_surface, allow_no_op=allow_no_op),
        _stage('execute', before_sha256=state, state=deepcopy(after), changed_surfaces=list(changes),
               outcome=('changed' if changes else 'skipped' if validation == 'rejected' or action == 'no_op'
                        else 'no_op')),
    ]
    return _envelope(card, stages)


def _cost(cost: Mapping[str, int | float] | None) -> dict[str, int | float] | None:
    # None means unreported; do not silently manufacture a zero measurement.
    if cost is None:
        return None
    if (not isinstance(cost, Mapping) or len(cost) > 16
            or any(type(k) is not str or not _COST_KEY.fullmatch(k) or type(v) not in (int, float)
                   or (type(v) is float and not math.isfinite(v)) or not 0 <= v <= 2**63 - 1
                   or (k in _COST_COUNTERS and type(v) is not int) for k, v in cost.items())):
        raise ValueError('Invalid MetaRSI kernel cost counters')
    return {key: cost[key] for key in sorted(cost)}


def _complete(execution: dict, *, selected: StateIdentity | None, selection: object,
              artifact: str | None, cost: Mapping[str, int | float] | None) -> dict:
    card = operator_card(execution['operator'])
    stages = deepcopy(execution['stages'])
    before, after = stages[0]['state'], stages[4]['state']
    selected = before if selected is None else selected
    validate_state_identity(selected)
    if selected not in (before, after):
        raise ValueError('MetaRSI kernel selected a state outside its bound comparison')
    promoted = selected != before
    stages.append(_stage('select', before_sha256=before['state_sha256'],
                         candidate_sha256=after['state_sha256'], selected_sha256=selected['state_sha256'],
                         candidate_id=selected['candidate_id'], decision='promoted' if promoted else 'retained',
                         selection_sha256=_reference(selection)))
    if artifact is not None and not _hash(artifact):
        raise ValueError('MetaRSI kernel export needs a SHA-256 content reference')
    content = selected['component_sha256'][card.write_surface] if artifact is None else artifact
    stages.append(_stage('export', state_sha256=selected['state_sha256'], artifact={
        'kind': card.artifact_kind, 'content_sha256': content,
        'parents': [before['candidate_id']] if promoted else [], 'supporting_evidence_sha256': _digest(stages),
        'cost': _cost(cost),
    }))
    return _envelope(card, stages)


def complete_kernel_evidence(execution: dict, *, selected: StateIdentity | None = None,
                             selection: object = None, artifact: str | None = None,
                             cost: Mapping[str, int | float] | None = None) -> dict:
    """Append the observed Select result and typed Export to a valid prefix.

    Selection defaults to retaining the incumbent. The optional artifact
    digest refers to separately materialized content; otherwise the exported
    component fingerprint is used. Neither implies Host-level release.
    """
    validate_kernel_execution(execution)
    return _complete(execution, selected=selected, selection=selection, artifact=artifact, cost=cost)


def build_kernel_evidence(operator: Operator, *, before: StateIdentity, after: StateIdentity | None = None,
                          selected: StateIdentity | None = None, action: Action | None = None,
                          signal: object = None, diagnosis: object = None, proposal: object = None,
                          validation: Validation = 'accepted', allow_no_op: bool = False,
                          selection: object = None, artifact: str | None = None,
                          cost: Mapping[str, int | float] | None = None) -> dict:
    execution = build_kernel_execution(operator, before=before, after=after, action=action, signal=signal,
                                       diagnosis=diagnosis, proposal=proposal, validation=validation,
                                       allow_no_op=allow_no_op)
    return _complete(execution, selected=selected, selection=selection, artifact=artifact, cost=cost)


def _wire_stages(evidence: object, vocabulary: tuple[str, ...]) -> list[dict]:
    if (type(evidence) is not dict or set(evidence) != {'protocol', 'operator', 'card', 'stages', 'evidence_sha256'}
            or type(evidence['stages']) is not list or len(evidence['stages']) != len(vocabulary)
            or any(type(stage) is not dict or stage.get('stage') != name
                   for stage, name in zip(evidence['stages'], vocabulary, strict=True))):
        raise ValueError('Invalid MetaRSI kernel stage sequence or envelope')
    return evidence['stages']


def _wire_ref(value: object) -> str | None:
    if value is not None and not _hash(value):
        raise ValueError('Invalid MetaRSI kernel stage evidence reference')
    return value


def _rebuild_execution(operator: Operator, stages: list[dict]) -> dict:
    return build_kernel_execution(operator, before=stages[0].get('state'), after=stages[4].get('state'),
                                  action=stages[2].get('action'), signal=_wire_ref(stages[0].get('signal_sha256')),
                                  diagnosis=_wire_ref(stages[1].get('diagnosis_sha256')),
                                  proposal=_wire_ref(stages[2].get('proposal_sha256')),
                                  validation=stages[3].get('decision'), allow_no_op=stages[3].get('allow_no_op'))


def _check_wire(evidence: dict, rebuilt: dict) -> None:
    # Canonical wire equality also rejects bool/int aliases and extra fields.
    if _digest(evidence) != _digest(rebuilt):
        raise ValueError('MetaRSI kernel evidence contains inconsistent or altered stage data')


def validate_kernel_execution(execution: dict) -> None:
    """Fail closed on an invalid or tampered execution prefix."""
    stages = _wire_stages(execution, EXECUTION_STAGES)
    _check_wire(execution, _rebuild_execution(execution['operator'], stages))


def validate_kernel_evidence(evidence: dict) -> None:
    """Fail closed on invalid states, scope, selection, or artifact lineage.

    This checks internal consistency, not provenance. A caller with external
    state or evidence must compare those trusted digests to the bound inputs.
    """
    stages = _wire_stages(evidence, KERNEL_STAGES)
    execution = _rebuild_execution(evidence['operator'], stages)
    before, after = execution['stages'][0]['state'], execution['stages'][4]['state']
    selected = next((state for state in (before, after) if state['state_sha256'] == stages[5].get('selected_sha256')),
                    None)
    artifact = stages[6].get('artifact')
    if selected is None or type(artifact) is not dict:
        raise ValueError('Invalid MetaRSI kernel selected state or exported artifact')
    rebuilt = _complete(execution, selected=selected, selection=_wire_ref(stages[5].get('selection_sha256')),
                        artifact=_wire_ref(artifact.get('content_sha256')), cost=artifact.get('cost'))
    _check_wire(evidence, rebuilt)
