"""Hash-only contracts for MetaRSI v2 §4.3, Figure 8 and Appendix C.

An artifact records where its evidence came from. A handoff records the live
state against which that evidence is consumed. Old Experience, for example,
can inform a newly extracted patch but does not make an old patch fresh.

Callers supply trusted state identities, evidence digests and a capability
epoch (the latest capability-altering step, or zero initially). Advance the
epoch on applied Harness/Model changes, even when restoring old contents;
Data changes do not advance it. These checks do not run the adapters, attest
to the contents behind hashes, schedule operators, or promote a result.
"""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import re
from types import MappingProxyType
from typing import Literal, TypedDict

from rsicomposer.runtime.storage.files import json_bytes

from .kernel_evidence import Operator, StateIdentity, validate_state_identity


TRANSITION_PROTOCOL = 'meta-rsi/transitions/v1'
ArtifactKind = Literal['experience', 'dataset', 'genome', 'checkpoint', 'training_report',
                       'harness_patch', 'materialized_dataset', 'learning_signal', 'model_learning_signal']


class TransitionValidationError(ValueError):
    """A fixed type or freshness check failed; code never echoes inputs."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class EdgeContract:
    adapter: str | None
    input_kind: ArtifactKind
    output_kind: ArtifactKind | None


EDGE_CONTRACTS = MappingProxyType({
    ('data', 'harness'): EdgeContract('experience_extraction', 'experience', 'harness_patch'),
    ('data', 'model'): EdgeContract('dataset_materialization', 'dataset', 'materialized_dataset'),
    ('harness', 'data'): EdgeContract('signal_recompilation', 'genome', 'learning_signal'),
    ('model', 'data'): EdgeContract('signal_recompilation', 'checkpoint', 'model_learning_signal'),
    ('model', 'harness'): EdgeContract('redundancy_reconciliation', 'training_report', 'harness_patch'),
    ('data', 'data'): EdgeContract(None, 'learning_signal', None),
    ('harness', 'harness'): EdgeContract(None, 'learning_signal', None),
})
FORBIDDEN_EDGES = frozenset({('harness', 'model'), ('model', 'model')})
ARTIFACT_EVIDENCE_FIELDS = MappingProxyType({
    'experience': frozenset({'experience_sha256'}),
    'dataset': frozenset({'cumulative_sha256', 'refresh_sha256'}),
    'genome': frozenset({'genome_sha256'}),
    'checkpoint': frozenset({'checkpoint_sha256'}),
    'training_report': frozenset({'training_report_sha256'}),
    'harness_patch': frozenset({'harness_patch_sha256'}),
    'materialized_dataset': frozenset({'masked_shards_sha256', 'recipe_sha256'}),
    'learning_signal': frozenset({'signal_sha256'}),
    'model_learning_signal': frozenset({'signal_sha256', 'learning_signatures_sha256'}),
})


class Artifact(TypedDict):
    protocol: str
    kind: ArtifactKind
    source_state: StateIdentity
    capability_epoch: int
    evidence_sha256: dict[str, str]
    derived_from: str | None
    artifact_sha256: str


class Transition(TypedDict):
    protocol: str
    source: Operator
    target: Operator
    adapter: str | None
    artifact: Artifact
    output: Artifact | None
    target_state_sha256: str
    capability_epoch: int
    transition_sha256: str


_OPERATORS = ('data', 'harness', 'model')
_SHA256 = re.compile(r'[0-9a-f]{64}', re.ASCII)
_ARTIFACT_FIELDS = frozenset(Artifact.__required_keys__)
_TRANSITION_FIELDS = frozenset(Transition.__required_keys__)


def _digest(value: object) -> str:
    return sha256(json_bytes(value)).hexdigest()


def _hash(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _fields(value: object, fields: frozenset[str]) -> bool:
    return type(value) is dict and value.keys() == fields


def _state(state: StateIdentity, epoch: int, *, role: str) -> None:
    if type(epoch) is not int or epoch < 0:
        raise TransitionValidationError('invalid_capability_epoch')
    try:
        validate_state_identity(state)
    except ValueError:
        raise TransitionValidationError(f'invalid_{role}_state') from None


def _artifact_fields(kind: ArtifactKind, source_state: StateIdentity, capability_epoch: int,
                     evidence_sha256: dict[str, str], derived_from: str | None) -> None:
    if type(kind) is not str or kind not in ARTIFACT_EVIDENCE_FIELDS:
        raise TransitionValidationError('invalid_artifact_kind')
    _state(source_state, capability_epoch, role='source')
    if (not _fields(evidence_sha256, ARTIFACT_EVIDENCE_FIELDS[kind])
            or not all(_hash(value) for value in evidence_sha256.values())):
        raise TransitionValidationError('invalid_artifact_evidence')
    if derived_from is not None and not _hash(derived_from):
        raise TransitionValidationError('invalid_artifact_lineage')
    if kind == 'dataset' and evidence_sha256['cumulative_sha256'] != source_state['component_sha256']['data']:
        raise TransitionValidationError('dataset_binding_mismatch')


def build_artifact(kind: ArtifactKind, *, source_state: StateIdentity, capability_epoch: int,
                   evidence_sha256: dict[str, str], derived_from: str | None = None) -> Artifact:
    """Bind a typed evidence reference to the state where it was produced.

    For a Data output this is the state containing the published cumulative
    data. The Lab's verified refresh proof contributes its cumulative hash
    and a digest of the proof itself, not the proof payload or records.
    For a derived input this is the evaluation or conversion context, not a
    state that hypothetically results from applying a proposed write. For a
    conversion output, pass the input reference's hash as 'derived_from'.
    """
    _artifact_fields(kind, source_state, capability_epoch, evidence_sha256, derived_from)
    body = {'protocol': TRANSITION_PROTOCOL, 'kind': kind, 'source_state': deepcopy(source_state),
            'capability_epoch': capability_epoch, 'evidence_sha256': evidence_sha256.copy(),
            'derived_from': derived_from}
    return {**body, 'artifact_sha256': _digest(body)}


def validate_artifact(artifact: Artifact) -> None:
    """Check a reference's type and internal binding, not its live freshness."""
    if not _fields(artifact, _ARTIFACT_FIELDS) or artifact['protocol'] != TRANSITION_PROTOCOL:
        raise TransitionValidationError('invalid_artifact')
    _artifact_fields(artifact['kind'], artifact['source_state'], artifact['capability_epoch'],
                     artifact['evidence_sha256'], artifact['derived_from'])
    if artifact['artifact_sha256'] != _digest({k: v for k, v in artifact.items() if k != 'artifact_sha256'}):
        raise TransitionValidationError('artifact_digest_mismatch')


def edge_contract(source: Operator, target: Operator) -> EdgeContract:
    """Return the fixed type signature; static legality is not live freshness."""
    if type(source) is not str or type(target) is not str or source not in _OPERATORS or target not in _OPERATORS:
        raise TransitionValidationError('invalid_operator')
    if (source, target) in FORBIDDEN_EDGES:
        raise TransitionValidationError('forbidden_edge')
    return EDGE_CONTRACTS[source, target]


def _fresh(artifact: Artifact, state: StateIdentity, epoch: int) -> bool:
    origin = artifact['source_state']
    return (artifact['capability_epoch'] == epoch and origin['protected_sha256'] == state['protected_sha256']
            and all(origin['component_sha256'][key] == state['component_sha256'][key]
                    for key in ('harness', 'model')))


def _check_input(artifact: Artifact, state: StateIdentity, epoch: int) -> None:
    kind, origin = artifact['kind'], artifact['source_state']
    if artifact['capability_epoch'] > epoch:
        raise TransitionValidationError('invalid_source_epoch')
    if kind == 'experience':
        return  # Verified experience is historical; its extracted patch must be current.
    if kind in ('dataset', 'learning_signal', 'model_learning_signal'):
        valid = _fresh(artifact, state, epoch) and (
            kind != 'dataset' or origin['component_sha256']['data'] == state['component_sha256']['data'])
    else:
        # These are provenance for the active genome/checkpoint, not compiled
        # signals. Recompilation/reconciliation uses the live target context.
        surface = 'harness' if kind == 'genome' else 'model'
        valid = origin['component_sha256'][surface] == state['component_sha256'][surface]
        if surface == 'model':
            valid = valid and origin['protected_sha256']['base_model'] == state['protected_sha256']['base_model']
    if not valid:
        raise TransitionValidationError('stale_input')


def build_transition(source: Operator, target: Operator, *, artifact: Artifact, target_state: StateIdentity,
                     capability_epoch: int, output: Artifact | None = None) -> Transition:
    """Typecheck a proposed handoff against the target's current state.

    Cross-edges require an independently bound output of their declared type.
    The self-loops reuse the current signal without an adapter. A Data-only
    change doesn't stale a signal or patch; a materialized dataset also binds
    the data surface. The receipt itself always records the exact target.
    """
    contract = edge_contract(source, target)
    _state(target_state, capability_epoch, role='target')
    validate_artifact(artifact)
    # A signal carrying weight-attributed signatures is still a learning signal.
    if artifact['kind'] != contract.input_kind and not (
            contract.input_kind == 'learning_signal' and artifact['kind'] == 'model_learning_signal'):
        raise TransitionValidationError('input_type_mismatch')
    _check_input(artifact, target_state, capability_epoch)
    if contract.output_kind is None:
        if output is not None:
            raise TransitionValidationError('unexpected_output')
    else:
        if output is None:
            raise TransitionValidationError('missing_output')
        validate_artifact(output)
        if output['kind'] != contract.output_kind:
            raise TransitionValidationError('output_type_mismatch')
        if output['derived_from'] != artifact['artifact_sha256']:
            raise TransitionValidationError('output_lineage_mismatch')
        if not _fresh(output, target_state, capability_epoch) or (
                output['kind'] == 'materialized_dataset' and
                output['source_state']['component_sha256']['data'] != target_state['component_sha256']['data']):
            raise TransitionValidationError('stale_output')
    body = {'protocol': TRANSITION_PROTOCOL, 'source': source, 'target': target, 'adapter': contract.adapter,
            'artifact': deepcopy(artifact), 'output': deepcopy(output), 'target_state_sha256': target_state['state_sha256'],
            'capability_epoch': capability_epoch}
    return {**body, 'transition_sha256': _digest(body)}


def validate_transition(transition: Transition, *, target_state: StateIdentity, capability_epoch: int) -> None:
    """Replay the type check and require the original exact target binding."""
    if not _fields(transition, _TRANSITION_FIELDS) or transition['protocol'] != TRANSITION_PROTOCOL:
        raise TransitionValidationError('invalid_transition')
    edge_contract(transition['source'], transition['target'])
    _state(target_state, capability_epoch, role='target')
    if (type(transition['capability_epoch']) is not int
            or transition['capability_epoch'] != capability_epoch
            or transition['target_state_sha256'] != target_state['state_sha256']):
        raise TransitionValidationError('stale_target_state')
    canonical = build_transition(transition['source'], transition['target'], artifact=transition['artifact'],
                                 target_state=target_state, capability_epoch=capability_epoch,
                                 output=transition['output'])
    if transition != canonical:
        raise TransitionValidationError('transition_binding_mismatch')
