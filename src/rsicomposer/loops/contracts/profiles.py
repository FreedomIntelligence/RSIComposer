"""Versioned Lab capabilities, metric identities and host protocol declarations.

A profile requires a metric declaration, never a metric name. Labs advertise
profile IDs with profiles() and declare primary_metric() for that instance.
The built-in catalog is module-owned; get_profile returns detached descriptions.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
import math
from typing import Literal

from .decision import OperationSpec


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label} must be nonempty text')


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: Literal['maximize', 'minimize']
    scorer: str

    def __post_init__(self):
        _text(self.name, 'Metric name')
        _text(self.scorer, 'Metric scorer')
        if self.direction not in ('maximize', 'minimize'):
            raise ValueError('Metric direction must be maximize or minimize')

    def value(self, data):
        """Read a receipt's measured primary value, without renaming metrics."""
        metrics = data.get('quality_metrics', {})
        value = metrics.get(self.name) if isinstance(metrics, Mapping) else None
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'Missing finite primary metric: quality_metrics.{self.name}')
        return value


@dataclass(frozen=True)
class SplitRole:
    inputs_visible: bool
    labels_visible: bool
    per_item_results: bool
    after_finish_only: bool

    def __post_init__(self):
        if any(type(value) is not bool for value in (
                self.inputs_visible, self.labels_visible, self.per_item_results, self.after_finish_only)):
            raise TypeError('Split role flags must be bools')
        if self.labels_visible and not self.inputs_visible:
            raise ValueError('Visible labels require visible inputs')
        if self.after_finish_only and (self.inputs_visible or self.labels_visible or self.per_item_results):
            raise ValueError('Sealed roles expose only aggregate results after finish')


TRAIN = SplitRole(True, True, True, False)
TRAIN_UNLABELED = SplitRole(True, False, True, False)
DEV = SplitRole(True, True, True, False)
SEALED = SplitRole(False, False, False, True)
SPLIT_ROLES = {'train': TRAIN, 'train_unlabeled': TRAIN_UNLABELED, 'dev': DEV, 'sealed': SEALED}


@dataclass(frozen=True)
class Invariant:
    """Declarative host rule. key names refer to request argument fields.

    baseline_before gates an operation on a successful baseline evaluation;
    min_candidates gates final sealed evaluation on distinct measured candidates;
    measured_selection and min_candidates require declared candidate identity fields;
    once_per gates an operation on a tuple of argument values. Failed attempts
    do not satisfy rules. Replaying the same call_id retains its earlier receipt.
    """
    kind: Literal['baseline_before', 'min_candidates', 'once_per', 'sealed_after_finish', 'measured_selection']
    operation: str = ''
    key: tuple[str, ...] = ()
    count: int = 0

    def __post_init__(self):
        if self.kind not in ('baseline_before', 'min_candidates', 'once_per', 'sealed_after_finish', 'measured_selection'):
            raise ValueError('Unknown profile invariant')
        if not isinstance(self.operation, str):
            raise TypeError('Invariant operation must be text')
        if (not isinstance(self.key, tuple) or any(not isinstance(k, str) or not k.strip() for k in self.key)
                or len(set(self.key)) != len(self.key)):
            raise ValueError('Invariant keys must be distinct argument names')
        if type(self.count) is not int or self.count < 0:
            raise ValueError('Invariant count must be a nonnegative integer')
        if self.kind == 'baseline_before':
            _text(self.operation, 'Gated operation')
        elif self.kind == 'min_candidates' and self.count < 1:
            raise ValueError('min_candidates requires a positive count')
        elif self.kind == 'once_per' and not self.key:
            raise ValueError('once_per requires argument keys')

    @classmethod
    def baseline_before(cls, operation):
        return cls('baseline_before', operation=operation)

    @classmethod
    def min_candidates(cls, count):
        return cls('min_candidates', count=count)

    @classmethod
    def once_per(cls, *key, operation=''):
        return cls('once_per', operation=operation, key=key)

    @classmethod
    def measured_selection(cls, operation='select'):
        return cls('measured_selection', operation=operation)

    @classmethod
    def sealed_after_finish(cls):
        return cls('sealed_after_finish')


@dataclass(frozen=True)
class LabProfile:
    name: str
    version: int
    operations: tuple[OperationSpec, ...]
    optional_operations: tuple[OperationSpec, ...] = ()
    receipt_fields: Mapping[str, dict] = field(default_factory=dict)
    primary_metric: bool = True
    split_roles: Mapping[str, SplitRole] = field(default_factory=dict)
    invariants: tuple[Invariant, ...] = ()
    evaluation_operations: tuple[str, ...] = ()
    candidate_identity_fields: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self):
        _text(self.name, 'Profile name')
        if '/' in self.name or type(self.version) is not int or self.version < 1:
            raise ValueError('Profile needs a simple name and positive version')
        if type(self.primary_metric) is not bool:
            raise TypeError('primary_metric indicates whether a declaration is required')
        specs = self.operations + self.optional_operations
        if (not isinstance(self.operations, tuple) or not isinstance(self.optional_operations, tuple)
                or any(not isinstance(spec, OperationSpec) for spec in specs)
                or len({spec.name for spec in specs}) != len(specs)):
            raise ValueError('Profile operations must be distinct OperationSpec tuples')
        if not isinstance(self.invariants, tuple) or any(not isinstance(rule, Invariant) for rule in self.invariants):
            raise TypeError('Profile invariants must be an Invariant tuple')
        names = {spec.name for spec in specs}
        if (not isinstance(self.evaluation_operations, tuple)
                or any(not isinstance(name, str) or name not in names for name in self.evaluation_operations)
                or len(set(self.evaluation_operations)) != len(self.evaluation_operations)):
            raise ValueError('Profile evaluations must name distinct profile operations')
        if (not isinstance(self.split_roles, Mapping)
                or any(not isinstance(name, str) or not name or not isinstance(role, SplitRole)
                       for name, role in self.split_roles.items())):
            raise TypeError('Profile split roles must map names to SplitRole')
        if (not isinstance(self.receipt_fields, Mapping)
                or any(name not in {spec.name for spec in specs} or not isinstance(fields, dict)
                       for name, fields in self.receipt_fields.items())):
            raise ValueError('Receipt fields must describe profile operations')
        object.__setattr__(self, 'receipt_fields', deepcopy(dict(self.receipt_fields)))
        object.__setattr__(self, 'split_roles', dict(self.split_roles))
        object.__setattr__(self, 'operations', deepcopy(self.operations))
        object.__setattr__(self, 'optional_operations', deepcopy(self.optional_operations))
        object.__setattr__(self, 'candidate_identity_fields', candidate_fields(self.candidate_identity_fields, names))

    @property
    def identity(self):
        return f'{self.name}/v{self.version}'

    def validate_catalog(self, catalog):
        """Check structural capabilities, including advertised parameter types.

        A Lab must accept the profile's fields without requiring extra arguments.
        Optional operations with no field declarations only advertise capability.
        Domain values and artifact lineage remain the Lab's responsibility.
        """
        specs = {spec.name: spec for spec in catalog}
        for expected in self.operations + self.optional_operations:
            actual = specs.get(expected.name)
            if actual is None:
                if expected in self.operations:
                    raise ValueError(f'{self.identity}: missing operation {expected.name}')
                continue
            schema = actual.input_schema
            if schema.get('type') != 'object':
                raise ValueError(f'{self.identity}: {expected.name} requires an object schema')
            fields = expected.input_schema.get('properties', {})
            if expected in self.operations or fields:
                extra = set(schema.get('required', ())) - set(expected.input_schema.get('required', ()))
                if extra:
                    raise ValueError(f'{self.identity}: {expected.name} requires extra parameters: ' + ', '.join(sorted(extra)))
            for name, requirement in fields.items():
                provided = schema.get('properties', {}).get(name)
                if provided is None:
                    raise ValueError(f'{self.identity}: {expected.name} missing parameter {name}')
                wanted = requirement.get('type')
                accepted = provided.get('type')
                accepted = accepted if isinstance(accepted, list) else [accepted]
                wanted = wanted if isinstance(wanted, list) else [wanted]
                if any(kind not in accepted for kind in wanted):
                    raise ValueError(f'{self.identity}: {expected.name}.{name} must accept type {wanted}')


def candidate_fields(declaration, operations):
    """Detach and validate ordered, dot-separated object paths before dispatch."""
    if not isinstance(declaration, Mapping):
        raise TypeError('candidate_identity_fields must map operation names to field tuples')
    result = {}
    for name, fields in declaration.items():
        if name not in operations:
            raise ValueError(f'candidate_identity_fields names an undeclared operation: {name}')
        if (not isinstance(fields, tuple) or not fields or any(not isinstance(path, str)
                or any(not part.strip() for part in path.split('.')) for path in fields)
                or len(set(fields)) != len(fields)):
            raise ValueError(f'{name} candidate identity requires distinct argument paths as a nonempty tuple')
        result[name] = fields
    return result


def _op(name, **fields):
    return OperationSpec(name, f'{name} profile contract', {
        'type': 'object', 'properties': {key: {'type': kind} for key, kind in fields.items()},
        'required': list(fields)})


WEIGHT_TRAINING = LabProfile('weight-training', 1, (
    _op('generate', model='string', dataset='string', samples_per_prompt='integer',
        temperature='number', condition='string'),
    _op('train', parent='string', dataset='string'),
    _op('evaluate', model='string', split='string'),
), tuple(_op(name) for name in ('filter', 'reward', 'adapt', 'select')), {
    'generate': {'dataset': 'string', 'count': 'integer'},
    'train': {'model': 'string'}, 'evaluate': {'quality_metrics': 'object'},
}, split_roles=SPLIT_ROLES, invariants=(Invariant.sealed_after_finish(),), evaluation_operations=('evaluate',),
    candidate_identity_fields={'evaluate': ('model',)})
ROUND_WEIGHT_TRAINING = LabProfile('round-weight-training', 1, (
    _op('generate', round='integer', model='string', dataset='string', samples_per_prompt='integer',
        temperature='number', condition='string'),
    _op('train', round='integer', parent='string', dataset='string'),
    _op('evaluate', round='integer', model='string', dataset='string', split='string'),
), WEIGHT_TRAINING.optional_operations, WEIGHT_TRAINING.receipt_fields,
    split_roles=WEIGHT_TRAINING.split_roles, invariants=WEIGHT_TRAINING.invariants,
    evaluation_operations=WEIGHT_TRAINING.evaluation_operations,
    candidate_identity_fields=WEIGHT_TRAINING.candidate_identity_fields)
REFERENCE_WEIGHT_TRAINING = replace(ROUND_WEIGHT_TRAINING, name='reference-weight-training', operations=(
    ROUND_WEIGHT_TRAINING.operations[0],
    _op('train', round='integer', parent='string', dataset='string', reference='string'),
    ROUND_WEIGHT_TRAINING.operations[2],
))
ARTIFACT_SEARCH = LabProfile('artifact-search', 1, (
    _op('register', candidate='object'),
    _op('evaluate', candidate_id='string', split='string', indices=['array', 'null']),
), tuple(_op(name) for name in ('task_spec', 'exemplars')), {
    'register': {'candidate_id': 'string'},
    'evaluate': {'valid': 'boolean', 'scored': 'boolean', 'scores': 'array',
                 'trace_refs': 'array', 'feedback': 'array', 'quality_metrics': 'object'},
}, split_roles=SPLIT_ROLES, invariants=(Invariant.sealed_after_finish(),), evaluation_operations=('evaluate',),
    candidate_identity_fields={'evaluate': ('candidate_id',)})
PROGRAM_EXECUTION = LabProfile('program-execution', 1, (
    _op('execute', program='string', split='string'),
), (_op('preview'),), {'execute': {'term_out': 'string', 'exc_type': ['string', 'null'],
    'exec_time': 'number', 'metrics': 'object'}}, primary_metric=False,
    split_roles=SPLIT_ROLES, invariants=(Invariant.sealed_after_finish(),), evaluation_operations=('execute',),
    candidate_identity_fields={'execute': ('program',)})
CODING_TASK = LabProfile('coding-task', 1, (
    _op('evaluate', artifact='object', split='string', trials='integer', task_ids='array'),
), receipt_fields={'evaluate': {'quality_metrics': 'object'}}, split_roles=SPLIT_ROLES,
    invariants=(Invariant.sealed_after_finish(),), evaluation_operations=('evaluate',),
    candidate_identity_fields={'evaluate': ('artifact.sha256',)})

_BUILTINS = {profile.identity: profile for profile in (
    WEIGHT_TRAINING, ROUND_WEIGHT_TRAINING, REFERENCE_WEIGHT_TRAINING,
    ARTIFACT_SEARCH, PROGRAM_EXECUTION, CODING_TASK)}


def get_profile(identity):
    """Resolve an exact name/vN; reject unknown versions rather than falling back."""
    try:
        return deepcopy(_BUILTINS[identity])
    except (KeyError, TypeError):
        raise ValueError(f'Unknown Lab profile: {identity}') from None
