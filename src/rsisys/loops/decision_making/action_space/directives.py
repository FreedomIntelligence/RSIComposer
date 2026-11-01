"""System instruction descriptions and payload parsing; no execution or I/O."""

from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType

from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.requests import SFTSpec, RLSpec, EvaluationSpec, PromptSpec


@dataclass(frozen=True)
class DirectiveSpec:
    mode: str
    description: str
    input_schema: dict[str, object]
    accepts_input_refs: bool = True

    def __post_init__(self):
        if type(self.accepts_input_refs) is not bool:
            raise TypeError('accepts_input_refs must be a bool')
        if not isinstance(self.mode, str) or not self.mode.strip():
            raise ValueError('A nonempty directive mode is required')
        if not isinstance(self.description, str) or not isinstance(self.input_schema, dict):
            raise TypeError('Directive descriptions require text and an input schema')


@dataclass(frozen=True)
class DirectiveDefinition:
    spec: DirectiveSpec
    parser: Callable[[dict[str, object], tuple[Candidate, ...]], object]

    def __post_init__(self):
        if not isinstance(self.spec, DirectiveSpec) or not callable(self.parser):
            raise TypeError('A directive definition requires a spec and payload parser')


class DirectiveDefinitions:
    """An explicitly supplied, per-assembly catalog. No global registration."""

    def __init__(self, definitions: Iterable[DirectiveDefinition] | None = None):
        self.is_default = definitions is None
        definitions = builtin_directives() if definitions is None else definitions
        catalog = {}
        for definition in definitions:
            if not isinstance(definition, DirectiveDefinition):
                raise TypeError('Expected a DirectiveDefinition')
            mode = definition.spec.mode
            if mode in catalog:
                raise ValueError(f'Duplicate directive definition: {mode}')
            # Copy mutable schema content; caller-owned metadata cannot alter the catalog.
            catalog[mode] = DirectiveDefinition(deepcopy(definition.spec), definition.parser)
        self._definitions = MappingProxyType(catalog)

    def modes(self) -> tuple[str, ...]:
        return tuple(self._definitions)

    def describe(self, modes: tuple[str, ...] | None = None) -> tuple[DirectiveSpec, ...]:
        selected = self.modes() if modes is None else modes
        return tuple(deepcopy(self._get(mode).spec) for mode in selected)

    def _get(self, mode: str) -> DirectiveDefinition:
        if not isinstance(mode, str) or mode not in self._definitions:
            raise ValueError(f'Unsupported directive mode: {mode}')
        return self._definitions[mode]

    def parse(self, mode: str, arguments: dict[str, object], *,
              candidates: tuple[Candidate, ...] = ()) -> object:
        if not isinstance(arguments, dict):
            raise TypeError('Directive arguments must be an object')
        return self._get(mode).parser(deepcopy(arguments), deepcopy(candidates))


def _arguments(arguments, required=(), optional=()):
    if not set(required) <= arguments.keys() or arguments.keys() - set(required) - set(optional):
        missing = sorted(set(required) - arguments.keys())
        unexpected = sorted(arguments.keys() - set(required) - set(optional))
        raise ValueError(f'Directive arguments do not match the declared fields: missing={missing}, unexpected={unexpected}')


def _execute(arguments, candidates):
    _arguments(arguments)
    return None


def _prompt(arguments, candidates):
    _arguments(arguments, ('instructions',))
    return PromptSpec(arguments['instructions'])


def _sft(arguments, candidates):
    _arguments(arguments, ('recipe_ref',), ('apply_result',))
    return SFTSpec(arguments['recipe_ref'], apply_result=arguments.get('apply_result', False))


def _rl(arguments, candidates):
    _arguments(arguments, ('recipe_ref',), ('updates',))
    return RLSpec(arguments['recipe_ref'], arguments.get('updates', 1))


def _evaluate(arguments, candidates):
    _arguments(arguments, ('candidate_id', 'protocol_ref'))
    available = {c.candidate_id: c for c in candidates}
    ident = arguments['candidate_id']
    if not isinstance(ident, str) or ident not in available:
        raise ValueError('Evaluation must reference an available candidate in decision context')
    return EvaluationSpec(available[ident], arguments['protocol_ref'])


def builtin_directives() -> tuple[DirectiveDefinition, ...]:
    def definition(mode, description, properties, required, parser):
        return DirectiveDefinition(DirectiveSpec(mode, description, {
            'type': 'object', 'properties': properties, 'required': list(required),
            'additionalProperties': False,
        }), parser)

    return (
        definition('execute', 'Run a bounded Actor episode', {}, (), _execute),
        definition('sft', 'Train using input artifacts; optionally apply the result', {
            'recipe_ref': {'type': 'string', 'minLength': 1, 'pattern': r'\S'},
            'apply_result': {'type': 'boolean', 'default': False},
        }, ('recipe_ref',), _sft),
        definition('rl', 'Alternate environment rollout and model training', {
            'recipe_ref': {'type': 'string', 'minLength': 1, 'pattern': r'\S'},
            'updates': {'type': 'integer', 'minimum': 1, 'default': 1},
        }, ('recipe_ref',), _rl),
        definition('evaluate', 'Evaluate an available candidate from decision context on selected inputs', {
            'candidate_id': {'type': 'string', 'minLength': 1, 'pattern': r'\S'},
            'protocol_ref': {'type': 'string', 'minLength': 1, 'pattern': r'\S'},
        }, ('candidate_id', 'protocol_ref'), _evaluate),
        definition('configure_prompt', 'Replace Actor instructions; an empty string clears them', {
            'instructions': {'type': 'string'},
        }, ('instructions',), _prompt),
    )


DEFAULT_DIRECTIVE_MODES = tuple(d.spec.mode for d in builtin_directives())


def training_directive(capabilities=None) -> DirectiveDefinition:
    """Explicitly enable backend-owned training; not advertised by default."""
    from rsisys.loops.contracts.requests import TrainSpec

    from rsisys.loops.contracts.training import TrainingOrigin, ResumeMode
    from rsisys.loops.contracts.model_service import ModelRef
    from rsisys.loops.contracts.artifacts import ArtifactRef
    from rsisys.loops.contracts.backend import validate_training_capabilities

    def parse(arguments, candidates):
        _arguments(arguments, ('recipe_ref', 'method'), ('apply_result', 'origin'))
        origin = arguments.get('origin')
        if origin is not None:
            if not isinstance(origin, dict):
                raise TypeError('origin must be an object')
            _arguments(origin, ('parent_system', 'initial_model'), ('resume_mode', 'checkpoint'))
            initial = origin['initial_model']
            if not isinstance(initial, dict):
                raise TypeError('initial_model must be an object')
            _arguments(initial, ('backend_id', 'model_id', 'revision'))
            checkpoint = origin.get('checkpoint')
            if checkpoint is not None:
                if not isinstance(checkpoint, dict):
                    raise TypeError('checkpoint must be an artifact reference object')
                checkpoint = ArtifactRef(**checkpoint)
            origin = TrainingOrigin(origin['parent_system'], ModelRef(**initial),
                ResumeMode(origin.get('resume_mode', 'restart')), checkpoint)
        if capabilities is not None:
            validate_training_capabilities(capabilities, method=arguments['method'], origin=origin)
        return TrainSpec(arguments['recipe_ref'], arguments['method'],
                         apply_result=arguments.get('apply_result', False), origin=origin)

    method_schema = {'type': 'string', 'minLength': 1, 'pattern': r'\S'}
    modes = [mode.value for mode in ResumeMode]
    if capabilities is not None:
        method_schema['enum'] = list(capabilities.methods)
        modes = [mode.value for mode in capabilities.resume_modes]

    # Encode initialization semantics in the generation contract as well as in
    # TrainingOrigin. Advertising a checkpoint object for restart lets a native
    # schema-constrained provider generate proposals the parser must reject.
    origin_branches = []
    for mode in modes:
        checkpoint_schema = {
            'type': 'object', 'properties': {
                key: {'type': ['string', 'null'] if key in ('version', 'digest') else 'string',
                      'pattern': r'\S'}
                for key in ('artifact_id', 'kind', 'uri', 'version', 'digest')
            }, 'required': ['artifact_id', 'kind', 'uri'], 'additionalProperties': False,
            'description': 'Weights/training state reference, not training data.',
        }
        required = ['parent_system', 'initial_model']
        resume_schema = {'type': 'string', 'enum': [mode]}
        if mode == 'restart':
            checkpoint_schema = {'type': 'null', 'description': 'Restart uses initial_model; omit checkpoint or use null.'}
            resume_schema['default'] = 'restart'
        else:
            required.append('resume_mode')
            if mode == 'full_state':
                required.append('checkpoint')
            else:
                checkpoint_schema['type'] = ['object', 'null']
        origin_branches.append({
            'type': ['object', 'null'] if mode == 'restart' else 'object',
            'properties': {
                'parent_system': {'type': 'string', 'minLength': 1},
                'initial_model': {'type': 'object', 'properties': {
                    key: {'type': 'string', 'minLength': 1} for key in ('backend_id', 'model_id', 'revision')
                }, 'required': ['backend_id', 'model_id', 'revision'], 'additionalProperties': False},
                'resume_mode': resume_schema, 'checkpoint': checkpoint_schema,
            }, 'required': required, 'additionalProperties': False,
        })
    origin_schema = origin_branches[0] if len(origin_branches) == 1 else {'oneOf': origin_branches}
    schema = {
        'type': 'object', 'properties': {
            'recipe_ref': {'type': 'string', 'minLength': 1, 'pattern': r'\S'}, 'method': method_schema,
            'apply_result': {'type': 'boolean', 'default': False},
            'origin': origin_schema,
        }, 'required': ['recipe_ref', 'method'], 'additionalProperties': False,
    }
    if 'restart' not in modes:
        schema['required'].append('origin')
    if not origin_branches:
        # A backend with no initialization modes cannot accept any training.
        schema['properties']['origin'] = {'not': {}}
    return DirectiveDefinition(DirectiveSpec('train',
        'Delegate training; top-level input_ids selects data. origin initializes weights; '
        'omit it for the configured restart initializer.', schema), parse)


def apply_candidate_directive() -> DirectiveDefinition:
    """Explicitly enable applying a known candidate; no job identity is required."""

    def parse(arguments, candidates):
        _arguments(arguments, ('candidate_id',))
        ident = arguments['candidate_id']
        if (not isinstance(ident, str) or not ident.strip()
                or not any(candidate.candidate_id == ident for candidate in candidates)):
            raise ValueError('Application must reference an available candidate in decision context')
        from rsisys.loops.contracts.requests import ApplyCandidateSpec

        return ApplyCandidateSpec(ident)

    return DirectiveDefinition(DirectiveSpec(
        'apply_candidate', 'Apply an available candidate from decision context', {
            'type': 'object', 'properties': {'candidate_id': {'type': 'string', 'minLength': 1}},
            'required': ['candidate_id'], 'additionalProperties': False,
        },
    ), parse)
