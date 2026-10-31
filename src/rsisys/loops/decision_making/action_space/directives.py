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

    def __post_init__(self):
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
        raise ValueError('Directive arguments do not match the declared fields')


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
        raise ValueError('Evaluation must reference a candidate from the latest feedback')
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
            'recipe_ref': {'type': 'string'}, 'apply_result': {'type': 'boolean', 'default': False},
        }, ('recipe_ref',), _sft),
        definition('rl', 'Alternate environment rollout and model training', {
            'recipe_ref': {'type': 'string'}, 'updates': {'type': 'integer', 'minimum': 1, 'default': 1},
        }, ('recipe_ref',), _rl),
        definition('evaluate', 'Evaluate a candidate from the latest feedback on selected inputs', {
            'candidate_id': {'type': 'string'}, 'protocol_ref': {'type': 'string'},
        }, ('candidate_id', 'protocol_ref'), _evaluate),
        definition('configure_prompt', 'Replace Actor instructions; an empty string clears them', {
            'instructions': {'type': 'string'},
        }, ('instructions',), _prompt),
    )


DEFAULT_DIRECTIVE_MODES = tuple(d.spec.mode for d in builtin_directives())
