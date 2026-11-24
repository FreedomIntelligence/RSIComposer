"""Host-selected component graphs; no discovery, dynamic imports or execution grants.

The catalog contains trusted Python factories. A JSON selection binds their
versions, native configuration and dependencies before any factory is called.
Factories construct only: research starts through the assembled public ports.
"""
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
import re
from types import MappingProxyType

from rsicomposer.runtime.capabilities import optional_method


def _name(value, label):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]{0,127}', value):
        raise ValueError(f'{label} must be a bounded component name')
    return value


def config_json(value):
    """Encode native JSON configuration, rejecting lossy coercions and nonfinite values."""
    # Round-tripping also rejects native objects and returns independent values.
    encoded = json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))
    if json.loads(encoded) != value:
        raise ValueError('Component configuration must contain native JSON values')
    return encoded


# Existing internal callers keep their spelling; plugins use api.config_json.
_json = config_json

@dataclass(frozen=True)
class ComponentDefinition:
    """One versioned host implementation, independent of its experiment slot.

    requires maps factory-local dependency names to advertised contract IDs.
    Contracts describe compatibility, not authority. validate checks the actual
    constructed value and must return None. An external state port is required
    unless the host explicitly declares the component stateless.
    """
    factory: Callable
    version: str
    provides: tuple[str, ...]
    requires: Mapping[str, str] = field(default_factory=dict)
    validate: Callable | None = None
    validate_config: Callable | None = None
    stateless: bool = False
    owned: bool = False

    def __post_init__(self):
        if not callable(self.factory) or any(v is not None and not callable(v)
                for v in (self.validate, self.validate_config)):
            raise TypeError('Component factory and optional validator must be callable')
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError('A component implementation version is required')
        if (not isinstance(self.provides, tuple) or not self.provides
                or any(not isinstance(v, str) or not v.strip() for v in self.provides)
                or len(set(self.provides)) != len(self.provides)):
            raise ValueError('provides must contain distinct nonempty contract IDs')
        if not isinstance(self.requires, Mapping):
            raise TypeError('requires must map local dependency names to contract IDs')
        requirements = dict(self.requires)
        for name, contract in requirements.items():
            _name(name, 'Dependency')
            if not isinstance(contract, str) or not contract.strip():
                raise ValueError('Dependency contract IDs must be nonempty strings')
        if type(self.stateless) is not bool or type(self.owned) is not bool:
            raise TypeError('stateless and owned must be booleans')
        object.__setattr__(self, 'requires', MappingProxyType(requirements))


class _StaticState:
    """An explicit no-state declaration plus immutable configuration identity."""
    state_id = 'rsicomposer-component-static/v1'

    def __init__(self, configuration):
        self._configuration = _json(configuration)

    def configuration(self):
        return json.loads(self._configuration)

    def capture(self):
        return {}

    def restore(self, state):
        if state != {}:
            raise ValueError('Stateless component received nonempty state')

    def validate_snapshot(self):
        return None


class ComponentInstances(Mapping):
    """Read-only bindings. A new combination constructs a new run, never a hot swap."""
    def __init__(self, values, manifest, owned=()):
        self._values = MappingProxyType(dict(values))
        self._manifest = _json(manifest)
        self._owned = frozenset(owned)

    def __getitem__(self, name):
        return self._values[name]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def configuration(self):
        return json.loads(self._manifest)

    def owns(self, value):
        return id(value) in self._owned

    def state_adapters(self):
        manifest = self.configuration()
        ports = {'component_manifest': _StaticState(manifest)}
        for name, entry in manifest['components'].items():
            if entry['stateless']:
                ports['component:' + name] = _StaticState(entry)
        return ports

    @property
    def required_state_adapters(self):
        return {'component_manifest', *('component:' + name for name in self)}


class ComponentGraph:
    """Config-driven assembly from an explicitly supplied, instance-local catalog.

    Slot names are free: data, training, inference, RSI algorithms and harnesses
    need not share an algorithm schema or become new fixed Lab tool categories.
    A factory receives (assembly_context, dependencies, configuration).
    """
    def __init__(self, catalog, selections):
        if not isinstance(catalog, Mapping) or not isinstance(selections, Mapping) or not selections:
            raise TypeError('A host catalog and nonempty component selections are required')
        self._catalog = dict(catalog)
        for name, definition in self._catalog.items():
            _name(name, 'Implementation')
            if not isinstance(definition, ComponentDefinition):
                raise TypeError('Catalog values must be ComponentDefinition instances')
        self._selections = {}
        for name, selection in selections.items():
            _name(name, 'Slot')
            if (not isinstance(selection, dict) or set(selection) - {'component', 'config', 'dependencies'}
                    or 'component' not in selection):
                raise ValueError(f'Invalid component selection: {name}')
            implementation = selection['component']
            if not isinstance(implementation, str) or implementation not in self._catalog:
                raise ValueError(f'Unknown host component: {implementation!r}')
            config = selection.get('config', {})
            dependencies = selection.get('dependencies', {})
            if not isinstance(config, dict) or not isinstance(dependencies, dict):
                raise TypeError('Component config and dependencies must be JSON objects')
            for alias, target in dependencies.items():
                _name(alias, 'Dependency')
                _name(target, 'Dependency target')
            self._selections[name] = json.loads(_json({
                'component': implementation, 'config': config, 'dependencies': dependencies}))
            validator = self._catalog[implementation].validate_config
            if validator is not None and validator(json.loads(_json(config))) is not None:
                raise TypeError(f'Component configuration validator must return None: {name}')
        for name, selection in self._selections.items():
            definition = self._catalog[selection['component']]
            dependencies = selection['dependencies']
            if dependencies.keys() != definition.requires.keys():
                raise ValueError(f'Dependency names differ from the host contract: {name}')
            for alias, target in dependencies.items():
                self.require(target, definition.requires[alias])
        order, visiting = [], set()

        def visit(name):
            if name in visiting:
                raise ValueError(f'Component dependency cycle: {name}')
            if name in order:
                return
            visiting.add(name)
            for dependency in sorted(self._selections[name]['dependencies'].values()):
                visit(dependency)
            visiting.remove(name)
            order.append(name)

        for name in sorted(self._selections):
            visit(name)
        self._order = tuple(order)
        self._configuration = _json({'schema_version': 1, 'components': {
            name: {**selection, 'version': self._catalog[selection['component']].version,
                   'provides': list(self._catalog[selection['component']].provides),
                   'requires': dict(self._catalog[selection['component']].requires),
                   'stateless': self._catalog[selection['component']].stateless,
                   'owned': self._catalog[selection['component']].owned}
            for name, selection in self._selections.items()}})

    def configuration(self):
        return json.loads(self._configuration)

    def require(self, name, contract):
        if name not in self._selections:
            raise ValueError(f'Missing component dependency/slot: {name}')
        definition = self._catalog[self._selections[name]['component']]
        if contract not in definition.provides:
            raise ValueError(f'Component {name} does not satisfy required contract {contract}')

    def build(self, context, *, bindings=None):
        if context._closed:
            raise RuntimeError('Assembly is closed')
        values, owned = {}, set()
        for name in self._order:
            selection = self._selections[name]
            definition = self._catalog[selection['component']]
            dependencies = MappingProxyType({alias: values[target]
                for alias, target in selection['dependencies'].items()})
            value = definition.factory(context, dependencies, json.loads(_json(selection['config'])))
            # Register ownership before any validation that can fail. Earlier
            # dependency resources remain retained if an owned close fails.
            if definition.owned:
                if id(value) in owned:
                    raise ValueError('A component resource cannot have multiple owners')
                close = optional_method(value, 'close')
                if close is None:
                    raise TypeError(f'Owned component {name} must implement close()')
                owned.add(id(value))
                context.add_owned_cleanup(close)
            if value is None:
                raise TypeError(f'Component {name} returned None')
            if definition.validate is not None and definition.validate(value) is not None:
                raise TypeError(f'Component validator must return None: {name}')
            values[name] = value
        return ComponentInstances(values, {**self.configuration(), 'bindings': dict(bindings or {})}, owned)
