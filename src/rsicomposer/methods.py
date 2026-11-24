"""Versioned whole-method selection from a trusted host catalog.

A method builds a ModuleAssembly, either as a whole implementation or as an
optional component graph. Selecting a method never runs its research lifecycle.
"""
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import json
import re
from copy import deepcopy
from typing import Protocol, TYPE_CHECKING

from rsicomposer.components import _json, _name


METHOD_PROTOCOL = 'rsicomposer-method/v1'

if TYPE_CHECKING:
    from rsicomposer.assembly import ModuleAssembly


class MethodFactory(Protocol):
    def __call__(self, config: dict) -> 'ModuleAssembly': ...


class RSIMethod(Protocol):
    """Method-level assembly contract.

    A method binds its controller and Lab together and may inject harness,
    memory, lab, verifier and backend components through the returned
    ModuleAssembly. The host does not prescribe how those components are
    implemented; individual component registries remain lower-level extension
    points for reusable pieces.
    """
    name: str
    version: str

    def build(self, config: dict) -> 'ModuleAssembly': ...


def validate_method_ports(decision, lab):
    """Check public assembly windows without prescribing internal phases/state."""
    windows = [
        ('decision endpoint', decision, ('step',)),
        ('Controller harness', getattr(decision, 'harness', None), ('window', 'observe')),
        ('action space', getattr(decision, 'action_space', None),
            ('describe', 'validate')),
    ]
    policy = getattr(decision, 'policy', None)
    if not callable(policy) and not callable(getattr(policy, 'decide', None)):
        raise TypeError('Method Controller policy must be callable or implement decide(window)')
    if lab is not None:
        windows.append(('Lab', lab, ('describe', 'execute')))
    for label, value, methods in windows:
        missing = [name for name in methods if not callable(getattr(value, name, None))]
        if missing:
            raise TypeError(f'Method {label} is missing callable ports: ' + ', '.join(missing))


class MethodIdentity:
    """Copied JSON identity; native implementation state belongs to its owner."""
    def __init__(self, name, version, config):
        _name(name, 'Method')
        if not isinstance(version, str) or not version.strip():
            raise ValueError('A method implementation version is required')
        if not isinstance(config, dict):
            raise TypeError('Method config must be a JSON object')
        self._configuration = _json({'protocol': METHOD_PROTOCOL, 'method': name,
                                    'version': version, 'config': config})

    def configuration(self):
        return json.loads(self._configuration)


@dataclass(frozen=True)
class ControllerDefinition:
    """A Controller: a Policy, plus an optional Harness and Memory.

    policy(context, config) returns a DecisionPolicy.
    harness(context, config) returns a ControllerHarness that builds the Policy's
    input window and supplies Controller-local tools, or None for the default.
    memory(config) returns a MemoryAssembly or None: the Controller Memory and
    feedback products, which the host installs once the loop is built.
    With endpoint=True, policy(context, execution, config) instead returns an
    assembled decision endpoint that owns its Harness; harness must be omitted.
    """
    policy: Callable
    memory: Callable | None = None
    endpoint: bool = False
    harness: Callable | None = None

    def __post_init__(self):
        if not callable(self.policy):
            raise TypeError('Controller policy factory must be callable')
        for name in ('memory', 'harness'):
            if getattr(self, name) is not None and not callable(getattr(self, name)):
                raise TypeError(f'Controller {name} factory must be callable')
        if type(self.endpoint) is not bool:
            raise TypeError('endpoint must be boolean')
        if self.endpoint and self.harness is not None:
            raise ValueError('An endpoint factory builds its own Harness; omit harness')


@dataclass(frozen=True)
class _FactoryDefinition:
    factory: object
    version: str
    validate_config: object = None
    endpoint: bool = False
    memory: object = None
    harness: object = None

    def validate(self, config):
        if self.validate_config is not None and self.validate_config(deepcopy(config)) is not None:
            raise TypeError('Configuration validators must return None')


@dataclass(frozen=True)
class MethodDefinition:
    """Pure config builder: factory(config) -> ModuleAssembly.

    Factories only describe assembly; acquire resources through the returned
    assembly factories. No fixed propose/train/evaluate/adopt pipeline is imposed.
    """
    factory: MethodFactory
    version: str
    protocol: str = METHOD_PROTOCOL
    validate_config: Callable | None = None
    observation_probes: tuple = ()

    @classmethod
    def from_factories(cls, *, controller, labs, version, validate_controller=None,
                       validate_labs=None, observation_probes=()):
        """Bind private factories without registering each implementation.

        controller is a ControllerDefinition or a policy factory accepting
        (context, config); Lab factories accept (context, config). Method config
        contains optional controller options and Lab options keyed by the aliases
        declared in ``labs``.
        Resources are only constructed at application launch. For custom
        assembly or component resolution use from_method() or a plain factory.
        """
        if not isinstance(labs, Mapping) or not labs:
            raise ValueError('A method requires at least one named Lab factory')
        if not isinstance(controller, ControllerDefinition):
            controller = ControllerDefinition(controller)
        if validate_labs is None:
            validate_labs = {}
        if not isinstance(validate_labs, Mapping) or set(validate_labs) - set(labs):
            raise ValueError('Lab validators must refer to declared Lab aliases')
        definitions = {}
        for alias, factory in labs.items():
            if not isinstance(alias, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', alias):
                raise ValueError('Lab aliases must be simple names without dots')
            definitions[alias] = _FactoryDefinition(factory, version, validate_labs.get(alias))
        policy = _FactoryDefinition(controller.policy, version, validate_controller,
                                    endpoint=controller.endpoint, memory=controller.memory,
                                    harness=controller.harness)
        for definition in (policy, *definitions.values()):
            if not callable(definition.factory):
                raise TypeError('Method component factories must be callable')
            if definition.validate_config is not None and not callable(definition.validate_config):
                raise TypeError('Configuration validators must be callable')

        def build(config):
            from rsicomposer.assembly import _build_policy_assembly
            if not isinstance(config, dict) or set(config) - {'controller', 'labs'}:
                raise ValueError('Method config accepts only controller and labs options')
            options = config.get('controller', {})
            lab_options = config.get('labs', {})
            if not isinstance(options, dict) or not isinstance(lab_options, dict):
                raise TypeError('Method controller and labs options must be objects')
            if set(lab_options) - set(definitions):
                raise ValueError('Unknown method Lab alias: ' + ', '.join(sorted(set(lab_options) - set(definitions))))
            policy.validate(options)
            selected = []
            for alias, definition in definitions.items():
                params = lab_options.get(alias, {})
                if not isinstance(params, dict):
                    raise TypeError('Method Lab config must be an object')
                capacity = params.get("max_concurrent", 1)
                if type(capacity) is not int or capacity < 1:
                    raise ValueError("Lab max_concurrent must be a positive integer")
                definition.validate({key: value for key, value in params.items() if key != "max_concurrent"})
                selected.append((alias, definition, deepcopy(params)))
            return _build_policy_assembly(policy, deepcopy(options), selected)

        return cls(build, version, observation_probes=observation_probes)

    @classmethod
    def from_method(cls, method: RSIMethod, *, validate_config=None, observation_probes=()):
        """Create a catalog definition from a method-level assembly object."""
        if not callable(getattr(method, 'build', None)):
            raise TypeError('RSIMethod must provide build(config)')
        return cls(method.build, method.version, validate_config=validate_config, observation_probes=observation_probes)

    def __post_init__(self):
        from rsicomposer.runtime.observation.facts import ObservationProbe
        probes = tuple(self.observation_probes)
        if (any(not isinstance(p, ObservationProbe) or p.kind.startswith('host.') for p in probes)
                or len({p.kind for p in probes}) != len(probes)):
            raise ValueError('Method observation probes must declare unique, non-host event kinds')
        object.__setattr__(self, 'observation_probes', probes)
        if not callable(self.factory) or (self.validate_config is not None and not callable(self.validate_config)):
            raise TypeError('Method factory and optional validator must be callable')
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError('A method implementation version is required')
        if not isinstance(self.protocol, str) or not self.protocol.strip():
            raise ValueError('A method protocol is required')


class MethodCatalog:
    """Host registration; JSON can select IDs and parameters, never import code."""
    def __init__(self, definitions):
        if not isinstance(definitions, Mapping):
            raise TypeError('A host method catalog is required')
        self._definitions = dict(definitions)
        for name, definition in self._definitions.items():
            _name(name, 'Method')
            if not isinstance(definition, MethodDefinition):
                raise TypeError('Catalog values must be MethodDefinition instances')

    def select(self, selection):
        from rsicomposer.assembly import ModuleAssembly
        if (not isinstance(selection, dict) or set(selection) - {'method', 'config'}
                or 'method' not in selection):
            raise ValueError('Select a method ID and optional config')
        name = selection['method']
        if not isinstance(name, str) or name not in self._definitions:
            raise ValueError(f'Unknown host method: {name!r}')
        definition = self._definitions[name]
        if definition.protocol != METHOD_PROTOCOL:
            raise ValueError(f'Unsupported method protocol: {definition.protocol}')
        identity = MethodIdentity(name, definition.version, selection.get('config', {}))
        config = identity.configuration()['config']
        if definition.validate_config is not None and definition.validate_config(json.loads(_json(config))) is not None:
            raise TypeError('Method configuration validator must return None')
        modules = definition.factory(json.loads(_json(config)))
        if not isinstance(modules, ModuleAssembly):
            raise TypeError('Method factory must return ModuleAssembly')
        if modules.method_identity is not None:
            raise ValueError('Method factory must not replace its host-selected identity')
        return replace(modules, method_identity=identity,
                       observation_probes=definition.observation_probes or modules.observation_probes)
