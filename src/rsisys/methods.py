"""Versioned whole-method selection from a trusted host catalog.

A method builds a ModuleAssembly, either as a whole implementation or as an
optional component graph. Selecting a method never runs its research lifecycle.
"""
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import json
from typing import Protocol, TYPE_CHECKING

from rsisys.components import _json, _name


METHOD_PROTOCOL = 'rsisys-method/v1'

if TYPE_CHECKING:
    from rsisys.assembly import ModuleAssembly


class MethodFactory(Protocol):
    def __call__(self, config: dict) -> 'ModuleAssembly': ...


class RSIMethod(Protocol):
    """Method-level assembly contract.

    A method binds its controller and Lab together and may inject harness,
    memory, scene, verifier and backend components through the returned
    ModuleAssembly. The host does not prescribe how those components are
    implemented; individual component registries remain lower-level extension
    points for reusable pieces.
    """
    name: str
    version: str

    def build(self, config: dict) -> 'ModuleAssembly': ...


def validate_method_ports(decision, exploration, lab):
    """Check public assembly windows without prescribing internal phases/state."""
    windows = [
        ('decision endpoint', decision, ('step', 'pending', 'receive_feedback')),
        ('decision harness', getattr(decision, 'harness', None), ('step', 'pending', 'receive_feedback')),
        ('action space', getattr(decision, 'action_space', None),
            ('describe', 'validate', 'directive_modes', 'validate_directive')),
        ('exploration endpoint', exploration, ('accept', 'advance', 'inspect', 'cancel', 'acknowledge')),
    ]
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
class MethodDefinition:
    """Pure config builder: factory(config) -> ModuleAssembly.

    Factories only describe assembly; acquire resources through the returned
    assembly factories. No fixed propose/train/evaluate/adopt pipeline is imposed.
    """
    factory: MethodFactory
    version: str
    protocol: str = METHOD_PROTOCOL
    validate_config: Callable | None = None

    @classmethod
    def from_method(cls, method: RSIMethod, *, validate_config=None):
        """Create a catalog definition from a method-level assembly object."""
        if not callable(getattr(method, 'build', None)):
            raise TypeError('RSIMethod must provide build(config)')
        return cls(method.build, method.version, validate_config=validate_config)

    def __post_init__(self):
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
        from rsisys.assembly import ModuleAssembly
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
        return replace(modules, method_identity=identity)
