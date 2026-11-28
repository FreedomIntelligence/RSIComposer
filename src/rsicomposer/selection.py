"""Data needed to rebuild a composition. No run state, plugin paths or live services."""
import importlib
import inspect
import json
import re
from dataclasses import dataclass, asdict, fields, replace
from pathlib import Path

from .components import config_json, _name
from .runtime.storage.files import file_hash


def _component(raw):
    if not isinstance(raw, dict) or set(raw) - {'name', 'config'} or 'name' not in raw:
        raise ValueError('Select a registered name with optional config')
    name = _name(raw['name'], 'Registration')
    config = raw.get('config', {})
    if not isinstance(config, dict):
        raise TypeError('Registration config must be an object')
    return {'name': name, 'config': json.loads(config_json(config))}


def _labs(raw):
    if not isinstance(raw, dict) or not raw:
        raise ValueError('At least one named Lab is required')
    for alias in raw:
        if not isinstance(alias, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', alias):
            raise ValueError('Lab aliases must be simple names without dots')
    return {alias: _component(item) for alias, item in sorted(raw.items())}


def factory_reference(factory):
    module, qualname = getattr(factory, '__module__', None), getattr(factory, '__qualname__', None)
    if not module or not qualname or '<' in qualname or module == '__main__':
        raise ValueError('Nested Python factories must be importable module-level definitions; closures are unsupported')
    target = importlib.import_module(module)
    for part in qualname.split('.'):
        target = getattr(target, part)
    if target is not factory:
        raise ValueError('Nested factory does not match its module-level binding')
    source = inspect.getsourcefile(factory)
    if source is None:
        raise ValueError('Nested factories need a verifiable Python source file')
    return {'module': module, 'qualname': qualname, 'sha256': file_hash(source)}


def load_factory(ref):
    if not isinstance(ref, dict) or ref.keys() != {'module', 'qualname', 'sha256'}:
        raise ValueError('Invalid nested factory reference')
    target = importlib.import_module(ref['module'])
    for part in ref['qualname'].split('.'):
        target = getattr(target, part)
    if factory_reference(target) != ref:
        raise ValueError('Nested factory source changed after admission')
    return target


@dataclass(frozen=True)
class RegistrySelection:
    """A root registration plus its effective overrides, never a chain of ancestors.

    Catalog values are slots in the pinned root (including a method-private current)
    or explicit Python definitions admitted by the parent. None for labs retains
    the root method's private Lab factories. Plugin identities exclude their paths.
    """
    framework: dict | None
    builtins: bool
    plugins: list
    catalog: dict | None = None
    controller: dict | None = None
    labs: dict | None = None
    lab_bindings: dict | None = None

    def __post_init__(self):
        if type(self.builtins) is not bool or not isinstance(self.plugins, list):
            raise ValueError('Invalid registry selection or plugin pins')
        for pin in self.plugins:
            if (not isinstance(pin, dict) or pin.keys() != {'name', 'sha256'}
                    or not isinstance(pin['name'], str) or not pin['name'] or Path(pin['name']).name != pin['name']
                    or not isinstance(pin['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', pin['sha256'])):
                raise ValueError('Invalid registry plugin identity')
        if self.framework is not None:
            if not isinstance(self.framework, dict):
                raise TypeError('framework must be an object')
            if self.framework.keys() == {'method'}:
                root = {'method': _component(self.framework['method'])}
            elif self.framework.keys() == {'controller', 'labs'}:
                root = {'controller': _component(self.framework['controller']), 'labs': _labs(self.framework['labs'])}
            else:
                raise ValueError('framework requires controller and labs, or one complete method selection')
            object.__setattr__(self, 'framework', root)
        if self.catalog is not None:
            if not isinstance(self.catalog, dict):
                raise ValueError('Invalid Controller catalog selection')
            for name, ref in self.catalog.items():
                _name(name, 'Controller')
                if not isinstance(ref, (str, dict)):
                    raise ValueError('Invalid Controller definition reference')
        if self.controller is not None and (not isinstance(self.controller, dict)
                or self.controller.keys() != {'name', 'config'} or not isinstance(self.controller['name'], str)
                or self.controller['name'] not in (self.catalog or {}) or not isinstance(self.controller['config'], dict)):
            raise ValueError('Invalid active Controller selection')
        if self.labs is not None and (self.controller is None or not isinstance(self.labs, dict) or not self.labs):
            raise ValueError('Nested labs must explicitly map aliases to registered selections, or be None')
        if self.labs is not None:
            object.__setattr__(self, 'labs', _labs(self.labs))

    def configuration(self):
        return asdict(self)

    @classmethod
    def parse(cls, value):
        if (not isinstance(value, dict) or value.keys() != {item.name for item in fields(cls)}
                or not isinstance(value['lab_bindings'], dict)):
            raise ValueError('Invalid recorded registry selection')
        return cls(**json.loads(config_json(value)))

    def restrict(self, allowed):
        return replace(self, catalog={k: ref for k, ref in (self.catalog or {}).items() if k in allowed})

    def child(self, controller, labs, definition):
        name = controller['name']
        catalog = dict(self.catalog or {})
        if definition is None:
            if name not in catalog:
                raise ValueError('Nested candidate selects an unknown Controller')
        else:
            catalog[name] = definition.describe_factories(factory_reference)
        if labs is not None:
            catalog = {'current': catalog[name], name: catalog[name]}
        return replace(self, catalog=catalog, controller=controller, labs=self.labs if labs is None else labs,
                       lab_bindings=self.lab_bindings if labs is None else None)
