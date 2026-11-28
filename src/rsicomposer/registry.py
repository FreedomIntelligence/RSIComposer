"""Register complete RSI methods and reusable components for host assembly.

Configuration selects registered names, never Python import strings. Plugins
are trusted host code explicitly loaded by the caller, not agent-written code.
"""
import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from .components import _json, _name
from .methods import ControllerDefinition, MethodCatalog, MethodDefinition, MethodIdentity
from .methods import _FactoryDefinition as _Registration
from .runtime.storage.files import file_hash
from .selection import RegistrySelection, load_factory


class Registry:
    """Per-host catalog. Factories accept (AssemblyContext, config).

    A controller factory returns a public decide(DecisionInput) implementation
    or a callable. A Lab factory returns describe()/execute(call). Harness,
    memory and engines may be injected from registered reusable components. Whole methods are the primary registration unit; registered
    Controllers and Labs can also be paired by a run config.
    Method-private implementations are not added to those global catalogs.
    """
    def __init__(self, *, builtins=True):
        self._controllers, self._labs, self._methods = {}, {}, {}
        self._builtins = bool(builtins)
        self._components = {}
        self._plugins = []
        self._plugin_paths = []
        if builtins:
            from .algorithms import register_builtins
            from .backends import register_components as register_backends
            from .harnesses import register_components as register_harnesses
            from .labs import register_builtins as register_labs
            register_backends(self)
            register_harnesses(self)
            register_labs(self)
            register_builtins(self)

    @staticmethod
    def _register(catalog, name, factory, version, validate_config):
        _name(name, 'Registration')
        if name in catalog:
            raise ValueError(f'Duplicate registration: {name}')
        if not callable(factory) or not isinstance(version, str) or not version.strip():
            raise TypeError('Registration requires a factory and nonempty version')
        if validate_config is not None and not callable(validate_config):
            raise TypeError('validate_config must be callable')
        catalog[name] = _Registration(factory, version, validate_config, registered_name=name)

    def register_controller(self, name, controller, *, version, validate_config=None):
        """Register a Controller that a run config can pair with any registered Labs.

        controller is a ControllerDefinition or a plain policy factory accepting
        (context, config), i.e. a Controller without a harness.
        """
        if not isinstance(controller, ControllerDefinition):
            controller = ControllerDefinition(controller)
        _name(name, 'Registration')
        if name in self._controllers:
            raise ValueError(f'Duplicate registration: {name}')
        if not isinstance(version, str) or not version.strip():
            raise TypeError('Registration requires a nonempty version')
        if validate_config is not None and not callable(validate_config):
            raise TypeError('validate_config must be callable')
        self._controllers[name] = _Registration(controller, version, validate_config)

    def register_lab(self, name, factory, *, version, validate_config=None):
        """Make a Lab available for explicit pairing and reuse inside methods."""
        self._register(self._labs, name, factory, version, validate_config)

    def lab(self, name):
        """Return the public Lab definition for a method, without constructing its resources."""
        try:
            return self._labs[name]
        except KeyError:
            raise ValueError(f'Unknown Lab: {name}') from None

    def register_component(self, kind, name, factory, *, version, validate_config=None):
        """Register a Harness, Memory or backend by role for use inside a method."""
        if kind not in ('harness', 'memory', 'backend'):
            raise ValueError(f'Unknown component role: {kind}')
        catalog = self._components.get(kind, {})
        self._register(catalog, name, factory, version, validate_config)
        self._components[kind] = catalog

    def resolve_component(self, kind, name, context, config=None):
        """Construct one registered component for a method assembly."""
        config = {} if config is None else config
        if not isinstance(config, dict):
            raise TypeError('Component config must be a JSON object')
        config = json.loads(_json(config))
        try:
            definition = self._components[kind][name]
        except KeyError as exc:
            raise ValueError(f'Unknown {kind} component: {name}') from exc
        definition.validate(config)
        return definition.factory(context, config)

    def register_method(self, name, definition):
        """Register a whole method with its private controller and Labs."""
        _name(name, 'Method')
        if name in self._methods:
            raise ValueError(f'Duplicate method: {name}')
        if not isinstance(definition, MethodDefinition):
            raise TypeError('Expected MethodDefinition')
        self._methods[name] = definition

    def describe(self):
        """Return detached registration metadata without constructing resources.

        A name/version catalog describes availability, not runtime compatibility
        or optional dependency readiness. Plugin loading remains explicit.
        """
        return {
            'controllers': {name: {'version': item.version}
                            for name, item in sorted(self._controllers.items())},
            'labs': {name: {'version': item.version}
                     for name, item in sorted(self._labs.items())},
            'methods': {name: {'version': item.version}
                        for name, item in sorted(self._methods.items())},
            'components': {kind: {name: {'version': item.version}
                                 for name, item in sorted(catalog.items())}
                           for kind, catalog in sorted(self._components.items())},
            'plugins': deepcopy(self._plugins),
        }

    def load_plugin(self, path, *, expected_sha256=None):
        """Load a trusted registration file or a complete method package directory."""
        path = Path(path).resolve(strict=True)
        package_root = path if path.is_dir() else None
        if package_root is not None:
            path = (package_root / '__init__.py').resolve(strict=True)
        if path.suffix != '.py':
            raise ValueError('A registration plugin must be a Python file')
        source = path.read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        if package_root is not None:
            # Include local package code/configuration, excluding generated bytecode.
            identity = [[p.relative_to(package_root).as_posix(), file_hash(p)]
                        for p in sorted(package_root.rglob('*'))
                        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
            digest = hashlib.sha256(_json(identity).encode()).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError('Registration plugin source changed since selection: ' + (package_root or path).name)
        name = '_rsicomposer_registration_' + digest
        spec = importlib.util.spec_from_file_location(name, path,
            submodule_search_locations=[str(package_root)] if package_root is not None else None)
        module = importlib.util.module_from_spec(spec)
        prior = (dict(self._controllers), dict(self._labs), dict(self._methods),
                 {kind: dict(catalog) for kind, catalog in self._components.items()}, list(self._plugins), list(self._plugin_paths))
        prior_modules = {key: value for key, value in sys.modules.items()
                         if key == name or key.startswith(name + '.')}
        sys.modules[name] = module
        try:
            # Execute the exact bytes whose hash becomes part of run identity.
            exec(compile(source, str(path), 'exec'), module.__dict__)
            register = getattr(module, 'register', None)
            if not callable(register) or register(self) is not None:
                raise TypeError('Plugin must define register(registry) returning None')
            self._plugins.append({'name': package_root.name if package_root is not None else path.name,
                                  'sha256': digest})
            self._plugin_paths.append(str(package_root or path))
        except BaseException:
            self._controllers, self._labs, self._methods, self._components, self._plugins, self._plugin_paths = prior
            for loaded in tuple(sys.modules):
                if loaded == name or loaded.startswith(name + '.'):
                    sys.modules.pop(loaded, None)
            sys.modules.update(prior_modules)
            raise

    @classmethod
    def reconstruct(cls, description, *, plugins=(), _nested=False):
        """Resolve a recorded selection. Portable callers provide trusted plugin paths themselves."""
        selection = RegistrySelection.parse(description)
        return cls._rebuild(selection, plugins, nested=_nested)

    @classmethod
    def _rebuild(cls, selection, plugins, *, nested):
        paths = tuple(Path(path).resolve(strict=True) for path in plugins)
        if [path.name for path in paths] != [pin['name'] for pin in selection.plugins]:
            raise ValueError('Supply the original registration plugins in order for this selection')
        if not nested and any(isinstance(ref, dict) for ref in (selection.catalog or {}).values()):
            raise TypeError('Supply the original explicit ModuleAssembly for a Python Controller definition')
        registry = cls(builtins=selection.builtins)
        for path, pin in zip(paths, selection.plugins, strict=True):
            registry.load_plugin(path, expected_sha256=pin['sha256'])
        return registry._resolve(selection)

    @classmethod
    def from_workspace(cls, directory, *, plugins=()):
        from .assembly import read_saved_module_layout
        selection = read_saved_module_layout(directory).get('registry_selection')
        if selection is None:
            raise TypeError('Supply the original explicit ModuleAssembly for this workspace')
        return cls.reconstruct(selection, plugins=plugins)

    def build(self, selection):
        """Validate the root selection now; construct owned resources only at launch."""
        if not isinstance(selection, dict):
            raise TypeError('framework must be an object')
        return self._resolve(RegistrySelection(selection, self._builtins, deepcopy(self._plugins)))

    def _resolve(self, selection):
        base = self._registered_root(selection.framework) if selection.framework is not None else None
        slots = base.controller_catalog if base is not None else {}
        origins = selection.catalog if selection.catalog is not None else {name: name for name in slots}
        catalog, validators = {}, {}
        for alias, ref in origins.items():
            if isinstance(ref, dict):
                catalog[alias] = ControllerDefinition.from_factory_refs(ref, load_factory)
            else:
                if ref not in slots:
                    raise ValueError('Controller slot is absent from the registered assembly: ' + ref)
                catalog[alias] = slots[ref]
                if ref in base.controller_validators:
                    validators[alias] = base.controller_validators[ref]
        active = selection.controller
        if selection.labs is not None:
            name = active['name']
            modules = self._bind_labs(catalog[name], active['config'], selection.labs,
                version='nested-1', validate_controller=validators.get(name))
        else:
            if base is None:
                raise ValueError('Nested composition has no Labs to inherit')
            modules = base
            if active is not None:
                check = validators.get(active['name'])
                if check is not None and check(deepcopy(active['config'])) is not None:
                    raise TypeError('Controller validators must return None')
        bindings = modules.registered_labs
        if selection.lab_bindings is not None and _json(bindings) != _json(selection.lab_bindings):
            raise ValueError('Resolved Lab bindings differ from the admitted registration')
        return replace(modules, controller_catalog=catalog, controller_validators=validators,
                       controller_selection=active if active is not None else modules.controller_selection,
                       _registration=replace(selection, catalog=origins, lab_bindings=bindings),
                       plugin_paths=tuple(self._plugin_paths),
                       _registered_bindings=None)

    def _with_controller_catalog(self, modules):
        if not modules.controller_catalog:
            return modules
        catalog = {name: item.factory for name, item in self._controllers.items()}
        catalog.update(modules.controller_catalog)
        validators = {name: item.validate_config for name, item in self._controllers.items()
                      if name not in modules.controller_catalog and item.validate_config is not None}
        validators.update(modules.controller_validators)
        return replace(modules, controller_catalog=catalog, controller_validators=validators)

    def _registered_root(self, selection):
        if set(selection) == {'method'}:
            method = selection['method']
            modules = MethodCatalog(self._methods).select({'method': method['name'], 'config': method['config']})
            identity = MethodIdentity('registered_framework', '1', {
                'method': modules.method_identity.configuration(), 'labs': modules.registered_labs,
                'plugins': self._plugins})
            return self._with_controller_catalog(replace(modules, method_identity=identity))
        name, options = selection['controller']['name'], selection['controller']['config']
        if name not in self._controllers:
            raise ValueError(f'Unknown controller: {name}')
        controller = self._controllers[name]
        modules = self._bind_labs(controller.factory, options, selection['labs'],
            version=controller.version, validate_controller=controller.validate_config)
        identity = MethodIdentity('registered_framework', '1', {
            'controller': {'name': name, 'version': controller.version, 'config': options},
            'labs': modules.registered_labs, 'plugins': self._plugins})
        return self._with_controller_catalog(replace(modules, method_identity=identity))

    def _bind_labs(self, controller, options, labs, *, version, validate_controller=None):
        """Bind registered Labs to a resolved Controller, without granting additional Controllers."""
        definitions, configs = {}, {}
        for alias, raw in labs.items():
            lab_name, config = raw['name'], raw['config']
            definitions[alias] = self.lab(lab_name)
            configs[alias] = config
        definition = MethodDefinition.from_factories(
            controller=controller, labs=definitions, version=version, validate_controller=validate_controller)
        return definition.factory({'controller': options, 'labs': configs})
