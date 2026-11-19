"""Register complete RSI methods and reusable components for host assembly.

Configuration selects registered names, never Python import strings. Plugins
are trusted host code explicitly loaded by the caller, not agent-written code.
"""
import hashlib
import importlib.util
import json
import re
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from .components import _json, _name
from .methods import ControllerDefinition, MethodCatalog, MethodDefinition, MethodIdentity
from .methods import _FactoryDefinition as _Registration
from .runtime.storage.files import file_hash


def _selection(raw):
    if not isinstance(raw, dict) or set(raw) - {'name', 'config'} or 'name' not in raw:
        raise ValueError('Select a registered name with optional config')
    _name(raw['name'], 'Registration')
    config = raw.get('config', {})
    if not isinstance(config, dict):
        raise TypeError('Registration config must be an object')
    return raw['name'], json.loads(_json(config))


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
        self._components = {}
        self._plugins = []
        self._plugin_sources = []
        if builtins:
            from .algorithms import register_builtins
            from .backends import register_components as register_backends
            from .harnesses import register_components as register_harnesses
            from .labs.scenes.math import register_components as register_math
            register_builtins(self)
            register_backends(self)
            register_harnesses(self)
            register_math(self)
            from .labs.scenes.coding import register_components as register_coding
            register_coding(self)
            from .labs.scenes.self_reflection import register as register_self_reflection
            register_self_reflection(self)

    @staticmethod
    def _register(catalog, name, factory, version, validate_config):
        _name(name, 'Registration')
        if name in catalog:
            raise ValueError(f'Duplicate registration: {name}')
        if not callable(factory) or not isinstance(version, str) or not version.strip():
            raise TypeError('Registration requires a factory and nonempty version')
        if validate_config is not None and not callable(validate_config):
            raise TypeError('validate_config must be callable')
        catalog[name] = _Registration(factory, version, validate_config)

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
        self._register(self._labs, name, factory, version, validate_config)

    def register_component(self, kind, name, factory, *, version, validate_config=None):
        """Register a reusable method component by role and stable name.

        Supported roles are harness, memory, backend and scene. Answer verifiers
        belong to the scene whose answer format they check, not to this catalog. The
        factory is intentionally opaque: the method assembly supplies its
        context and configuration when it resolves the component.
        """
        allowed = {'harness', 'memory', 'backend', 'scene'}
        if kind not in allowed:
            raise ValueError(f'Unknown component role: {kind}')
        _name(name, 'Component')
        if not callable(factory) or not isinstance(version, str) or not version.strip():
            raise TypeError('Component requires a factory and nonempty version')
        if validate_config is not None and not callable(validate_config):
            raise TypeError('validate_config must be callable')
        catalog = self._components.setdefault(kind, {})
        if name in catalog:
            raise ValueError(f'Duplicate {kind} component: {name}')
        catalog[name] = _Registration(factory, version, validate_config)

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
            raise ValueError('Nested registration plugin source changed after admission')
        name = '_rsisys_registration_' + digest
        spec = importlib.util.spec_from_file_location(name, path,
            submodule_search_locations=[str(package_root)] if package_root is not None else None)
        module = importlib.util.module_from_spec(spec)
        prior = (dict(self._controllers), dict(self._labs), dict(self._methods),
                 {kind: dict(catalog) for kind, catalog in self._components.items()}, list(self._plugins), list(self._plugin_sources))
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
            self._plugin_sources.append({'path': str(package_root or path), 'sha256': digest})
        except BaseException:
            self._controllers, self._labs, self._methods, self._components, self._plugins, self._plugin_sources = prior
            for loaded in tuple(sys.modules):
                if loaded == name or loaded.startswith(name + '.'):
                    sys.modules.pop(loaded, None)
            sys.modules.update(prior_modules)
            raise

    def _with_controller_catalog(self, modules, selection):
        if not modules.controller_catalog:
            return modules
        catalog = {name: item.factory for name, item in self._controllers.items()}
        catalog.update(modules.controller_catalog)
        validators = {name: item.validate_config for name, item in self._controllers.items()
                      if item.validate_config is not None}
        validators.update(modules.controller_validators)
        return replace(modules, controller_catalog=catalog, controller_validators=validators,
                       registry_source={'framework': deepcopy(selection), 'plugins': deepcopy(self._plugin_sources)})

    def build(self, selection):
        """Validate selection now; construct owned resources only at launch."""
        selection = json.loads(_json(selection))
        if not isinstance(selection, dict):
            raise TypeError('framework must be an object')
        if set(selection) == {'method'}:
            method_name, method_config = _selection(selection['method'])
            modules = MethodCatalog(self._methods).select({'method': method_name, 'config': method_config})
            identity = MethodIdentity('registered_framework', '1', {
                'method': modules.method_identity.configuration(), 'plugins': self._plugins})
            return self._with_controller_catalog(replace(modules, method_identity=identity), selection)
        if set(selection) != {'controller', 'labs'}:
            raise ValueError('framework requires controller and labs, or one complete method selection')
        name, options = _selection(selection['controller'])
        if name not in self._controllers:
            raise ValueError(f'Unknown controller: {name}')
        controller = self._controllers[name]
        labs = selection['labs']
        if not isinstance(labs, dict) or not labs:
            raise ValueError('At least one named Lab is required')
        selected_labs, manifest = [], {}
        for alias, raw in labs.items():
            if not isinstance(alias, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', alias):
                raise ValueError('Lab aliases must be simple names without dots')
            lab_name, config = _selection(raw)
            if lab_name not in self._labs:
                raise ValueError(f'Unknown Lab: {lab_name}')
            definition = self._labs[lab_name]
            selected_labs.append((alias, definition, config))
            manifest[alias] = {'name': lab_name, 'version': definition.version, 'config': config}
        identity = MethodIdentity('registered_framework', '1', {
            'controller': {'name': name, 'version': controller.version, 'config': options},
            'labs': manifest, 'plugins': self._plugins})

        # Controller/Lab composition builds through the same method path as whole methods.
        definition = MethodDefinition.from_factories(
            controller=controller.factory, labs={alias: item.factory for alias, item, _ in selected_labs},
            version=controller.version, validate_controller=controller.validate_config,
            validate_labs={alias: item.validate_config for alias, item, _ in selected_labs
                           if item.validate_config is not None})
        modules = definition.factory({'controller': options,
            'labs': {alias: config for alias, _, config in selected_labs}})
        return self._with_controller_catalog(replace(modules, method_identity=identity), selection)
