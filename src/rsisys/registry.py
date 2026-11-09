"""Register RSI controllers and Labs; assemble them through the host runtime.

Configuration selects registered names, never Python import strings. Plugins
are trusted host code explicitly loaded by the caller, not agent-written code.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

from .assembly import _build_policy_assembly
from .components import _json, _name
from .methods import (MethodCatalog, MethodDefinition, MethodIdentity,
                      _FactoryDefinition as _Registration)


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
    memory, engines and verifiers may be injected from registered reusable
    components. Whole methods are the primary registration unit; separate
    controller/Lab registrations remain available for compatibility and reuse.
    """
    def __init__(self, *, builtins=True):
        self._controllers, self._labs, self._methods = {}, {}, {}
        self._components = {}
        self._plugins = []
        if builtins:
            from .algorithms import register_builtins
            register_builtins(self)
            self.register_lab('local_process', self._local_lab, version='1', validate_config=self._local_config)

    @staticmethod
    def _local_config(config):
        from rsisys.runtime.orchestration.run_policy import positive
        allowed = {'timeout_seconds', 'max_jobs', 'max_parallel', 'max_file_bytes', 'max_poll_wait_seconds'}
        if set(config) - allowed:
            raise ValueError('Unknown local_process Lab option')
        for key, value in config.items():
            if key in {'max_jobs', 'max_parallel', 'max_file_bytes'}:
                if type(value) is not int or value < 1:
                    raise ValueError(f'{key} must be a positive integer')
            else:
                positive(value, key)

    @staticmethod
    def _local_lab(context, config):
        from rsisys.labs.local_process.lab import LocalProcessLab
        if context.execution != 'local_process':
            raise ValueError('local_process Lab requires host execution=local_process')
        return LocalProcessLab(context.sandbox, context.artifacts, **config)

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

    def register_controller(self, name, factory, *, version, validate_config=None,
                            endpoint=False, research_factory=None):
        """Register a policy, or an advanced assembled decision endpoint.

        Standard factories accept (context, config) and return a policy.
        endpoint=True factories accept (context, execution, config) and own the
        assembled endpoint. Optional research_factory(config) supplies method-owned
        research integration; the registry makes no harness or memory choices.
        """
        if type(endpoint) is not bool:
            raise TypeError('endpoint must be boolean')
        if research_factory is not None and not callable(research_factory):
            raise TypeError('research_factory must be callable')
        self._register(self._controllers, name, factory, version, validate_config)
        self._controllers[name] = replace(self._controllers[name], endpoint=endpoint,
                                          research_factory=research_factory)

    def register_lab(self, name, factory, *, version, validate_config=None):
        self._register(self._labs, name, factory, version, validate_config)

    def register_component(self, kind, name, factory, *, version, validate_config=None):
        """Register a reusable method component by role and stable name.

        Supported roles are harness, memory, backend, scene and verifier. The
        factory is intentionally opaque: the method assembly supplies its
        context and configuration when it resolves the component.
        """
        allowed = {'harness', 'memory', 'backend', 'scene', 'verifier'}
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

    def load_plugin(self, path):
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
            identity = [[p.relative_to(package_root).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest()]
                        for p in sorted(package_root.rglob('*'))
                        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
            digest = hashlib.sha256(_json(identity).encode()).hexdigest()
        name = '_rsisys_registration_' + digest
        spec = importlib.util.spec_from_file_location(name, path,
            submodule_search_locations=[str(package_root)] if package_root is not None else None)
        module = importlib.util.module_from_spec(spec)
        prior = (dict(self._controllers), dict(self._labs), dict(self._methods),
                 {kind: dict(catalog) for kind, catalog in self._components.items()}, list(self._plugins))
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
        except BaseException:
            self._controllers, self._labs, self._methods, self._components, self._plugins = prior
            for loaded in tuple(sys.modules):
                if loaded == name or loaded.startswith(name + '.'):
                    sys.modules.pop(loaded, None)
            sys.modules.update(prior_modules)
            raise

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
            return replace(modules, method_identity=identity)
        if set(selection) != {'controller', 'labs'}:
            raise ValueError('framework requires controller and labs, or one complete method selection')
        name, options = _selection(selection['controller'])
        if name not in self._controllers:
            raise ValueError(f'Unknown controller: {name}')
        controller = self._controllers[name]
        controller.validate(options)
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
            definition.validate(config)
            selected_labs.append((alias, definition, config))
            manifest[alias] = {'name': lab_name, 'version': definition.version, 'config': config}
        identity = MethodIdentity('registered_framework', '1', {
            'controller': {'name': name, 'version': controller.version, 'config': options},
            'labs': manifest, 'plugins': self._plugins})

        modules = _build_policy_assembly(controller, options, selected_labs)
        return replace(modules, method_identity=identity)
