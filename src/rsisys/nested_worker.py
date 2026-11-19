"""Trusted composition bootstrap for independent, explicitly budgeted child runs."""
from dataclasses import asdict, is_dataclass, replace
from hashlib import sha256
import importlib
import inspect
import json
import os
from pathlib import Path
import signal
import sys

from rsisys.loops.contracts.controller_version import ControllerVersion, EvidenceScope
from rsisys.runtime.storage.codec import decode, encode
from rsisys.runtime.storage.files import atomic_write, json_bytes


def _factory_manifest(factory):
    if factory is None:
        return None
    module, qualname = getattr(factory, '__module__', None), getattr(factory, '__qualname__', None)
    if not module or not qualname or '<' in qualname or module == '__main__':
        raise ValueError('Nested Python factories must be importable module-level definitions; closures are unsupported')
    imported = importlib.import_module(module)
    target = imported
    for part in qualname.split('.'):
        target = getattr(target, part)
    if target is not factory:
        raise ValueError('Nested factory does not match its module-level binding')
    source = inspect.getsourcefile(factory)
    if source is None:
        raise ValueError('Nested factories need a verifiable Python source file')
    return {'module': module, 'qualname': qualname, 'sha256': sha256(Path(source).read_bytes()).hexdigest()}


def _load_factory(spec):
    target = importlib.import_module(spec['module'])
    for part in spec['qualname'].split('.'):
        target = getattr(target, part)
    if _factory_manifest(target) != spec:
        raise ValueError('Nested factory source changed after admission')
    return target


def build_nested_manifest(controller, labs, host_config, *, catalog=None, inherited_history=()):
    """Serialize trusted selections; arbitrary JSON import strings are forbidden."""
    from rsisys.methods import ControllerDefinition
    definition = None
    if isinstance(controller, ControllerDefinition):
        definition = {'policy': _factory_manifest(controller.policy),
                      'memory': _factory_manifest(controller.memory),
                      'harness': _factory_manifest(controller.harness), 'endpoint': controller.endpoint}
        version = ControllerVersion('nested-initial', {'name': 'nested_python', 'config': {}})
    else:
        version = controller if isinstance(controller, ControllerVersion) else ControllerVersion('nested-initial', controller)
        if catalog is not None and version.config['name'] not in catalog:
            raise ValueError('Nested candidate selects an unknown Controller')
    if labs is not None and (not isinstance(labs, dict) or not labs):
        raise ValueError('Nested labs must explicitly map aliases to registered selections, or be None')
    source = host_config.get('registry_source')
    if source is None and definition is None:
        raise ValueError('Registered NestedRun needs trusted registry_source from its parent assembly')
    if source is None and labs is None:
        raise ValueError('An explicit Python Controller needs explicit registered Labs')
    manifest = {'schema': 1, 'version': encode(version), 'source': source, 'labs': labs, 'definition': definition,
                'inherited_history': encode(tuple(inherited_history))}
    # Validate in the parent before reserving budget or launching any worker.
    _build_modules(manifest)
    return json.loads(json.dumps(manifest, allow_nan=False))


def _build_modules(manifest):
    from rsisys.methods import ControllerDefinition, MethodDefinition
    from rsisys.registry import Registry
    registry = Registry()
    source = manifest['source']
    if source is not None:
        for plugin in source['plugins']:
            registry.load_plugin(plugin['path'], expected_sha256=plugin['sha256'])
            if registry._plugin_sources[-1] != plugin:
                raise ValueError('Nested registration plugin source changed after admission')
        original = registry.build(source['framework'])
    else:
        original = None
    version = decode(manifest['version'])
    if not isinstance(version, ControllerVersion):
        raise TypeError('Nested manifest needs an explicit ControllerVersion')
    if manifest['definition'] is not None:
        raw = manifest['definition']
        definition = ControllerDefinition(_load_factory(raw['policy']),
            memory=_load_factory(raw['memory']) if raw['memory'] else None, endpoint=raw['endpoint'],
            harness=_load_factory(raw['harness']) if raw['harness'] else None)
    elif version.config['name'] == 'current':
        definition = original.controller_catalog['current']
    elif version.config['name'] in registry._controllers:
        definition = registry._controllers[version.config['name']].factory
    else:
        raise ValueError('Nested Controller is not available in the trusted registry')
    labs = manifest['labs']
    if labs is None:
        # Initial version injection by Application selects/configures this
        # catalog entry before any decision, including memory/model restoration.
        catalog = dict(original.controller_catalog)
        catalog[version.config['name']] = definition
        return replace(original, controller_catalog=catalog,
                       controller_selection=version.config), version
    selected, validators, options = {}, {}, {}
    for alias, selection in labs.items():
        if (not isinstance(selection, dict) or set(selection) - {'name', 'config'}
                or selection.get('name') not in registry._labs or not isinstance(selection.get('config', {}), dict)):
            raise ValueError('Nested Lab must select a registered name and JSON config')
        entry = registry._labs[selection['name']]
        selected[alias], options[alias] = entry.factory, selection.get('config', {})
        if entry.validate_config is not None:
            validators[alias] = entry.validate_config
    modules = MethodDefinition.from_factories(controller=definition, labs=selected,
        version='nested-1', validate_labs=validators).factory({'controller': version.config['config'], 'labs': options})
    catalog = dict(modules.controller_catalog)
    catalog[version.config['name']] = definition
    return replace(modules, controller_catalog=catalog, controller_selection=version.config,
                   registry_source=source), version


def _export_evidence(view, store):
    """Export selection evidence, excluding host-final measurements again here."""
    from rsisys.loops.contracts.artifacts import ArtifactRef
    index = {key: ref for ref in store.refs() for key in (ref.artifact_id, ref.uri)}
    exports, visiting = {}, set()
    fields = {'artifact_id', 'kind', 'uri', 'version', 'digest'}
    def project(value):
        if is_dataclass(value):
            value = asdict(value)
        if isinstance(value, dict) and set(value) in ({'$map'}, {'$list'}, {'$tuple'}, {'$set'},
                {'$frozenset'}, {'$type', 'fields'}, {'$enum', 'value'}):
            try:
                return project(decode(value))
            except (KeyError, TypeError, ValueError):
                return '[unavailable encoded child evidence]'
        ref = (ArtifactRef(**value) if isinstance(value, dict) and set(value) == fields else
               index.get(value) if isinstance(value, str) else None)
        if ref is not None:
            if ref.artifact_id in visiting:
                return '[unavailable cyclic child evidence]'
            if ref.artifact_id not in exports:
                visiting.add(ref.artifact_id)
                try:
                    raw = view.read(ref).decode('utf-8')
                    try:
                        raw = json.loads(raw)
                    except ValueError:
                        pass
                    exports[ref.artifact_id] = {'kind': ref.kind, 'digest': ref.digest, 'content': project(raw)}
                except (PermissionError, KeyError, ValueError, UnicodeDecodeError):
                    return '[unavailable restricted child evidence]'
                finally:
                    visiting.remove(ref.artifact_id)
            return {'$nested_artifact': ref.artifact_id}
        if isinstance(value, dict):
            return {key: project(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [project(item) for item in value]
        return value
    events = []
    for event in view.events():
        request = event.request if isinstance(event.request, dict) else {}
        arguments = request.get('arguments', {})
        output = event.output if isinstance(event.output, dict) else {}
        if (event.event_id.startswith('sealed-') or output.get('phase') == 'host_finish'
                or isinstance(arguments, dict) and arguments.get('split') == 'sealed'):
            continue
        events.append(event)
    return project(events), exports


def main(path):
    from rsisys.application import Application, ApplicationConfig
    from rsisys.runtime.execution.budget import HostBudget
    bootstrap = json.loads(Path(path).read_text())
    if bootstrap.get('schema') != 1:
        raise ValueError('Unsupported nested bootstrap schema')
    from rsisys.runtime.execution.process_identity import process_identity
    atomic_write(Path(bootstrap['output']).with_name('worker_started.json'), json_bytes({
        'pid': os.getpid(), 'identity': process_identity(os.getpid()),
        'supervisor_pid': os.getppid(), 'supervisor_identity': process_identity(os.getppid())}))
    # The coordinator is CPU-only, while its child workers acquire GPU leases
    # independently. Executor's empty CPU mask must not constrain those leases.
    for key, value in bootstrap['coordinator_visibility'].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    modules, version = _build_modules(bootstrap['manifest'])
    raw = dict(bootstrap['host']['application'])
    raw.update(objective=bootstrap['objective'], data_dir=bootstrap['child_dir'], run_id=bootstrap['run_id'],
               parent_run_id=raw['run_id'], budget=bootstrap['budget'])
    if version.model is not None:
        raw['roles'] = {**raw['roles'], 'decision_maker': asdict(version.model)}
    config = ApplicationConfig.from_dict(raw, base_dir=Path('/'))
    budget = HostBudget(bootstrap['budget_root'], bootstrap['budget'], restoring=True)
    output = {'schema': 1, 'status': 'failed', 'parent_call_id': bootstrap['parent_call_id']}
    def interrupted(signum, frame):
        raise InterruptedError('Nested run cancellation requested')
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    app = None
    try:
        app = Application(config, modules=modules, _host_budget=budget,
                          _initial_controller_version=version,
                          _delegated_models=bootstrap['delegated_models'],
                          _inherited_controller_history=decode(bootstrap['manifest']['inherited_history']))
        result = app.run(wait_for_jobs=True)
        evidence = app.assembly.evidence_view(EvidenceScope())
        events, artifacts = _export_evidence(evidence, app.artifacts)
        output.update(status='succeeded', evidence=events, artifacts=artifacts,
            validation={'status': result.status, 'rounds_completed': result.rounds_completed,
                        'decision_steps': result.decision_steps, 'shutdown_complete': result.shutdown_complete})
    finally:
        # Publication is proof of successful close, not just a returned run().
        if app is not None:
            app.close()
        atomic_write(Path(bootstrap['output']), json_bytes(output))


if __name__ == '__main__':
    main(sys.argv[1])
