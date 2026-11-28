"""Trusted composition bootstrap for independent, explicitly budgeted child runs."""
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import signal
import sys

from rsicomposer.components import config_json
from rsicomposer.loops.contracts.controller_version import ControllerVersion, EvidenceScope
from rsicomposer.loops.decision_making.action_space.access import restore_action_permissions
from rsicomposer.runtime.storage.codec import decode, encode
from rsicomposer.runtime.storage.files import atomic_write, json_bytes
from rsicomposer.selection import RegistrySelection


_MANIFEST_SCHEMA = 4


def build_nested_manifest(controller, labs, host_config, *, permissions, catalog, inherited_history=()):
    """Serialize trusted selections; arbitrary JSON import strings are forbidden."""
    from rsicomposer.methods import ControllerDefinition
    definition = controller if isinstance(controller, ControllerDefinition) else None
    version = (controller if isinstance(controller, ControllerVersion) else ControllerVersion(
        'nested-initial', {'name': 'nested_python'} if definition is not None else controller))
    source = host_config.get('registry_source')
    if source is None and definition is None:
        raise ValueError('Registered NestedRun needs trusted registry_source from its parent assembly')
    if source is None and labs is None:
        raise ValueError('An explicit Python Controller needs explicit registered Labs')
    parent = RegistrySelection.parse(source['selection']) if source is not None else RegistrySelection(None, True, [])
    child = parent.restrict(catalog).child(version.config, labs, definition)
    from rsicomposer.registry import Registry
    paths = list(source['paths']) if source is not None else []
    # Inherit the parent's pins or resolve explicit replacements once at admission.
    # Workers reconstruct this exact selected inventory, never an unpinned override.
    selected = Registry._rebuild(child, paths, nested=True).registration
    return {'schema': _MANIFEST_SCHEMA, 'version': encode(version), 'selection': selected.configuration(),
            'plugins': paths, 'action_permissions': permissions.to_dict(),
            'inherited_history': encode(tuple(inherited_history))}


def _build_modules(manifest):
    if not isinstance(manifest, dict) or type(manifest.get('schema')) is not int or manifest['schema'] != _MANIFEST_SCHEMA:
        raise ValueError('Unsupported nested manifest schema')
    from rsicomposer.registry import Registry
    version = decode(manifest['version'])
    if not isinstance(version, ControllerVersion):
        raise TypeError('Nested manifest needs an explicit ControllerVersion')

    if config_json(manifest['selection']['controller']) != config_json(version.config):
        raise ValueError('Nested Controller version differs from the admitted selection')
    return Registry.reconstruct(manifest['selection'], plugins=manifest['plugins'], _nested=True), version


def _export_evidence(view, store):
    """Export selection evidence, excluding host-final measurements again here."""
    from rsicomposer.loops.contracts.artifacts import ArtifactRef
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
    from rsicomposer.application import Application, ApplicationConfig
    from rsicomposer.runtime.execution.budget import HostBudget
    bootstrap = json.loads(Path(path).read_text())
    if bootstrap.get('schema') != 1:
        raise ValueError('Unsupported nested bootstrap schema')
    from rsicomposer.runtime.execution.process_identity import process_identity
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
                          action_permissions=restore_action_permissions(bootstrap['manifest']),
                          _initial_controller_version=version,
                          _delegated_models=bootstrap['delegated_models'],
                          _inherited_controller_history=decode(bootstrap['manifest']['inherited_history']))
        result = app.run(wait_for_jobs=True)
        evidence = app.assembly.evidence_view(EvidenceScope())
        events, artifacts = _export_evidence(evidence, app.artifacts)
        output.update(status='succeeded', evidence=events, artifacts=artifacts,
            validation={'status': result.status, 'rounds_completed': result.rounds_completed,
                        'decision_steps': result.decision_steps, 'shutdown_complete': result.shutdown_complete,
                        'stop_reason': result.stop_reason, 'allocated_budget': bootstrap['budget']})
    finally:
        # Publication is proof of successful close, not just a returned run().
        if app is not None:
            app.close()
        atomic_write(Path(bootstrap['output']), json_bytes(output))


if __name__ == '__main__':
    main(sys.argv[1])
