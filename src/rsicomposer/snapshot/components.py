"""Portable snapshots through explicit component state ports, without introspection."""
import json
import sys
from dataclasses import replace
from time import time

from rsicomposer.loops.decision_making.action_space.access import restore_action_permissions
from rsicomposer.runtime.capabilities import optional_method
from rsicomposer.runtime.state.recovery import (
    capture_runner,
    policy_state,
    restore_policy,
    restore_runner,
)
from rsicomposer.runtime.state.contracts import select_contract, validate_contract

from .codec import decode, encode
from .storage import capture_bundle, publish_restore


def _validate(contract, app):
    if app._closed or app.store is None:
        raise RuntimeError('Application is closed')
    if app._requires_parent:
        raise RuntimeError('Parent-bound runs cannot be captured as independent snapshots')
    if app.recovery is not None and app.recovery.intervention is not None:
        raise RuntimeError('Reconcile interrupted durable recovery before creating a snapshot')
    if contract.validate_snapshot(app) is not None:
        raise TypeError('Snapshot validation must return None')


def capture_components(app, store, contract, label):
    validate_contract(contract, portable=True)
    action_permissions = app._action_permissions.to_dict()
    _validate(contract, app)
    # Encode before staging. The integration must reject unknown effects and
    # capture dependent memory/files here instead of copying its workspace.
    components = contract.capture(app)
    state = encode({'contract_id': contract.contract_id, 'components': components,
                    'runner': capture_runner(app.runner),
                    'policy': policy_state(app.runner.run_policy, external=True),
                    'action_permissions': action_permissions})
    export = optional_method(contract, 'export_snapshot_files')
    write_files = (lambda stage: export(app, stage / 'component_files', components)) if export is not None else None
    return capture_bundle(app, store, label=label, config=app.config, state=state,
        runtime={'python': list(sys.version_info[:2]), 'saved_at': time()},
        loop_state_refs={'assembly': 'state.json'},
        component_refs={'assembly': 'state.json', 'experience': 'experience.sqlite3', 'artifacts': 'artifacts/'},
        resume_policy={'active_jobs': 'forbidden', 'effects': 'never_replay',
            'models': 'external_references_only', 'state': 'explicit_component_adapters'},
        kind='component_quiescent', write_extra=write_files)


def restore_components(directory, target, manifest, *, run_id, modules, contract, run_policy):
    from rsicomposer.application import Application, ApplicationConfig
    from rsicomposer.assembly import ModuleAssembly
    from rsicomposer.runtime.storage.artifacts import FileArtifactStore
    if not isinstance(modules, ModuleAssembly):
        raise TypeError('modules must be a ModuleAssembly')
    if contract is None and modules.state_factory is None:
        raise TypeError('Supply the original modules/state_factory or an explicit snapshot contract')
    modules.validate_saved(directory)
    # State adapters are available after constructing the supplied assembly.
    contract = select_contract(contract, has_state_adapters=modules.state_factory is not None, portable=True)
    state = decode(json.loads((directory / 'state.json').read_text()))
    fields = {'contract_id', 'components', 'runner', 'policy'}
    if (not isinstance(state, dict) or set(state) not in (fields, {*fields, 'action_permissions'})
            or state['contract_id'] != contract.contract_id):
        raise TypeError('Snapshot requires the original component state contract')
    action_permissions = restore_action_permissions(state)
    config = ApplicationConfig.from_dict(json.loads((directory / 'config.json').read_text()), base_dir=directory)
    if run_id is not None:
        raise ValueError('Component snapshots preserve run identity; forking needs explicit domain integration')
    from rsicomposer.runtime.state.controller_versions import restore_controller_seed
    restore_controller_seed(json.loads((directory / 'run.json').read_text()), config)
    if config.roles != manifest.role_bindings:
        raise ValueError('Snapshot model bindings are inconsistent')
    runtime = json.loads((directory / 'runtime.json').read_text())
    if runtime['python'] != list(sys.version_info[:2]):
        raise ValueError('Snapshot requires the same Python major/minor version')
    import math
    if type(runtime.get('saved_at')) not in (int, float) or not math.isfinite(runtime['saved_at']):
        raise ValueError('Invalid snapshot timestamp')
    policy = restore_policy(state['policy'], run_policy)
    if not (directory / 'artifacts').is_dir():
        raise ValueError('Snapshot artifacts directory is missing')
    FileArtifactStore(directory / 'artifacts').refs()
    config = replace(config, data_dir=target)

    def construct():
        # Snapshot restore is not a new durable run. The same state contract
        # still authorizes reconstruction of Lab/feedback markers if supplied.
        return Application(config, modules=modules, run_policy=policy, _restore=True,
                           action_permissions=action_permissions)

    def restore(app):
        restore_runner(app.runner, state['runner'], max(0.0, time() - runtime['saved_at']))
        portable = optional_method(contract, 'restore_portable')
        result = (portable(app, state['components'], directory / 'component_files') if portable is not None
                  else contract.restore(app, state['components']))
        if result is not None:
            raise TypeError('Component snapshot restore must return None')
        _validate(contract, app)

    return publish_restore(directory, target, config, manifest, ('module_assembly.json', 'run.json'), construct, restore)
