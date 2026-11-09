"""Portable snapshots through explicit component state ports, without introspection."""
import json
import shutil
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from time import time
from uuid import uuid4

from rsisys.runtime.capabilities import optional_method
from rsisys.runtime.state.recovery import (
    capture_runner,
    policy_state,
    restore_policy,
    restore_runner,
)
from rsisys.runtime.storage.files import (
    atomic_write,
    json_bytes,
    sync_directory,
    sync_tree,
)

from .codec import decode, encode
from .manifest import SnapshotManifest


def _contract(contract):
    if (not isinstance(getattr(contract, 'contract_id', None), str) or not contract.contract_id
            or any(optional_method(contract, method) is None
                   for method in ('capture', 'restore', 'validate_snapshot'))):
        raise TypeError('Portable snapshot contract needs contract_id, capture, restore, validate_snapshot')
    return contract


def _validate(contract, app):
    if app._closed or app.store is None:
        raise RuntimeError('Application is closed')
    if app.recovery is not None and app.recovery.intervention is not None:
        raise RuntimeError('Reconcile interrupted durable recovery before creating a snapshot')
    if contract.validate_snapshot(app) is not None:
        raise TypeError('Snapshot validation must return None')


def capture_components(app, store, contract, label):
    from rsisys.runtime.state.provenance import code_revision
    from rsisys.runtime.storage.files import file_hash
    _contract(contract)
    _validate(contract, app)
    # Encode before staging. The integration must reject unknown effects and
    # capture dependent memory/files here instead of copying its workspace.
    state = encode({'contract_id': contract.contract_id, 'components': contract.capture(app),
                    'runner': capture_runner(app.runner),
                    'policy': policy_state(app.runner.run_policy, external=True)})
    ident = uuid4().hex
    store.root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.capture-', dir=store.root))
    try:
        atomic_write(staging / 'state.json', json_bytes(state))
        atomic_write(staging / 'config.json', json_bytes(app.config.to_dict()))
        atomic_write(staging / 'run.json', json_bytes({**app.config.source.to_dict(),
                                                       'objective': app.config.objective}))
        atomic_write(staging / 'runtime.json', json_bytes({'python': list(sys.version_info[:2]),
                                                          'saved_at': time()}))
        marker = app.config.data_dir / 'module_assembly.json'
        if marker.exists():
            if json.loads(marker.read_text()) != app.modules.configuration():
                raise TypeError('Saved module assembly differs from the running assembly')
            shutil.copyfile(marker, staging / marker.name)
        app.store.backup(staging / 'experience.sqlite3')
        (staging / 'artifacts').mkdir()
        for ref in app.artifacts.refs():
            shutil.copytree(app.artifacts.root / ref.artifact_id, staging / 'artifacts' / ref.artifact_id)
        digests = {p.relative_to(staging).as_posix(): file_hash(p)
                   for p in staging.rglob('*') if p.is_file()}
        manifest = SnapshotManifest(snapshot_id=ident,
            parent_snapshot_id=getattr(app, 'parent_snapshot_id', None),
            role_bindings=dict(app.config.roles), loop_state_refs={'assembly': 'state.json'},
            learner_state_refs={}, environment_state_refs={},
            experience_revision=digests['experience.sqlite3'], runtime_state_ref='runtime.json',
            code_revision=code_revision(), config_ref='config.json',
            dependency_lock_ref='',
            component_refs={'assembly': 'state.json', 'experience': 'experience.sqlite3', 'artifacts': 'artifacts/'},
            random_state_refs=(), resume_policy={'active_jobs': 'forbidden', 'effects': 'never_replay',
                'models': 'external_references_only', 'state': 'explicit_component_adapters'},
            kind='component_quiescent', label=label, file_digests=digests)
        atomic_write(staging / 'manifest.json', json_bytes(asdict(manifest)))
        sync_tree(staging)
        staging.rename(store.directory(ident))
        sync_directory(store.root)
        app.parent_snapshot_id = ident
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def restore_components(directory, target, manifest, *, run_id, modules, contract, run_policy):
    from rsisys.application import Application, ApplicationConfig
    from rsisys.assembly import module_assembly
    from rsisys.runtime.state.component_state import AssemblyRecovery
    from rsisys.runtime.storage.artifacts import FileArtifactStore
    from rsisys.snapshot.local import publish_restore
    modules = module_assembly(modules)
    if contract is None and modules.state_factory is None:
        raise TypeError('Supply the original modules/state_factory or an explicit snapshot contract')
    marker = directory / 'module_assembly.json'
    if marker.exists():
        if json.loads(marker.read_text()) != modules.configuration():
            raise TypeError('Supply the original modules (module assembly) when restoring this snapshot')
    elif modules.method_identity is not None:
        raise TypeError('Snapshot has no method identity')
    contract = _contract(contract if contract is not None else AssemblyRecovery())
    state = decode(json.loads((directory / 'state.json').read_text()))
    if (not isinstance(state, dict) or set(state) != {'contract_id', 'components', 'runner', 'policy'}
            or state['contract_id'] != contract.contract_id):
        raise TypeError('Snapshot requires the original component state contract')
    config = ApplicationConfig.from_dict(json.loads((directory / 'config.json').read_text()), base_dir=directory)
    if run_id is not None:
        raise ValueError('Component snapshots preserve run identity; forking needs explicit domain integration')
    identity = json.loads((directory / 'run.json').read_text())
    if identity != {**config.source.to_dict(), 'objective': config.objective}:
        raise ValueError('Snapshot run identity does not match configuration')
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
        return Application(config, modules=modules, run_policy=policy, _restore=True)

    def restore(app):
        restore_runner(app.runner, state['runner'], max(0.0, time() - runtime['saved_at']))
        if contract.restore(app, state['components']) is not None:
            raise TypeError('Component snapshot restore must return None')
        _validate(contract, app)

    files = (marker.name,) if marker.exists() else ()
    return publish_restore(directory, target, config, manifest, files, construct, restore)
