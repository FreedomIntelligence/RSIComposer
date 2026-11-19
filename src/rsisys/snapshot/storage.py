"""Shared atomic bundle capture and restore for all portable snapshot kinds."""
import fcntl
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from rsisys.runtime.state.provenance import code_revision
from rsisys.runtime.storage.files import atomic_write, file_hash, json_bytes, sync_directory, sync_tree
from .manifest import SnapshotManifest


def capture_bundle(app, store, *, label, config, state, runtime, loop_state_refs,
                   component_refs, resume_policy, kind='component_quiescent', write_extra=None,
                   include_assembly=False):
    """Stage, checksum, sync and publish once; state codecs belong to the adapter."""
    ident = uuid4().hex
    store.root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.capture-', dir=store.root))
    try:
        atomic_write(staging / 'state.json', json_bytes(state))
        atomic_write(staging / 'config.json', json_bytes(config.to_dict()))
        atomic_write(staging / 'run.json', json_bytes({**config.source.to_dict(), 'objective': config.objective}))
        atomic_write(staging / 'runtime.json', json_bytes(runtime))
        if include_assembly:
            app.modules.validate_saved(app.config.data_dir)
            marker = app.config.data_dir / 'module_assembly.json'
            if marker.exists():
                shutil.copyfile(marker, staging / marker.name)
        if write_extra is not None:
            write_extra(staging)
        accounts = app.config.data_dir / 'model_accounts'
        if accounts.exists():
            shutil.copytree(accounts, staging / 'model_accounts')
        app.assembly.resources.validate_snapshot()
        shutil.copyfile(app.config.data_dir / 'host_execution' / 'budget.json', staging / 'host_budget.json')
        app.store.backup(staging / 'experience.sqlite3')
        (staging / 'artifacts').mkdir()
        for ref in app.artifacts.refs():
            shutil.copytree(app.artifacts.root / ref.artifact_id, staging / 'artifacts' / ref.artifact_id)
        digests = {p.relative_to(staging).as_posix(): file_hash(p)
                   for p in staging.rglob('*') if p.is_file()}
        manifest = SnapshotManifest(snapshot_id=ident,
            parent_snapshot_id=getattr(app, 'parent_snapshot_id', None), role_bindings=dict(config.roles),
            loop_state_refs=loop_state_refs, learner_state_refs={}, environment_state_refs={},
            experience_revision=digests['experience.sqlite3'], runtime_state_ref='runtime.json',
            code_revision=code_revision(), config_ref='config.json', dependency_lock_ref='',
            component_refs=component_refs, random_state_refs=(), resume_policy=resume_policy,
            kind=kind, label=label, file_digests=digests)
        atomic_write(staging / 'manifest.json', json_bytes(asdict(manifest)))
        sync_tree(staging)
        staging.rename(store.directory(ident))
        sync_directory(store.root)
        app.parent_snapshot_id = ident
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def publish_restore(directory, target, config, manifest, files, construct, restore):
    """Publish either snapshot kind with the same atomicity and ownership checks."""
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.restore-', dir=target.parent))
    app, published, restore_lock = None, False, None
    directory_identity = staging.stat()
    try:
        # Retain this exact inode across publication, so failed initialization
        # can clean up only our workspace and only while holding its lock.
        restore_lock = (staging / '.lock').open('a+b')
        shutil.copyfile(directory / 'experience.sqlite3', staging / 'experience.sqlite3')
        (staging / 'host_execution').mkdir()
        shutil.copyfile(directory / 'host_budget.json', staging / 'host_execution' / 'budget.json')
        for name in files:
            shutil.copyfile(directory / name, staging / name)
        shutil.copytree(directory / 'artifacts', staging / 'artifacts')
        if (directory / 'model_accounts').exists():
            shutil.copytree(directory / 'model_accounts', staging / 'model_accounts')
        atomic_write(staging / 'run.json', json_bytes({**config.source.to_dict(), 'objective': config.objective}))
        atomic_write(staging / 'config.json', json_bytes(config.to_dict()))
        sync_tree(staging)
        staging.rename(target)
        published = True
        sync_directory(target.parent)
        app = construct()
        restore(app)
        app.parent_snapshot_id = manifest.snapshot_id
        return app
    except BaseException as error:
        cleanup_pending = False
        if app is not None:
            try:
                app.close()
            except BaseException as cleanup_error:
                error.add_note(f'Restored application cleanup also failed: {cleanup_error}')
                cleanup_pending = app.assembly is not None and app.assembly.cleanup_pending
        if published and cleanup_pending:
            error.add_note('Restore target retained: owned resource cleanup is still pending')
        elif published:
            try:
                fcntl.flock(restore_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                current, lock = target.lstat(), (target / '.lock').lstat()
                original_lock = os.fstat(restore_lock.fileno())
                if ((current.st_dev, current.st_ino) != (directory_identity.st_dev, directory_identity.st_ino)
                        or (lock.st_dev, lock.st_ino) != (original_lock.st_dev, original_lock.st_ino)):
                    raise RuntimeError('Restore target or workspace lock identity changed; target retained')
                shutil.rmtree(target)
                sync_directory(target.parent)
            except BlockingIOError:
                error.add_note('Restore target cleanup skipped: another owner holds the workspace lock')
            except BaseException as cleanup_error:
                error.add_note(f'Restore target cleanup also failed: {cleanup_error}')
        raise
    finally:
        if restore_lock is not None:
            restore_lock.close()
        if staging.exists():
            shutil.rmtree(staging)
