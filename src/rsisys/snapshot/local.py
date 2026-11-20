"""Portable quiescent snapshots through AssemblyRecovery state ports."""
import json
from dataclasses import asdict
from pathlib import Path
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.runtime.state.provenance import code_revision
from rsisys.runtime.storage.files import atomic_write, file_hash, json_bytes
from .manifest import SNAPSHOT_SCHEMA_VERSION, SnapshotManifest


class FileSnapshotStore:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def directory(self, snapshot_id):
        import re
        if not isinstance(snapshot_id, str) or not re.fullmatch(r'[0-9a-f]{32}', snapshot_id):
            raise ValueError('Invalid snapshot identity')
        return self.root / snapshot_id

    def save(self, manifest: SnapshotManifest):
        directory = self.directory(manifest.snapshot_id)
        if not directory.is_dir() or (directory / 'manifest.json').exists():
            raise ValueError('Snapshot must be staged once before its manifest is saved')
        atomic_write(directory / 'manifest.json', json_bytes(asdict(manifest)))

    def load(self, snapshot_id):
        return load_manifest(self.directory(snapshot_id))


def load_manifest(directory: Path) -> SnapshotManifest:
    raw = json.loads((directory / 'manifest.json').read_text())
    if not isinstance(raw, dict) or raw.get('schema_version') != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError('Snapshot uses an earlier RSISys protocol; it cannot be restored')
    raw['role_bindings'] = {key: ModelRef(**value) for key, value in raw['role_bindings'].items()}
    raw['random_state_refs'] = tuple(raw['random_state_refs'])
    manifest = SnapshotManifest(**raw)
    if manifest.kind != 'component_quiescent':
        raise ValueError('Unsupported snapshot kind')
    if manifest.code_revision != code_revision():
        raise ValueError('Snapshot requires the same framework source revision')
    files = set()
    for path in directory.rglob('*'):
        if path.is_symlink():
            raise ValueError('Snapshots may not contain symlinks')
        if path.is_file() and path.relative_to(directory).as_posix() != 'manifest.json':
            files.add(path.relative_to(directory).as_posix())
    if files != set(manifest.file_digests):
        raise ValueError('Snapshot file inventory does not match the manifest')
    for name, digest in manifest.file_digests.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Snapshot file escapes its directory')
        if file_hash(directory / relative) != digest:
            raise ValueError(f'Snapshot digest mismatch: {name}')
    required = {'state.json', 'config.json', 'run.json', 'runtime.json', 'experience.sqlite3', 'host_budget.json'}
    if not required <= files:
        raise ValueError('Snapshot is missing required local state')
    return manifest


class LocalSnapshotCapture:
    """Capture the explicit assembly; never substitute a durable-only contract."""
    def __init__(self, application):
        self.application = application
        self.store = FileSnapshotStore(application.config.data_dir / 'snapshots')

    def capture(self, label: str) -> SnapshotManifest:
        from rsisys.runtime.state.component_state import AssemblyRecovery
        from .components import capture_components
        app = self.application
        if app.recovery is not None and type(app.recovery.contract) is not AssemblyRecovery:
            raise TypeError('Portable snapshots require AssemblyRecovery; durable-only contracts cannot be substituted')
        return capture_components(app, self.store, AssemblyRecovery(), label)


def restore_snapshot(directory, target, *, run_id=None, modules=None, run_policy=None, plugins=()):
    """Restore an assembly snapshot into a new workspace, preserving run identity.

    Registered methods can be reconstructed from their manifest. Direct Python
    assemblies require the original explicit ModuleAssembly. Plugins are trusted
    only when explicitly supplied by the caller.
    """
    directory, target = Path(directory).resolve(), Path(target).resolve()
    if target.exists():
        raise FileExistsError('Restore target must be a new directory')
    manifest = load_manifest(directory)
    if modules is None:
        from rsisys.assembly import read_saved_module_layout
        layout = read_saved_module_layout(directory) or {}
        method = layout.get('method')
        if method is None:
            raise TypeError('Supply the original explicit ModuleAssembly for this snapshot')
        from rsisys.registry import Registry
        registry = Registry()
        for plugin in plugins:
            registry.load_plugin(Path(plugin))
        if method['method'] == 'registered_framework':
            config = method['config']
            if 'method' in config:
                selected = config['method']
                selection = {'method': {'name': selected['method'], 'config': selected['config']}}
            else:
                selection = {'controller': {k: v for k, v in config['controller'].items() if k != 'version'},
                    'labs': {alias: {k: v for k, v in value.items() if k != 'version'}
                             for alias, value in config['labs'].items()}}
        else:
            selection = {'method': {'name': method['method'], 'config': method['config']}}
        modules = registry.build(selection)

    from rsisys.runtime.state.component_state import AssemblyRecovery
    from .components import restore_components
    return restore_components(directory, target, manifest, run_id=run_id, modules=modules,
                              contract=AssemblyRecovery(), run_policy=run_policy)
