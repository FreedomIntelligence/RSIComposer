"""Durable, quiescent checkpoints for the built-in local Application."""

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import tempfile
from uuid import uuid4

from rsisys.runtime.files import atomic_write, json_bytes, sync_directory, sync_tree
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.decision_making.state import DecisionState
from rsisys.loops.exploration.state import ExplorationState
from rsisys.loops.decision_making.memory import DecisionMemory
from rsisys.loops.exploration.memory import ExplorationMemory
from .codec import encode, decode
from .manifest import SnapshotManifest


def digest_file(path: Path) -> str:
    digest = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def code_revision() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def check_states(decision, actor):
    if not isinstance(decision, DecisionState) or not isinstance(actor, ExplorationState):
        raise ValueError('Invalid loop state types')
    if (decision.pending_directive_id is not None or decision.pending_directive is not None
            or decision.pending_job is not None or decision.pending_operation_record is not None
            or actor.active_directive_id is not None
            or actor.pending_episode is not None):
        raise RuntimeError('Snapshot requires a boundary with no pending directive or operation')
    if actor.rl is not None and not actor.rl.terminal:
        raise RuntimeError('An RL cycle is still active')
    for value in (decision.issued, decision.completed, actor.completed):
        if type(value) is not int or value < 0:
            raise ValueError('Invalid loop counters')


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
    raw['role_bindings'] = {key: ModelRef(**value) for key, value in raw['role_bindings'].items()}
    raw['random_state_refs'] = tuple(raw['random_state_refs'])
    manifest = SnapshotManifest(**raw)
    if manifest.schema_version != 2 or manifest.kind != 'local_quiescent':
        raise ValueError('Unsupported snapshot schema')
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
        if digest_file(directory / relative) != digest:
            raise ValueError(f'Snapshot digest mismatch: {name}')
    required = {'state.json', 'config.json', 'run.json', 'runtime.json', 'experience.sqlite3',
                'decision_memory.json', 'exploration_memory.json'}
    if not required <= files:
        raise ValueError('Snapshot is missing required local state')
    return manifest


class LocalSnapshotCapture:
    def __init__(self, application):
        self.application = application
        self.store = FileSnapshotStore(application.config.data_dir / 'snapshots')

    def capture(self, label: str) -> SnapshotManifest:
        app = self.application
        if app.store is None:
            raise RuntimeError('Application is closed')
        dm, actor = app.runner.decision_making, app.runner.exploration
        if app.runner.busy or actor.command_busy:
            raise RuntimeError('Snapshot requires no pending coordination delivery')
        check_states(dm.state, actor.state)
        if dm.training.busy or actor.training.busy or actor.evaluation.busy or app.sandbox.active_count:
            raise RuntimeError('Snapshot requires all local jobs and sessions to be idle')
        from rsisys.loops.decision_making.policy import ModelDecisionPolicy
        from rsisys.loops.exploration.policy import ModelActorPolicy
        from rsisys.loops.decision_making.decision_space.local import LocalDecisionSpace
        from rsisys.loops.exploration.environment import SandboxEchoEnvironment
        from rsisys.loops.exploration.environment.local import LocalProcessEnvironment, CommandActionSpace
        from rsisys.loops.exploration.action_space import EchoActionSpace
        expected_env = LocalProcessEnvironment if app.config.execution == 'local_process' else SandboxEchoEnvironment
        expected_space = CommandActionSpace if app.config.execution == 'local_process' else EchoActionSpace
        for component, expected in ((dm.policy, ModelDecisionPolicy), (actor.policy, ModelActorPolicy),
                                    (dm.space, LocalDecisionSpace), (actor.space, expected_space),
                                    (actor.environment, expected_env)):
            if type(component) is not expected:
                raise TypeError('Custom components must provide their own snapshot integration')
        if type(dm.memory) is not DecisionMemory or type(actor.memory) is not ExplorationMemory:
            raise TypeError('Custom loop memories need explicit snapshot integration')
        if dm.experience is not app.experience or actor.experience is not app.experience:
            raise TypeError('Custom experience systems need explicit snapshot integration')
        if dm.space.artifacts is not app.artifacts or actor.environment.artifacts is not app.artifacts:
            raise TypeError('Custom artifact stores need explicit snapshot integration')
        if dm.policy.allowed_actions != tuple(spec.name for spec in actor.space.describe()):
            raise TypeError('Custom decision action bindings need explicit snapshot integration')
        from rsisys.runtime.model import MockModelBackend, RemoteModelBackend
        from rsisys.loops.decision_making.learner.backends.mock import MockDecisionLearner
        from rsisys.loops.exploration.learner.backends.mock import MockActorLearner
        from rsisys.loops.exploration.evaluation.mock import MockEvaluator
        for policy in (dm.policy, actor.policy):
            binding = app.config.backends[policy.model.backend_id]
            expected_backend = MockModelBackend if binding.kind == 'mock' else RemoteModelBackend
            if type(policy.backend) is not expected_backend:
                raise TypeError('Custom inference backends need explicit snapshot integration')
            if isinstance(policy.backend, RemoteModelBackend) and policy.backend.binding != binding:
                raise ValueError('Live inference configuration differs from the checkpoint configuration')
        learning_types = (MockDecisionLearner, MockActorLearner, MockEvaluator) if app.config.learning == 'mock' else (type(None),) * 3
        for actual, expected in zip((dm.training.backend, actor.training.backend, actor.evaluation.backend), learning_types):
            if type(actual) is not expected:
                raise TypeError('Custom learning backends need explicit snapshot integration')
        roles = {'decision_maker': dm.policy.model, 'actor': actor.policy.model}
        config = replace(app.config, roles=roles)
        ident = uuid4().hex
        self.store.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.capture-', dir=self.store.root))
        try:
            atomic_write(staging / 'state.json', json_bytes({'decision': encode(dm.state), 'actor': encode(actor.state)}))
            atomic_write(staging / 'config.json', json_bytes(config.to_dict()))
            shutil.copyfile(app.config.data_dir / 'run.json', staging / 'run.json')
            atomic_write(staging / 'runtime.json', json_bytes({
                'python': list(sys.version_info[:2]),
                'decision_experience_limit': dm.experience_limit,
                'memory_limits': {'decision_maker': dm.memory_limit, 'actor': actor.memory_limit},
            }))
            app.store.backup(staging / 'experience.sqlite3')
            dm.memory.backup(staging / 'decision_memory.json')
            actor.memory.backup(staging / 'exploration_memory.json')
            (staging / 'artifacts').mkdir()
            for ref in app.artifacts.refs():
                shutil.copytree(app.artifacts.root / ref.artifact_id, staging / 'artifacts' / ref.artifact_id)
            lock = Path(__file__).resolve().parents[3] / 'pixi.lock'
            if lock.is_file():
                shutil.copyfile(lock, staging / 'pixi.lock')
            digests = {path.relative_to(staging).as_posix(): digest_file(path)
                       for path in staging.rglob('*') if path.is_file()}
            manifest = SnapshotManifest(
                snapshot_id=ident, parent_snapshot_id=getattr(app, 'parent_snapshot_id', None),
                role_bindings=roles, loop_state_refs={'decision_maker': 'state.json#decision', 'actor': 'state.json#actor'},
                learner_state_refs={}, environment_state_refs={}, experience_revision=digests['experience.sqlite3'],
                runtime_state_ref='runtime.json', code_revision=code_revision(), config_ref='config.json',
                dependency_lock_ref='pixi.lock' if lock.is_file() else '',
                component_refs={'artifacts': 'artifacts/', 'experience': 'experience.sqlite3',
                                'decision_memory': 'decision_memory.json',
                                'exploration_memory': 'exploration_memory.json'}, random_state_refs=(),
                resume_policy={'active_jobs': 'forbidden', 'completed_jobs': 'archived_in_experience',
                               'models': 'external_references_only', 'environment': 'fresh_session',
                               'random_state': 'not_captured'}, label=label, file_digests=digests,
            )
            atomic_write(staging / 'manifest.json', json_bytes(asdict(manifest)))
            sync_tree(staging)
            staging.rename(self.store.directory(ident))
            sync_directory(self.store.root)
            app.parent_snapshot_id = ident
            return manifest
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def restore_snapshot(directory, target):
    """Fork a checkpoint into a new local workspace; never overwrite the source run."""
    from rsisys.application import Application, ApplicationConfig
    from rsisys.runtime.artifacts import FileArtifactStore
    directory, target = Path(directory).resolve(), Path(target).resolve()
    if target.exists():
        raise FileExistsError('Restore target must be a new directory')
    manifest = load_manifest(directory)
    states = json.loads((directory / 'state.json').read_text())
    decision, actor = decode(states['decision']), decode(states['actor'])
    check_states(decision, actor)
    config = ApplicationConfig.from_dict(json.loads((directory / 'config.json').read_text()), base_dir=directory)
    if config.roles != manifest.role_bindings:
        raise ValueError('Snapshot model bindings are inconsistent')
    runtime = json.loads((directory / 'runtime.json').read_text())
    if runtime['python'] != list(sys.version_info[:2]):
        raise ValueError('Snapshot requires the same Python major/minor version')
    if set(runtime['memory_limits']) != {'decision_maker', 'actor'} or any(
        type(n) is not int or n < 1 for n in runtime['memory_limits'].values()
    ):
        raise ValueError('Invalid restored memory limits')
    if type(runtime['decision_experience_limit']) is not int or runtime['decision_experience_limit'] < 1:
        raise ValueError('Invalid restored experience limit')
    # Validate both owners and payloads before creating the restore target.
    DecisionMemory(directory / 'decision_memory.json')
    ExplorationMemory(directory / 'exploration_memory.json')
    if not (directory / 'artifacts').is_dir():
        raise ValueError('Snapshot artifacts directory is missing')
    FileArtifactStore(directory / 'artifacts').refs()
    config = replace(config, data_dir=target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.restore-', dir=target.parent))
    try:
        shutil.copyfile(directory / 'experience.sqlite3', staging / 'experience.sqlite3')
        for name in ('decision_memory.json', 'exploration_memory.json'):
            shutil.copyfile(directory / name, staging / name)
        shutil.copytree(directory / 'artifacts', staging / 'artifacts')
        shutil.copyfile(directory / 'run.json', staging / 'run.json')
        atomic_write(staging / 'config.json', json_bytes(config.to_dict()))
        sync_tree(staging)
        staging.rename(target)
        sync_directory(target.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    app = Application(config, _restore=True)
    app.runner.decision_making.state = decision
    app.runner.exploration.state = actor
    app.runner.decision_making.experience_limit = runtime['decision_experience_limit']
    app.runner.decision_making.memory_limit = runtime['memory_limits']['decision_maker']
    app.runner.exploration.memory_limit = runtime['memory_limits']['actor']
    app.parent_snapshot_id = manifest.snapshot_id
    return app
