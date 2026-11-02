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
from rsisys.loops.decision_making.harness.state import DecisionState
from rsisys.loops.exploration.harness.state import ExplorationState
from rsisys.loops.decision_making.harness.memory import DecisionMemory
from rsisys.loops.exploration.harness.memory import ExplorationMemory
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.exploration.harness.engine import DefaultExplorationHarness
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
            or decision.pending_review_record is not None
            or decision.review_session is not None
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
                'decision_memory.json', 'exploration_memory.json', 'decision_research.json'}
    if not required <= files:
        raise ValueError('Snapshot is missing required local state')
    return manifest


def research_configuration(dm):
    from rsisys.loops.decision_making.harness.research import ResearchAgenda
    from rsisys.loops.decision_making.harness.context import BasicContextCompactor, IdentityContextCompactor
    if type(dm.research_agenda) is not ResearchAgenda:
        raise TypeError('Custom research agendas need explicit snapshot integration')
    compactor = dm.context_compactor
    if type(compactor) is IdentityContextCompactor:
        return {'kind': 'identity'}
    if type(compactor) is not BasicContextCompactor:
        raise TypeError('Custom context compactors need explicit snapshot integration')
    return {'kind': 'basic', 'trigger_tokens': compactor.trigger_tokens,
            'target_tokens': compactor.target_tokens, 'summary_tokens': compactor.summary_tokens,
            'state': compactor.export_state()}


def restore_research(dm, agenda, compactor):
    from rsisys.loops.decision_making.harness.context import BasicContextCompactor, IdentityContextCompactor
    config = dict(compactor)
    kind = config.pop('kind')
    state = config.pop('state', None)
    if kind not in ('identity', 'basic'):
        raise ValueError('Unsupported context compactor')
    dm.context_compactor = (IdentityContextCompactor(**config) if kind == 'identity'
                            else BasicContextCompactor(**config))
    if state is not None:
        if kind != 'basic':
            raise ValueError('Only basic compactor has local rolling state')
        dm.context_compactor.restore_state(state)
    dm.research_agenda.restore(agenda)


def validate_local_application(app, *, ongoing=False):
    """Validate explicit built-in composition; ongoing recovery supplies its own state boundary."""
    if app.store is None:
        raise RuntimeError('Application is closed')
    from rsisys.runtime.recovery import DurableRecovery
    if not ongoing and isinstance(getattr(app, 'recovery', None), DurableRecovery) and app.recovery.intervention is not None:
        raise RuntimeError('Reconcile interrupted durable recovery before creating a snapshot')
    decision_loop, exploration_loop = app.runner.decision_making, app.runner.exploration
    dm, actor = decision_loop.harness, exploration_loop.harness
    from rsisys.runtime.recovery import decision_extension
    extension = ongoing and decision_extension(dm)
    research_configuration(dm)
    if app.custom_assembly and not (extension and not app.custom_exploration):
        raise TypeError('Custom assembly factories need explicit snapshot integration or recovery contract')
    if (type(dm) is not DefaultDecisionHarness and not extension) or type(actor) is not DefaultExplorationHarness:
        raise TypeError('Custom harnesses need explicit snapshot integration')
    from rsisys.loops.exploration.harness.environment import EpisodeExecution
    from rsisys.loops.exploration.harness.learner import LearningExecution
    from rsisys.loops.exploration.harness.evaluation import EvaluationExecution
    from rsisys.loops.exploration.harness.rl import LocalRLExecution
    for component, expected in ((actor.episodes, EpisodeExecution), (actor.learning, LearningExecution),
                                (actor.evaluations, EvaluationExecution), (actor.local_rl, LocalRLExecution)):
        if type(component) is not expected:
            raise TypeError('Custom execution components need explicit snapshot integration')
    bindings = ((actor.learning.state, actor.state), (actor.evaluations.state, actor.state),
                (actor.local_rl.state, actor.state), (actor.learning.policy, actor.policy),
                (actor.learning.memory, actor.memory), (actor.learning.experience, actor.experience),
                (actor.evaluations.experience, actor.experience),
                (actor.local_rl.episodes, actor.episodes), (actor.local_rl.learning, actor.learning))
    if any(actual is not expected for actual, expected in bindings):
        raise TypeError('Custom execution component bindings need explicit snapshot integration')
    from rsisys.runtime.run_policy import ResearcherRunPolicy
    if not ongoing and (type(app.runner.run_policy) is not ResearcherRunPolicy
            or app.runner.run_policy != ResearcherRunPolicy()
            or app.runner.cleanup_timeout_seconds != 30):
        raise TypeError('Custom run policies/budgets need explicit snapshot integration')
    if app.runner.parallel or dm.max_outstanding != 1:
        raise TypeError('Concurrent decisions need explicit snapshot integration')
    from rsisys.experience.retrieval import RecentFeedbackSelector
    if type(dm.feedback_selector) is not RecentFeedbackSelector:
        raise TypeError('Custom feedback selectors need explicit snapshot integration')
    if dm.model_activator is not None or actor.model_activator is not None:
        raise TypeError('Custom model activators need explicit snapshot integration')
    if not ongoing and (app.runner.busy or actor.command_busy):
        raise RuntimeError('Snapshot requires no pending coordination delivery')
    if not ongoing:
        check_states(dm.state, actor.state)
    if not ongoing and (dm.training.busy or actor.training.busy or actor.evaluation.busy or app.sandbox.active_count):
        raise RuntimeError('Snapshot requires all local jobs and sessions to be idle')
    from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
    from rsisys.loops.exploration.harness.policy import ModelActorPolicy
    from rsisys.loops.decision_making.action_space import LocalDecisionActionSpace
    from rsisys.loops.decision_making.harness.tools.local import LocalDecisionTools
    from rsisys.loops.decision_making.harness.tools.queries import QueryTools, ArtifactReader
    from rsisys.loops.exploration.harness.environment import SandboxEchoEnvironment
    from rsisys.loops.exploration.harness.environment.local import LocalProcessEnvironment
    from rsisys.loops.exploration.action_space.local import CommandActionSpace
    from rsisys.loops.exploration.action_space import EchoActionSpace
    expected_env = LocalProcessEnvironment if app.config.execution == 'local_process' else SandboxEchoEnvironment
    expected_space = CommandActionSpace if app.config.execution == 'local_process' else EchoActionSpace
    for component, expected in ((dm.policy, ModelDecisionPolicy), (actor.policy, ModelActorPolicy),
                                (dm.tools, LocalDecisionTools), (decision_loop.action_space, LocalDecisionActionSpace),
                                (exploration_loop.action_space, expected_space),
                                (actor.environment, expected_env)):
        if type(component) is not expected and not (extension and component is dm.policy):
            raise TypeError('Custom components must provide their own snapshot integration')
    if (actor.commands.customized or (not extension and not dm.policy.directives.is_default)
            or not decision_loop.action_space.directives.is_default):
        raise TypeError('Custom directive definitions or handlers need explicit snapshot integration')
    if type(dm.memory) is not DecisionMemory or type(actor.memory) is not ExplorationMemory:
        raise TypeError('Custom loop memories need explicit snapshot integration')
    if dm.experience is not app.experience or actor.experience is not app.experience:
        raise TypeError('Custom experience systems need explicit snapshot integration')
    if dm.tools.artifacts is not app.artifacts or actor.environment.artifacts is not app.artifacts:
        raise TypeError('Custom artifact stores need explicit snapshot integration')
    if not extension and dm.policy.max_validation_attempts != 1:
        raise ValueError('Custom policy retry budgets require an explicit recovery contract')
    if not extension and dm.policy.structured_output:
        raise ValueError('Structured policy backends require an explicit recovery contract')
    if not extension and dm.policy.project_context:
        raise ValueError('Project context requires an explicit recovery contract; refusing to lose project allocation')
    if not extension and dm.policy.allowed_actions != tuple(spec.name for spec in exploration_loop.action_space.describe()):
        raise TypeError('Custom decision action bindings need explicit snapshot integration')
    expected_modes = ("execute", "configure_prompt") + (("sft", "rl", "evaluate") if app.config.learning == 'mock' else ())
    from rsisys.loops.decision_making.action_space.access import ActionPermissions
    if type(decision_loop.action_space.permissions) is not ActionPermissions:
        raise TypeError('Custom action permissions need explicit snapshot integration')
    action_permissions = decision_loop.action_space.permissions.to_dict()
    query_tools = getattr(dm.tools, 'query_tools', None)
    expected_queries = QueryTools(exploration_loop, artifacts=app.artifacts)
    if (type(query_tools) is not QueryTools or query_tools.provider is not exploration_loop
            or query_tools.artifacts is not app.artifacts or type(query_tools.reader) is not ArtifactReader
            or query_tools.reader.artifacts is not app.artifacts
            or query_tools.specs != expected_queries.specs):
        raise TypeError('Custom query bindings need explicit snapshot integration')
    from rsisys.loops.decision_making.harness.tools.research import ResearchTools
    research_tools = getattr(dm.tools, 'research_tools', None)
    if (type(research_tools) is not ResearchTools or research_tools.agenda is not dm.research_agenda
            or research_tools.memory is not dm.memory or research_tools.experience is not app.experience):
        raise TypeError('Custom research tool bindings need explicit snapshot integration')
    expected_bindings = {
        'prepare_data': dm.tools._prepare_data, 'search_literature': dm.tools._search_literature,
        'store_strategy': dm.tools._store_strategy, **query_tools.bindings, **research_tools.bindings,
    }
    if dm.tools._bindings != expected_bindings:
        raise TypeError('Custom tool bindings need explicit snapshot integration')
    if (decision_loop.action_space.directive_modes() != expected_modes
            or decision_loop.action_space.describe() != LocalDecisionActionSpace(
                extra_definitions=(*expected_queries.definitions, *research_tools.definitions)).describe()):
        raise TypeError('Custom decision actions need explicit snapshot integration')
    from rsisys.runtime.model import MockModelBackend, RemoteModelBackend
    from rsisys.loops.decision_making.harness.learner.backends.mock import MockDecisionLearner
    from rsisys.loops.exploration.harness.learner.backends.mock import MockActorLearner
    from rsisys.loops.exploration.harness.evaluation.mock import MockEvaluator
    for policy in ((actor.policy,) if extension else (dm.policy, actor.policy)):
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
    roles = {'decision_maker': app.config.roles['decision_maker'] if extension else dm.policy.model,
             'actor': actor.policy.model}
    config = replace(app.config, roles=roles)
    return dm, actor, config, action_permissions


class LocalSnapshotCapture:
    def __init__(self, application):
        self.application = application
        self.store = FileSnapshotStore(application.config.data_dir / 'snapshots')

    def capture(self, label: str) -> SnapshotManifest:
        app = self.application
        dm, actor, config, action_permissions = validate_local_application(app)
        roles = config.roles
        ident = uuid4().hex
        self.store.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.capture-', dir=self.store.root))
        try:
            atomic_write(staging / 'state.json', json_bytes({'decision': encode(dm.state), 'actor': encode(actor.state)}))
            atomic_write(staging / 'config.json', json_bytes(config.to_dict()))
            shutil.copyfile(app.config.data_dir / 'run.json', staging / 'run.json')
            atomic_write(staging / 'runtime.json', json_bytes({
                'python': list(sys.version_info[:2]),
                'context_compactor': research_configuration(dm),
                'decision_experience_limit': dm.experience_limit,
                'review_tool_limit': dm.review_tool_limit,
                'action_permissions': action_permissions,
                'memory_limits': {'decision_maker': dm.memory_limit, 'actor': actor.memory_limit},
            }))
            app.store.backup(staging / 'experience.sqlite3')
            dm.memory.backup(staging / 'decision_memory.json')
            actor.memory.backup(staging / 'exploration_memory.json')
            atomic_write(staging / 'decision_research.json', json_bytes(dm.research_agenda.export()))
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
                                'exploration_memory': 'exploration_memory.json',
                                'research_agenda': 'decision_research.json'}, random_state_refs=(),
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


def restore_snapshot(directory, target, *, run_id: str | None = None):
    """Restore to a new workspace; a new run_id forks with automatic ancestry."""
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
    identity = json.loads((directory / 'run.json').read_text())
    if any(identity.get(key) != value for key, value in {
            **config.source.to_dict(), 'objective': config.objective}.items()):
        raise ValueError('Snapshot run identity does not match configuration')
    if run_id is not None:
        source = config.source.child(run_id)
        config = replace(config, **source.to_dict())
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
    if type(runtime['review_tool_limit']) is not int or runtime['review_tool_limit'] < 0:
        raise ValueError('Invalid restored review tool limit')
    from rsisys.loops.decision_making.action_space.access import ActionPermissions
    permissions = ActionPermissions.from_dict(runtime['action_permissions'])
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
        for name in ('decision_memory.json', 'exploration_memory.json', 'decision_research.json'):
            shutil.copyfile(directory / name, staging / name)
        shutil.copytree(directory / 'artifacts', staging / 'artifacts')
        atomic_write(staging / 'run.json', json_bytes({**config.source.to_dict(), 'objective': config.objective}))
        atomic_write(staging / 'config.json', json_bytes(config.to_dict()))
        sync_tree(staging)
        staging.rename(target)
        sync_directory(target.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    app = Application(config, _restore=True, action_permissions=permissions)
    restore_research(app.runner.decision_making.harness,
                     json.loads((directory / 'decision_research.json').read_text()), runtime['context_compactor'])
    app.runner.decision_making.harness.state = decision
    app.runner.exploration.harness.state = actor
    app.runner.decision_making.harness.experience_limit = runtime['decision_experience_limit']
    app.runner.decision_making.harness.review_tool_limit = runtime['review_tool_limit']
    app.runner.decision_making.harness.memory_limit = runtime['memory_limits']['decision_maker']
    app.runner.exploration.harness.memory_limit = runtime['memory_limits']['actor']
    app.parent_snapshot_id = manifest.snapshot_id
    return app
