"""Persistent composition root. Loop business logic remains in its own module."""

from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
import fcntl
import json
from pathlib import Path
from uuid import uuid4

from rsisys.assembly import (
    AssemblyContext, DecisionFactory, ExplorationFactory, LabExecutionFactory, assemble_runner, module_assembly,
)
from rsisys.experience.sqlite import SQLiteExperienceStore
from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.loops.contracts.model_service import BackendBinding, ModelRef
from rsisys.loops.decision_making.action_space.access import ActionPermissions
from rsisys.runtime.run_policy import RunPolicy
from rsisys.runtime.artifacts import FileArtifactStore
from rsisys.runtime.model import MockModelBackend, RemoteModelBackend
from rsisys.runtime.sandbox import MockSandboxBackend
from rsisys.runtime.local_process import LocalProcessBackend
from rsisys.runtime.files import atomic_write, json_bytes
from rsisys.runtime.provenance import RunSource


@dataclass(frozen=True)
class ApplicationConfig:
    objective: str
    data_dir: Path
    backends: dict[str, BackendBinding]
    roles: dict[str, ModelRef]
    execution: str = 'mock'
    learning: str = 'none'
    rounds: int = 1
    experience_limit: int = 32
    memory_limit: int = 32
    run_id: str = field(default_factory=lambda: uuid4().hex)
    parent_run_id: str | None = None
    max_decision_steps: int | None = None

    def __post_init__(self):
        if not isinstance(self.objective, str) or not self.objective.strip():
            raise ValueError('An objective is required')
        if self.execution not in ('mock', 'local_process') or self.learning not in ('none', 'mock'):
            raise ValueError('Unsupported execution or learning backend')
        if type(self.rounds) is not int or self.rounds < 1 or type(self.experience_limit) is not int or self.experience_limit < 1:
            raise ValueError('rounds and experience_limit must be positive integers')
        if type(self.memory_limit) is not int or self.memory_limit < 1:
            raise ValueError('memory_limit must be a positive integer')
        if self.max_decision_steps is not None and (
                type(self.max_decision_steps) is not int or self.max_decision_steps < 1):
            raise ValueError('max_decision_steps must be a positive integer or None')
        if set(self.roles) - {'decision_maker', 'actor'}:
            raise ValueError('Unknown model role; use decision_maker or actor')
        RunSource(self.run_id, self.parent_run_id)
        for key, backend in self.backends.items():
            if key != backend.backend_id or backend.kind not in ('mock', 'http', 'chat_completions'):
                raise ValueError('Invalid backend configuration')
        for model in self.roles.values():
            if model.backend_id not in self.backends:
                raise ValueError('Model refers to an unknown backend')

    @property
    def source(self) -> RunSource:
        return RunSource(self.run_id, self.parent_run_id)

    @classmethod
    def from_dict(cls, raw: dict, *, base_dir: Path):
        raw = dict(raw)
        if raw.pop('mode', 'local') != 'local':
            raise ValueError('Expected local application mode')
        path = Path(raw.pop('data_dir'))
        data_dir = (base_dir / path).resolve() if not path.is_absolute() else path.resolve()
        backends = {key: BackendBinding(backend_id=key, **value) for key, value in raw.pop('backends', {}).items()}
        roles = {key: ModelRef(**value) for key, value in raw.pop('roles', {}).items()}
        return cls(data_dir=data_dir, backends=backends, roles=roles, **raw)

    @classmethod
    def load(cls, path: Path):
        return cls.from_dict(json.loads(path.read_text()), base_dir=path.resolve().parent)

    def to_dict(self):
        raw = asdict(self)
        raw['data_dir'] = str(self.data_dir)
        raw['mode'] = 'local'
        for backend in raw['backends'].values():
            backend.pop('backend_id')
        return raw


class Application:
    def __init__(self, config: ApplicationConfig, *, decision_factory: DecisionFactory | None = None,
                 exploration_factory: ExplorationFactory | None = None,
                 lab_factory: LabExecutionFactory | None = None,
                 action_permissions: ActionPermissions | None = None, run_policy: RunPolicy | None = None,
                 cleanup_timeout_seconds: float = 30, durable: bool = False,
                 recovery_contract=None, research=None, research_backend=None,
                 feedback_adapter_factory=None, modules=None, _restore: bool = False, _resume: bool = False):
        modules = module_assembly(modules, decision_factory=decision_factory,
            exploration_factory=exploration_factory, lab_factory=lab_factory,
            research=research, research_backend=research_backend, feedback_adapter_factory=feedback_adapter_factory)
        decision_factory, exploration_factory, lab_factory = (
            modules.decision_factory, modules.exploration_factory, modules.lab_factory)
        research = modules.research
        module_layout = modules.configuration()
        if modules.method_identity is not None:
            if durable and recovery_contract is None and modules.state_factory is None:
                raise TypeError('Methods require explicit state_factory or recovery_contract for durable=True')
        if modules.components is not None:
            if durable and recovery_contract is None and modules.state_factory is None:
                raise TypeError('Components require explicit state_factory or recovery_contract for durable=True')
        if research is not None:
            if durable and recovery_contract is None and modules.state_factory is None:
                raise ValueError('Research assembly requires an explicit recovery_contract for durable=True')
        if action_permissions is not None and not isinstance(action_permissions, ActionPermissions):
            raise TypeError('Expected ActionPermissions')
        if type(durable) is not bool:
            raise TypeError('durable must be boolean')
        if durable and lab_factory is not None and recovery_contract is None and modules.state_factory is None:
            raise TypeError('Lab execution requires explicit recovery integration; durable execution is unsupported')
        if recovery_contract is not None and not durable:
            raise ValueError('recovery_contract requires durable=True')
        self.config = config
        self.modules = modules
        self.custom_assembly = (decision_factory is not None or exploration_factory is not None
                                or lab_factory is not None or modules.components is not None
                                or modules.method_identity is not None)
        self.custom_exploration = exploration_factory is not None
        self.recovery = None
        self.assembly: AssemblyContext | None = None
        self._resources = ExitStack()
        self._closed = False
        self.store = None
        self.sandbox = None
        root = config.data_dir
        try:
            # Validate transports before touching storage. Keys are read only at call time.
            self.backends = {}
            for key, binding in config.backends.items():
                backend = MockModelBackend() if binding.kind == 'mock' else RemoteModelBackend(binding)
                self.backends[key] = backend
                if callable(getattr(backend, 'close', None)):
                    self._resources.callback(backend.close)
            root.mkdir(parents=True, exist_ok=True)
            self._lock = (root / '.lock').open('a+b')
            self._resources.callback(self._lock.close)
            try:
                fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('This workspace is already open in another process') from None
            contents = {p.name for p in root.iterdir()} - {'.lock'}
            if contents and not _restore:
                raise ValueError('Workspace is not empty; use Application.resume for durable runs or restore a snapshot')
            if _restore:
                if (root / 'lab_execution.json').exists() and (
                        lab_factory is None or (recovery_contract is None and modules.state_factory is None)):
                    raise TypeError('Lab execution requires explicit recovery integration; automatic restore is unsupported')
                if (root / 'research_assembly.json').exists() and research is None:
                    raise ValueError('Supply the original research assembly when restoring this workspace')
                marker = root / 'module_assembly.json'
                if marker.exists() and json.loads(marker.read_text()) != module_layout:
                    raise TypeError('Supply the original module assembly when restoring this workspace')
                existing = json.loads((root / 'run.json').read_text())
                if existing['run_id'] != config.run_id or existing['objective'] != config.objective:
                    raise ValueError('Restored run identity does not match configuration')
                if existing.get('parent_run_id') != config.parent_run_id:
                    raise ValueError('Restored run source does not match configuration')
            self.store = SQLiteExperienceStore(root / 'experience.sqlite3')
            self._resources.callback(self.store.close)
            self.experience = BasicExperienceSystem(self.store, run_id=config.run_id)
            self.artifacts = FileArtifactStore(root / 'artifacts')
            if config.execution == 'local_process':
                self.sandbox = LocalProcessBackend(root / 'workspaces')
            else:
                self.sandbox = MockSandboxBackend()
            self.assembly = AssemblyContext(
                experience=self.experience, artifacts=self.artifacts, backends=self.backends,
                sandbox=self.sandbox, roles=dict(config.roles), data_dir=root, execution=config.execution,
                with_learning=config.learning == 'mock', experience_limit=config.experience_limit,
                memory_limit=config.memory_limit,
                action_permissions=action_permissions if action_permissions is not None else ActionPermissions(),
            )
            # A shared executor can own work beyond the active Lab/environment.
            # Failed close must retain its callback, storage and workspace lock.
            if callable(getattr(self.sandbox, 'close_all', None)):
                self.assembly.add_owned_cleanup(self.sandbox.close_all)
            self.runner = assemble_runner(self.assembly, modules=modules, run_policy=run_policy,
                                          cleanup_timeout_seconds=cleanup_timeout_seconds)
            if any(module_layout.values()):
                atomic_write(root / 'module_assembly.json', json_bytes(module_layout))
            if self.assembly.lab_execution is not None:
                atomic_write(root / 'lab_execution.json', json_bytes({'restore': 'requires_explicit_integration'}))
            if not _restore:
                atomic_write(root / 'run.json', json_bytes({**config.source.to_dict(), 'objective': config.objective}))
                atomic_write(root / 'config.json', json_bytes(config.to_dict()))
            if durable:
                from rsisys.runtime.recovery import DurableRecovery
                self.recovery = DurableRecovery(self, recovery_contract)
                if _resume:
                    self.recovery.load(run_policy)
                else:
                    self.recovery.initialize()
                self.runner.recovery = self.recovery
        except BaseException as exc:
            try:
                self.close()
            except BaseException as cleanup_error:
                exc.add_note(f"Application cleanup also failed: {cleanup_error}")
            raise

    @classmethod
    def resume(cls, data_dir, *, decision_factory=None, exploration_factory=None,
               lab_factory=None,
               recovery_contract=None, run_policy=None, research=None, research_backend=None,
               feedback_adapter_factory=None, modules=None):
        """Reopen the same durable workspace; uncertain effects remain blocked.

        Factories must reconstruct resources without starting work. The workspace
        lock is acquired before checkpoint state is read/applied by the constructor.
        """
        root = Path(data_dir).resolve()
        config = ApplicationConfig.load(root / 'config.json')
        if config.data_dir.resolve() != root:
            raise ValueError('Resume uses the original workspace; recovery is not a portable snapshot')
        return cls(config, decision_factory=decision_factory, exploration_factory=exploration_factory,
                   lab_factory=lab_factory,
                   recovery_contract=recovery_contract, run_policy=run_policy,
                   research=research, research_backend=research_backend,
                   feedback_adapter_factory=feedback_adapter_factory, modules=modules,
                   durable=True, _restore=True, _resume=True)

    def run(self, *, rounds: int | None = None, max_decision_steps: int | None = None):
        if self._closed:
            raise RuntimeError('Application is closed')
        return self.runner.run(self.config.objective, self.config.rounds if rounds is None else rounds,
                               max_decision_steps=(self.config.max_decision_steps
                                                   if max_decision_steps is None else max_decision_steps))

    def controller_report(self):
        """Read optional method-owned progress, distinct from transport lifecycle.

        The host validates JSON transport only; it does not infer success from a
        finish decision or select models from the method's reported metrics.
        """
        report = self.assembly.controller_report if self.assembly is not None else None
        if report is None:
            return None
        value = report()
        if not isinstance(value, dict):
            raise TypeError('Controller report must return a JSON object')
        return json.loads(json.dumps(value, allow_nan=False))

    def close(self):
        self._closed = True
        try:
            if self.assembly is not None:
                self.assembly.close()
        finally:
            # Keep the lock/store/backends available while an owned Lab close
            # is pending. ExitStack cannot retain a callback that has raised.
            if self.assembly is None or not self.assembly.cleanup_pending:
                try:
                    self._resources.close()
                finally:
                    self.store = None

    def __enter__(self):
        if self._closed:
            raise RuntimeError('Application is closed')
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f"Application cleanup also failed: {cleanup_error}")
