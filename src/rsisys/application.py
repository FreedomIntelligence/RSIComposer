"""Persistent composition root. Loop business logic remains in its own module."""

from rsisys.runtime.capabilities import optional_method

import fcntl
import json
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from uuid import uuid4

from rsisys.assembly import AssemblyContext, ModuleAssembly
from rsisys.loops.contracts.model_service import BackendBinding, ModelRef
from rsisys.loops.decision_making.action_space.access import ActionPermissions
from rsisys.runtime.evidence.retrieval import BasicEvidenceRepository
from rsisys.runtime.evidence.sqlite import SQLiteExperienceStore
from rsisys.runtime.execution.local_process import LocalProcessBackend
from rsisys.runtime.execution.sandbox import MockSandboxBackend
from rsisys.runtime.orchestration.run_policy import RunPolicy
from rsisys.runtime.state.provenance import RunSource
from rsisys.runtime.storage.artifacts import FileArtifactStore
from rsisys.runtime.storage.files import atomic_write, json_bytes
import rsisys.snapshot.codec  # noqa: F401  registers built-in checkpoint state types


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
    resources: dict = field(default_factory=dict)
    budget: dict = field(default_factory=dict)

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
        from rsisys.runtime.execution.resources import validate_resources
        from rsisys.runtime.execution.budget import validate_budget
        validate_resources(self.resources)
        validate_budget(self.budget)
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
    def __init__(self, config: ApplicationConfig, *, modules: ModuleAssembly | None = None,
                 action_permissions: ActionPermissions | None = None, run_policy: RunPolicy | None = None,
                 cleanup_timeout_seconds: float = 30, durable: bool = False,
                 recovery_contract=None, _restore: bool = False, _resume: bool = False, _delegated_models=None,
                 _host_budget=None, _initial_controller_version=None, _inherited_controller_history=()):
        if not isinstance(modules, ModuleAssembly):
            raise TypeError('modules must be a ModuleAssembly')
        has_lab = modules.has_port('lab')
        harness = modules.harness
        module_layout = modules.configuration()
        if modules.method_identity is not None:
            if durable and recovery_contract is None and modules.state_factory is None:
                raise TypeError('Methods require explicit state_factory or recovery_contract for durable=True')
        if modules.components is not None:
            if durable and recovery_contract is None and modules.state_factory is None:
                raise TypeError('Components require explicit state_factory or recovery_contract for durable=True')
        if harness is not None:
            if durable and recovery_contract is None and modules.state_factory is None:
                raise ValueError('A controller harness requires an explicit recovery_contract for durable=True')
        if action_permissions is not None and not isinstance(action_permissions, ActionPermissions):
            raise TypeError('Expected ActionPermissions')
        if type(durable) is not bool:
            raise TypeError('durable must be boolean')
        if durable and has_lab and recovery_contract is None and modules.state_factory is None:
            raise TypeError('Lab execution requires explicit recovery integration; durable execution is unsupported')
        if recovery_contract is not None and not durable:
            raise ValueError('recovery_contract requires durable=True')
        self.config = config
        self.modules = modules
        self.recovery = None
        self.assembly: AssemblyContext | None = None
        self._resources = ExitStack()
        self._closed = False
        self.store = None
        self.sandbox = None
        root = config.data_dir
        try:
            # Validate transports before touching storage. Keys are read only at call time.
            from rsisys.backends.serving.chat_completions import (
                MockModelBackend,
                RemoteModelBackend,
            )
            self.backends = {}
            for key, binding in config.backends.items():
                if _delegated_models is not None and key in _delegated_models.get('backends', {}):
                    route = _delegated_models['backends'][key]
                    binding = replace(binding, endpoint=route['endpoint'], credential_env=route['credential_env'])
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
                        not has_lab or (recovery_contract is None and modules.state_factory is None)):
                    raise TypeError('Lab execution requires explicit recovery integration; automatic restore is unsupported')
                if (root / 'research_assembly.json').exists() and harness is None:
                    raise ValueError('Supply the original controller harness when restoring this workspace')
                modules.validate_saved(root)
                existing = json.loads((root / 'run.json').read_text())
                if existing['run_id'] != config.run_id or existing['objective'] != config.objective:
                    raise ValueError('Restored run identity does not match configuration')
                if existing.get('parent_run_id') != config.parent_run_id:
                    raise ValueError('Restored run source does not match configuration')
            self.store = SQLiteExperienceStore(root / 'experience.sqlite3')
            self._resources.callback(self.store.close)
            self.experience = BasicEvidenceRepository(self.store, run_id=config.run_id)
            self.artifacts = FileArtifactStore(root / 'artifacts')
            if config.execution == 'local_process':
                self.sandbox = LocalProcessBackend(root / 'workspaces')
            else:
                self.sandbox = MockSandboxBackend()
            self.assembly = AssemblyContext(
                experience=self.experience, artifacts=self.artifacts, backends=self.backends,
                sandbox=self.sandbox, roles=dict(config.roles), data_dir=root, execution=config.execution,
                with_learning=config.learning == 'mock', experience_limit=config.experience_limit,
                memory_limit=config.memory_limit, restoring=_restore, resuming=_resume,
                action_permissions=action_permissions if action_permissions is not None else ActionPermissions(),
            )
            self.assembly._delegated_models = _delegated_models
            self.assembly.budget = _host_budget
            self.assembly._initial_controller_version = _initial_controller_version
            self.assembly._inherited_controller_history = deepcopy(tuple(_inherited_controller_history))
            self.assembly.run_id, self.assembly.parent_run_id = config.run_id, config.parent_run_id
            self.assembly.host_config = {'application': config.to_dict(), 'registry_source': modules.registry_source}
            self.assembly._resource_config = config.resources
            self.assembly._budget_config = config.budget
            # A shared executor can own work beyond the active Lab/environment.
            # Failed close must retain its callback, storage and workspace lock.
            if callable(getattr(self.sandbox, 'close_all', None)):
                self.assembly.add_owned_cleanup(self.sandbox.close_all)
            self.runner = modules.build(self.assembly, run_policy=run_policy,
                                        cleanup_timeout_seconds=cleanup_timeout_seconds)
            profile_identity = optional_method(self.assembly.lab_execution, 'profile_identity')
            declared = profile_identity() if profile_identity is not None else {}
            if declared:
                module_layout['lab_profiles'] = declared
            if _restore and (root / 'module_assembly.json').exists():
                if json.loads((root / 'module_assembly.json').read_text()) != module_layout:
                    raise ValueError('Lab profile or primary metric differs from saved method identity')
            if any(module_layout.values()):
                atomic_write(root / 'module_assembly.json', json_bytes(module_layout))
            if self.assembly.lab_execution is not None:
                atomic_write(root / 'lab_execution.json', json_bytes({'restore': 'requires_explicit_integration'}))
            if not _restore:
                atomic_write(root / 'run.json', json_bytes({**config.source.to_dict(), 'objective': config.objective}))
                atomic_write(root / 'config.json', json_bytes(config.to_dict()))
            if durable:
                from rsisys.runtime.state.recovery import DurableRecovery
                self.recovery = DurableRecovery(self, recovery_contract)
                if _resume:
                    self.recovery.load(run_policy)
                else:
                    self.recovery.initialize()
                self.runner.recovery = self.recovery
                execution = self.assembly.host_execution
                if self.assembly.lab_execution is not None:
                    execution.in_flight_hook = self.recovery.commit_in_flight
        except BaseException as exc:
            try:
                self.close()
            except BaseException as cleanup_error:
                exc.add_note(f"Application cleanup also failed: {cleanup_error}")
            raise

    @classmethod
    def resume(cls, data_dir, *, modules=None, recovery_contract=None, run_policy=None):
        """Reopen the same durable workspace; uncertain effects remain blocked.

        Factories must reconstruct resources without starting work. The workspace
        lock is acquired before checkpoint state is read/applied by the constructor.
        """
        root = Path(data_dir).resolve()
        config = ApplicationConfig.load(root / 'config.json')
        if config.data_dir.resolve() != root:
            raise ValueError('Resume uses the original workspace; recovery is not a portable snapshot')
        return cls(config, modules=modules, recovery_contract=recovery_contract, run_policy=run_policy,
                   durable=True, _restore=True, _resume=True)

    def run(self, *, rounds: int | None = None, max_decision_steps: int | None = None,
            wait_for_jobs: bool = False):
        if self._closed:
            raise RuntimeError('Application is closed')
        return self.runner.run(self.config.objective, self.config.rounds if rounds is None else rounds,
                               max_decision_steps=(self.config.max_decision_steps
                                                   if max_decision_steps is None else max_decision_steps),
                               wait_for_jobs=wait_for_jobs)

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
