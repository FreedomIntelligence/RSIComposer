"""Persistent composition root. Loop business logic remains in its own module."""

from rsicomposer.runtime.capabilities import optional_method

import fcntl
import json
import os
from copy import deepcopy
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from uuid import uuid4

from rsicomposer.assembly import AssemblyContext, ModuleAssembly, read_saved_module_layout
from rsicomposer.loops.contracts.model_service import BackendBinding, ModelRef
from rsicomposer.loops.decision_making.action_space.access import ActionPermissions, restore_action_permissions
from rsicomposer.runtime.evidence.retrieval import BasicEvidenceRepository
from rsicomposer.runtime.evidence.sqlite import SQLiteExperienceStore
from rsicomposer.runtime.execution.local_process import LocalProcessBackend
from rsicomposer.runtime.execution.sandbox import MockSandboxBackend
from rsicomposer.runtime.orchestration.run_policy import RunPolicy
from rsicomposer.runtime.state.provenance import RunSource
from rsicomposer.runtime.storage.artifacts import FileArtifactStore
from rsicomposer.runtime.storage.files import atomic_write, json_bytes
import rsicomposer.snapshot.codec  # noqa: F401  registers built-in checkpoint state types


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
        from rsicomposer.runtime.execution.resources import validate_resources
        from rsicomposer.runtime.execution.budget import validate_budget
        validate_resources(self.resources)
        validate_budget(self.budget)
        RunSource(self.run_id, self.parent_run_id)
        from rsicomposer.backends.serving import is_serving_kind
        for key, backend in self.backends.items():
            if key != backend.backend_id or not is_serving_kind(backend.kind):
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
            if backend['engine_version'] is None:
                backend.pop('engine_version')
            if backend['enable_thinking'] is None:
                backend.pop('enable_thinking')
        return raw


class Application:
    def __init__(self, config: ApplicationConfig, *, modules: ModuleAssembly | None = None,
                 action_permissions: ActionPermissions | None = None, run_policy: RunPolicy | None = None,
                 cleanup_timeout_seconds: float = 30, durable: bool = False,
                 recovery_contract=None, _restore: bool = False, _resume: bool = False, _delegated_models=None,
                 _host_budget=None, _initial_controller_version=None, _inherited_controller_history=(),
                 _cli_launch: bytes | None = None, _initial_literature: tuple[Path, ...] = ()):
        if (_cli_launch is not None or _initial_literature) and (not durable or _restore or _resume):
            raise ValueError('CLI launch metadata requires a new durable run')
        if not isinstance(modules, ModuleAssembly):
            raise TypeError('modules must be a ModuleAssembly')
        has_lab = modules.has_port('lab')
        memory = modules.memory
        module_layout = modules.configuration()
        if modules.method_identity is not None:
            if durable and recovery_contract is None and modules.state_factory is None:
                raise TypeError('Methods require explicit state_factory or recovery_contract for durable=True')
        if modules.components is not None:
            if durable and recovery_contract is None and modules.state_factory is None:
                raise TypeError('Components require explicit state_factory or recovery_contract for durable=True')
        if memory is not None:
            if durable and recovery_contract is None and modules.state_factory is None:
                raise ValueError('Controller Memory requires an explicit recovery_contract for durable=True')
        if action_permissions is not None and not isinstance(action_permissions, ActionPermissions):
            raise TypeError('Expected ActionPermissions')
        if type(durable) is not bool:
            raise TypeError('durable must be boolean')
        if durable and has_lab and recovery_contract is None and modules.state_factory is None:
            raise TypeError('Lab execution requires explicit recovery integration; durable execution is unsupported')
        if recovery_contract is not None and not durable:
            raise ValueError('recovery_contract requires durable=True')
        if _resume and action_permissions is not None:
            raise ValueError('Cannot replace recovered action permissions')
        permissions = action_permissions if action_permissions is not None else ActionPermissions()
        if durable and not _resume:
            # Do not build a durable assembly that cannot reproduce its admission policy.
            permissions.to_dict()
        root = Path(config.data_dir).resolve()
        if root != config.data_dir:
            config = replace(config, data_dir=root)
        self.config = config
        self.modules = modules
        self.recovery = None
        self.assembly: AssemblyContext | None = None
        self._resources = ExitStack()
        self._closed = False
        self.store = None
        self.sandbox = None
        try:
            # Validate transports before touching storage. Keys are read only at call time.
            from rsicomposer.backends.serving import build_serving_backend, is_native_serving_kind
            self.backends = {}
            for key, binding in config.backends.items():
                route = (_delegated_models or {}).get('backends', {}).get(key)
                if route is not None:
                    binding = replace(binding, endpoint=route['endpoint'], credential_env=route['credential_env'])
                backend = build_serving_backend(binding, gateway=route is not None and (
                    is_native_serving_kind(binding.kind) or binding.kind in ('anthropic_messages', 'openai_chat_completions')),
                    models={model: model.model_id for model in config.roles.values() if model.backend_id == key})
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
            if durable:
                from rsicomposer.runtime.state.recovery import DurableRecovery
                if _resume:
                    checkpoint = DurableRecovery.read(root)
                    DurableRecovery.validate_workspace(checkpoint, config)
                    permissions = restore_action_permissions(checkpoint)
            self._action_permissions = permissions
            if _restore:
                if (root / 'lab_execution.json').exists() and (
                        not has_lab or (recovery_contract is None and modules.state_factory is None)):
                    raise TypeError('Lab execution requires explicit recovery integration; automatic restore is unsupported')
                if (root / 'research_assembly.json').exists() and memory is None:
                    # Registered Controllers construct Memory only after the
                    # restored workspace and permissions are established.
                    selection = (_initial_controller_version.config if _initial_controller_version is not None
                                 else modules.controller_selection)
                    definition = modules.controller_catalog.get(selection['name']) if selection is not None else None
                    if (definition is None or definition.memory is None
                            or (modules.registry_source is None and modules.method_identity is None)):
                        raise ValueError('Supply the original Controller Memory when restoring this workspace')
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
                action_permissions=self._action_permissions,
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
            if self.runner._declared_evaluations:
                module_layout['evaluation_operations'] = sorted(self.runner._declared_evaluations)
            saved_layout = read_saved_module_layout(root) if _restore else None
            if saved_layout is not None and saved_layout != module_layout:
                raise ValueError('Lab profile, primary metric or evaluation operations differ from saved method identity')
            atomic_write(root / 'module_assembly.json', json_bytes(module_layout))
            if self.assembly.lab_execution is not None:
                atomic_write(root / 'lab_execution.json', json_bytes({'restore': 'requires_explicit_integration'}))
            if not _restore:
                atomic_write(root / 'run.json', json_bytes({**config.source.to_dict(), 'objective': config.objective}))
                atomic_write(root / 'config.json', json_bytes(config.to_dict()))
            if durable:
                self.recovery = DurableRecovery(self, recovery_contract)
                if _resume:
                    self.recovery.load(checkpoint, run_policy)
                else:
                    # Publish the CLI's reconstruction input under the workspace
                    # lock, before the first checkpoint can advertise a valid run.
                    for path in _initial_literature:
                        self.artifacts.put_file(path, kind='literature')
                    if _cli_launch is not None:
                        atomic_write(root / 'framework.json', _cli_launch)
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
        raw = json.loads((root / 'config.json').read_text())
        config = ApplicationConfig.from_dict(raw, base_dir=root)
        if config.data_dir != root:
            if Path(raw['data_dir']).is_absolute():
                raise ValueError('Resume uses the original workspace; recovery is not a portable snapshot')
            # Older applications saved relative paths from their launch directory.
            # The locked checkpoint verifies the original absolute path before assembly.
            config = replace(config, data_dir=root)
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

    @contextmanager
    def _hold_workspace_lock(self):
        """Let the CLI publish the cleanup outcome under the original lock."""
        # flock is attached to the open file description. Keep a reference
        # through close() without reacquiring the lock in a race with resume.
        fd = os.dup(self._lock.fileno())
        try:
            yield
        finally:
            os.close(fd)

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
