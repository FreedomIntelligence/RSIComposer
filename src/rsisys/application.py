"""Persistent composition root. Loop business logic remains in its own module."""

from dataclasses import asdict, dataclass, field
import fcntl
import json
from pathlib import Path
from uuid import uuid4

from rsisys.experience.sqlite import SQLiteExperienceStore
from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.loops.contracts.model_service import BackendBinding, ModelRef
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.loops.decision_making.harness.memory import DecisionMemory
from rsisys.loops.exploration.harness.memory import ExplorationMemory
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.action_space import LocalDecisionActionSpace
from rsisys.loops.decision_making.harness.tools.local import LocalDecisionTools
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.exploration.harness.engine import DefaultExplorationHarness
from rsisys.loops.decision_making.harness.learner.backends.mock import MockDecisionLearner
from rsisys.loops.exploration.loop import ExplorationLoop
from rsisys.loops.exploration.harness.policy import ModelActorPolicy
from rsisys.loops.exploration.action_space import EchoActionSpace
from rsisys.loops.exploration.harness.environment import SandboxEchoEnvironment
from rsisys.loops.exploration.harness.environment.local import LocalProcessEnvironment
from rsisys.loops.exploration.action_space.local import CommandActionSpace
from rsisys.loops.exploration.harness.learner.backends.mock import MockActorLearner
from rsisys.loops.exploration.harness.evaluation.mock import MockEvaluator
from rsisys.runtime.artifacts import FileArtifactStore
from rsisys.runtime.model import MockModelBackend, RemoteModelBackend
from rsisys.runtime.sandbox import MockSandboxBackend
from rsisys.runtime.local_process import LocalProcessBackend
from rsisys.runtime.runner import Runner
from rsisys.runtime.files import atomic_write, json_bytes


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

    def __post_init__(self):
        if not isinstance(self.objective, str) or not self.objective.strip():
            raise ValueError('An objective is required')
        if self.execution not in ('mock', 'local_process') or self.learning not in ('none', 'mock'):
            raise ValueError('Unsupported execution or learning backend')
        if type(self.rounds) is not int or self.rounds < 1 or type(self.experience_limit) is not int or self.experience_limit < 1:
            raise ValueError('rounds and experience_limit must be positive integers')
        if type(self.memory_limit) is not int or self.memory_limit < 1:
            raise ValueError('memory_limit must be a positive integer')
        if set(self.roles) != {'decision_maker', 'actor'}:
            raise ValueError('Both decision_maker and actor role bindings are required')
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError('run_id is required')
        for key, backend in self.backends.items():
            if key != backend.backend_id or backend.kind not in ('mock', 'http', 'chat_completions'):
                raise ValueError('Invalid backend configuration')
        for model in self.roles.values():
            if model.backend_id not in self.backends:
                raise ValueError('Model refers to an unknown backend')

    @classmethod
    def from_dict(cls, raw: dict, *, base_dir: Path):
        raw = dict(raw)
        if raw.pop('mode', 'local') != 'local':
            raise ValueError('Expected local application mode')
        path = Path(raw.pop('data_dir'))
        data_dir = (base_dir / path).resolve() if not path.is_absolute() else path.resolve()
        backends = {key: BackendBinding(backend_id=key, **value) for key, value in raw.pop('backends').items()}
        roles = {key: ModelRef(**value) for key, value in raw.pop('roles').items()}
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
    def __init__(self, config: ApplicationConfig, *, _restore: bool = False):
        self.config = config
        root = config.data_dir
        # Validate transports before touching storage. Keys themselves are read only at call time.
        backends = {key: MockModelBackend() if binding.kind == 'mock' else RemoteModelBackend(binding)
                    for key, binding in config.backends.items()}
        root.mkdir(parents=True, exist_ok=True)
        self._lock = (root / '.lock').open('a+b')
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise RuntimeError('This workspace is already open in another process') from None
        self.store = None
        self.sandbox = None
        try:
            contents = {p.name for p in root.iterdir()} - {'.lock'}
            if contents and not _restore:
                raise ValueError('Workspace is not empty; resume a snapshot into a new directory')
            if _restore:
                existing = json.loads((root / 'run.json').read_text())
                if existing['run_id'] != config.run_id or existing['objective'] != config.objective:
                    raise ValueError('Restored run identity does not match configuration')
            self.store = SQLiteExperienceStore(root / 'experience.sqlite3')
            self.experience = BasicExperienceSystem(self.store, run_id=config.run_id)
            self.artifacts = FileArtifactStore(root / 'artifacts')
            if config.execution == 'local_process':
                self.sandbox = LocalProcessBackend(root / 'workspaces')
                space, environment = CommandActionSpace(), LocalProcessEnvironment(self.sandbox, self.artifacts)
            else:
                self.sandbox = MockSandboxBackend()
                space, environment = EchoActionSpace(), SandboxEchoEnvironment(self.sandbox, self.artifacts)
            dm, actor = config.roles['decision_maker'], config.roles['actor']
            modes = ("execute", "configure_prompt") + (("sft", "rl", "evaluate") if config.learning == 'mock' else ())
            decision = DecisionMakingLoop(
                DefaultDecisionHarness(
                    ModelDecisionPolicy(backends[dm.backend_id], dm,
                                        allowed_actions=tuple(spec.name for spec in space.describe())),
                    LocalDecisionTools(self.artifacts), self.experience,
                    learner=MockDecisionLearner(self.artifacts) if config.learning == 'mock' else None,
                    experience_limit=config.experience_limit,
                    memory=DecisionMemory(root / "decision_memory.json"), memory_limit=config.memory_limit,
                ), LocalDecisionActionSpace(directive_modes=modes),
            )
            exploration = ExplorationLoop(
                DefaultExplorationHarness(
                    ModelActorPolicy(backends[actor.backend_id], actor), environment, self.experience,
                    learner=MockActorLearner(self.artifacts) if config.learning == 'mock' else None,
                    evaluator=MockEvaluator(self.artifacts) if config.learning == 'mock' else None,
                    memory=ExplorationMemory(root / "exploration_memory.json"), memory_limit=config.memory_limit,
                ), space,
            )
            self.runner = Runner(decision, exploration)
            if not _restore:
                atomic_write(root / 'run.json', json_bytes({'run_id': config.run_id, 'objective': config.objective}))
                atomic_write(root / 'config.json', json_bytes(config.to_dict()))
        except BaseException as exc:
            try:
                self.close()
            except Exception as cleanup_error:
                exc.add_note(f"Application cleanup also failed: {cleanup_error}")
            raise

    def run(self, *, rounds: int | None = None, max_decision_steps: int | None = None):
        return self.runner.run(self.config.objective, self.config.rounds if rounds is None else rounds,
                               max_decision_steps=max_decision_steps)

    def close(self):
        try:
            if self.sandbox is not None and hasattr(self.sandbox, 'close_all'):
                self.sandbox.close_all()
        finally:
            try:
                if self.store is not None:
                    self.store.close()
                    self.store = None
            finally:
                if not self._lock.closed:
                    fcntl.flock(self._lock, fcntl.LOCK_UN)
                    self._lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
