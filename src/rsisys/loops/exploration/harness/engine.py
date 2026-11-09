"""Default assembly and compatibility API for exploration execution components."""

from collections.abc import Mapping
from rsisys.runtime.evidence.retrieval import ExperienceSystem
from rsisys.loops._memory import MemoryStore
from rsisys.loops.contracts.coordination import ExecutionView, ExplorationProgress
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.jobs import JobRef, JobOutcome
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.contracts.training import TrainingResult, ModelActivator
from rsisys.runtime.orchestration.jobs import JobBook
from ..action_space import ActionSpace
from .commands import ExplorationCommands, CommandHandler
from .environment import Environment, EpisodeExecution
from .evaluation import Evaluator, EvaluationRequest, EvaluationResult, EvaluationExecution
from .learner import ActorLearner, ActorTrainingRequest, LearningExecution
from .memory import ExplorationMemory
from .policy import ActorPolicy
from .rl import RLPlan, RLState, LocalRLExecution
from .state import ExplorationState


class DefaultExplorationHarness:
    """Coordinate command ownership and explicitly composed execution components."""

    def __init__(self, policy: ActorPolicy, environment: Environment,
                 experience: ExperienceSystem, learner: ActorLearner | None = None,
                 evaluator: Evaluator | None = None, *, memory: MemoryStore | None = None, memory_limit: int = 32,
                 command_handlers: Mapping[str, CommandHandler] | None = None,
                 model_activator: ModelActivator | None = None):
        if type(memory_limit) is not int or memory_limit < 1:
            raise ValueError("memory_limit must be a positive integer")
        state = ExplorationState()
        memory = memory if memory is not None else ExplorationMemory()
        self.episodes = EpisodeExecution(policy, environment, memory, experience, state,
                                         memory_limit=memory_limit)
        self.learning = LearningExecution(policy, memory, experience, state, JobBook(learner), model_activator)
        self.evaluations = EvaluationExecution(experience, state, JobBook(evaluator))
        self.local_rl = LocalRLExecution(self.episodes, self.learning, state)
        self.commands = ExplorationCommands(self, command_handlers)

    @property
    def episodes(self):
        return self._episodes

    @episodes.setter
    def episodes(self, value):
        self._episodes = value
        if hasattr(self, 'local_rl'):
            self.local_rl.episodes = value

    @property
    def learning(self):
        return self._learning

    @learning.setter
    def learning(self, value):
        self._learning = value
        if hasattr(self, 'local_rl'):
            self.local_rl.learning = value

    # Preserve mutable public dependencies, including snapshot state replacement.
    # Execution components retain only their declared dependencies, never the harness.
    @property
    def state(self):
        return self.episodes.state

    @state.setter
    def state(self, value):
        self.episodes.state = self.learning.state = self.evaluations.state = self.local_rl.state = value

    @property
    def policy(self):
        return self.episodes.policy

    @policy.setter
    def policy(self, value):
        self.episodes.policy = self.learning.policy = value

    @property
    def environment(self):
        return self.episodes.environment

    @environment.setter
    def environment(self, value):
        self.episodes.environment = value

    @property
    def memory(self):
        return self.episodes.memory

    @memory.setter
    def memory(self, value):
        self.episodes.memory = self.learning.memory = value

    @property
    def experience(self):
        return self.episodes.experience

    @experience.setter
    def experience(self, value):
        self.episodes.experience = self.learning.experience = self.evaluations.experience = value

    @property
    def memory_limit(self):
        return self.episodes.memory_limit

    @memory_limit.setter
    def memory_limit(self, value):
        self.episodes.memory_limit = value

    @property
    def model_activator(self):
        return self.learning.model_activator

    @model_activator.setter
    def model_activator(self, value):
        self.learning.model_activator = value

    @property
    def training(self):
        return self.learning.jobs

    @training.setter
    def training(self, value):
        self.learning.jobs = value

    @property
    def evaluation(self):
        return self.evaluations.jobs

    @evaluation.setter
    def evaluation(self, value):
        self.evaluations.jobs = value

    @property
    def learner(self):
        return self.learning.jobs.backend

    @learner.setter
    def learner(self, value):
        self.learning.jobs.backend = value

    @property
    def command_busy(self) -> bool:
        return self.commands.busy

    def accept(self, directive):
        return self.commands.accept(directive)

    def advance(self, directive_id, action_space: ActionSpace):
        return self.commands.advance(directive_id, action_space)

    def inspect(self, directive_id):
        return self.commands.inspect(directive_id)

    def pending(self) -> ExplorationProgress | None:
        """Return the unacknowledged receipt, including terminal feedback."""
        active = self.commands.active
        return self.commands.inspect(active) if active is not None else None

    def execution_view(self) -> ExecutionView:
        model = getattr(self.policy, 'model', None)
        return ExecutionView(model if isinstance(model, ModelRef) else None, self.pending())

    def describe_queries(self):
        from .queries import ExplorationQueries
        return ExplorationQueries(self).describe_queries()

    def query(self, request):
        from .queries import ExplorationQueries
        return ExplorationQueries(self).query(request)

    def update(self, request):
        return self.commands.update(request)

    def cancel(self, directive_id):
        return self.commands.cancel(directive_id)

    def acknowledge(self, directive_id):
        self.commands.acknowledge(directive_id)

    def _require_local_control(self):
        if self.command_busy:
            raise RuntimeError('A directive owns exploration; advance or cancel it through its lifecycle')

    def run(self, directive: Directive, action_space: ActionSpace) -> Feedback:
        self._require_local_control()
        if self.state.rl is not None and not self.state.rl.terminal:
            raise RuntimeError("An RL cycle is active; advance it with step_rl")
        if self.training.busy:
            raise RuntimeError("A training job is still active")
        return self.episodes.run(directive, action_space)

    def submit_learning(self, request: ActorTrainingRequest, *, idempotency_key: str) -> JobRef:
        """Submit local learning explicitly; no cross-loop scheduling is implied."""
        self._require_local_control()
        if self.state.rl is not None and not self.state.rl.terminal:
            raise RuntimeError("An RL cycle owns the learner; advance it with step_rl")
        return self.learning.submit(request, idempotency_key=idempotency_key)

    def apply_learning(self, job: JobRef) -> ModelRef:
        """Explicitly bind a learned version to this policy only."""
        self._require_local_control()
        if self.state.rl is not None and not self.state.rl.terminal:
            raise RuntimeError("An RL cycle owns model updates; advance it with step_rl")
        return self.learning.apply(job)

    def apply_candidate(self, candidate_id: str) -> ModelRef:
        """Explicit local selection; evaluation acceptance belongs to the caller."""
        self._require_local_control()
        return self.learning.apply_candidate(candidate_id)

    def start_rl(self, plan: RLPlan) -> RLState:
        self._require_local_control()
        return self.local_rl.start(plan)

    def step_rl(self, action_space: ActionSpace) -> RLState:
        self._require_local_control()
        return self.local_rl.step(action_space)

    def cancel_rl(self) -> RLState:
        self._require_local_control()
        return self.local_rl.cancel()

    def submit_evaluation(self, request: EvaluationRequest, *, idempotency_key: str) -> JobRef:
        self._require_local_control()
        return self.evaluations.submit(request, idempotency_key=idempotency_key)

    def poll_learning(self, job: JobRef) -> JobOutcome[TrainingResult]:
        return self.learning.poll(job)

    def cancel_learning(self, job: JobRef) -> JobOutcome[TrainingResult]:
        return self.learning.cancel(job)

    def candidate_result(self, candidate_id: str) -> TrainingResult:
        return self.learning.candidate_result(candidate_id)

    def apply_candidate_for_command(self, candidate_id: str, *, directive_id: str) -> ModelRef:
        return self.commands.apply_candidate(candidate_id, directive_id=directive_id)

    def poll_evaluation(self, job: JobRef) -> JobOutcome[EvaluationResult]:
        return self.evaluations.poll(job)

    def cancel_evaluation(self, job: JobRef) -> JobOutcome[EvaluationResult]:
        return self.evaluations.cancel(job)
