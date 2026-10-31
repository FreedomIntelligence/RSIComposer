"""Python assembly hooks sharing resources owned by an application or demo."""

from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

from rsisys.experience.retrieval import ExperienceSystem
from rsisys.loops.contracts.model_service import ModelBackend, ModelRef, ModelUpdatable
from rsisys.loops.decision_making.action_space import DemoDecisionActionSpace, LocalDecisionActionSpace
from rsisys.loops.decision_making.action_space.directives import DirectiveDefinitions
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.learner.backends.mock import MockDecisionLearner
from rsisys.loops.decision_making.harness.memory import DecisionMemory
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.harness.tools import DemoDecisionTools
from rsisys.loops.decision_making.harness.tools.local import LocalDecisionTools
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.loops.exploration.action_space import EchoActionSpace
from rsisys.loops.exploration.action_space.local import CommandActionSpace
from rsisys.loops.exploration.harness.engine import DefaultExplorationHarness
from rsisys.loops.exploration.harness.commands import ExplorationCommands
from rsisys.loops.exploration.harness.environment import SandboxEchoEnvironment
from rsisys.loops.exploration.harness.environment.local import LocalProcessEnvironment
from rsisys.loops.exploration.harness.evaluation.mock import MockEvaluator
from rsisys.loops.exploration.harness.learner.backends.mock import MockActorLearner
from rsisys.loops.exploration.harness.memory import ExplorationMemory
from rsisys.loops.exploration.harness.policy import ModelActorPolicy
from rsisys.loops.exploration.loop import ExplorationLoop
from rsisys.runtime.artifacts import ArtifactStore
from rsisys.runtime.runner import Runner
from rsisys.runtime.sandbox import SandboxBackend


@dataclass
class AssemblyContext:
    experience: ExperienceSystem
    artifacts: ArtifactStore
    backends: Mapping[str, ModelBackend]
    sandbox: SandboxBackend
    roles: Mapping[str, ModelRef]
    data_dir: Path | None = None
    execution: str = "mock"
    with_learning: bool = False
    experience_limit: int = 32
    memory_limit: int = 32
    # Explicit executable modes/parsers supplied by an exploration factory.
    exploration_directives: DirectiveDefinitions | None = None
    _cleanup: ExitStack = field(default_factory=ExitStack, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def add_cleanup(self, callback: Callable[[], object]) -> None:
        """Register cleanup immediately after acquiring a factory-owned resource."""
        if self._closed:
            raise RuntimeError("Assembly is closed")
        if not callable(callback):
            raise TypeError("Cleanup must be callable")
        self._cleanup.callback(callback)

    def close(self) -> None:
        """Run every callback once, in reverse order, even if one raises."""
        if not self._closed:
            self._closed = True
            self._cleanup.close()


ExplorationFactory = Callable[[AssemblyContext], ExplorationLoop]
DecisionFactory = Callable[[AssemblyContext, ExplorationLoop], DecisionMakingLoop]


def build_default_exploration(context: AssemblyContext) -> ExplorationLoop:
    if context.execution == "local_process":
        space = CommandActionSpace()
        environment = LocalProcessEnvironment(context.sandbox, context.artifacts)
    else:
        space = EchoActionSpace()
        environment = SandboxEchoEnvironment(context.sandbox, context.artifacts)
    context.add_cleanup(environment.close)
    model = context.roles["actor"]
    memory_path = context.data_dir / "exploration_memory.json" if context.data_dir is not None else None
    return ExplorationLoop(
        DefaultExplorationHarness(
            ModelActorPolicy(context.backends[model.backend_id], model), environment, context.experience,
            learner=MockActorLearner(context.artifacts) if context.with_learning else None,
            evaluator=MockEvaluator(context.artifacts) if context.with_learning else None,
            memory=ExplorationMemory(memory_path), memory_limit=context.memory_limit,
        ), space,
    )


def _decision_directives(context: AssemblyContext, exploration: ExplorationLoop):
    """Infer only known built-in handlers; custom execution requires a declaration."""
    declared = context.exploration_directives
    if declared is not None and not isinstance(declared, DirectiveDefinitions):
        raise TypeError("exploration_directives must be a DirectiveDefinitions catalog")
    harness = exploration.harness
    if (type(harness) is not DefaultExplorationHarness
            or type(harness.commands) is not ExplorationCommands or harness.commands.loop is not harness):
        if declared is None:
            raise ValueError("Custom exploration harness requires decision_factory or explicit "
                             "context.exploration_directives")
        return declared, declared.modes()

    executable = callable(getattr(harness.policy, "act", None)) and all(
        callable(getattr(harness.environment, method, None)) for method in ("reset", "step", "close")
    )
    trainable = (harness.learner is not None and harness.training.backend is not None
                 and isinstance(getattr(harness.policy, "model", None), ModelRef))
    available = {
        "execute": executable,
        "configure_prompt": True,
        "sft": trainable,
        "rl": trainable and executable and isinstance(harness.policy, ModelUpdatable),
        "evaluate": harness.evaluation.backend is not None,
    }
    handlers = harness.commands.handlers
    if declared is not None:
        for mode in declared.modes():
            if mode not in handlers:
                raise ValueError(f"Declared directive has no exploration handler: {mode}")
            if mode in harness.commands.builtin_modes and not available.get(mode, False):
                raise ValueError(f"Declared directive lacks required exploration capabilities: {mode}")
        return declared, declared.modes()

    # Keep the default catalog identity for built-in snapshot compatibility.
    # Replaced/new handlers are only advertised through an explicit declaration.
    modes = tuple(mode for mode in handlers
                  if mode in harness.commands.builtin_modes and available.get(mode, False))
    return DirectiveDefinitions(), modes


def build_default_decision(context: AssemblyContext, exploration: ExplorationLoop) -> DecisionMakingLoop:
    directives, modes = _decision_directives(context, exploration)
    if context.data_dir is None:
        tools = DemoDecisionTools(context.sandbox, context.artifacts)
        space = DemoDecisionActionSpace(directive_modes=modes, directives=directives)
    else:
        tools = LocalDecisionTools(context.artifacts)
        space = LocalDecisionActionSpace(directive_modes=modes, directives=directives)
    model = context.roles["decision_maker"]
    memory_path = context.data_dir / "decision_memory.json" if context.data_dir is not None else None
    return DecisionMakingLoop(
        DefaultDecisionHarness(
            ModelDecisionPolicy(context.backends[model.backend_id], model,
                                allowed_actions=tuple(spec.name for spec in exploration.action_space.describe()),
                                directives=directives),
            tools, context.experience,
            learner=MockDecisionLearner(context.artifacts) if context.with_learning else None,
            experience_limit=context.experience_limit,
            memory=DecisionMemory(memory_path), memory_limit=context.memory_limit,
        ), space,
    )


class AssemblyRunner(Runner):
    """Demo runner owning its assembly callbacks; close explicitly or use with."""

    def __init__(self, decision: DecisionMakingLoop, exploration: ExplorationLoop, assembly: AssemblyContext):
        super().__init__(decision, exploration)
        self.assembly = assembly

    def close(self) -> None:
        self.assembly.close()

    def __enter__(self):
        if self.assembly._closed:
            raise RuntimeError("Assembly is closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f"Assembly cleanup also failed: {cleanup_error}")

    def step(self, objective: str):
        if self.assembly._closed:
            raise RuntimeError("Assembly is closed")
        return super().step(objective)
