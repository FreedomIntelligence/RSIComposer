"""Python assembly hooks sharing resources owned by an application or demo."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from rsisys.components import ComponentGraph, ComponentInstances
from rsisys.methods import MethodIdentity
from rsisys.experience.retrieval import ExperienceSystem
from rsisys.loops.contracts.execution import LabExecution
from rsisys.loops.contracts.model_service import ModelBackend, ModelRef, ModelUpdatable
from rsisys.loops.contracts.queries import QueryProvider
from rsisys.loops.contracts.backend import BackendCapabilityProvider
from rsisys.loops.contracts.training import ResumeMode
from rsisys.loops.decision_making.action_space import (
    DemoDecisionActionSpace, LocalDecisionActionSpace, ModularDecisionActionSpace,
)
from rsisys.loops.decision_making.action_space.access import ActionPermissions
from rsisys.loops.decision_making.action_space.directives import DirectiveDefinitions
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.learner.backends.mock import MockDecisionLearner
from rsisys.loops.decision_making.harness.memory import DecisionMemory
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.harness.research import ResearchAgenda
from rsisys.loops.decision_making.harness.tools.research import ResearchTools
from rsisys.loops.decision_making.harness.tools import DemoDecisionTools
from rsisys.loops.decision_making.harness.tools.local import LocalDecisionTools
from rsisys.loops.decision_making.harness.tools.queries import QueryTools
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
from rsisys.runtime.run_policy import RunPolicy
from rsisys.runtime.artifacts import ArtifactStore
from rsisys.runtime.runner import Runner
from rsisys.runtime.sandbox import SandboxBackend
from rsisys.runtime.capabilities import optional_method


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
    action_permissions: ActionPermissions = field(default_factory=ActionPermissions)
    project_context: dict = field(default_factory=dict)
    lab_execution: LabExecution | None = None
    model_session: object | None = None
    controller_report: Callable[[], dict] | None = field(default=None, repr=False)
    components: ComponentInstances | None = field(default=None, init=False)
    method_identity: MethodIdentity | None = field(default=None, init=False)
    state_adapters: dict = field(default_factory=dict, init=False, repr=False)
    _cleanup: list[tuple[Callable[[], object], bool]] = field(default_factory=list, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def add_cleanup(self, callback: Callable[[], object]) -> None:
        """Register cleanup immediately after acquiring a factory-owned resource."""
        if self._closed:
            raise RuntimeError("Assembly is closed")
        if not callable(callback):
            raise TypeError("Cleanup must be callable")
        self._cleanup.append((callback, False))

    def add_owned_cleanup(self, callback: Callable[[], object]) -> None:
        """Register an idempotent owned-resource close, retaining dependencies on failure."""
        self.add_cleanup(callback)
        self._cleanup[-1] = (callback, True)

    @property
    def cleanup_pending(self) -> bool:
        return bool(self._cleanup)

    def close(self) -> None:
        """Stop admission; retry only failed owned closes before releasing dependencies.

        Ordinary callbacks run once in LIFO order, even on failure. An owned
        close is a barrier: failure retains it and the earlier dependencies.
        Successful callbacks, and ordinary callbacks with uncertain effects,
        are never replayed by a later close.
        """
        self._closed = True
        errors = []
        while self._cleanup:
            callback, owned = self._cleanup[-1]
            try:
                callback()
            except BaseException as exc:
                errors.append(exc)
                if owned:
                    break
            self._cleanup.pop()
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup('Assembly cleanup failed', errors)


ExplorationFactory = Callable[[AssemblyContext], ExplorationLoop]
DecisionFactory = Callable[[AssemblyContext, ExplorationLoop], DecisionMakingLoop]
LabExecutionFactory = Callable[[AssemblyContext], LabExecution]


@dataclass(frozen=True)
class ModuleAssembly:
    """Replaceable modules, shared by every launcher; factories only construct.

    state_factory(context, decision, exploration) optionally returns named state
    adapters. Their private state format belongs to the integration, not the host.
    """

    decision_factory: DecisionFactory | None = None
    exploration_factory: ExplorationFactory | None = None
    lab_factory: LabExecutionFactory | None = None
    research: object = None
    research_backend: object = None
    feedback_adapter_factory: Callable | None = None
    feedback_options: Mapping = field(default_factory=dict)
    state_factory: Callable | None = None
    components: ComponentGraph | None = None
    component_bindings: Mapping = field(default_factory=dict)
    method_identity: MethodIdentity | None = None
    model_session: object | None = None

    def __post_init__(self):
        if self.model_session is not None:
            from rsisys.runtime.model_session import HostModelSession
            if not isinstance(self.model_session, HostModelSession):
                raise TypeError('model_session must be a host-accounted HostModelSession')
        if self.method_identity is not None and not isinstance(self.method_identity, MethodIdentity):
            raise TypeError('method_identity must be a MethodIdentity')
        if self.components is not None and not isinstance(self.components, ComponentGraph):
            raise TypeError('components must be a ComponentGraph')
        bindings = dict(self.component_bindings)
        if bindings and self.components is None:
            raise ValueError('Component bindings require a component graph')
        for role, name in bindings.items():
            contracts = {'decision': 'decision-factory/v1', 'lab': 'lab-execution/v1',
                         'exploration': 'exploration-factory/v1'}
            if role not in contracts:
                raise ValueError(f'Unknown component binding: {role}')
            self.components.require(name, contracts[role])
        object.__setattr__(self, 'component_bindings', MappingProxyType(bindings))
        for name in ('decision_factory', 'exploration_factory', 'lab_factory',
                     'feedback_adapter_factory', 'state_factory'):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f'{name} must be callable')
        if self.research is not None:
            from rsisys.research import ResearchAssembly
            if not isinstance(self.research, ResearchAssembly):
                raise TypeError('research must be a ResearchAssembly')
            self.research.validate()
        elif self.research_backend is not None or self.feedback_adapter_factory is not None or self.feedback_options:
            raise ValueError('Research backend/adapter/options requires a research assembly')
        if not isinstance(self.feedback_options, Mapping):
            raise TypeError('feedback_options must be a mapping')
        reserved = {'loop', 'assembly', 'owner', 'adapter', 'backend'}
        if reserved.intersection(self.feedback_options):
            raise ValueError('Feedback assembly bindings cannot be overridden by feedback_options')

    @classmethod
    def from_components(cls, components, *, decision, lab=None, exploration=None, **options):
        """Bind graph outputs to existing loop/Lab ports without new launcher code.

        decision/exploration outputs are factories using the existing public
        context signatures; lab is an already constructed LabExecution. Factories
        remain trusted host code and do not acquire model modification rights.
        """
        if not isinstance(components, ComponentGraph):
            raise TypeError('components must be a ComponentGraph')
        if set(options) & {'components', 'component_bindings', 'decision_factory', 'lab_factory', 'exploration_factory'}:
            raise ValueError('Component bindings cannot override assembly factories')
        components.require(decision, 'decision-factory/v1')
        if lab is not None:
            components.require(lab, 'lab-execution/v1')
        if exploration is not None:
            components.require(exploration, 'exploration-factory/v1')
        bindings = {role: name for role, name in (
            ('decision', decision), ('lab', lab), ('exploration', exploration)) if name is not None}
        def selected(context, name, methods):
            value = context.components[name]
            if any(optional_method(value, method) is None for method in methods):
                raise TypeError(f'Component {name} does not implement its assembly binding')
            return value
        def decision_factory(context, execution):
            value = context.components[decision]
            if not callable(value):
                raise TypeError('Decision component must return a factory')
            return value(context, execution)
        def exploration_factory(context):
            value = context.components[exploration]
            if not callable(value):
                raise TypeError('Exploration component must return a factory')
            return value(context)
        return cls(components=components, component_bindings=bindings,
            decision_factory=decision_factory,
            lab_factory=(lambda context: selected(context, lab, ('describe', 'execute'))) if lab is not None else None,
            exploration_factory=exploration_factory if exploration is not None else None, **options)

    def component_configuration(self):
        return {**self.components.configuration(), 'bindings': dict(self.component_bindings)}

    def configuration(self):
        """Stable assembly identity, shared by durable and portable restore."""
        layout = {name: getattr(self, name) is not None for name in (
            'decision_factory', 'exploration_factory', 'lab_factory', 'research', 'state_factory')}
        if self.components is not None:
            layout['components'] = self.component_configuration()
        if self.method_identity is not None:
            layout['method'] = self.method_identity.configuration()
        if self.model_session is not None:
            from dataclasses import asdict
            layout['model_session'] = {'models': [asdict(m) for m in self.model_session.models],
                'protocol': 'host-accounted-model-session/v1'}
        return layout


def module_assembly(modules=None, **legacy):
    """Accept old keyword APIs without silently overriding a module bundle."""
    if modules is not None:
        if not isinstance(modules, ModuleAssembly):
            raise TypeError('modules must be a ModuleAssembly')
        if any(value is not None for value in legacy.values()):
            raise ValueError('Supply modules or individual module arguments, not both')
        return modules
    return ModuleAssembly(**legacy)


def assemble_runner(context: AssemblyContext, *, modules: ModuleAssembly | None = None,
                    run_policy=None, cleanup_timeout_seconds=30):
    """Compose loops, feedback, state ports and lifecycle through one entry.

    The caller owns shared resources and closes context on construction failure.
    No factory, state restore or feedback setup may submit research work here.
    """
    modules = module_assembly(modules)
    if context._closed:
        raise RuntimeError('Assembly is closed')
    context.method_identity = modules.method_identity
    if modules.model_session is not None:
        context.model_session = modules.model_session
        if any(model not in context.model_session.models for model in context.roles.values()):
            raise ValueError('Assembly role is outside the host model session')
        context.backends = dict(context.model_session.backends)
    if modules.components is not None:
        if context.components is not None:
            raise RuntimeError('Components are already assembled')
        context.components = modules.components.build(context, bindings=modules.component_bindings)
    decision, exploration = assemble_loops(context, decision_factory=modules.decision_factory,
        exploration_factory=modules.exploration_factory, lab_factory=modules.lab_factory)
    # Resolve the current harness at close, including domain configuration before
    # the first step. Owned close retains its dependencies if cleanup must retry.
    def close_decision():
        harness = getattr(decision, 'harness', None)
        close = optional_method(harness, 'close')
        if close is not None and not (context.components is not None and context.components.owns(decision.harness)):
            close()
    context.add_owned_cleanup(close_decision)
    if context.method_identity is not None:
        from rsisys.methods import validate_method_ports
        validate_method_ports(decision, exploration, context.lab_execution)
    if modules.research is not None:
        from rsisys.research import assemble_research
        options = dict(modules.feedback_options)
        options.setdefault('path', context.data_dir / 'feedback_learning.json'
                           if context.data_dir is not None else None)
        assemble_research(decision, assembly=modules.research, backend=modules.research_backend,
            owner=context, adapter=modules.feedback_adapter_factory(decision.harness)
                if modules.feedback_adapter_factory else None, **options)
    if modules.state_factory is not None:
        from rsisys.runtime.component_state import validate_adapters, assembly_bindings
        adapters = validate_adapters(modules.state_factory(context, decision, exploration))
        automatic = {}
        if context.components is not None:
            automatic.update(context.components.state_adapters())
        if context.method_identity is not None:
            from rsisys.components import _StaticState
            automatic['method_manifest'] = _StaticState(context.method_identity.configuration())
        if automatic.keys() & adapters.keys():
            raise ValueError('Explicit state adapters cannot replace method/component manifest/stateless ports')
        adapters = {**automatic, **adapters}
        context.state_adapters = adapters
        context._state_bindings = assembly_bindings(context, decision, exploration)
        context._state_port_bindings = dict(context.state_adapters)
    return AssemblyRunner(decision, exploration, context, run_policy=run_policy,
                          cleanup_timeout_seconds=cleanup_timeout_seconds)


def assemble_loops(context: AssemblyContext, *, decision_factory=None,
                   exploration_factory=None, lab_factory: LabExecutionFactory | None = None):
    """One assembly path for persistent applications and in-memory demos."""
    if lab_factory is not None:
        context.lab_execution = lab_factory(context)
        if context.lab_execution is None:
            raise TypeError('lab_factory must return a LabExecution provider')
        close = optional_method(context.lab_execution, 'close')
        if close is not None and not (context.components is not None and context.components.owns(context.lab_execution)):
            context.add_owned_cleanup(close)
    if exploration_factory is not None:
        exploration = exploration_factory(context)
    elif context.lab_execution is not None:
        from rsisys.adapters.execution import OperationOnlyExploration
        exploration = OperationOnlyExploration()
    else:
        exploration = build_default_exploration(context)
    decision = (decision_factory if decision_factory is not None else build_default_decision)(context, exploration)
    return decision, exploration


def build_default_exploration(context: AssemblyContext) -> ExplorationLoop:
    if 'actor' not in context.roles:
        raise ValueError('Legacy exploration requires an actor role; supply lab_factory for direct Lab execution')
    if context.execution == "local_process":
        space = CommandActionSpace()
        environment = LocalProcessEnvironment(context.sandbox, context.artifacts)
    else:
        space = EchoActionSpace()
        environment = SandboxEchoEnvironment(context.sandbox, context.artifacts)
    context.add_owned_cleanup(environment.close)
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
    if isinstance(harness.training.backend, BackendCapabilityProvider):
        capabilities = harness.training.backend.capabilities()
        # Built-in SFT/local RL start fresh; an unrelated backend method or
        # resume-only backend does not make either built-in command executable.
        for mode in ('sft', 'rl'):
            available[mode] = (available[mode] and mode in capabilities.methods
                               and ResumeMode.RESTART in capabilities.resume_modes)
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


def build_default_decision(context: AssemblyContext, exploration: ExplorationLoop, *, policy=None) -> DecisionMakingLoop:
    """Reuse default tools/harness with an independently supplied public RSI policy.

    Custom policies receive DecisionInput through ContextDecisionPolicy. The
    host retains its optional model identity, permissions and execution loop. A policy
    making model calls must use the host's accounted backend integration.
    Whole-harness replacement remains the decision_factory boundary.
    """
    model = context.roles.get("decision_maker")
    if policy is None and model is None:
        raise ValueError('The meta-agent controller requires a decision_maker model; '
                         'supply a policy or decision_factory for a model-free controller')
    open_lab = context.lab_execution is not None
    directives, modes = (DirectiveDefinitions(()), ()) if open_lab else _decision_directives(context, exploration)
    query_tools = (QueryTools(exploration, artifacts=context.artifacts)
                   if not open_lab and isinstance(exploration, QueryProvider) and exploration.describe_queries() else None)
    memory_path = context.data_dir / "decision_memory.json" if context.data_dir is not None else None
    memory = DecisionMemory(memory_path)
    agenda = ResearchAgenda(context.data_dir / "decision_research.json" if context.data_dir is not None else None)
    research_tools = ResearchTools(agenda, memory, experience=context.experience)
    definitions = (*(query_tools.definitions if query_tools is not None else ()), *research_tools.definitions)
    bindings = {**(query_tools.bindings if query_tools is not None else {}), **research_tools.bindings}
    if open_lab:
        from rsisys.adapters.execution import LabExecutionTools
        tools = LabExecutionTools(context.lab_execution, extra_definitions=definitions, extra_bindings=bindings)
        space = ModularDecisionActionSpace(tools.definitions, directive_modes=(), directives=directives,
                                           permissions=context.action_permissions)
    elif context.data_dir is None:
        tools = DemoDecisionTools(context.sandbox, context.artifacts, extra_bindings=bindings)
        space = DemoDecisionActionSpace(directive_modes=modes, directives=directives, extra_definitions=definitions,
                                        permissions=context.action_permissions)
    else:
        tools = LocalDecisionTools(context.artifacts, extra_bindings=bindings)
        space = LocalDecisionActionSpace(directive_modes=modes, directives=directives, extra_definitions=definitions,
                                         permissions=context.action_permissions)
    tools.query_tools = query_tools
    tools.research_tools = research_tools
    if policy is None:
        policy = ModelDecisionPolicy(context.backends[model.backend_id], model,
            allowed_actions=() if open_lab else tuple(spec.name for spec in exploration.action_space.describe()),
            directives=directives, project_context=context.project_context)
    else:
        from rsisys.adapters.decision import ContextDecisionPolicy, DecisionInputAdapter
        policy = ContextDecisionPolicy(policy, input_adapter=DecisionInputAdapter(directives=directives))
    return DecisionMakingLoop(
        DefaultDecisionHarness(
            policy,
            tools, context.experience,
            learner=MockDecisionLearner(context.artifacts) if context.with_learning else None,
            experience_limit=context.experience_limit,
            memory=memory, memory_limit=context.memory_limit, research_agenda=agenda,
            model=model,
        ), space,
    )


class AssemblyRunner(Runner):
    """Runner sharing its assembly's admission and owned cleanup lifecycle."""

    def __init__(self, decision: DecisionMakingLoop, exploration: ExplorationLoop, assembly: AssemblyContext,
                 *, run_policy: RunPolicy | None = None, cleanup_timeout_seconds: float = 30):
        super().__init__(decision, exploration, run_policy=run_policy,
                         cleanup_timeout_seconds=cleanup_timeout_seconds,
                         execution_shutdown=optional_method(assembly.lab_execution, 'shutdown'))
        self.assembly = assembly

    @property
    def closed(self) -> bool:
        return self.assembly._closed

    def close(self) -> None:
        self.assembly.close()

    def __enter__(self):
        if self.closed:
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
        if self.closed:
            raise RuntimeError("Assembly is closed")
        return super().step(objective)
