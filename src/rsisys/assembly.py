"""Composition and resource lifecycle for explicitly supplied RSI method ports."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from rsisys.components import ComponentGraph, ComponentInstances
from rsisys.loops.contracts.execution import LabExecution
from rsisys.loops.contracts.model_service import ModelBackend, ModelRef
from rsisys.loops.decision_making.action_space.access import ActionPermissions
from rsisys.loops.decision_making.action_space.directives import DirectiveDefinitions
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.loops.exploration.loop import ExplorationLoop
from rsisys.methods import MethodIdentity
from rsisys.runtime.capabilities import optional_method
from rsisys.runtime.evidence.retrieval import ExperienceSystem
from rsisys.runtime.execution.sandbox import SandboxBackend
from rsisys.runtime.orchestration.run_policy import RunPolicy
from rsisys.runtime.orchestration.runner import Runner
from rsisys.runtime.storage.artifacts import ArtifactStore


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
            from rsisys.runtime.models.model_session import HostModelSession
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


def _build_policy_assembly(controller, options, selected_labs):
    """Share lifecycle wiring between whole methods and legacy selections."""
    from rsisys.runtime.orchestration.lab_group import LabGroup

    def lab_factory(context):
        group = LabGroup()
        # Retain partial constructions if a later factory/catalog fails.
        context.add_owned_cleanup(group.close)
        for alias, definition, config in selected_labs:
            group.add(alias, definition.factory(context, deepcopy(config)))
        return group

    def decision_factory(context, execution):
        if controller.endpoint:
            return controller.factory(context, execution, deepcopy(options))
        policy = controller.factory(context, deepcopy(options))
        close = optional_method(policy, 'close')
        if close is not None:
            context.add_owned_cleanup(close)
        if not callable(policy) and not callable(getattr(policy, 'decide', None)):
            raise TypeError('Controller factory must return decide(input) or a callable')
        report = optional_method(policy, 'report')
        if report is not None:
            context.controller_report = report
        required = optional_method(policy, 'required_operations')
        if required is not None:
            names = required()
            if (not isinstance(names, tuple) or any(not isinstance(n, str) or not n for n in names)
                    or len(set(names)) != len(names)):
                raise TypeError('required_operations() must return distinct operation names as a tuple')
            provider = context.lab_execution
            available = {spec.name for spec in provider.describe()} if provider is not None else set()
            missing = sorted(set(names) - available)
            if missing:
                raise ValueError('Controller requires missing Lab operations: ' + ', '.join(missing))
        return build_default_decision(context, execution, policy=policy)

    research = (controller.research_factory(deepcopy(options))
                if controller.research_factory is not None else None)
    return ModuleAssembly(decision_factory=decision_factory, lab_factory=lab_factory,
                          research=research)


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
        from rsisys.runtime.state.component_state import (
            assembly_bindings,
            validate_adapters,
        )
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
        from rsisys.loops.exploration.operation_only import OperationOnlyExploration
        exploration = OperationOnlyExploration()
    else:
        exploration = build_default_exploration(context)
    decision = (decision_factory if decision_factory is not None else build_default_decision)(context, exploration)
    return decision, exploration


def build_default_exploration(context: AssemblyContext) -> ExplorationLoop:
    """Select the bundled Actor harness; custom exploration factories bypass it."""
    from rsisys.harnesses.defaults import build_default_exploration as build
    return build(context)


def build_default_decision(context: AssemblyContext, exploration: ExplorationLoop, *, policy=None) -> DecisionMakingLoop:
    """Select the bundled policy adapter/harness; custom decision factories bypass it."""
    from rsisys.harnesses.defaults import build_default_decision as build
    return build(context, exploration, policy=policy)


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
