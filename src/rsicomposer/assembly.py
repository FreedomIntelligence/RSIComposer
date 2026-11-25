"""Composition and resource lifecycle for explicitly supplied RSI method ports."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4
import warnings

from rsicomposer.components import ComponentGraph, ComponentInstances
from rsicomposer.component_context import LabContext, controller_learning, nested_run
from rsicomposer.loops.contracts.execution import LabExecution
from rsicomposer.loops.contracts.model_service import ModelBackend, ModelRef
from rsicomposer.loops.decision_making.action_space.access import ActionPermissions
from rsicomposer.loops.decision_making.loop import DecisionMakingLoop
from rsicomposer.methods import ControllerDefinition, MethodIdentity
from rsicomposer.runtime.capabilities import optional_method
from rsicomposer.runtime.evidence.retrieval import EvidenceRepository
from rsicomposer.runtime.execution.sandbox import SandboxBackend
from rsicomposer.runtime.orchestration.run_policy import RunPolicy
from rsicomposer.runtime.orchestration.runner import Runner
from rsicomposer.runtime.storage.artifacts import ArtifactStore


_MODULE_ASSEMBLY_SCHEMA = 1


def read_saved_module_layout(directory):
    """Read the assembly marker only after verifying its format."""
    marker = Path(directory) / 'module_assembly.json'
    if not marker.exists():
        return None
    layout = json.loads(marker.read_text())
    if (not isinstance(layout, dict) or type(layout.get('schema_version')) is not int
            or layout['schema_version'] != _MODULE_ASSEMBLY_SCHEMA):
        raise ValueError('Unsupported module assembly schema; start a new run')
    return layout


def _factory_identity(value):
    target = value if hasattr(value, '__qualname__') else type(value)
    return str(getattr(target, '__module__', '')) + ':' + target.__qualname__


@dataclass
class AssemblyContext:
    experience: EvidenceRepository
    artifacts: ArtifactStore
    backends: Mapping[str, ModelBackend]
    sandbox: SandboxBackend
    roles: Mapping[str, ModelRef]
    data_dir: Path | None = None
    execution: str = "mock"
    with_learning: bool = False
    experience_limit: int = 32
    memory_limit: int = 32
    action_permissions: ActionPermissions = field(default_factory=ActionPermissions)
    project_context: dict = field(default_factory=dict)
    lab_execution: LabExecution | None = None
    model_session: object | None = None
    resources: object | None = None
    weight_sync: Mapping[str, object] = field(default_factory=dict, repr=False)
    executor: object | None = None
    host_execution: object | None = None
    budget: object | None = None
    # True when factories rebuild an existing workspace (resume or snapshot restore).
    restoring: bool = False
    resuming: bool = False
    controller_report: Callable[[], dict] | None = field(default=None, repr=False)
    components: ComponentInstances | None = field(default=None, init=False)
    method_identity: MethodIdentity | None = field(default=None, init=False)
    state_adapters: dict = field(default_factory=dict, init=False, repr=False)
    _memory_observer: Callable | None = field(default=None, init=False, repr=False)
    _evidence_secrets: set[str] = field(default_factory=set, init=False, repr=False)
    _cleanup: list[tuple[Callable[[], object], bool]] = field(default_factory=list, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _assembled: bool = field(default=False, init=False, repr=False)

    def record(self, kind: str, payload=None, *, references=(), decision_id: str | None = None):
        """Record a Controller-side fact as evidence, then offer it to the Controller's Memory.

        Policies use this for what they produce or conclude, e.g. kind='candidate'
        for generated code or kind='interpretation' for an LLM verdict on a receipt,
        so evidence holds more than Lab receipts. payload must be JSON; large data
        belongs in artifacts named by references. The evidence row is written first;
        a failing Memory.observe only warns, since Memory is a view rebuilt from evidence.
        """
        from rsicomposer.loops.contracts.feedback_context import EvidenceEvent
        from rsicomposer.runtime.evidence.records import ExperienceRecord
        from rsicomposer.runtime.storage.files import json_bytes
        if self._closed:
            raise RuntimeError('Assembly is closed')
        host = getattr(self, 'controller_host', None)
        version = getattr(self, '_controller_version', None) or (host.version_id if host is not None else None)
        event = EvidenceEvent(event_id=str(uuid4()), source='controller', kind=kind,
                              output=deepcopy(payload), references=deepcopy(tuple(references)),
                              decision_id=decision_id, identity={'controller_version': version} if version else {})
        details = {'event': asdict(event)}
        json_bytes(details)
        self.experience.record(ExperienceRecord(event.event_id, 'controller_event', 'decision_maker', None,
                                                f'Controller {kind}', False, details=details))
        if self._memory_observer is not None:
            try:
                self._memory_observer(deepcopy(event))
            except Exception as exc:
                warnings.warn(f'Memory.observe failed for controller event {event.event_id}: '
                              f'{type(exc).__name__}: {exc}', RuntimeWarning, stacklevel=2)
        return event

    def evidence_view(self, scope=None):
        from rsicomposer.loops.contracts.controller_version import EvidenceScope
        from rsicomposer.runtime.evidence.view import ReadOnlyEvidenceView
        from rsicomposer.runtime.evidence.credentials import host_secret_values
        owner = getattr(self, '_evidence_owner', self)
        return ReadOnlyEvidenceView(owner.experience, owner._artifact_access,
            EvidenceScope() if scope is None else scope, run_id=getattr(owner, 'run_id', None),
            secret_values=host_secret_values(owner))

    def nested_run(self):
        return nested_run(self, self)

    def controller_learning(self):
        return controller_learning(self)

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


DecisionFactory = Callable[[AssemblyContext, LabExecution | None], DecisionMakingLoop]
LabExecutionFactory = Callable[[AssemblyContext], LabExecution]


@dataclass(frozen=True)
class ModuleAssembly:
    """Replaceable modules, shared by every launcher; factories only construct.

    build(context) is the runtime entry; reuse this description with fresh contexts.
    Direct factories and graph bindings are mutually exclusive for each role.
    Application owns context cleanup; direct callers must close it on failure too.

    state_factory(context, decision) optionally returns a nonempty
    mapping of adapters with state_id, capture() and restore(state) -> None.
    Portable adapters also supply validate_snapshot() -> None. Dependent private
    state belongs to the integration; graph/method manifest ports are added by the
    host and must not be replaced by state_factory.
    """

    decision_factory: DecisionFactory | None = None
    lab_factory: LabExecutionFactory | None = None
    memory: object = None
    state_factory: Callable | None = None
    components: ComponentGraph | None = None
    component_bindings: Mapping = field(default_factory=dict)
    method_identity: MethodIdentity | None = None
    model_session: object | None = None
    controller_catalog: Mapping = field(default_factory=dict)
    controller_selection: dict | None = None
    controller_validators: Mapping = field(default_factory=dict, repr=False)
    registry_source: dict | None = field(default=None, repr=False)
    controller_activator: object | None = field(default=None, repr=False)
    controller_learner: object | None = field(default=None, repr=False)
    observation_probes: tuple = field(default=(), repr=False, compare=False)

    def __post_init__(self):
        if self.decision_factory is None and 'decision' not in self.component_bindings:
            raise ValueError('ModuleAssembly requires an explicit decision_factory or decision component binding')
        if self.model_session is not None:
            from rsicomposer.runtime.models.model_session import HostModelSession
            if not isinstance(self.model_session, HostModelSession):
                raise TypeError('model_session must be a host-accounted HostModelSession')
        if self.method_identity is not None and not isinstance(self.method_identity, MethodIdentity):
            raise TypeError('method_identity must be a MethodIdentity')
        if self.components is not None and not isinstance(self.components, ComponentGraph):
            raise TypeError('components must be a ComponentGraph')
        if self.controller_activator is not None and optional_method(self.controller_activator, 'activate') is None:
            raise TypeError('Controller activator must implement activate(result, idempotency_key=...)')
        if self.controller_learner is not None and any(optional_method(self.controller_learner, name) is None
                for name in ('submit', 'poll', 'cancel')):
            raise TypeError('Controller learner must implement submit, poll and cancel')
        catalog = dict(self.controller_catalog)
        if any(not isinstance(name, str) or not name or not isinstance(item, ControllerDefinition)
               for name, item in catalog.items()):
            raise TypeError('Controller catalog needs explicit names and ControllerDefinition values')
        validators = dict(self.controller_validators)
        if set(validators) - set(catalog) or any(not callable(value) for value in validators.values()):
            raise TypeError('Controller validators must refer to declared catalog entries')
        selection = deepcopy(self.controller_selection)
        if selection is not None and (not isinstance(selection, dict) or set(selection) != {'name', 'config'}
                or selection['name'] not in catalog or not isinstance(selection['config'], dict)):
            raise ValueError('Initial Controller selection must name an explicit catalog entry')
        if catalog and selection is None:
            raise ValueError('Controller catalog requires its initial selection')
        object.__setattr__(self, 'controller_catalog', MappingProxyType(catalog))
        object.__setattr__(self, 'controller_validators', MappingProxyType(validators))
        object.__setattr__(self, 'controller_selection', selection)
        object.__setattr__(self, 'registry_source', deepcopy(self.registry_source))
        bindings = dict(self.component_bindings)
        if bindings and self.components is None:
            raise ValueError('Component bindings require a component graph')
        for role, name in bindings.items():
            contracts = {'decision': 'decision-factory/v1', 'lab': 'lab-execution/v1'}
            if role not in contracts:
                raise ValueError(f'Unknown component binding: {role}')
            self.components.require(name, contracts[role])
            if getattr(self, role + '_factory') is not None:
                raise ValueError(f'Choose a factory or a graph binding for {role}, not both')
        object.__setattr__(self, 'component_bindings', MappingProxyType(bindings))
        for name in ('decision_factory', 'lab_factory', 'state_factory'):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f'{name} must be callable')
        if self.memory is not None:
            from rsicomposer.research import MemoryAssembly
            if not isinstance(self.memory, MemoryAssembly):
                raise TypeError('memory must be a MemoryAssembly')

    @classmethod
    def from_components(cls, components, *, decision, lab=None, **options):
        """Bind graph outputs to existing loop/Lab ports without new launcher code.

        decision outputs are factories using the existing public
        context signatures; lab is an already constructed LabExecution. Factories
        remain trusted host code and do not acquire model modification rights.
        """
        if not isinstance(components, ComponentGraph):
            raise TypeError('components must be a ComponentGraph')
        if set(options) & {'components', 'component_bindings', 'decision_factory', 'lab_factory'}:
            raise ValueError('Component bindings cannot override assembly factories')
        components.require(decision, 'decision-factory/v1')
        if lab is not None:
            components.require(lab, 'lab-execution/v1')
        bindings = {role: name for role, name in (
            ('decision', decision), ('lab', lab)) if name is not None}
        return cls(components=components, component_bindings=bindings, **options)

    def has_port(self, role):
        """Whether a loop/Lab factory is supplied directly or through the graph."""
        if role not in ('decision', 'lab'):
            raise ValueError(f'Unknown assembly role: {role}')
        return getattr(self, role + '_factory') is not None or role in self.component_bindings

    def _factories(self, context):
        """Adapt graph outputs once, then use the same factory assembly path."""
        factories = {role: getattr(self, role + '_factory')
                     for role in ('decision', 'lab')}
        for role, name in self.component_bindings.items():
            value = context.components[name]
            if role == 'lab':
                if any(optional_method(value, method) is None for method in ('describe', 'execute')):
                    raise TypeError(f'Component {name} does not implement Lab execution')
                factories[role] = lambda context, value=value: value
            else:
                if not callable(value):
                    raise TypeError(f'Component {name} must return a {role} factory')
                factories[role] = value
        return factories

    def build(self, context: AssemblyContext, *, run_policy=None, cleanup_timeout_seconds=30):
        """The runtime composition entry for direct, registered and graph methods.

        The caller owns the context and closes it if construction fails. Factories
        construct only; no state restore or research execution happens here.
        """
        return _assemble_runner(context, self, run_policy=run_policy,
                                cleanup_timeout_seconds=cleanup_timeout_seconds)

    def validate_saved(self, directory):
        """One assembly identity check for durable and snapshot restore."""
        saved = read_saved_module_layout(directory)
        if saved is not None:
            saved.pop('lab_profiles', None)
            saved.pop('evaluation_operations', None)  # Validated after constructing the Lab.
            if saved != self.configuration():
                raise TypeError('Supply the original module assembly when restoring this workspace')
        elif self.method_identity is not None:
            raise TypeError('Workspace has no method identity')

    def component_configuration(self):
        return {**self.components.configuration(), 'bindings': dict(self.component_bindings)}

    def configuration(self):
        """Stable assembly identity, shared by durable and portable restore."""
        layout = {'schema_version': _MODULE_ASSEMBLY_SCHEMA,
                  **{role + '_factory': self.has_port(role) for role in ('decision', 'lab')}}
        selected = (self.controller_catalog.get(self.controller_selection['name'])
                    if self.controller_selection else None)
        layout.update(memory=self.memory is not None or bool(selected is not None and selected.memory is not None),
                      state_factory=self.state_factory is not None)
        if self.components is not None:
            layout['components'] = self.component_configuration()
        if self.method_identity is not None:
            layout['method'] = self.method_identity.configuration()
        if self.controller_activator is not None or self.controller_learner is not None:
            layout['controller_learning'] = {name: _factory_identity(type(value)) if value is not None else None
                for name, value in (('activator', self.controller_activator), ('learner', self.controller_learner))}
        if self.controller_catalog:
            layout['controller_catalog'] = {name: {
                'policy': _factory_identity(item.policy),
                'memory': _factory_identity(item.memory) if item.memory else None,
                'harness': _factory_identity(item.harness) if item.harness else None,
                'endpoint': item.endpoint,
            } for name, item in sorted(self.controller_catalog.items())}
            layout['controller_selection'] = deepcopy(self.controller_selection)
        if self.model_session is not None:
            from dataclasses import asdict
            layout['model_session'] = {'models': [asdict(m) for m in self.model_session.configured_models],
                'protocol': 'host-accounted-model-session/v1'}
        return layout


def _build_policy_assembly(controller, options, selected_labs):
    """Share lifecycle wiring between whole methods and explicit Controller/Lab selections."""
    from rsicomposer.runtime.orchestration.lab_group import LabGroup

    def lab_factory(context):
        group = LabGroup(receipts=context.data_dir / 'lab_receipts' if context.data_dir is not None else None,
                         artifacts=context.artifacts, experience=context.experience,
                         artifact_access=context._artifact_access)
        # Retain partial constructions if a later factory/catalog fails.
        context.add_owned_cleanup(group.close)
        for alias, definition, config in selected_labs:
            scoped = LabContext(context, alias)
            options = deepcopy(config)
            capacity = options.pop('max_concurrent', 1)
            scoped.models.max_concurrent = capacity
            lab = definition.factory(scoped, options)
            group.add(alias, lab, model_scope=scoped.models.scope, max_concurrent=capacity,
                      declare_roles=scoped.resources.declare)
        return group

    def decision_factory(context, execution):
        from rsicomposer.host.lifecycle import build_controller, initial_controller_version
        definition = ControllerDefinition(controller.factory, memory=controller.memory,
                                          endpoint=controller.endpoint, harness=controller.harness)
        selection = {'name': 'current', 'config': deepcopy(options)}
        override = getattr(context, '_initial_controller_version', None)
        version = override or initial_controller_version(selection, context.roles.get('decision_maker'))
        params = options
        initial_memory = getattr(context, '_initial_memory', None)
        if override is not None:
            if version.config['name'] not in context._controller_catalog:
                raise ValueError('Initial version selects an unavailable Controller')
            definition = context._controller_catalog[version.config['name']]
            params = version.config['config']
            validator = context._controller_validators.get(version.config['name'])
            if validator is not None and validator(deepcopy(params)) is not None:
                raise TypeError('Controller validators must return None')
            if definition.memory is not None:
                initial_memory = None
        instance = build_controller(context, definition, params, version, initial=True, memory=initial_memory)
        context.add_owned_cleanup(instance.close)
        context._initial_controller_instance = instance
        context._initial_controller_version = version
        return instance.endpoint

    from rsicomposer.method_state import method_state_factory
    return ModuleAssembly(decision_factory=decision_factory, lab_factory=lab_factory,
                          state_factory=method_state_factory,
                          controller_catalog={'current': ControllerDefinition(controller.factory,
                              memory=controller.memory, endpoint=controller.endpoint,
                              harness=controller.harness)},
                          controller_selection={'name': 'current', 'config': deepcopy(options)},
                          controller_validators=({'current': controller.validate_config}
                              if controller.validate_config is not None else {}))


def _ray_service_binding(config, nodes, node_ip):
    """Choose reachable, explicit host service bindings for the current topology."""
    import ipaddress
    import socket
    settings = config.get('host', {})
    client = settings.get('mode', 'head') == 'client'
    spread = any(pool.get('strategy') == 'STRICT_SPREAD' and len(pool.get('bundles', ())) > 1
                 for pool in config.get('pools', {}).values())
    distributed = client or sum(bool(node.get('Alive')) for node in nodes) > 1 or spread
    if not distributed:
        advertise = settings.get('advertise_address', '127.0.0.1')
        if advertise != '127.0.0.1':
            raise ValueError('Single-node Ray head advertise_address must be 127.0.0.1')
        bind = '127.0.0.1'
    else:
        if client and not settings.get('advertise_address'):
            raise ValueError('Ray client services require a reachable advertise_address')
        advertise = settings.get('advertise_address') or node_ip
        if (not isinstance(advertise, str) or not advertise or
                any(character in advertise for character in '/:@?#') or
                any(character.isspace() for character in advertise)):
            raise ValueError('Ray advertise_address must be a reachable IPv4 address or hostname')
        try:
            addresses = [ipaddress.ip_address(advertise)]
        except ValueError:
            try:
                addresses = [ipaddress.ip_address(row[4][0]) for row in
                             socket.getaddrinfo(advertise, 0, socket.AF_INET, socket.SOCK_STREAM)]
            except socket.gaierror as error:
                raise ValueError('Ray advertise_address must resolve on the host') from error
        if not addresses or any(address.is_loopback or address.is_unspecified for address in addresses):
            raise ValueError('Distributed Ray advertise_address must be reachable from cluster nodes')
        bind = advertise
    if settings.get('bind_address', bind) != bind:
        raise ValueError('Ray bind_address must match the required service bind: ' + bind)
    return {'bind_host': bind, 'advertise_host': advertise}


def _assemble_runner(context, modules, *, run_policy=None, cleanup_timeout_seconds=30):
    """Construct loops, integrations, state and cleanup once for a run."""
    if context._closed:
        raise RuntimeError('Assembly is closed')
    if context._assembled:
        raise RuntimeError('Assembly context has already been used')
    context._assembled = True
    context.method_identity = modules.method_identity
    if modules.model_session is not None:
        context.model_session = modules.model_session
        if any(model not in context.model_session.models for model in context.roles.values()):
            raise ValueError('Assembly role is outside the host model session')
        context.backends = dict(context.model_session.backends)
    from rsicomposer.runtime.storage.artifact_access import ArtifactAuthority
    context._artifact_access = ArtifactAuthority(context.artifacts)
    from rsicomposer.runtime.execution.budget import HostBudget
    from rsicomposer.runtime.execution.resources import ResourceManager
    from rsicomposer.runtime.execution.executor import Executor
    if context.data_dir is None:
        from tempfile import TemporaryDirectory
        temporary = TemporaryDirectory(prefix='rsicomposer-host-')
        context.add_cleanup(temporary.cleanup)
        host_root = Path(temporary.name)
    else:
        host_root = context.data_dir / 'host_execution'
    if context.budget is None:
        context.budget = HostBudget(host_root,
            getattr(context, '_budget_config', {}), restoring=context.restoring)
    if context.resources is None:
        context.resources = ResourceManager(getattr(context, '_resource_config', {}), budget=context.budget)
        context.add_owned_cleanup(context.resources.close)
    from rsicomposer.runtime.execution.weight_sync import build_weight_sync, validate_weight_sync_roles
    context.weight_sync = build_weight_sync(getattr(context, '_resource_config', {}).get('weight_sync', {}),
        getattr(context.artifacts, 'root', host_root / 'weight_sync'), context.artifacts)
    environment_provider = None
    if getattr(context, '_resource_config', {}).get('executor', 'local') == 'ray':
        from rsicomposer.runtime.execution.remote_services import HostWorkerServices
        import ray
        context._network_host = _ray_service_binding(context._resource_config, ray.nodes(),
                                                     ray.util.get_node_ip_address())
        context._worker_services = HostWorkerServices(context, **context._network_host)
        context.add_owned_cleanup(context._worker_services.close)
        environment_provider = context._worker_services.environment
        if hasattr(context.artifacts, 'uri_scheme'):
            context.artifacts.uri_scheme = 'file'
    if context.executor is None:
        context.executor = Executor(host_root / 'workers', context.resources, context.budget,
                                    environment_provider=environment_provider)
        context.add_owned_cleanup(context.executor.close)
    from rsicomposer.runtime.models.component_access import HostModelAccess
    from rsicomposer.backends.serving.chat import ChatPool
    from rsicomposer.backends.serving import build_serving_backend
    context._model_access = HostModelAccess(context, pool_factory=ChatPool, transport_factory=build_serving_backend)
    context.add_owned_cleanup(context._model_access.close)
    from rsicomposer.host.execution import HostExecution
    if context.host_execution is None:
        context.host_execution = HostExecution(experience=context.experience)
        context.add_owned_cleanup(context.host_execution.close)
    if modules.components is not None:
        if context.components is not None:
            raise RuntimeError('Components are already assembled')
        context.components = modules.components.build(context, bindings=modules.component_bindings)
    context._controller_catalog = dict(modules.controller_catalog)
    context._controller_validators = dict(modules.controller_validators)
    context._initial_memory = modules.memory
    if modules.controller_catalog:
        from rsicomposer.host.lifecycle import controller_capabilities
        context._host_capabilities = controller_capabilities(lambda arguments: context.controller_host.request(arguments))
    from rsicomposer.runtime.models.delegation import prepare_nested_models
    context.prepare_nested_models = lambda child_budget, call_id, host_config, version, labs: prepare_nested_models(
        context._model_access, child_budget, call_id, host_config, version, labs)
    def nested_manifest(controller, labs):
        from rsicomposer.nested_worker import build_nested_manifest
        from rsicomposer.loops.contracts.controller_version import ControllerVersion
        host = getattr(context, 'controller_host', None)
        inherited = host.approve_memory_history(controller) if host is not None and isinstance(controller, ControllerVersion) else ()
        return build_nested_manifest(controller, labs, context.host_config, catalog=context._controller_catalog,
                                     inherited_history=inherited)
    context.nested_manifest = nested_manifest
    factories = modules._factories(context)
    decision = _assemble_loops(context, decision_factory=factories['decision'], lab_factory=factories['lab'])
    # Resolve the current harness at close, including domain configuration before
    # the first step. Owned close retains its dependencies if cleanup must retry.
    def close_decision():
        harness = getattr(decision, 'harness', None)
        close = optional_method(harness, 'close')
        if close is not None and not (context.components is not None and context.components.owns(decision.harness)):
            close()
    if not hasattr(context, '_initial_controller_instance'):
        host_loop = isinstance(decision, DecisionMakingLoop) and isinstance(decision.host_execution, HostExecution)
        if host_loop and decision.host_execution is not context.host_execution:
            raise ValueError('Controller endpoint belongs to another run execution')
        context.add_owned_cleanup(close_decision)
        if host_loop:
            if decision.operation_router.lab is not context.lab_execution:
                raise ValueError('Controller endpoint must retain the assembled Lab routing')
            from rsicomposer.loops.decision_making.action_space.access import enforce_host_permissions
            decision.action_space = enforce_host_permissions(decision.action_space, context.action_permissions)
            if context.host_execution.endpoint is not decision:
                context.host_execution.activate(decision, getattr(decision, 'controller_version', None))
            context.controller_report = optional_method(decision.policy, 'report')
    context.resources.validate_roles()
    validate_weight_sync_roles(context.weight_sync, context.resources)
    profiles = optional_method(context.lab_execution, 'profile_identity')
    if context.method_identity is not None and profiles is not None and profiles():
        declared = context.method_identity.configuration()
        context.method_identity = MethodIdentity(declared['method'], declared['version'],
            {**declared['config'], 'lab_profiles': profiles()})
    if context.method_identity is not None:
        from rsicomposer.methods import validate_method_ports
        validate_method_ports(decision, context.lab_execution)
    if modules.memory is not None and not hasattr(context, '_initial_controller_instance'):
        modules.memory.install(decision, owner=context, path=context.data_dir / 'feedback_learning.json'
                                          if context.data_dir is not None else None)
    if hasattr(context, '_initial_controller_instance'):
        from rsicomposer.host.lifecycle import ControllerEndpoint, build_controller
        from rsicomposer.host.lifecycle import ControllerHost
        initial = context._initial_controller_instance
        def construct_version(version):
            name, options = version.config['name'], version.config['config']
            if name not in context._controller_catalog:
                raise ValueError('Unknown host Controller: ' + name)
            validator = context._controller_validators.get(name)
            if validator is not None and validator(deepcopy(options)) is not None:
                raise TypeError('Controller validators must return None')
            definition = context._controller_catalog[name]
            inherited = modules.memory if definition.memory is None and name == 'current' else None
            return build_controller(context, definition, options, version, memory=inherited)
        host = ControllerHost(context, initial, context._initial_controller_version, construct_version,
                              activator=modules.controller_activator, learner=modules.controller_learner)
        context.add_owned_cleanup(host.close)
        context.controller_host = host
        if getattr(context, '_initial_controller_version', None) is not None:
            version = context._initial_controller_version
            policy = initial.policy
            if version.policy_state is not None:
                restore = optional_method(policy, 'restore')
                if restore is None or restore(deepcopy(version.policy_state)) is not None:
                    raise TypeError('Initial Controller must restore its version state explicitly')
            host._restore_memory(initial, version.memory_state)
            if version.model is not None:
                initial.bind_model(version.model)
            current_model = initial.endpoint.current_model()
            if current_model is not None:
                initial.bind_model(current_model)
        decision = ControllerEndpoint(host)
        context.controller_report = lambda: (host.active.context.controller_report()
            if host.active.context.controller_report is not None else {})
        context._memory_observer = lambda event: (host.active.context._memory_observer(event)
            if host.active.context._memory_observer is not None else None)
    if modules.state_factory is not None:
        from rsicomposer.runtime.state.component_state import (
            assembly_bindings,
            validate_adapters,
        )
        adapters = validate_adapters(modules.state_factory(context, decision))
        automatic = {}
        if context.components is not None:
            automatic.update(context.components.state_adapters())
        if context.method_identity is not None:
            from rsicomposer.components import _StaticState
            automatic['method_manifest'] = _StaticState(context.method_identity.configuration())
        if automatic.keys() & adapters.keys():
            raise ValueError('Explicit state adapters cannot replace method/component manifest/stateless ports')
        adapters = {**automatic, **adapters}
        context.state_adapters = adapters
        context._state_bindings = assembly_bindings(context, decision)
        context._state_port_bindings = dict(context.state_adapters)
    return AssemblyRunner(decision, context, run_policy=run_policy,
                          cleanup_timeout_seconds=cleanup_timeout_seconds)


def _assemble_loops(context: AssemblyContext, *, decision_factory, lab_factory=None):
    """Construct only explicitly supplied Controller/Lab ports."""
    if lab_factory is not None:
        context.lab_execution = lab_factory(context)
        if context.lab_execution is None:
            raise TypeError('lab_factory must return a LabExecution provider')
        close = optional_method(context.lab_execution, 'close')
        if close is not None and not (context.components is not None and context.components.owns(context.lab_execution)):
            context.add_owned_cleanup(close)
    return decision_factory(context, context.lab_execution)


def build_default_decision(context: AssemblyContext, execution: LabExecution, *, policy=None) -> DecisionMakingLoop:
    """Wrap an explicitly supplied Controller policy with the host harness."""
    from rsicomposer.harnesses.defaults import build_default_decision as build
    return build(context, execution, policy=policy)


class AssemblyRunner(Runner):
    """Runner sharing its assembly's admission and owned cleanup lifecycle."""

    def __init__(self, decision: DecisionMakingLoop, assembly: AssemblyContext,
                 *, run_policy: RunPolicy | None = None, cleanup_timeout_seconds: float = 30):
        super().__init__(decision, run_policy=run_policy,
                         cleanup_timeout_seconds=cleanup_timeout_seconds,
                         execution_shutdown=optional_method(assembly.lab_execution, 'shutdown'),
                         finish_execution=optional_method(assembly.lab_execution, 'finish_step'))
        self.assembly = assembly
        self._evaluation_operations = optional_method(assembly.lab_execution, 'is_evaluation')
        self._declared_evaluations = None
        declared = optional_method(assembly.lab_execution, 'evaluation_operations')
        if self._evaluation_operations is None and declared is not None:
            from rsicomposer.runtime.orchestration.profiles import evaluation_operations
            self._declared_evaluations = evaluation_operations(assembly.lab_execution,
                                                               assembly.lab_execution.describe(), 'execution')
            self._evaluation_operations = self._declared_evaluations.__contains__
        self.host_budget = assembly.budget
        self._guard_actions = True

    def _classify_outcome(self, receipt, decision_id):
        classify = self._evaluation_operations
        from rsicomposer.host.execution import HostExecution
        execution = self.assembly.host_execution
        if classify is not None and decision_id is not None and isinstance(execution, HostExecution):
            record = execution.state.decision_records.get(decision_id, {})
            if record.get('operation_outcomes', {}).get(receipt.call_id) == receipt.status.value:
                call = next((call for call in record['proposal']['operations']
                             if call['call_id'] == receipt.call_id), None)
                if call is not None and classify(call['name']) is True:
                    from rsicomposer.loops.contracts.decision import OperationCall
                    from rsicomposer.runtime.orchestration.run_policy import EvaluationOutcome
                    outcome = EvaluationOutcome.from_call(OperationCall(**call), receipt)
                    if outcome is not None:
                        return outcome
        return super()._classify_outcome(receipt, decision_id)

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
