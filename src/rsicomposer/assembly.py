"""Composition and resource lifecycle for explicitly supplied RSI method ports."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import inspect
import json
import marshal
from pathlib import Path
from types import MappingProxyType

from rsicomposer.components import ComponentGraph, ComponentInstances, config_json
from rsicomposer.component_context import (
    LabContext, NestedRunServices, ResourceScope, build_nested_services, controller_learning,
    nested_run, record_controller_event, require_nested_services,
)
from rsicomposer.loops.contracts.execution import LabExecution
from rsicomposer.loops.contracts.model_service import ModelBackend, ModelRef
from rsicomposer.loops.decision_making.action_space.access import ActionPermissions
from rsicomposer.loops.decision_making.loop import DecisionMakingLoop
from rsicomposer.methods import ControllerDefinition, MethodIdentity
from rsicomposer.selection import RegistrySelection
from rsicomposer.runtime.capabilities import optional_method
from rsicomposer.runtime.evidence.retrieval import EvidenceRepository
from rsicomposer.runtime.execution.sandbox import SandboxBackend
from rsicomposer.runtime.orchestration.run_policy import RunPolicy
from rsicomposer.runtime.orchestration.runner import Runner
from rsicomposer.runtime.storage.artifacts import ArtifactStore


_MODULE_ASSEMBLY_SCHEMA = 5


def read_saved_module_layout(directory):
    """Read the assembly marker only after verifying its format."""
    marker = Path(directory) / 'module_assembly.json'
    if not marker.is_file():
        raise ValueError('Missing module assembly marker; cannot verify the saved assembly')
    layout = json.loads(marker.read_text())
    if (not isinstance(layout, dict) or type(layout.get('schema_version')) is not int
            or layout['schema_version'] != _MODULE_ASSEMBLY_SCHEMA):
        raise ValueError('Unsupported module assembly schema; start a new run')
    return layout


def _definition_code(code):
    # Locations do not identify the definition. Strip them recursively, including
    # generators and comprehensions; their repr includes a process-local address.
    return code.replace(co_filename='', co_firstlineno=1, co_linetable=b'', co_consts=tuple(
        _definition_code(value) if inspect.iscode(value) else value for value in code.co_consts))


def _factory_identity(value):
    """Name a factory by importable location plus its definition, never by captured state.

    A qualified name alone is not an identity: every closure built by one
    ``from_factories`` call reports the same ``<locals>.build``. Hashing the code
    and source tells two different factories apart while staying stable across
    processes, so a resume still matches the assembly that wrote the workspace.
    Captured values are deliberately excluded: callbacks and paths differ between
    a run and its resume without the architecture having changed.
    """
    target = value if hasattr(value, '__qualname__') else type(value)
    module = str(getattr(target, '__module__', ''))
    qualname = str(getattr(target, '__qualname__', ''))
    if not module or not qualname:
        raise TypeError('Assembly identities need an importable factory or class')
    digest = sha256()
    code = getattr(target, '__code__', None)
    if code is not None:
        # Format 2 encodes values without process-dependent shared-object references.
        digest.update(marshal.dumps(_definition_code(code), 2))
    try:
        # The definition's own text, so an unrelated edit elsewhere in the module
        # does not invalidate every workspace that uses this factory.
        digest.update(inspect.getsource(target).encode())
    except (OSError, TypeError):
        if code is None:
            raise TypeError('Assembly identities need a verifiable factory definition: '
                            + module + ':' + qualname) from None
    return module + ':' + qualname + ':' + digest.hexdigest()[:16]


@dataclass
class AssemblyContext(ResourceScope):
    """The run owns long-lived services; a Controller version has its own ResourceScope."""
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
    _assembled: bool = field(default=False, init=False, repr=False)
    _nested_services: NestedRunServices | None = field(default=None, init=False, repr=False)
    durable: bool = False
    host_config: dict | None = None

    def record(self, kind: str, payload=None, *, references=(), decision_id: str | None = None):
        """Record a Controller-side fact as evidence, then offer it to the Controller's Memory.

        Policies use this for what they produce or conclude, e.g. kind='candidate'
        for generated code or kind='interpretation' for an LLM verdict on a receipt,
        so evidence holds more than Lab receipts. payload must be JSON; large data
        belongs in artifacts named by references. The evidence row is written first;
        a failing Memory.observe only warns, since Memory is a view rebuilt from evidence.
        """
        host = getattr(self, 'controller_host', None)
        return record_controller_event(self, host.version_id if host is not None else None,
                                       kind, payload, references, decision_id)

    def evidence_view(self, scope=None):
        from rsicomposer.loops.contracts.controller_version import EvidenceScope
        from rsicomposer.runtime.evidence.view import ReadOnlyEvidenceView
        from rsicomposer.runtime.evidence.credentials import host_secret_values
        return ReadOnlyEvidenceView(self.experience, self._artifact_access,
            EvidenceScope() if scope is None else scope, run_id=getattr(self, 'run_id', None),
            secret_values=host_secret_values(self))

    def nested_run(self):
        return nested_run(self, self)

    def nested_manifest(self, controller, labs):
        return require_nested_services(self).manifest(controller, labs)

    def controller_learning(self):
        return controller_learning(self)


DecisionFactory = Callable[[AssemblyContext, LabExecution | None], DecisionMakingLoop]
LabExecutionFactory = Callable[[AssemblyContext], LabExecution]


@dataclass(frozen=True)
class ControllerPort:
    """The Controller a run starts with, and enough identity to replace it later.

    Supplying this is what makes a run's Controller replaceable: the Host wraps
    the instance in a ControllerHost, so a run-control request or a trained
    candidate can swap the active Controller at a checkpoint boundary. A run
    that supplies only a decision factory keeps the single Controller that
    factory returns for its whole life.

    instance is a trusted BuiltController exposing endpoint, context,
    capture/restore and close. The assembly takes ownership of it once
    returned, so the factory must not close it.
    """
    instance: object
    version: object

    def __post_init__(self):
        if getattr(self.instance, 'endpoint', None) is None:
            raise TypeError('ControllerPort instance must expose its endpoint')
        if getattr(self.instance, 'context', None) is None:
            raise TypeError('ControllerPort instance must expose the context it was built in')
        for name in ('capture', 'restore', 'close'):
            if optional_method(self.instance, name) is None:
                raise TypeError(f'ControllerPort instance must provide {name}()')
        if getattr(self.version, 'version_id', None) is None:
            raise TypeError('ControllerPort requires the initial ControllerVersion')


ControllerFactory = Callable[[AssemblyContext, LabExecution | None], ControllerPort]


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
    controller_factory: ControllerFactory | None = None
    memory: object = None
    state_factory: Callable | None = None
    components: ComponentGraph | None = None
    component_bindings: Mapping = field(default_factory=dict)
    method_identity: MethodIdentity | None = None
    model_session: object | None = None
    controller_catalog: Mapping = field(default_factory=dict)
    controller_selection: dict | None = None
    controller_validators: Mapping = field(default_factory=dict, repr=False)
    _registration: RegistrySelection | None = field(default=None, repr=False)
    plugin_paths: tuple[str, ...] = field(default=(), repr=False, compare=False)
    _registered_bindings: tuple | None = field(default=None, repr=False, compare=False)
    controller_activator: object | None = field(default=None, repr=False)
    controller_learner: object | None = field(default=None, repr=False)
    observation_probes: tuple = field(default=(), repr=False, compare=False)

    def __post_init__(self):
        if (self.decision_factory is None and self.controller_factory is None
                and 'decision' not in self.component_bindings):
            raise ValueError('ModuleAssembly requires a decision_factory, a controller_factory '
                             'or a decision component binding')
        if self.controller_factory is not None and (
                self.decision_factory is not None or 'decision' in self.component_bindings):
            raise ValueError('Choose controller_factory or a fixed decision factory/binding, not both')
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
        if self.controller_factory is not None and not catalog:
            raise ValueError('controller_factory requires the controller_catalog it builds from')
        if catalog and self.controller_factory is None:
            # A catalog with no port is inert: nothing would construct the Host
            # that serves adopt_controller or rebuilds a listed entry, so a
            # declaration would silently mean "listed but unreachable".
            raise ValueError('controller_catalog requires controller_factory: a catalog without a '
                             'Controller Host cannot adopt or rebuild Controllers')
        object.__setattr__(self, 'controller_catalog', MappingProxyType(catalog))
        object.__setattr__(self, 'controller_validators', MappingProxyType(validators))
        object.__setattr__(self, 'controller_selection', selection)
        self._check_registration(catalog, validators, selection)
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
        for name in ('decision_factory', 'lab_factory', 'state_factory', 'controller_factory'):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f'{name} must be callable')
        if self.memory is not None:
            from rsicomposer.research import MemoryAssembly
            if not isinstance(self.memory, MemoryAssembly):
                raise TypeError('memory must be a MemoryAssembly')

    def _check_registration(self, catalog, validators, selection):
        if self._registration is None:
            return
        if not isinstance(self._registration, RegistrySelection):
            raise TypeError('registration must be a RegistrySelection')
        # A caller may dataclasses.replace a compiled assembly. Narrowing a catalog
        # is replayable; swapping even two closures with identical code is not.
        # These identities are checked only in this process, never written to disk.
        ports = tuple(id(port) for port in (self.decision_factory, self.lab_factory, self.controller_factory,
            self.state_factory, self.memory, self.components, self.method_identity, self.model_session,
            self.controller_learner, self.controller_activator, *self.observation_probes))
        shape = (ports, dict(self.component_bindings), config_json(selection))
        slots = frozenset((name, id(item), id(validators.get(name))) for name, item in catalog.items())
        previous = self._registered_bindings
        if previous is not None and (shape != previous[0] or not slots <= previous[1]):
            object.__setattr__(self, '_registration', None)
            object.__setattr__(self, '_registered_bindings', None)
            object.__setattr__(self, 'plugin_paths', ())
        else:
            object.__setattr__(self, '_registration', deepcopy(self._registration.restrict(catalog)))
            object.__setattr__(self, '_registered_bindings', (shape, slots))
            object.__setattr__(self, 'plugin_paths', tuple(self.plugin_paths))

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
        """Whether a loop/Lab/Controller factory is supplied directly or through the graph."""
        if role not in ('decision', 'lab'):
            raise ValueError(f'Unknown assembly role: {role}')
        if role == 'decision':
            return self.decision_factory is not None or 'decision' in self.component_bindings
        return getattr(self, role + '_factory') is not None or role in self.component_bindings

    def _factories(self, context):
        """Adapt graph outputs once, then use the same factory assembly path."""
        factories = {role: getattr(self, role + '_factory')
                     for role in ('decision', 'lab', 'controller')}
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

    @property
    def registration(self):
        """Detached selection for inspection, or None for an explicit Python assembly."""
        return deepcopy(self._registration)

    @property
    def registry_source(self):
        """Trusted parent bootstrap: portable selection plus this host's explicitly loaded paths."""
        if self._registration is not None:
            return {'selection': self._registration.configuration(), 'paths': list(self.plugin_paths)}
        return None

    @property
    def registered_labs(self):
        """Public bindings of the actual Lab factory; private callables have no registration."""
        return self.lab_factory.registrations() if isinstance(self.lab_factory, _BoundLabs) else {}

    def validate_saved(self, directory):
        """One assembly identity check for durable and snapshot restore."""
        saved = read_saved_module_layout(directory)
        saved.pop('lab_profiles', None)
        saved.pop('evaluation_operations', None)  # Validated after constructing the Lab.
        saved.pop('candidate_identity_fields', None)
        if config_json(saved) != config_json(self.configuration()):
            raise TypeError('Supply the original module assembly when restoring this workspace')

    def component_configuration(self):
        return {**self.components.configuration(), 'bindings': dict(self.component_bindings)}

    def configuration(self):
        """Stable assembly identity, shared by durable and portable restore."""
        layout = {'schema_version': _MODULE_ASSEMBLY_SCHEMA,
                  **{role + '_factory': self.has_port(role) for role in ('decision', 'lab')},
                  'controller_port': self.controller_factory is not None}
        selected = (self.controller_catalog.get(self.controller_selection['name'])
                    if self.controller_selection else None)
        layout.update(memory=self.memory is not None or bool(selected is not None and selected.memory is not None),
                      state_factory=self.state_factory is not None)
        if self.components is not None:
            layout['components'] = self.component_configuration()
        if self.method_identity is not None:
            layout['method'] = self.method_identity.configuration()
        if self._registration is not None:
            layout['registry_selection'] = self._registration.configuration()
        elif bindings := self.registered_labs:
            # Direct Python MethodDefinition.factory()/MethodCatalog.select() paths.
            layout['registered_labs'] = bindings
        if self.controller_activator is not None or self.controller_learner is not None:
            layout['controller_learning'] = {name: _factory_identity(type(value)) if value is not None else None
                for name, value in (('activator', self.controller_activator), ('learner', self.controller_learner))}
        if self.controller_catalog:
            layout['controller_catalog'] = {name: item.describe_factories(_factory_identity)
                                             for name, item in sorted(self.controller_catalog.items())}
            layout['controller_selection'] = deepcopy(self.controller_selection)
        if self.model_session is not None:
            from dataclasses import asdict
            layout['model_session'] = {'models': [asdict(m) for m in self.model_session.configured_models],
                'protocol': 'host-accounted-model-session/v1'}
        return layout


@dataclass(frozen=True)
class _BoundLabs:
    """The same frozen options describe the binding and construct its independent instances."""
    selected: tuple

    def registrations(self):
        return {alias: {'name': definition.registered_name, 'version': definition.version, 'config': json.loads(params)}
                for alias, definition, params in self.selected if definition.registered_name is not None}

    def __call__(self, context):
        from rsicomposer.runtime.orchestration.lab_group import LabGroup
        group = LabGroup(receipts=context.data_dir / 'lab_receipts' if context.data_dir is not None else None,
                         artifacts=context.artifacts, experience=context.experience,
                         artifact_access=context._artifact_access)
        # Retain partial constructions if a later factory/catalog fails.
        context.add_owned_cleanup(group.close)
        for alias, definition, config in self.selected:
            scoped = LabContext(context, alias)
            options = json.loads(config)
            capacity = options.pop('max_concurrent', 1)
            scoped.models.max_concurrent = capacity
            lab = definition.factory(scoped, options)
            group.add(alias, lab, model_scope=scoped.models.scope, max_concurrent=capacity,
                      declare_roles=scoped.resources.declare)
        return group


def _build_policy_assembly(controller: ControllerDefinition, options, selected_labs, validate_controller):
    """Share lifecycle wiring between whole methods and explicit Controller/Lab selections."""
    def controller_factory(context, execution):
        from rsicomposer.host.lifecycle import build_controller, initial_controller_version
        selection = context._controller_selection
        override = getattr(context, '_initial_controller_version', None)
        version = override or initial_controller_version(selection, context.roles.get('decision_maker'))
        name, params = version.config['name'], version.config['config']
        if name not in context._controller_catalog:
            raise ValueError('Initial version selects an unavailable Controller')
        definition = context._controller_catalog[name]
        initial_memory = getattr(context, '_initial_memory', None)
        if override is not None:
            validator = context._controller_validators.get(name)
            if validator is not None and validator(deepcopy(params)) is not None:
                raise TypeError('Controller validators must return None')
            if definition.memory is not None or name != 'current':
                initial_memory = None
        instance = build_controller(context, definition, params, version, initial=True, memory=initial_memory)
        return ControllerPort(instance, version)

    from rsicomposer.method_state import method_state_factory
    return ModuleAssembly(controller_factory=controller_factory,
                          lab_factory=_BoundLabs(tuple(selected_labs)), state_factory=method_state_factory,
                          controller_catalog={'current': controller},
                          controller_selection={'name': 'current', 'config': deepcopy(options)},
                          controller_validators=({'current': validate_controller}
                                                 if validate_controller is not None else {}))


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
    from rsicomposer.host.lifecycle import validate_controller_requirements, validate_host_endpoint
    if context.host_execution is None:
        context.host_execution = HostExecution(experience=context.experience)
        context.add_owned_cleanup(context.host_execution.close)
    context._controller_selection = modules.controller_selection
    context._controller_catalog = dict(modules.controller_catalog)
    context._controller_validators = dict(modules.controller_validators)
    context._initial_memory = modules.memory
    context._nested_services = build_nested_services(context)
    if modules.components is not None:
        if context.components is not None:
            raise RuntimeError('Components are already assembled')
        context.components = modules.components.build(context, bindings=modules.component_bindings)
    factories = modules._factories(context)
    _assemble_lab(context, factories['lab'])
    port = None
    if factories['controller'] is None:
        decision = factories['decision'](context, context.lab_execution)
        host_loop = isinstance(decision, DecisionMakingLoop) and isinstance(context.host_execution, HostExecution)
        if host_loop and decision.host_execution is not context.host_execution:
            # A borrowed endpoint's harness is still owned by its original run.
            raise ValueError('Controller endpoint belongs to another run execution')

        def close_decision():
            harness = getattr(decision, 'harness', None)
            close = optional_method(harness, 'close')
            if close is not None and not (context.components is not None and context.components.owns(harness)):
                close()
        context.add_owned_cleanup(close_decision)
        if host_loop:
            # Memory may need this execution's history while installing its own
            # tools. Check routing before binding, but authorize the final catalog.
            if decision.operation_router.lab is not context.lab_execution:
                raise ValueError('Controller endpoint must retain the assembled Lab routing')
            if context.host_execution.endpoint is not decision:
                context.host_execution.activate(decision, getattr(decision, 'controller_version', None))
            context.controller_report = optional_method(decision.policy, 'report')
    else:
        _advertise_controller_host(context)
        port = factories['controller'](context, context.lab_execution)
        if not isinstance(port, ControllerPort):
            raise TypeError('controller_factory must return a ControllerPort')
        # Keep one retryable owner across construction. Until ControllerHost is
        # ready, every failure closes the returned instance; afterwards the Host
        # closes whichever version is active, without retaining the initial one.
        controller_owner = port.instance
        context.add_owned_cleanup(lambda: controller_owner.close())
        decision = port.instance.endpoint
        validate_host_endpoint(decision, context)
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
    if port is None:
        if modules.memory is not None:
            modules.memory.install(decision, owner=context, path=context.data_dir / 'feedback_learning.json'
                                              if context.data_dir is not None else None)
        if host_loop:
            validate_host_endpoint(decision, context)
    else:
        from rsicomposer.host.lifecycle import ControllerEndpoint, build_controller
        from rsicomposer.host.lifecycle import ControllerHost
        initial = port.instance
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
        host = ControllerHost(context, initial, port.version, construct_version,
                              activator=modules.controller_activator, learner=modules.controller_learner)
        controller_owner = host
        context.controller_host = host
        version = port.version
        # Rebuild a seeded run at its original version/root. Its state adapters
        # restore the checkpoint; replaying the initial seed could rewind live Memory.
        restoring_seed = context.restoring and getattr(context, '_initial_controller_version', None) is not None
        if not restoring_seed:
            if version.policy_state is not None:
                restore = optional_method(initial.policy, 'restore')
                if restore is None or restore(deepcopy(version.policy_state)) is not None:
                    raise TypeError('Initial Controller must restore its version state explicitly')
            host._restore_memory(initial, version.memory_state)
        if version.model is not None:
            initial.bind_model(version.model)
        current_model = initial.endpoint.current_model()
        if current_model is not None:
            initial.bind_model(current_model)
        if not restoring_seed:
            validate_controller_requirements(initial.endpoint, context.lab_execution)
        decision = ControllerEndpoint(host)
        host.bind_context_callbacks()
    if modules.state_factory is not None:
        from rsicomposer.runtime.state.component_state import (
            assembly_bindings,
            validate_adapters,
        )
        adapters = validate_adapters(modules.state_factory(context, decision))
        automatic = {}
        if context.components is not None:
            automatic.update(context.components.state_adapters(adapters, lab_execution=context.lab_execution))
        if context.method_identity is not None:
            from rsicomposer.components import _StaticState
            automatic['method_manifest'] = _StaticState(context.method_identity.configuration())
        if automatic.keys() & adapters.keys():
            raise ValueError('Explicit state adapters cannot replace method/component manifest/stateless ports')
        adapters = {**automatic, **adapters}
        context.state_adapters = adapters
        context._state_bindings = assembly_bindings(context, decision)
        context._state_port_bindings = dict(context.state_adapters)
    runner = AssemblyRunner(decision, context, run_policy=run_policy,
                            cleanup_timeout_seconds=cleanup_timeout_seconds)
    if port is None:
        # On direct and graph paths, join Host workers before any dependency
        # (including graph-owned Lab/Harness resources) is released. Keep the
        # original Host close for cleanup callbacks and partial assembly failure.
        # A host-owned run needs none of this: ControllerHost.close() reaches the
        # same execution, and it is registered where the Host is built.
        context.add_owned_cleanup(context.host_execution.operation_scheduler.close)
    return runner


def _assemble_lab(context: AssemblyContext, factory):
    if factory is not None:
        context.lab_execution = factory(context)
        if context.lab_execution is None:
            raise TypeError('lab_factory must return a LabExecution provider')
        close = optional_method(context.lab_execution, 'close')
        if close is not None and not (context.components is not None and context.components.owns(context.lab_execution)):
            context.add_owned_cleanup(close)


def _advertise_controller_host(context):
    """Expose host.adopt_controller only where a Controller Host can serve it.

    The first Controller is built before the Host exists, and it still has to
    see the capability in its own action space. Rebuilding with the same
    definition and a ready Host produces the identical catalog entry.
    """
    from rsicomposer.host.lifecycle import controller_capabilities

    def request_controller_host(arguments):
        host = getattr(context, 'controller_host', None)
        if host is None:
            raise RuntimeError('host.adopt_controller requires an assembled Controller Host')
        return host.request(arguments)
    context._host_capabilities = controller_capabilities(request_controller_host)


def build_default_decision(context: AssemblyContext, execution: LabExecution, *, policy=None) -> DecisionMakingLoop:
    """Wrap an explicitly supplied Controller policy with the host harness."""
    from rsicomposer.harnesses.defaults import build_default_decision as build
    return build(context, execution, policy=policy)


class AssemblyRunner(Runner):
    """Runner sharing its assembly's admission and owned cleanup lifecycle."""

    def __init__(self, decision: DecisionMakingLoop, assembly: AssemblyContext,
                 *, run_policy: RunPolicy | None = None, cleanup_timeout_seconds: float = 30):
        from rsicomposer.runtime.orchestration.lab_group import LabGroup
        lab = assembly.lab_execution
        if not isinstance(lab, LabGroup):
            from collections.abc import Mapping
            from rsicomposer.loops.contracts.profiles import Invariant, SplitRole
            if (declare := optional_method(lab, 'split_roles')) is not None:
                roles = declare()
                if not isinstance(roles, Mapping) or any(not isinstance(role, SplitRole) for role in roles.values()):
                    raise TypeError('Lab split_roles() must map split names to SplitRole')
                if any(role.after_finish_only for role in roles.values()):
                    raise ValueError('Sealed splits require a Lab composed through LabGroup or the Registry')
            if (declare := optional_method(lab, 'invariants')) is not None:
                rules = declare()
                if not isinstance(rules, tuple) or any(not isinstance(rule, Invariant) for rule in rules):
                    raise TypeError('Lab invariants() must return an Invariant tuple')
                if any(rule.kind in ('min_candidates', 'measured_selection') for rule in rules):
                    raise ValueError('Candidate history invariants require a Lab composed through LabGroup or the Registry')
        super().__init__(decision, run_policy=run_policy,
                         cleanup_timeout_seconds=cleanup_timeout_seconds,
                         execution_shutdown=optional_method(lab, 'shutdown'),
                         finish_execution=optional_method(lab, 'finish_step'))
        self.assembly = assembly
        self._evaluation_operations = None
        self._declared_evaluations = None
        if isinstance(assembly.lab_execution, LabGroup):
            self._evaluation_operations = assembly.lab_execution.is_evaluation
        elif optional_method(assembly.lab_execution, 'evaluation_operations') is not None:
            from rsicomposer.runtime.orchestration.profiles import evaluation_operations
            self._declared_evaluations = evaluation_operations(assembly.lab_execution,
                                                               assembly.lab_execution.describe(), 'execution')
            self._evaluation_operations = self._declared_evaluations.__contains__
        from rsicomposer.host.execution import HostExecution
        if isinstance(assembly.host_execution, HostExecution):
            assembly.host_execution.bind_evaluations(lab, self._evaluation_operations, grouped=isinstance(lab, LabGroup))
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
                    outcome = execution.evaluation_outcome(OperationCall(**call), receipt)
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
