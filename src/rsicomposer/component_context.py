"""Explicit services bound to registered method components."""
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from uuid import uuid4
import warnings

from rsicomposer.runtime.capabilities import optional_method
from rsicomposer.runtime.evidence.retrieval import controller_experience


@dataclass(frozen=True)
class _Cleanup:
    callback: Callable
    owned: bool
    claims: tuple = ()


@dataclass(eq=False)
class ResourceScope:
    """One owner's cleanup stack. A failed owned close retains earlier dependencies."""
    _cleanup: list[_Cleanup] = field(default_factory=list, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def _register(self, callback, *, owned, claims=()):
        if self._closed:
            raise RuntimeError('Resource scope is closed')
        if not callable(callback):
            raise TypeError('Cleanup must be callable')
        if not any(callback in entry.claims for entry in self._cleanup):
            self._cleanup.append(_Cleanup(callback, owned, claims))

    def claim_close(self, callback, *, aliases=()):
        """Take a returned part from its factory, once, without reordering an existing claim."""
        if callback is not None and any(callback in entry.claims for entry in self._cleanup):
            return
        claims = tuple(close for close in (callback, *aliases) if close is not None)
        self.relinquish_cleanups(lambda close: close in claims)
        if callback is not None:
            self._register(callback, owned=True, claims=claims)

    def add_cleanup(self, callback):
        """Run a callback once, including when its effects are uncertain after failure."""
        self._register(callback, owned=False)

    def add_owned_cleanup(self, callback):
        """Retry the close on failure before releasing any earlier dependencies."""
        self._register(callback, owned=True)

    def owned_resources(self):
        """Bound resources already claimed by this scope, not arbitrary callback state."""
        return tuple(resource for entry in self._cleanup if entry.owned
                     and (resource := getattr(entry.callback, '__self__', None)) is not None)

    def relinquish_cleanups(self, matches):
        """Forget matching registrations without running any close."""
        self._cleanup[:] = [entry for entry in self._cleanup if not matches(entry.callback)]

    @property
    def cleanup_pending(self):
        return bool(self._cleanup)

    def close(self):
        self._closed = True
        errors = []
        while self._cleanup:
            entry = self._cleanup[-1]
            try:
                entry.callback()
            except BaseException as exc:
                errors.append(exc)
                if entry.owned:
                    break
            self._cleanup.pop()
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup('Resource cleanup failed', errors)


def record_controller_event(owner, version, kind, payload, references, decision_id):
    """Append before notifying this version's observer: Memory can rebuild from evidence."""
    from rsicomposer.loops.contracts.feedback_context import EvidenceEvent
    from rsicomposer.runtime.evidence.records import ExperienceRecord
    from rsicomposer.runtime.storage.files import json_bytes
    if owner._closed:
        raise RuntimeError('Controller evidence owner is closed')
    event = EvidenceEvent(event_id=str(uuid4()), source='controller', kind=kind,
                          output=deepcopy(payload), references=deepcopy(tuple(references)),
                          decision_id=decision_id, identity={'controller_version': version} if version else {})
    details = {'event': asdict(event)}
    json_bytes(details)
    owner.experience.record(ExperienceRecord(event.event_id, 'controller_event', 'decision_maker', None,
                                            f'Controller {kind}', False, details=details))
    if owner._memory_observer is not None:
        try:
            owner._memory_observer(deepcopy(event))
        except Exception as exc:
            warnings.warn(f'Memory.observe failed for controller event {event.event_id}: '
                          f'{type(exc).__name__}: {exc}', RuntimeWarning, stacklevel=3)
    return event


@dataclass(eq=False)
class ControllerVersionScope(ResourceScope):
    """Resources and mutable bindings retired together; run services are not owned here."""
    version_id: str
    data_dir: Path | None
    roles: dict
    experience: object
    validate_model: Callable
    restoring: bool
    resuming: bool
    controller_report: Callable | None = None
    _memory_observer: Callable | None = field(default=None, init=False, repr=False)

    def record(self, kind, payload=None, *, references=(), decision_id=None):
        return record_controller_event(self, self.version_id, kind, payload, references, decision_id)


class LabCatalog:
    """A detached view of installed operations; grouping and metrics are optional."""
    def __init__(self, execution):
        self._specs = deepcopy(tuple(execution.describe())) if execution is not None else ()
        metadata = optional_method(execution, 'catalog_metrics')
        if metadata is None:
            self.aliases = tuple(dict.fromkeys(spec.name.rpartition('.')[0]
                                               for spec in self._specs if '.' in spec.name))
            self._metrics = {}
        else:
            metrics = metadata()
            self.aliases = tuple(metrics)
            self._metrics = {alias: deepcopy(metric) for alias, metric in metrics.items() if metric is not None}

    def describe(self):
        return deepcopy(self._specs)

    def primary_metric(self, alias):
        try:
            return self._metrics[alias]
        except KeyError:
            mounted = ', '.join(sorted(self.aliases)) or 'none'
            raise KeyError(f'Lab alias {alias!r} declares no primary metric; mounted aliases: {mounted}') from None


class _ComponentContext:
    def __init__(self, run, name, owner):
        from rsicomposer.runtime.execution.weight_sync import component_weight_sync
        self._roles = lambda: MappingProxyType(dict(owner.roles))
        self.data_dir, self.restoring, self.resuming = owner.data_dir, owner.restoring, owner.resuming
        self.execution, self.sandbox = run.execution, run.sandbox
        self.artifacts = run._artifact_access.component(name)
        self.models = run._model_access.component(name)
        self.backends, self.model_session = self.models.backends, self.models
        self.resources = run.resources.component(name)
        self.weight_sync = component_weight_sync(run.weight_sync, name, self.artifacts)
        self.executor = run.executor.component(name) if callable(getattr(run.executor, 'component', None)) else run.executor
        self.evidence_view = run.evidence_view
        self.add_cleanup, self.add_owned_cleanup = owner.add_cleanup, owner.add_owned_cleanup
        if getattr(run, '_worker_services', None) is not None:
            self.forward_socket = partial(run._worker_services.add_socket, name)

    @property
    def roles(self):
        return self._roles()


class ControllerContext(_ComponentContext):
    def __init__(self, run, version):
        super().__init__(run, 'controller', version)
        self.lab_catalog = LabCatalog(run.lab_execution)
        self.record = version.record
        self.parent_run_id = getattr(run, 'parent_run_id', None)
        self._wiring = decision_wiring(run)


class EndpointContext(ControllerContext):
    @property
    def host_execution(self):
        return self._wiring.host_execution


@dataclass(frozen=True)
class DecisionWiring:
    lab_execution: object
    host_execution: object
    host_capabilities: object
    action_permissions: object
    experience: object
    experience_limit: int
    memory_limit: int
    project_context: object


def decision_wiring(context):
    """Supply the built-in Harness with only the runtime ports it consumes."""
    if isinstance(context, ControllerContext):
        return context._wiring
    return DecisionWiring(context.lab_execution, context.host_execution, getattr(context, '_host_capabilities', None),
                          context.action_permissions, controller_experience(context.experience),
                          context.experience_limit, context.memory_limit, context.project_context)


def controller_model_access(context):
    return context.models if isinstance(context, ControllerContext) else context._model_access.component('controller')


def controller_learning(owner):
    service = owner.controller_host.learning
    return SimpleNamespace(train=service.train, candidate=service.candidate, poll=service.poll,
                           cancel=service.cancel, reattach=service.training.request,
                           validate_snapshot=service.validate_snapshot)


@dataclass(frozen=True)
class NestedRunServices:
    host_config: dict
    manifest: Callable
    prepare_models: Callable


def build_nested_services(owner):
    if owner.host_config is None:
        return None
    if not isinstance(owner.host_config, dict) or not isinstance(owner.host_config.get('application'), dict):
        raise ValueError('Nested runs require host_config with a parent application configuration')
    from rsicomposer.loops.contracts.controller_version import ControllerVersion
    from rsicomposer.nested_worker import build_nested_manifest
    from rsicomposer.runtime.models.delegation import prepare_nested_models

    config = deepcopy(owner.host_config)

    def manifest(controller, labs):
        host = getattr(owner, 'controller_host', None)
        inherited = host.approve_memory_history(controller) if host is not None and isinstance(controller, ControllerVersion) else ()
        return build_nested_manifest(controller, labs, config, catalog=owner._controller_catalog,
                                     permissions=owner.action_permissions, inherited_history=inherited)

    return NestedRunServices(config, manifest, partial(prepare_nested_models, owner._model_access))


def require_nested_services(owner):
    if owner._nested_services is None:
        raise ValueError('Nested runs require an Application or an AssemblyContext with explicit host_config')
    return owner._nested_services


def nested_run(owner, component):
    from rsicomposer.runtime.execution.nested import NestedRun
    services = require_nested_services(owner)
    if component.data_dir is None:
        raise ValueError('Nested runs require an explicit data_dir')
    return NestedRun(SimpleNamespace(data_dir=component.data_dir, resources=component.resources,
        executor=component.executor, artifacts=component.artifacts, budget=owner.budget,
        nested_manifest=services.manifest, prepare_nested_models=services.prepare_models,
        host_config=services.host_config))


class LabContext(_ComponentContext):
    def __init__(self, owner, alias):
        super().__init__(owner, 'lab:' + alias, owner)
        self.nested_manifest = owner.nested_manifest
        self.nested_run = lambda: nested_run(owner, self)
        self.controller_learning = lambda: controller_learning(owner)
