"""Explicit services bound to registered method components."""
from copy import deepcopy
from types import MappingProxyType, SimpleNamespace


class LabCatalog:
    def __init__(self, group):
        self.aliases = group.aliases
        self._specs = group.describe()
        self._metrics = {alias: group.primary_metric(alias) for alias in self.aliases}

    def describe(self):
        return deepcopy(self._specs)

    def primary_metric(self, alias):
        return self._metrics[alias]


class _ComponentContext:
    def __init__(self, owner, name):
        from rsisys.runtime.execution.weight_sync import component_weight_sync
        self._roles = lambda: MappingProxyType(dict(owner.roles))
        self.data_dir, self.restoring, self.resuming = owner.data_dir, owner.restoring, owner.resuming
        self.execution, self.sandbox = owner.execution, owner.sandbox
        self.artifacts = owner._artifact_access.component(name)
        self.models = owner._model_access.component(name)
        self.backends, self.model_session = self.models.backends, self.models
        self.resources = owner.resources.component(name)
        self.weight_sync = component_weight_sync(owner.weight_sync, name, self.artifacts)
        self.executor = owner.executor.component(name) if callable(getattr(owner.executor, 'component', None)) else owner.executor
        self.evidence_view = owner.evidence_view
        self.add_cleanup, self.add_owned_cleanup = owner.add_cleanup, owner.add_owned_cleanup
        if getattr(owner, '_worker_services', None) is not None:
            self.forward_socket = lambda path: owner._worker_services.add_socket(name, path)

    @property
    def roles(self):
        return self._roles()


class ControllerContext(_ComponentContext):
    def __init__(self, owner, wiring):
        super().__init__(owner, 'controller')
        self.lab_catalog = LabCatalog(wiring.lab_execution)
        self.record = owner.record
        self.parent_run_id = getattr(wiring, 'parent_run_id', None)
        self._wiring = wiring


class EndpointContext(ControllerContext):
    @property
    def host_execution(self):
        return self._wiring.host_execution


def controller_wiring(context):
    return context._wiring if isinstance(context, ControllerContext) else context


def controller_learning(owner):
    service = owner.controller_host.learning
    return SimpleNamespace(train=service.train, candidate=service.candidate, poll=service.poll,
                           cancel=service.cancel, reattach=service.training.request,
                           validate_snapshot=service.validate_snapshot)


def nested_run(owner, component):
    from rsisys.runtime.execution.nested import NestedRun
    return NestedRun(SimpleNamespace(data_dir=component.data_dir, resources=component.resources,
        executor=component.executor, artifacts=component.artifacts, budget=owner.budget,
        nested_manifest=owner.nested_manifest, prepare_nested_models=owner.prepare_nested_models,
        host_config=owner.host_config))


class LabContext(_ComponentContext):
    def __init__(self, owner, alias):
        super().__init__(owner, 'lab:' + alias)
        self.nested_manifest = owner.nested_manifest
        self.nested_run = lambda: nested_run(owner, self)
        self.controller_learning = lambda: controller_learning(owner)
