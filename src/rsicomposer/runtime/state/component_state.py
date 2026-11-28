"""Aggregate explicit module state ports for recovery and portable snapshots."""
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256

from rsicomposer.runtime.capabilities import optional_method
from rsicomposer.runtime.storage.codec import encode, same_metadata
from rsicomposer.runtime.storage.files import json_bytes


def validate_adapters(adapters):
    if not isinstance(adapters, Mapping) or not adapters:
        raise TypeError('state_factory must return a nonempty mapping of component state adapters')
    result = dict(adapters)
    for name, adapter in result.items():
        if not isinstance(name, str) or not name.strip():
            raise TypeError('Component state names must be nonempty strings')
        if (not isinstance(getattr(adapter, 'state_id', None), str) or not adapter.state_id.strip()
                or optional_method(adapter, 'capture') is None or optional_method(adapter, 'restore') is None):
            raise TypeError(f'Component state adapter {name} needs state_id, capture(), restore(state)')
    return result


def assembly_bindings(context, decision):
    # Public composition boundaries only. Internal dependency state is the
    # adapter's responsibility; the host never walks a plugin's object graph.
    bindings = optional_method(decision, 'recovery_bindings')
    controller = (bindings() if bindings is not None else
                  (decision, getattr(decision, 'harness', None), getattr(decision, 'action_space', None),
                   getattr(decision, 'integration', None), getattr(decision, 'host_execution', None)))
    if not isinstance(controller, tuple):
        raise TypeError('Controller recovery bindings must be an explicit tuple')
    return (*controller, context.host_execution, context.lab_execution, context.method_identity, context.components,
            *(context.components.values() if context.components is not None else ()))


class AssemblyRecovery:
    """One state envelope for registered modules, not an implicit serializer.

    Every top-level component needs an explicit adapter, even if it is stateless.
    An adapter may cover dependent policies/tools/memory/backends. Extra named
    ports are allowed. Configuration compatibility belongs in state_id or the
    adapter's optional configuration() metadata and restore validation.

    Required ports include decision and installed Lab/host execution owners.
    The decision adapter owns its explicit integration, including Memory and feedback.
    A graph-bound Lab uses the top-level lab_execution adapter. Other graph
    slots (including the decision factory) need component:<slot> ports;
    manifests and explicitly stateless independent slots have host-owned adapters.
    restore and each portable validate_snapshot hook must return None.
    """
    contract_id = 'rsicomposer-assembly-state/v9'

    def _adapters(self, app):
        if app.store is None or app._closed:
            raise RuntimeError('Application is closed')
        context = app.assembly
        live = assembly_bindings(context, app.runner.decision_making)
        bound = getattr(context, '_state_bindings', ())
        if len(bound) != len(live) or any(a is not b for a, b in zip(bound, live)):
            raise TypeError('Assembled components changed; rebuild their state integration')
        adapters = validate_adapters(context.state_adapters)
        if any(context._state_port_bindings.get(k) is not v for k, v in adapters.items()) or (
                adapters.keys() != context._state_port_bindings.keys()):
            raise TypeError('Component state adapters changed after assembly')
        required = {'decision'}
        execution = getattr(app.runner.decision_making, 'host_execution', None)
        if execution is not None:
            if execution is not context.host_execution:
                raise TypeError('Controller execution differs from the run execution owner')
            required.add('execution')
        if context.method_identity is not None:
            required.add('method_manifest')
        if context.components is not None:
            required.update(context.components.required_state_adapters)
        if context.lab_execution is not None:
            required.add('lab_execution')
        missing = required - adapters.keys()
        if missing:
            raise TypeError('Missing component state adapters: ' + ', '.join(sorted(missing)))
        return adapters

    def _metadata(self, adapter):
        configuration = optional_method(adapter, 'configuration')
        metadata = {'state_id': adapter.state_id,
                    'configuration': deepcopy(configuration()) if configuration else None}
        json_bytes(encode(metadata))
        return metadata

    def capture(self, app):
        adapters = self._adapters(app)
        result = {name: {**self._metadata(adapter), 'state': deepcopy(adapter.capture())}
                  for name, adapter in sorted(adapters.items())}
        json_bytes(encode(result))
        return result

    @staticmethod
    def _file_ports(adapters, directory):
        for name, adapter in adapters.items():
            export = optional_method(adapter, 'export_snapshot_files')
            load = optional_method(adapter, 'import_snapshot_files')
            if (export is None) != (load is None):
                raise TypeError(f'Component {name} must provide both portable file ports')
            if export is not None:
                yield name, directory / sha256(name.encode()).hexdigest(), export, load

    def export_snapshot_files(self, app, directory, state):
        """Bundle component-owned files on portable capture, never on a durable tick."""
        for name, path, export, _ in self._file_ports(self._adapters(app), directory):
            path.mkdir(parents=True)
            if export(path, state[name]['state']) is not None:
                raise TypeError(f'Component {name} file export must return None')

    def restore(self, app, state):
        self._restore(app, state)

    def restore_portable(self, app, state, directory):
        self._restore(app, state, directory)

    def _restore(self, app, state, directory=None):
        adapters = self._adapters(app)
        if not isinstance(state, dict) or state.keys() != adapters.keys():
            raise TypeError('Component state inventory differs from the saved assembly')
        # Check every identity/configuration before invoking any restore hook.
        for name, adapter in adapters.items():
            entry = state[name]
            if (not isinstance(entry, dict) or set(entry) != {'state_id', 'configuration', 'state'}
                    or not same_metadata({k: entry[k] for k in ('state_id', 'configuration')}, self._metadata(adapter))):
                raise TypeError(f'Component state identity/configuration mismatch: {name}')
        if directory is not None:
            for name, path, _, load in self._file_ports(adapters, directory):
                if not path.is_dir():
                    raise ValueError(f'Missing portable files for component {name}')
                if load(path, deepcopy(state[name]['state'])) is not None:
                    raise TypeError(f'Component {name} file import must return None')
        for name, adapter in adapters.items():
            if adapter.restore(deepcopy(state[name]['state'])) is not None:
                raise TypeError(f'Component restore must return None: {name}')
        # A custom adapter can restore the Lab after the Controller. Validate
        # the actual pair once both ports and their dependencies hold saved state.
        from rsicomposer.host.execution import HostExecution
        from rsicomposer.host.lifecycle import ControllerEndpoint, validate_controller_requirements
        from rsicomposer.loops.decision_making.loop import DecisionMakingLoop
        decision = app.runner.decision_making
        if isinstance(decision, ControllerEndpoint):
            decision = decision.host.active.endpoint
        elif not (isinstance(decision, DecisionMakingLoop) and isinstance(app.assembly.host_execution, HostExecution)
                  and decision.host_execution is app.assembly.host_execution):
            return
        validate_controller_requirements(decision, app.assembly.lab_execution)

    def validate_snapshot(self, app):
        if app.recovery is not None and app.recovery.intervention is not None:
            raise RuntimeError('Reconcile interrupted durable recovery before creating a snapshot')
        pending_job = optional_method(app.runner.decision_making, 'pending_research_job')
        pending_records = optional_method(app.runner.decision_making, 'pending_records')
        if (app.runner.busy or app.sandbox.active_count
                or (pending_job is not None and pending_job() is not None)
                or (pending_records is not None and pending_records())):
            raise RuntimeError('Snapshot requires a quiescent assembly with no pending work')
        app.assembly.resources.validate_snapshot()
        app.assembly.executor.validate_snapshot()
        for name, adapter in self._adapters(app).items():
            validate = optional_method(adapter, 'validate_snapshot')
            if validate is None:
                raise TypeError(f'Component {name} needs explicit portable snapshot integration')
            if validate() is not None:
                raise TypeError(f'Component snapshot validation must return None: {name}')


def snapshot_integration(app):
    """Whether current components permit a portable snapshot, without capturing any."""
    if app._requires_parent or not app.assembly.state_adapters:
        return False
    try:
        for adapter in AssemblyRecovery()._adapters(app).values():
            validate = optional_method(adapter, 'validate_snapshot')
            if validate is None or validate() is not None:
                return False
    except (TypeError, RuntimeError):
        return False
    return True
