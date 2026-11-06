"""Aggregate explicit module state ports for recovery and portable snapshots."""
from collections.abc import Mapping
from copy import deepcopy

from .capabilities import optional_method


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


def assembly_bindings(context, decision, exploration):
    # Public composition boundaries only. Internal dependency state is the
    # adapter's responsibility; the host never walks a plugin's object graph.
    return (decision, getattr(decision, 'harness', None), getattr(decision, 'action_space', None),
            exploration, getattr(exploration, 'harness', None), getattr(exploration, 'action_space', None),
            context.lab_execution, getattr(decision, 'feedback_service', None),
            context.components, *(context.components.values() if context.components is not None else ()))


class AssemblyRecovery:
    """One state envelope for registered modules, not an implicit serializer.

    Every top-level component needs an explicit adapter, even if it is stateless.
    An adapter may cover dependent policies/tools/memory/backends. Extra named
    ports are allowed. Configuration compatibility belongs in state_id or the
    adapter's optional configuration() metadata and restore validation.
    """
    contract_id = 'rsisys-assembly-state/v1'

    def _adapters(self, app):
        if app.store is None or app._closed:
            raise RuntimeError('Application is closed')
        context = app.assembly
        live = assembly_bindings(context, app.runner.decision_making, app.runner.exploration)
        bound = getattr(context, '_state_bindings', ())
        if len(bound) != len(live) or any(a is not b for a, b in zip(bound, live)):
            raise TypeError('Assembled components changed; rebuild their state integration')
        adapters = validate_adapters(context.state_adapters)
        if any(context._state_port_bindings.get(k) is not v for k, v in adapters.items()) or (
                adapters.keys() != context._state_port_bindings.keys()):
            raise TypeError('Component state adapters changed after assembly')
        required = {'decision', 'exploration'}
        if context.components is not None:
            required.update(context.components.required_state_adapters)
        if context.lab_execution is not None:
            required.add('lab_execution')
        if getattr(app.runner.decision_making, 'feedback_service', None) is not None:
            required.add('feedback')
        if getattr(app.runner.decision_making.harness, 'agent_memory', None) is not None:
            required.add('agent_memory')
        missing = required - adapters.keys()
        if missing:
            raise TypeError('Missing component state adapters: ' + ', '.join(sorted(missing)))
        return adapters

    def _metadata(self, adapter):
        from rsisys.snapshot.codec import encode
        from .files import json_bytes
        configuration = optional_method(adapter, 'configuration')
        metadata = {'state_id': adapter.state_id,
                    'configuration': deepcopy(configuration()) if configuration else None}
        json_bytes(encode(metadata))
        return metadata

    def capture(self, app):
        from rsisys.snapshot.codec import encode
        from .files import json_bytes
        adapters = self._adapters(app)
        result = {name: {**self._metadata(adapter), 'state': deepcopy(adapter.capture())}
                  for name, adapter in sorted(adapters.items())}
        json_bytes(encode(result))
        return result

    def restore(self, app, state):
        adapters = self._adapters(app)
        if not isinstance(state, dict) or state.keys() != adapters.keys():
            raise TypeError('Component state inventory differs from the saved assembly')
        # Check every identity/configuration before invoking any restore hook.
        for name, adapter in adapters.items():
            entry = state[name]
            if (not isinstance(entry, dict) or set(entry) != {'state_id', 'configuration', 'state'}
                    or {k: entry[k] for k in ('state_id', 'configuration')} != self._metadata(adapter)):
                raise TypeError(f'Component state identity/configuration mismatch: {name}')
        for name, adapter in adapters.items():
            if adapter.restore(deepcopy(state[name]['state'])) is not None:
                raise TypeError(f'Component restore must return None: {name}')

    def validate_snapshot(self, app):
        if app.recovery is not None and app.recovery.intervention is not None:
            raise RuntimeError('Reconcile interrupted durable recovery before creating a snapshot')
        if (app.runner.parallel or app.runner.busy or app.sandbox.active_count
                or app.runner.decision_making.pending_research_job() is not None
                or app.runner.decision_making.pending_records()):
            raise RuntimeError('Snapshot requires a quiescent assembly with no pending work')
        for name, adapter in self._adapters(app).items():
            validate = optional_method(adapter, 'validate_snapshot')
            if validate is None:
                raise TypeError(f'Component {name} needs explicit portable snapshot integration')
            if validate() is not None:
                raise TypeError(f'Component snapshot validation must return None: {name}')
