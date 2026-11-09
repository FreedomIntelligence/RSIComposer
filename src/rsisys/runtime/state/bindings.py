"""Journaled in-place component/model activation, independent of any RSI method or scene."""
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

from rsisys.runtime.state.systems import SystemActivationRejected, SystemRegistryState
from rsisys.runtime.storage.files import atomic_write, json_bytes


@dataclass(frozen=True)
class ActivationPort:
    """Trusted host implementation. prepare/read are observational.

    apply/restore must be bounded, idempotent local binding operations. External
    deployment requires its own durable activator with readiness reconciliation.
    Values are copied JSON; readback must equal the prepared value exactly.
    """
    version: str
    prepare: object
    read: object
    apply: object
    role: str | None = None

    def __post_init__(self):
        if not self.version or not all(callable(v) for v in (self.prepare, self.read, self.apply)):
            raise TypeError('Activation ports need a version and prepare/read/apply callbacks')


class BindingSystemActivator:
    def __init__(self, ports, artifacts, *, models=None, path=None):
        self.ports, self.artifacts, self.models = dict(ports), artifacts, dict(models or {})
        self.initial_values = json.loads(json.dumps({
            **{name: port.read() for name, port in self.ports.items()},
            **{'model:' + name: port.read() for name, port in self.models.items()}}, allow_nan=False))
        self.path = Path(path) if path is not None else None
        self.state = {'pending': None, 'receipts': {}}
        if self.path is not None and self.path.exists():
            self.state = json.loads(self.path.read_text())
        else:
            self._save()

    def configuration(self):
        return {'ports': {k: {'version': v.version, 'role': v.role} for k, v in self.ports.items()},
                'models': {k: v.version for k, v in self.models.items()},
                'initial_values_sha256': hashlib.sha256(json_bytes(self.initial_values)).hexdigest()}

    def _save(self):
        if self.path is not None:
            atomic_write(self.path, json_bytes(self.state))

    def _prepare(self, system):
        values = deepcopy(self.initial_values)
        for name, ref in system.components.items():
            if name not in self.ports:
                raise SystemActivationRejected('No installed host activation port: ' + name)
            raw = self.artifacts.get(ref)
            if ref.digest != hashlib.sha256(raw).hexdigest():
                raise SystemActivationRejected('Candidate component digest differs')
            values[name] = self.ports[name].prepare(json.loads(raw))
        for name, model in system.models.items():
            if name not in self.models:
                raise SystemActivationRejected('No installed model binding port: ' + name)
            values['model:' + name] = self.models[name].prepare(asdict(model))
        # Complete preparation and JSON validation before the first mutation.
        return json.loads(json.dumps(values, allow_nan=False))

    def _port(self, name):
        return self.models[name[6:]] if name.startswith('model:') else self.ports[name]

    def activate(self, system, *, idempotency_key):
        identity = hashlib.sha256(json_bytes(asdict(system))).hexdigest()
        receipt = self.state['receipts'].get(idempotency_key)
        if receipt is not None:
            if receipt['identity'] != identity:
                raise ValueError('Activation key identifies another manifest')
            if any(self._port(k).read() != v for k, v in receipt['values'].items()):
                raise RuntimeError('Activation receipt differs from live bindings; reconcile before retry')
            return deepcopy(system)
        if self.state['pending'] is not None:
            # A crash between two in-place assignments cannot be retried as if
            # nothing happened. Keep the intent for evidence-backed recovery.
            raise RuntimeError('Unconfirmed binding activation requires reconciliation')
        try:
            values = self._prepare(system)
        except (ValueError, TypeError, KeyError) as exc:
            raise SystemActivationRejected(str(exc)) from exc
        previous = {k: deepcopy(self._port(k).read()) for k in values}
        self.state['pending'] = {'key': idempotency_key, 'identity': identity,
                                 'previous': previous, 'values': values}
        self._save()
        try:
            for name, value in values.items():
                self._port(name).apply(deepcopy(value))
            if any(self._port(k).read() != v for k, v in values.items()):
                raise RuntimeError('Activation did not confirm requested bindings')
        except BaseException as exc:
            try:
                for name, value in reversed(tuple(previous.items())):
                    self._port(name).apply(deepcopy(value))
                if any(self._port(k).read() != v for k, v in previous.items()):
                    raise RuntimeError('Activation rollback readback differs')
            except BaseException as rollback:
                exc.add_note('Rollback unconfirmed: ' + str(rollback))
                raise
            self.state['pending'] = None
            self._save()
            raise SystemActivationRejected('Bindings rolled back: ' + str(exc)) from exc
        self.state['receipts'][idempotency_key] = {'identity': identity, 'values': values}
        self.state['pending'] = None
        self._save()
        return deepcopy(system)

    def restore_bindings(self, system):
        if self.state['pending'] is not None:
            raise RuntimeError('Unconfirmed binding activation requires reconciliation')
        for name, value in self._prepare(system).items():
            self._port(name).apply(deepcopy(value))
            if self._port(name).read() != value:
                raise RuntimeError('Restored binding differs from current system')


class EvolutionState(SystemRegistryState):
    state_id = 'live-system-evolution/v1'

    def __init__(self, registry):
        super().__init__(registry, configuration=registry.activator.configuration())

    def capture(self):
        activator = self.registry.activator
        if activator.state['pending'] is None:
            expected = activator._prepare(self.registry.current)
            if any(activator._port(k).read() != v for k, v in expected.items()):
                raise TypeError('Live bindings changed outside the versioned activation boundary')
        return {'registry': super().capture(), 'activator': deepcopy(self.registry.activator.state)}

    def restore(self, state):
        if self.registry.activator.state != state['activator']:
            from rsisys.runtime.state.recovery import RecoveryRequired
            raise RecoveryRequired('Activation journal advanced after checkpoint')
        super().restore(state['registry'])
        self.registry.activator.restore_bindings(self.registry.current)
