"""Persistent host admission: wall time, provider attempts and GPU lease time."""
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import threading
import time

from rsicomposer.runtime.storage.files import atomic_write, json_bytes


class BudgetExhausted(RuntimeError):
    pass


def validate_budget(limits):
    if not isinstance(limits, dict) or set(limits) - {'seconds', 'model_calls', 'gpu_seconds'}:
        raise ValueError('budget supports seconds, model_calls, gpu_seconds')
    for key, value in limits.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'budget.{key} must be finite and positive')
        if key == 'model_calls' and type(value) is not int:
            raise ValueError('budget.model_calls must be an integer')


class HostBudget:
    def __init__(self, root, limits=None, *, restoring=False, _started_at=None):
        self.root, self.limits = Path(root).resolve(), dict(limits or {})
        validate_budget(self.limits)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'budget.json'
        self._lock = threading.RLock()
        with self._transaction(create=True) as state:
            if state:
                if not restoring or state.get('schema') != 3 or state.get('limits') != self.limits:
                    raise ValueError('Cannot reset or change host budget identity')
                if _started_at is not None and state['started_at'] != _started_at:
                    raise ValueError('Child budget allocation identity changed')
            elif restoring:
                raise ValueError('Saved host budget is missing')
            else:
                state.update(schema=3, limits=self.limits, started_at=time.time() if _started_at is None else _started_at,
                             model_calls=0, gpu_seconds=0., gpu_lease_seconds=0., leases={}, allocations={})

    @contextmanager
    def _transaction(self, *, create=False):
        with self._lock, (self.root / 'budget.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else {}
            if not state and not create:
                raise ValueError('Saved host budget is missing')
            if state:
                self._validate(state)
            previous = json_bytes(state)
            yield state
            self._validate(state)
            if json_bytes(state) != previous:
                atomic_write(self.path, json_bytes(state))

    def _validate(self, state):
        if set(state) != {'schema', 'limits', 'started_at', 'model_calls', 'gpu_seconds',
                          'gpu_lease_seconds', 'leases', 'allocations'}:
            raise ValueError('Unsupported host budget state fields')
        if state['schema'] != 3 or state['limits'] != self.limits:
            raise ValueError('Host budget identity changed')
        if (type(state['model_calls']) is not int or state['model_calls'] < 0
                or state['model_calls'] > self.limits.get('model_calls', float('inf'))
                or type(state['gpu_seconds']) not in (int, float) or not math.isfinite(state['gpu_seconds'])
                or state['gpu_seconds'] < 0 or type(state['started_at']) not in (int, float)
                or type(state['gpu_lease_seconds']) not in (int, float)
                or not math.isfinite(state['gpu_lease_seconds']) or state['gpu_lease_seconds'] < 0
                or not math.isfinite(state['started_at']) or not isinstance(state['leases'], dict)
                or not isinstance(state['allocations'], dict)):
            raise ValueError('Invalid host budget counters')
        for ident, value in state['leases'].items():
            if (not isinstance(value, dict) or set(value) != {'started_at', 'gpus', 'marker'}
                    or type(value['gpus']) is not int or value['gpus'] < 0
                    or type(value['started_at']) not in (int, float) or not math.isfinite(value['started_at'])
                    or not isinstance(ident, str) or len(ident) != 32
                    or any(c not in '0123456789abcdef' for c in ident)
                    or value['marker'] != str(self.root / 'leases' / (ident + '.lock'))):
                raise ValueError('Invalid persisted resource lease')

        for ident, value in state['allocations'].items():
            if (not self._valid_allocation_id(ident) or not isinstance(value, dict)
                    or set(value) != {'root', 'limits', 'requested', 'created_at', 'state', 'used', 'gpu_overrun'}
                    or not isinstance(value['root'], str) or not Path(value['root']).is_absolute()
                    or '..' in Path(value['root']).parts or Path(value['root']) == self.root
                    or type(value['created_at']) not in (int, float) or not math.isfinite(value['created_at'])
                    or value['state'] not in ('reserved', 'settled')
                    or not isinstance(value['used'], dict) or set(value['used']) != {'model_calls', 'gpu_seconds'}
                    or type(value['used']['model_calls']) is not int or value['used']['model_calls'] < 0
                    or type(value['used']['gpu_seconds']) not in (int, float)
                    or not math.isfinite(value['used']['gpu_seconds']) or value['used']['gpu_seconds'] < 0
                    or type(value['gpu_overrun']) not in (int, float)
                    or not math.isfinite(value['gpu_overrun']) or value['gpu_overrun'] < 0):
                raise ValueError('Invalid nested budget allocation')
            validate_budget(value['limits'])
            validate_budget(value['requested'])
            if set(value['limits']) != set(value['requested']):
                raise ValueError('Nested allocation requested dimensions changed')
            if value['state'] == 'reserved' and any(value['used'].values()):
                raise ValueError('An unsettled allocation cannot carry settled usage')
            if set(self.limits) - set(value['limits']):
                raise ValueError('Nested budget allocation omits a parent limit')
            if value['used']['model_calls'] > value['limits'].get('model_calls', float('inf')):
                raise ValueError('Nested model calls exceed their allocated budget')
            overrun = max(0., value['used']['gpu_seconds'] - value['limits'].get('gpu_seconds', float('inf')))
            if not math.isclose(value['gpu_overrun'], overrun, rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError('Nested GPU seconds exceed their recorded allocation accounting')
        metered = state['gpu_lease_seconds'] + sum(value['used']['gpu_seconds']
            for value in state['allocations'].values() if value['state'] == 'settled')
        if not math.isclose(state['gpu_seconds'], metered, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError('Host GPU seconds differ from metered leases and settled child usage')
        for key in ('model_calls', 'gpu_seconds'):
            reserved = sum(value['limits'].get(key, 0) for value in state['allocations'].values()
                           if value['state'] == 'reserved')
            # GPU caps gate admission, while recorded usage includes the guard's
            # polling and process-tree cleanup latency. Retain that actual cost.
            admitted = state[key] + reserved if key == 'model_calls' else reserved
            if admitted > self.limits.get(key, float('inf')):
                raise ValueError('Nested reservations exceed the parent ' + key + ' budget')

    @staticmethod
    def _valid_allocation_id(ident):
        return (isinstance(ident, str) and len(ident) == 32
                and all(character in '0123456789abcdef' for character in ident))

    def _reconcile(self, state):
        # Lease marker OFDs are inherited by supervisors. An unlocked marker proves
        # both the host and every worker dropped it; never infer release from a PID.
        for ident, value in tuple(state['leases'].items()):
            marker = Path(value['marker'])
            with marker.open('a+b') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                ended = min(time.time(), marker.stat().st_mtime)
                self._charge_gpu_lease(state, value, ended)
                del state['leases'][ident]

    @staticmethod
    def _charge_gpu_lease(state, lease, ended):
        measured = max(0., ended - lease['started_at']) * lease['gpus']
        state['gpu_lease_seconds'] += measured
        state['gpu_seconds'] += measured

    def _remaining(self, state, *, reservations=True):
        self._reconcile(state)
        now = time.time()
        used = {'seconds': max(0., now - state['started_at']), 'model_calls': state['model_calls'],
                'gpu_seconds': state['gpu_seconds'] + sum(
                    max(0., now - v['started_at']) * v['gpus'] for v in state['leases'].values())}
        for allocation in state['allocations'].values():
            if reservations and allocation['state'] == 'reserved':
                for key in ('model_calls', 'gpu_seconds'):
                    used[key] += allocation['limits'].get(key, 0)
        return {k: max(0, value - used[k]) for k, value in self.limits.items()}

    def remaining(self):
        with self._transaction() as state:
            return self._remaining(state)

    def enforcement_remaining(self):
        # Reserved child quotas are not consumed resources. Each child enforces
        # its own deadline/caps while its parent may continue observing it.
        with self._transaction() as state:
            return self._remaining(state, reservations=False)

    def check(self):
        remaining = self.enforcement_remaining()
        if any(v <= 0 for v in remaining.values()):
            raise BudgetExhausted('Host budget exhausted: ' + ', '.join(k for k, v in remaining.items() if v <= 0))

    def reserve_model_call(self):
        with self._transaction() as state:
            remaining = self._remaining(state)
            if any(v <= 0 for v in remaining.values()):
                raise BudgetExhausted('Host model admission budget exhausted')
            state['model_calls'] += 1

    def start_lease(self, ident, gpus, marker):
        with self._transaction() as state:
            if any(v <= 0 for v in self._remaining(state, reservations=bool(gpus)).values()):
                raise BudgetExhausted('Host resource admission budget exhausted')
            state['leases'][ident] = dict(started_at=time.time(), gpus=gpus, marker=marker)

    def end_lease(self, ident):
        with self._transaction() as state:
            value = state['leases'].pop(ident, None)
            if value:
                self._charge_gpu_lease(state, value, time.time())

    def timeout(self, requested=None, *, gpus=0):
        with self._transaction() as state:
            remaining = self._remaining(state)
            # All active leases spend the same balance. The caller's lease is
            # normally already registered; max also covers pre-admission calls.
            gpu_rate = max(gpus, sum(value['gpus'] for value in state['leases'].values()))
        candidates = [v for v in (requested, remaining.get('seconds'),
                      remaining.get('gpu_seconds', float('inf')) / gpu_rate if gpus else None) if v is not None]
        value = min(candidates) if candidates else None
        if value is not None and value <= 0:
            raise BudgetExhausted('Host worker time budget exhausted')
        return value

    def reserve_child(self, ident, root, limits):
        """Reserve consumable quotas before creating any child or model route.

        Wall time is a shared deadline, not an additive resource: the child's
        duration is bounded by the parent's remaining wall time. Model/GPU caps
        are unavailable to the parent until actual child usage is settled.
        """
        if not self._valid_allocation_id(ident):
            raise ValueError('Nested allocation identity must be 32 lowercase hexadecimal characters')
        root, requested = Path(root).resolve(), dict(limits)
        validate_budget(requested)
        if root == self.root.resolve():
            raise ValueError('A nested run needs its own budget ledger')
        with self._transaction() as state:
            existing = state['allocations'].get(ident)
            if existing is not None:
                if existing['root'] != str(root) or existing['state'] != 'reserved':
                    raise ValueError('Nested allocation identity was already used')
                # Wall-clock admission may have narrowed seconds slightly. The
                # originally allocated caps are the durable identity thereafter.
                if existing['requested'] != requested:
                    raise ValueError('Nested allocation budget identity changed')
                allocated, created = dict(existing['limits']), existing['created_at']
            else:
                if (root / 'budget.json').exists():
                    raise ValueError('A nested allocation cannot reuse an existing budget ledger')
                remaining = self._remaining(state)
                if any(value <= 0 for value in remaining.values()):
                    raise BudgetExhausted('Host nested-run admission budget exhausted')
                if set(self.limits) - set(requested):
                    raise ValueError('Nested budget must declare every parent budget dimension')
                for key in ('model_calls', 'gpu_seconds'):
                    if key in remaining and requested[key] > remaining[key]:
                        raise BudgetExhausted('Nested budget exceeds parent remaining ' + key)
                allocated = dict(requested)
                if 'seconds' in remaining:
                    allocated['seconds'] = min(allocated['seconds'], remaining['seconds'])
                created = time.time()
                state['allocations'][ident] = {'root': str(root), 'limits': allocated, 'requested': requested,
                    'created_at': created, 'state': 'reserved', 'used': {'model_calls': 0, 'gpu_seconds': 0.},
                    'gpu_overrun': 0.}
        # The reservation is durable before child initialization. A crash here
        # leaves a reserved entry that can recreate this same ledger, never a
        # launched child whose quota was not deducted from the parent.
        return HostBudget(root, allocated, restoring=(root / 'budget.json').exists(), _started_at=created)

    def settle_child(self, ident, *, unstarted=False, closed_tree=False):
        """Return unused quota only after child-owned resources have exited."""
        with self._transaction() as state:
            allocation = state['allocations'].get(ident)
            if allocation is None:
                raise ValueError('Unknown nested budget allocation')
            if allocation['state'] == 'settled':
                return dict(allocation['used'])
            path = Path(allocation['root']) / 'budget.json'
            if not path.exists():
                if not unstarted:
                    raise RuntimeError('Child budget ledger is missing; allocation remains reserved')
                used = {'model_calls': 0, 'gpu_seconds': 0.}
            else:
                child = HostBudget(allocation['root'], allocation['limits'], restoring=True,
                                   _started_at=allocation['created_at'])
                if closed_tree:
                    # Only a caller that has revoked its delegated capabilities
                    # and confirmed the owning worker's exit may use this path.
                    # A child supervisor owns a lease in this ledger: until all
                    # such leases end, no descendant allocation can be settled.
                    with child._transaction() as child_state:
                        child._reconcile(child_state)
                        if child_state['leases']:
                            raise RuntimeError('Closed nested tree still owns live resource leases')
                        descendants = [key for key, value in child_state['allocations'].items()
                                       if value['state'] == 'reserved']
                    for descendant in descendants:
                        child.settle_child(descendant, unstarted=True, closed_tree=True)
                child.validate_snapshot()
                with child._transaction() as child_state:
                    child._reconcile(child_state)
                    used = {key: child_state[key] for key in ('model_calls', 'gpu_seconds')}
            if used['model_calls'] > allocation['limits'].get('model_calls', float('inf')):
                raise BudgetExhausted('Nested usage exceeds allocated model_calls')
            if state['model_calls'] + used['model_calls'] > self.limits.get('model_calls', float('inf')):
                raise BudgetExhausted('Nested settlement exceeds parent model_calls')
            # The child ledger was validated above. Its GPU usage is actual
            # metering, including a possible finite shutdown overrun, not an
            # additional quota grant. No future admission uses this overrun.
            allocation['gpu_overrun'] = max(0., used['gpu_seconds'] - allocation['limits'].get('gpu_seconds', float('inf')))
            allocation['used'], allocation['state'] = used, 'settled'
            state['model_calls'] += used['model_calls']
            state['gpu_seconds'] += used['gpu_seconds']
            return dict(used)

    def allocation(self, ident):
        with self._transaction() as state:
            if ident not in state['allocations']:
                raise ValueError('Unknown nested budget allocation')
            return json.loads(json.dumps(state['allocations'][ident]))

    def validate_snapshot(self):
        with self._transaction() as state:
            self._reconcile(state)
            if state['leases']:
                raise RuntimeError('Snapshot requires all resource leases released')
            if any(value['state'] == 'reserved' for value in state['allocations'].values()):
                raise RuntimeError('Snapshot requires all nested budget allocations settled')
