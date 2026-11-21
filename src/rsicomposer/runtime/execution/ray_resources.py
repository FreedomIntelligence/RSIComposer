"""Optional Ray admission: one native placement group per named bundle pool.

Ray places bundles. This broker grants whole bundle indexes, with strict
priority/FIFO ordering, and shares those indexes only for explicit colocation.
"""
import fcntl
from hashlib import sha256
import ipaddress
import math
import os
from pathlib import Path
import threading
import time
from uuid import uuid4

from .budget import HostBudget
from .errors import WorkerLostError
from .resources import ResourceManager, ResourceRequest


_BROKER_VERSION = sha256(Path(__file__).read_bytes()).hexdigest()
_CPU_POOL = '__rsisys_cpu__'
_STRATEGIES = {'PACK', 'SPREAD', 'STRICT_PACK', 'STRICT_SPREAD'}


def load_ray():
    try:
        import ray
    except ImportError as exc:
        raise ImportError("resources.executor=ray requires Ray; pip install 'rsicomposer[ray]'") from exc
    return ray


def _priority(value):
    names = {'high': 1, 'normal': 0, 'low': -1}
    if isinstance(value, str) and value in names:
        return names[value]
    if type(value) is not int:
        raise ValueError('Ray placement priority must be an integer or high/normal/low')
    return value


def _heartbeat_grace(host):
    minimum = (host.get('heartbeat_interval', 5) + 5) * host.get('heartbeat_failures', 3) + 2
    grace = host.get('heartbeat_timeout', minimum)
    if grace < minimum:
        raise ValueError('heartbeat_timeout must cover tolerated heartbeat RPC failures: at least ' + str(minimum))
    return grace


def _run_identity(request):
    return request.get('budget_root') or request['manager']


def _pool_spec(pool):
    if (not isinstance(pool, dict) or set(pool) - {'bundles', 'strategy'}
            or not isinstance(pool.get('bundles'), list) or not pool['bundles']
            or pool.get('strategy', 'PACK') not in _STRATEGIES):
        raise ValueError('Ray pools require bundles and a native placement strategy')
    bundles = []
    for bundle in pool['bundles']:
        if (not isinstance(bundle, dict) or type(bundle.get('CPU')) is not int or bundle['CPU'] < 1
                or type(bundle.get('GPU', 0)) is not int or bundle.get('GPU', 0) < 0
                or any(not isinstance(key, str) or not key or type(value) not in (int, float)
                       or not math.isfinite(value) or value < 0 for key, value in bundle.items())):
            raise ValueError('Ray bundles require positive integer CPU and nonnegative integer GPU resources')
        bundles.append({key: value for key, value in bundle.items() if value})
    return {'bundles': bundles, 'strategy': pool.get('strategy', 'PACK')}


def _select_bundles(bundles, gpus, cpus, available=None):
    """Select the fewest whole bundles that can cover the request."""
    available = range(len(bundles)) if available is None else available
    choices = {(0, 0): ()}
    for index in available:
        bundle = bundles[index]
        count = bundle.get('GPU', 0)
        if bool(gpus) != bool(count):
            continue
        for (used_gpus, used_cpus), indexes in tuple(choices.items()):
            candidate = indexes + (index,)
            if len(candidate) > (gpus or cpus):
                continue  # A selected GPU bundle must contribute at least one GPU.
            key = min(gpus, used_gpus + count), min(cpus, used_cpus + bundle['CPU'])
            previous = choices.get(key)
            if previous is None or (len(candidate), candidate) < (len(previous), previous):
                choices[key] = candidate
    return choices.get((gpus, cpus))


def _grant(allocations, gpus, cpus, *, keep_devices=False):
    """Account for only the requested resources within the reserved bundles."""
    granted = []
    for index, allocation in enumerate(allocations):
        gpu_count = allocation['gpu_count'] if keep_devices else 0
        if not keep_devices and gpus:
            gpu_count = min(allocation['bundle_gpus'], gpus - (len(allocations) - index - 1))
        cpu_count = min(cpus, allocation['bundle_cpus'])
        granted.append(dict(allocation, gpu_count=gpu_count, cpus=cpu_count))
        gpus -= gpu_count
        cpus -= cpu_count
    return granted


def validate_ray_resources(config):
    if set(config) - {'executor', 'pools', 'placement', 'weight_sync', 'host'}:
        raise ValueError('Unknown Ray resources configuration field')
    host = config.get('host', {})
    timeouts = {'heartbeat_interval', 'heartbeat_timeout', 'owner_ready_timeout', 'gpu_lock_timeout'}
    if not isinstance(host, dict) or set(host) - {
            'mode', 'address', 'advertise_address', 'bind_address', 'namespace', 'heartbeat_failures', *timeouts}:
        raise ValueError('Invalid Ray host configuration')
    if host.get('mode', 'head') not in ('head', 'client'):
        raise ValueError('Ray host.mode must be head or client')
    for key in ('address', 'advertise_address', 'bind_address', 'namespace'):
        if key in host and (not isinstance(host[key], str) or not host[key].strip()):
            raise ValueError('Ray host.' + key + ' must be nonempty text')
    for key in timeouts:
        if key in host and (type(host[key]) not in (int, float) or not math.isfinite(host[key]) or host[key] <= 0):
            raise ValueError('Ray host.' + key + ' must be finite and positive')
    if type(host.get('heartbeat_failures', 3)) is not int or host.get('heartbeat_failures', 3) < 1:
        raise ValueError('heartbeat_failures must be a positive integer')
    _heartbeat_grace(host)
    for key in ('advertise_address', 'bind_address'):
        value = host.get(key)
        if value is not None and (any(c in value for c in '/:@?#') or any(c.isspace() for c in value)):
            raise ValueError('Ray host.' + key + ' must be an address without a port')
    if host.get('mode') == 'client':
        if not host.get('address', '').startswith('ray://') or not host.get('advertise_address'):
            raise ValueError('Ray client requires ray:// address and advertise_address')
        advertised = host['advertise_address'].rstrip('.').lower()
        try:
            address = ipaddress.ip_address(advertised)
        except ValueError:
            address = None
        if advertised == 'localhost' or address is not None and (address.is_loopback or address.is_unspecified):
            raise ValueError('Ray client advertise_address must be reachable from cluster nodes')
    pools, placement = config.get('pools', {}), config.get('placement', {})
    if not isinstance(pools, dict) or not isinstance(placement, dict):
        raise ValueError('Ray pools and placement must be mappings')
    specs = {}
    for name, pool in pools.items():
        if not isinstance(name, str) or not name or name == _CPU_POOL:
            raise ValueError('Ray pool name is empty or reserved')
        specs[name] = _pool_spec(pool)
    for role, entry in placement.items():
        if (not isinstance(role, str) or not role or not isinstance(entry, dict) or set(entry) - {
                'pool', 'gpus', 'cpus', 'timeout', 'max_concurrent', 'colocate', 'with', 'gpu_fraction', 'priority'}
                or entry.get('pool') not in pools):
            raise ValueError('Invalid Ray role placement')
        request = ResourceRequest(**{key: value for key, value in entry.items() if key in {'gpus', 'cpus', 'timeout'}})
        bundles = specs[entry['pool']]['bundles']
        # An omitted count comes from the Lab. Check that the configured dimensions are feasible.
        counts = (request.gpus,) if 'gpus' in entry else (0, sum(bundle.get('GPU', 0) for bundle in bundles))
        if all(_select_bundles(bundles, gpus, request.cpus) is None for gpus in counts):
            raise ValueError('Ray role placement cannot fit whole bundles; exceeds pool capacity: ' + role)
        if type(entry.get('max_concurrent', 1)) is not int or entry.get('max_concurrent', 1) < 1:
            raise ValueError('max_concurrent must be positive')
        _priority(entry.get('priority', 0))
        mode, fraction = entry.get('colocate', 'exclusive'), entry.get('gpu_fraction', 1.)
        if mode not in ('exclusive', 'shared', 'hybrid') or type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError('Invalid Ray colocation mode or gpu_fraction')
        if mode == 'exclusive':
            if 'with' in entry:
                raise ValueError('Exclusive roles cannot specify with')
            continue
        target = entry.get('with')
        anchor = placement.get(target)
        if (target == role or not isinstance(anchor, dict) or anchor.get('pool') != entry['pool']
                or anchor.get('colocate', 'exclusive') != 'exclusive'):
            raise ValueError('Colocation requires an exclusive anchor in the same pool')
        if mode == 'shared' and ('gpu_fraction' not in entry or 'gpu_fraction' not in anchor
                or type(anchor['gpu_fraction']) not in (int, float) or not math.isfinite(anchor['gpu_fraction'])
                or not 0 < anchor['gpu_fraction'] <= 1 or fraction + anchor['gpu_fraction'] > 1):
            raise ValueError('Shared roles require valid fractions totaling at most one')
    sync = config.get('weight_sync', {})
    if not isinstance(sync, dict) or any(not isinstance(key, str) or ' -> ' not in key or value != 'filesystem'
                                       for key, value in sync.items()):
        raise ValueError('Ray weight_sync supports role -> role: filesystem')


class _PoolBroker:
    """Namespace admission authority; physical placement belongs entirely to Ray."""
    def __init__(self):
        self.version = _BROKER_VERSION
        self.pools, self.managers, self.pending, self.leases, self.budgets = {}, {}, {}, {}, {}
        self.failed, self.sequence = {}, 0
        self.lock, self._stop = threading.RLock(), threading.Event()
        threading.Thread(target=self._maintain, daemon=True).start()

    def code_version(self):
        return self.version

    def _maintain(self):
        while not self._stop.wait(1):
            try:
                with self.lock:
                    self._refresh()
                    for ident, manager in tuple(self.managers.items()):
                        if time.monotonic() - manager['heartbeat'] > manager['timeout']:
                            self._unregister(ident)
                    self._cleanup_pools()
            except Exception:
                # A transient GCS/owner failure cannot disable future maintenance.
                continue

    def _refresh(self):
        from ray.util.placement_group import placement_group_table
        alive = {node['NodeID'] for node in load_ray().nodes() if node['Alive']}
        for name, pool in self.pools.items():
            nodes = placement_group_table(pool['group']).get('bundles_to_node_id', {})
            pool['node_ids'] = [nodes.get(index) if nodes.get(index) in alive else None
                                for index in range(len(pool['spec']['bundles']))]
            for ident, lease in tuple(self.leases.items()):
                if lease['pool'] != name:
                    continue
                lost = {a['bundle_index'] for a in lease['allocations'] if a['node_id'] not in alive
                        or pool['node_ids'][a['bundle_index']] and pool['node_ids'][a['bundle_index']] != a['node_id']}
                if lost:
                    exited = {a['bundle_index'] for a in lease['allocations'] if a['node_id'] not in alive}
                    self._lose(ident, 'Ray bundle node lost; operation effects uncertain', exited=exited)

    def _lose(self, ident, reason, *, exited=()):
        lease = self.leases[ident]
        lease['lost'] = reason
        for worker, bundle in tuple(lease['workers'].items()):
            if bundle in exited or bundle is None and all(a['bundle_index'] in exited for a in lease['allocations']):
                del lease['workers'][worker]
        self._maybe_release(ident)

    def register(self, ident, pools, version=_BROKER_VERSION, heartbeat_timeout=20):
        from ray.util.placement_group import placement_group, remove_placement_group
        if version != self.version:
            raise RuntimeError('Ray broker code version differs; restart the rsisys.resource-broker actor')
        with self.lock:
            specs = {name: _pool_spec(pool) for name, pool in pools.items()}
            for name, spec in specs.items():
                if name in self.pools and self.pools[name]['spec'] != spec:
                    raise ValueError('Named Ray pool already has different bundles: ' + name)
            created = {}
            try:
                for name, spec in specs.items():
                    if name not in self.pools:
                        group = placement_group(spec['bundles'], strategy=spec['strategy'],
                                                lifetime='detached', name='rsisys-' + uuid4().hex)
                        created[name] = dict(spec=spec, group=group, node_ids=[None] * len(spec['bundles']),
                                             owners={})
            except BaseException:
                for pool in created.values():
                    remove_placement_group(pool['group'])
                raise
            self.pools.update(created)
            self.managers[ident] = dict(heartbeat=time.monotonic(), timeout=heartbeat_timeout, pools=set(pools))

    def heartbeat(self, manager):
        with self.lock:
            if manager not in self.managers:
                raise RuntimeError('Ray resource manager expired')
            self.managers[manager]['heartbeat'] = time.monotonic()

    def _unregister(self, manager):
        self.managers.pop(manager, None)
        for ident, request in tuple(self.pending.items()):
            if request['manager'] == manager:
                del self.pending[ident]
        for ident, (owner, _) in tuple(self.failed.items()):
            if owner == manager:
                del self.failed[ident]
        for ident, lease in tuple(self.leases.items()):
            if lease['manager'] == manager:
                lease['host'] = False
                self._maybe_release(ident)

    def unregister(self, manager):
        with self.lock:
            self._unregister(manager)
            self._cleanup_pools()

    @staticmethod
    def _dispose(pool):
        ray = load_ray()
        from ray.util.placement_group import remove_placement_group
        for owner, _ in pool['owners'].values():
            try:
                ray.get(owner.close.remote(), timeout=10)
            except Exception:
                pass
            try:
                ray.kill(owner, no_restart=True)
            except Exception:
                pass
        remove_placement_group(pool['group'])

    def _cleanup_pools(self):
        registered = {name for manager in self.managers.values() for name in manager['pools']}
        for name, pool in tuple(self.pools.items()):
            if name not in registered and not any(lease['pool'] == name for lease in self.leases.values()):
                del self.pools[name]
                threading.Thread(target=self._dispose, args=(pool,), daemon=True).start()

    def register_owner(self, pool, bundle_index, owner, node_id):
        with self.lock:
            state = self.pools[pool]
            if state['node_ids'][bundle_index] != node_id:
                raise WorkerLostError('Ray bundle moved before owner registration; operation effects uncertain')
            winner, _ = state['owners'].setdefault(bundle_index, (owner, node_id))
            return winner

    def unregister_owner(self, pool, bundle_index, expected):
        with self.lock:
            owners = self.pools[pool]['owners']
            current = owners.get(bundle_index)
            if current is None or current[0]._actor_id != expected._actor_id:
                return False
            _, node_id = current
            del owners[bundle_index]
            for ident, lease in tuple(self.leases.items()):
                if (lease['pool'] == pool and not lease['lost'] and any(
                        a['bundle_index'] == bundle_index and a['node_id'] == node_id for a in lease['allocations'])):
                    self._lose(ident, 'Ray bundle owner lost; operation effects uncertain')
            return True

    def owner(self, pool, bundle_index):
        with self.lock:
            current = self.pools[pool]['owners'].get(bundle_index)
            return current[0] if current is not None else None

    def enqueue(self, manager, ident, request):
        with self.lock:
            if manager not in self.managers:
                raise RuntimeError('Ray resource manager expired')
            if request['pool'] not in self.managers[manager]['pools']:
                raise ValueError('Ray pool is outside the registered manager')
            if _select_bundles(self.pools[request['pool']]['spec']['bundles'], request['gpus'], request['cpus']) is None:
                raise ValueError('Resource request cannot fit whole Ray bundles; exceeds pool capacity')
            self.sequence += 1
            self.pending[ident] = dict(request, manager=manager, sequence=self.sequence)

    def _active_leases(self):
        return [lease for lease in self.leases.values() if not lease.get('lost') or lease['workers']]

    @staticmethod
    def _at_concurrency_limit(request, active):
        return sum(_run_identity(lease) == _run_identity(request) and lease['role'] == request['role']
                   for lease in active) >= request['max_concurrent']

    def _candidate(self, request, active=None):
        if active is None:
            active = self._active_leases()
            if self._at_concurrency_limit(request, active):
                return None
        pool = self.pools[request['pool']]
        if request.get('with'):
            for anchor in active:
                if (_run_identity(anchor) != _run_identity(request) or anchor['pool'] != request['pool']
                        or anchor['role'] != request['with'] or anchor['gpus'] != request['gpus'] or anchor.get('lost')):
                    continue
                family = [v for v in active if v['group'] == anchor['group']]
                if any(v['mode'] not in ('exclusive', request['mode']) for v in family):
                    continue
                if request['mode'] == 'shared' and sum(v['fraction'] for v in family) + request['fraction'] > 1 + 1e-9:
                    continue
                if sum(a['bundle_cpus'] for a in anchor['allocations']) >= request['cpus']:
                    return anchor['group'], _grant(anchor['allocations'], request['gpus'], request['cpus'], keep_devices=True)
            return None
        occupied = {a['bundle_index'] for lease in active if lease['pool'] == request['pool'] for a in lease['allocations']}
        available = [i for i, node in enumerate(pool['node_ids']) if node and i not in occupied]
        indexes = _select_bundles(pool['spec']['bundles'], request['gpus'], request['cpus'], available)
        if indexes is None:
            return None
        group = uuid4().hex
        allocations = [dict(pool=request['pool'], bundle_index=i, node_id=pool['node_ids'][i],
            placement_group=pool['group'], bundle_gpus=pool['spec']['bundles'][i].get('GPU', 0),
            bundle_cpus=pool['spec']['bundles'][i]['CPU'], group=group) for i in indexes]
        return group, _grant(allocations, request['gpus'], request['cpus'])

    def _budget(self, request):
        root = request.get('budget_root')
        if root and root not in self.budgets:
            self.budgets[root] = HostBudget(root, request['budget_limits'], restoring=True)
        return self.budgets.get(root)

    def _admit(self, ident, request, group, allocations):
        budget, marker = self._budget(request), None
        if budget is not None:
            path = budget.root / 'leases' / (ident + '.lock')
            path.parent.mkdir(exist_ok=True)
            marker = path.open('a+b')
            fcntl.flock(marker, fcntl.LOCK_SH)
            try:
                budget.start_lease(ident, request['gpus'], str(path))
            except BaseException:
                marker.close()
                raise
        self.leases[ident] = dict(request, allocations=allocations, group=group, host=True,
                                  workers={}, budget=budget, marker=marker, lost=None)

    def poll(self, ident):
        with self.lock:
            self._refresh()
            blocked = set()
            for key, request in sorted(tuple(self.pending.items()), key=lambda pair: (-pair[1]['priority'], pair[1]['sequence'])):
                if request['pool'] in blocked:
                    continue
                try:
                    budget = self._budget(request)
                    if budget is not None:
                        budget.check()
                    active = self._active_leases()
                    if self._at_concurrency_limit(request, active):
                        continue
                    candidate = self._candidate(request, active)
                    if candidate is None:
                        # A colocated role depends on an anchor rather than free bundles.
                        if not request.get('with'):
                            blocked.add(request['pool'])
                        continue
                    self._admit(key, request, *candidate)
                except Exception as error:
                    self.failed[key] = (request['manager'], error)
                del self.pending[key]
            if ident in self.failed:
                _, error = self.failed.pop(ident)
                raise error
            if ident in self.leases:
                if self.leases[ident]['lost']:
                    raise WorkerLostError(self.leases[ident]['lost'])
                return {key: self.leases[ident][key] for key in ('allocations', 'group')}
            if ident not in self.pending:
                raise RuntimeError('Ray resource request was cancelled')
            return None

    def cancel(self, ident):
        with self.lock:
            self.pending.pop(ident, None)
            self.failed.pop(ident, None)
            self.release(ident)

    def retain(self, ident, worker, bundle_index=None):
        with self.lock:
            lease = self.leases[ident]
            if not lease['host']:
                raise RuntimeError('Cannot launch on a released Ray lease')
            if lease['lost']:
                raise WorkerLostError(lease['lost'])
            if worker in lease['workers']:
                raise ValueError('Ray worker reference was already registered')
            if bundle_index is not None and bundle_index not in {a['bundle_index'] for a in lease['allocations']}:
                raise ValueError('Worker bundle is outside the Ray lease')
            lease['workers'][worker] = bundle_index
            return worker

    def worker_done(self, ident, worker):
        with self.lock:
            if ident in self.leases:
                self.leases[ident]['workers'].pop(worker, None)
                self._maybe_release(ident)
            self._cleanup_pools()

    def release(self, ident):
        with self.lock:
            if ident in self.leases:
                self.leases[ident]['host'] = False
                self._maybe_release(ident)
            self._cleanup_pools()

    def _maybe_release(self, ident):
        lease = self.leases[ident]
        if lease['workers'] or (lease['host'] and not lease['lost']):
            return
        if not lease['host']:
            del self.leases[ident]
        marker = lease['marker']
        if marker is not None:
            try:
                Path(marker.name).touch()
            finally:
                marker.close()
                lease['marker'] = None
            lease['budget'].remaining()

    def status(self, ident):
        with self.lock:
            self._refresh()
            lease = self.leases.get(ident)
            return {'valid': lease is not None and not lease['lost'],
                    'lost': lease['lost'] if lease is not None else None}

    def active(self, manager):
        with self.lock:
            return [ident for ident, lease in self.leases.items()
                    if lease['manager'] == manager and (not lease['lost'] or lease['workers'])]

    def describe(self):
        with self.lock:
            return {'pools': tuple(self.pools), 'leases': tuple(self.leases), 'pending': tuple(self.pending),
                    'node_id': load_ray().get_runtime_context().get_node_id()}


class RayLease:
    def __init__(self, manager, ident, role, request, mode, allocation):
        self.manager, self.ident, self.role, self.mode = manager, ident, role, mode
        self.timeout, self.allocations, self.group = request.timeout, tuple(allocation['allocations']), allocation['group']
        self.devices = tuple(range(sum(a['gpu_count'] for a in self.allocations)))
        self.released, self._launched_count = False, 0
        self.marker_path = str(manager.budget.root / 'leases' / (ident + '.lock')) if manager.budget else None

    def retain(self, worker_id=None, *, bundle_index=None):
        if self.released:
            raise RuntimeError('Cannot launch on a released Ray lease')
        return self.manager.ray.get(self.manager.broker.retain.remote(self.ident, worker_id or uuid4().hex, bundle_index))

    def worker_done(self, worker_id):
        return self.manager.ray.get(self.manager.broker.worker_done.remote(self.ident, worker_id))

    def worker_registration(self):
        return {'broker': self.manager.broker, 'lease_id': self.ident, 'marker_path': self.marker_path,
                'budget_root': str(self.manager.budget.root) if self.manager.budget else None}

    def release(self):
        if not self.released:
            self.manager.ray.get(self.manager.broker.release.remote(self.ident))
            self.released = True
            self.manager._leases.discard(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


class RayResourceManager(ResourceManager):
    def __init__(self, config=None, *, budget=None, lock_dir=None):
        self.config = dict(config or {})
        validate_ray_resources(self.config)
        self.ray = ray = load_ray()
        host = self.config.get('host', {})
        self.namespace, self.budget = host.get('namespace', 'rsicomposer'), budget
        if not ray.is_initialized():
            ray.init(address=host.get('address', 'auto'), namespace=self.namespace)
        self._roles, self._leases, self._closed = {}, set(), False
        self._lock, self.ident = threading.RLock(), uuid4().hex
        actor = ray.remote(num_cpus=0)(_PoolBroker)
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        head = next(node['NodeID'] for node in ray.nodes()
                    if node['Alive'] and 'node:__internal_head__' in node['Resources'])
        self.broker = actor.options(name='rsisys.resource-broker', namespace=self.namespace,
            lifetime='detached', get_if_exists=True,
            scheduling_strategy=NodeAffinitySchedulingStrategy(head, soft=False), runtime_env={'env_vars': {
                'PYTHONPATH': str(Path(__file__).resolve().parents[3]) + os.pathsep + os.environ.get('PYTHONPATH', '')}}).remote()
        self.declared_pools = dict(self.config.get('pools', {}))
        self._pools = {**self.declared_pools, _CPU_POOL: {'bundles': [{'CPU': 1}], 'strategy': 'PACK'}}
        self.default_pool = _CPU_POOL
        self._verify_broker()
        self._register()
        self._registered = threading.Event()
        self._registered.set()
        self._unregistered = False
        self._stop = threading.Event()
        self._heartbeats = threading.Thread(target=self._heartbeat, daemon=True)
        self._heartbeats.start()

    @property
    def cluster_address(self):
        return self.ray.get_runtime_context().gcs_address

    def _verify_broker(self):
        try:
            version = self.ray.get(self.broker.code_version.remote(), timeout=5)
        except (AttributeError, TypeError) as error:
            raise RuntimeError('Ray broker has no current version handshake; restart rsisys.resource-broker') from error
        if version != _BROKER_VERSION:
            raise RuntimeError('Ray broker code version differs; restart rsisys.resource-broker')

    def _register(self, *, timeout=None):
        self._rpc(self.broker.register, self.ident, self._pools, _BROKER_VERSION,
                  _heartbeat_grace(self.config.get('host', {})), timeout=timeout)

    def _rpc(self, method, *args, timeout=None):
        try:
            return self.ray.get(method.remote(*args), timeout=timeout)
        except self.ray.exceptions.RayTaskError as error:
            # A rejected configuration/budget is a broker exception, not a lost worker.
            raise error.cause from error

    def _heartbeat(self):
        failures = 0
        host = self.config.get('host', {})
        interval = delay = host.get('heartbeat_interval', 5)
        while not self._stop.wait(delay):
            try:
                if self._registered.is_set():
                    self.ray.get(self.broker.heartbeat.remote(self.ident), timeout=5)
                else:
                    self._register(timeout=5)
                    self._registered.set()
                failures = 0
                delay = interval
            except Exception:
                failures += 1
                if failures >= host.get('heartbeat_failures', 3):
                    self._registered.clear()
                    delay = max(interval, min(30., 2 * delay))

    def _check_admission(self):
        if self._closed:
            raise RuntimeError('Resource admission is closed')
        if not self._registered.is_set():
            raise RuntimeError('Ray resource admission is waiting for broker registration; retry later')

    def acquire(self, role, request=None):
        self._check_admission()
        expected = self.request(role)
        request = expected if request is None else request
        if not isinstance(request, ResourceRequest):
            raise TypeError('request must be ResourceRequest')
        if request.gpus > expected.gpus or request.cpus > expected.cpus:
            raise ValueError('Request exceeds role placement')
        timeouts = [v for v in (expected.timeout, request.timeout) if v is not None]
        request = ResourceRequest(request.gpus, request.cpus, min(timeouts) if timeouts else None)
        entry = self.config.get('placement', {}).get(role, {})
        if request.gpus and 'pool' not in entry:
            raise ValueError('GPU role requires explicit Ray pool/placement: ' + role)
        if self.budget:
            self.budget.check()
        ident, started = uuid4().hex, time.monotonic()
        payload = dict(role=role, pool=entry.get('pool', self.default_pool), gpus=request.gpus, cpus=request.cpus,
            priority=_priority(entry.get('priority', 0)), max_concurrent=entry.get('max_concurrent', 1),
            mode=entry.get('colocate', 'exclusive'), fraction=entry.get('gpu_fraction', 1.),
            **({'with': entry['with']} if 'with' in entry else {}),
            budget_root=str(self.budget.root) if self.budget else None,
            budget_limits=self.budget.limits if self.budget else {})
        try:
            self._rpc(self.broker.enqueue, self.ident, ident, payload)
            while True:
                self._check_admission()
                if self.budget:
                    self.budget.check()
                allocation = self._rpc(self.broker.poll, ident)
                if allocation is not None:
                    with self._lock:
                        self._check_admission()
                        lease = RayLease(self, ident, role, request, payload['mode'], allocation)
                        self._leases.add(lease)
                        return lease
                if request.timeout is not None and time.monotonic() - started >= request.timeout:
                    raise TimeoutError('Timed out waiting for Ray resource role ' + role)
                time.sleep(.05)
        except BaseException:
            try:
                self.ray.get(self.broker.cancel.remote(ident), timeout=5)
            except self.ray.exceptions.RayError:
                pass  # Preserve the original failure; manager expiry also drops pending requests.
            raise

    def close(self):
        with self._lock:
            if self._unregistered:
                return
            self._closed = True
            self._stop.set()
            self._heartbeats.join(timeout=6)
            try:
                self.ray.get(self.broker.unregister.remote(self.ident), timeout=5)
            except self.ray.exceptions.RayError:
                return  # A repeated close can retry; heartbeat expiry is the fallback.
            self._unregistered = True
            for lease in tuple(self._leases):
                lease.released = True
            self._leases.clear()

    def validate_snapshot(self):
        if self.ray.get(self.broker.active.remote(self.ident)):
            raise RuntimeError('Snapshot requires all Ray resource leases released')
        if self.budget:
            self.budget.validate_snapshot()
