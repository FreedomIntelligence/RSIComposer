"""Local, cross-run resource leases. Device locks never live in run directories."""
from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import dataclass
import fcntl
from hashlib import sha256
import os
from pathlib import Path
import tempfile
import threading
import time
from uuid import uuid4


@dataclass(frozen=True)
class ResourceRequest:
    gpus: int = 0
    cpus: int = 1
    timeout: float | None = None

    def __post_init__(self):
        import math
        if type(self.gpus) is not int or self.gpus < 0 or type(self.cpus) is not int or self.cpus < 1:
            raise ValueError('Resource requests require nonnegative GPUs and positive CPUs')
        if self.timeout is not None and (type(self.timeout) not in (int, float)
                or not math.isfinite(self.timeout) or self.timeout <= 0):
            raise ValueError('Resource timeout must be finite and positive')


def validate_resources(config):
    if isinstance(config, dict) and config.get('executor') == 'ray':
        from .ray_resources import validate_ray_resources
        return validate_ray_resources(config)
    if not isinstance(config, dict) or set(config) - {'executor', 'pools', 'placement', 'weight_sync'}:
        raise ValueError('resources supports executor, pools, placement, weight_sync')
    if config.get('executor', 'local') != 'local':
        raise ValueError('resources.executor must be local or ray')
    pools = config.get('pools', {})
    if not isinstance(pools, dict):
        raise ValueError('resources.pools must be a mapping')
    for name, pool in pools.items():
        if not isinstance(name, str) or not name or not isinstance(pool, dict) or set(pool) - {
                'nodes', 'gpus_per_node', 'cpus_per_node', 'gpus'}:
            raise ValueError('Invalid local resource pool')
        if pool.get('nodes', 1) != 1:
            raise ValueError('Local resource pools require nodes=1')
        count = pool.get('gpus_per_node', 0)
        if type(count) is not int or count < 0:
            raise ValueError('gpus_per_node must be nonnegative')
        gpus = pool.get('gpus', list(range(count)))
        if (not isinstance(gpus, list) or any(type(i) is not int or i < 0 for i in gpus)
                or len(gpus) != len(set(gpus))):
            raise ValueError('Local pool gpus must be distinct nonnegative physical indices')
        if 'gpus_per_node' in pool and (type(pool['gpus_per_node']) is not int or pool['gpus_per_node'] != len(gpus)):
            raise ValueError('gpus_per_node must equal the local GPU inventory length')
        if type(pool.get('cpus_per_node', os.cpu_count() or 1)) is not int or pool.get('cpus_per_node', 1) < 1:
            raise ValueError('cpus_per_node must be positive')
    placement = config.get('placement', {})
    if not isinstance(placement, dict):
        raise ValueError('resources.placement must be a mapping')
    for role, entry in placement.items():
        if (not isinstance(role, str) or not role or not isinstance(entry, dict) or set(entry) - {
                'pool', 'gpus', 'cpus', 'timeout', 'max_concurrent', 'colocate'}):
            raise ValueError('Invalid local role placement')
        if entry.get('pool') not in pools:
            raise ValueError(f'Unknown resource pool for {role}')
        if entry.get('colocate', 'exclusive') != 'exclusive':
            raise ValueError('hybrid/shared colocate is scheduled for step 7b')
        ResourceRequest(**{k: v for k, v in entry.items() if k in {'gpus', 'cpus', 'timeout'}})
        if type(entry.get('max_concurrent', 1)) is not int or entry.get('max_concurrent', 1) < 1:
            raise ValueError('max_concurrent must be positive')
    sync = config.get('weight_sync', {})
    if not isinstance(sync, dict) or any(v != 'filesystem' or ' -> ' not in k for k, v in sync.items()):
        raise ValueError('Local weight_sync only supports role -> role: filesystem')


class Lease:
    def __init__(self, manager, role, devices, streams, timeout):
        self.manager, self.role, self.devices = manager, role, tuple(devices)
        self._streams, self.timeout = streams, timeout
        self.ident, self.released, self.transferred = uuid4().hex, False, False
        self._launched_count = 0
        if manager.budget:
            marker = manager.budget.root / 'leases' / (self.ident + '.lock')
            marker.parent.mkdir(exist_ok=True)
            stream = marker.open('a+b')
            fcntl.flock(stream, fcntl.LOCK_EX)
            self._streams.append(stream)
            manager.budget.start_lease(self.ident, len(devices), str(marker))

    @property
    def fds(self):
        return tuple(s.fileno() for s in self._streams)

    def release(self):
        if self.released:
            return
        # close, not LOCK_UN: detached supervisors hold duplicates of these OFDs.
        if self.manager.budget and not self.transferred:
            marker = self.manager.budget.root / 'leases' / (self.ident + '.lock')
            marker.touch()
        for stream in self._streams:
            stream.close()
        self.released = True
        self.manager._leases.discard(self)
        if self.manager.budget and not self.transferred:
            # Asynchronous owned workers still hold the marker OFD. Reconcile
            # only once every holder has closed it, so early release keeps billing.
            self.manager.budget.remaining()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


class ResourceManager:
    def __new__(cls, config=None, **kwargs):
        if cls is ResourceManager and isinstance(config, dict) and config.get('executor') == 'ray':
            from .ray_resources import RayResourceManager
            return object.__new__(RayResourceManager)
        return object.__new__(cls)

    def __init__(self, config=None, *, budget=None, lock_dir=None):
        self.config = {} if config is None else config
        validate_resources(self.config)
        self.budget = budget
        self.root = Path(lock_dir or (Path(tempfile.gettempdir()) / 'rsisys-resources'))
        self.root.mkdir(parents=True, exist_ok=True)
        self._roles, self._leases, self._closed = {}, set(), False
        self._lock = threading.RLock()

    def component(self, name):
        return ComponentResources(self, name.removeprefix('lab:'))

    def declare(self, role, request):
        if not isinstance(role, str) or not role or not isinstance(request, ResourceRequest):
            raise TypeError('resource_roles() must map role names to ResourceRequest')
        with self._lock:
            if role in self._roles and self._roles[role] != request:
                raise ValueError('Conflicting resource role declaration: ' + role)
            self._roles[role] = request

    def request(self, role):
        if role not in self._roles:
            raise ValueError('Undeclared resource role: ' + role)
        entry = self.config.get('placement', {}).get(role, {})
        declared = self._roles[role]
        return ResourceRequest(entry.get('gpus', declared.gpus), entry.get('cpus', declared.cpus),
                               entry.get('timeout', declared.timeout))

    def _try(self, key):
        stream = (self.root / (sha256(key.encode()).hexdigest() + '.lock')).open('a+b')
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            return None
        return stream

    def acquire(self, role, request=None):
        expected = self.request(role)
        request = expected if request is None else request
        if not isinstance(request, ResourceRequest):
            raise TypeError('request must be ResourceRequest')
        if request.gpus > expected.gpus or request.cpus > expected.cpus:
            raise ValueError('Request exceeds role placement')
        timeouts = [v for v in (expected.timeout, request.timeout) if v is not None]
        request = ResourceRequest(request.gpus, request.cpus, min(timeouts) if timeouts else None)
        entry = self.config.get('placement', {}).get(role, {})
        pools = self.config.get('pools', {})
        pool_name = entry.get('pool')
        if pool_name is None:
            if request.gpus:
                raise ValueError(f'GPU role {role} requires resources.placement with an explicit local pool')
            pool = {'gpus': [], 'cpus_per_node': os.cpu_count() or 1}
            pool_name = 'cpu'
        else:
            pool = pools[pool_name]
        devices = pool.get('gpus', list(range(pool.get('gpus_per_node', 0))))
        cpu_count = pool.get('cpus_per_node', os.cpu_count() or 1)
        if request.gpus > len(devices) or request.cpus > cpu_count:
            raise ValueError(f'Resource request for {role} exceeds pool capacity')
        started = time.monotonic()
        while True:
            if self._closed:
                raise RuntimeError('Resource admission is closed')
            if self.budget:
                self.budget.check()
            streams, allocated = [], []
            # Shared gate makes multi-device acquisition atomic and avoids deadlocks.
            with (self.root / 'admission.lock').open('a+b') as gate:
                fcntl.flock(gate, fcntl.LOCK_EX)
                cap = entry.get('max_concurrent')
                if cap:
                    prefix = 'role:' + pool_name + ':' + role + ':'
                    slot = next((s for i in range(cap) if (s := self._try(prefix + str(i))) is not None), None)
                    if slot is not None:
                        streams.append(slot)
                if cap is None or streams:
                    for device in devices if request.gpus else ():
                        stream = self._try('gpu:' + str(device))
                        if stream:
                            streams.append(stream)
                            allocated.append(device)
                        if len(allocated) == request.gpus:
                            break
                    cpus = []
                    for index in range(cpu_count):
                        stream = self._try('cpu:' + str(index))
                        if stream:
                            cpus.append(stream)
                        if len(cpus) == request.cpus:
                            break
                    streams.extend(cpus)
                    if len(allocated) == request.gpus and len(cpus) == request.cpus:
                        try:
                            lease = Lease(self, role, allocated, streams, request.timeout)
                        except BaseException:
                            for stream in streams:
                                stream.close()
                            raise
                        self._leases.add(lease)
                        return lease
                for stream in streams:
                    stream.close()
            if request.timeout is not None and time.monotonic() - started >= request.timeout:
                raise TimeoutError('Timed out waiting for resource role ' + role)
            time.sleep(.05)

    def release(self, lease):
        if lease.manager is not self:
            raise ValueError('Lease belongs to another resource manager')
        lease.release()

    def validate_roles(self):
        unknown = set(self.config.get('placement', {})) - self._roles.keys()
        if unknown:
            raise ValueError('Placement refers to undeclared roles: ' + ', '.join(sorted(unknown)))

    def close(self):
        self._closed = True
        # Executor cleanup must precede this. Transferred leases outlive the host.
        for lease in tuple(self._leases):
            lease.release()

    def validate_snapshot(self):
        if self._leases:
            raise RuntimeError('Snapshot requires all resource leases released')
        if self.budget:
            self.budget.validate_snapshot()


class ComponentResources:
    def __init__(self, owner, name):
        self.owner, self.name = owner, name

    def declare(self, roles):
        if not isinstance(roles, Mapping):
            raise TypeError('resource_roles() must return a mapping')
        for role, request in roles.items():
            self.owner.declare(self.name + '.' + role, request)

    def request(self, role):
        return self.owner.request(self.name + '.' + role)

    def acquire(self, role, request=None):
        return self.owner.acquire(self.name + '.' + role, request)

    def release(self, lease):
        return self.owner.release(lease)

    @contextmanager
    def allocation(self, role):
        with self.acquire(role) as lease:
            yield lease
