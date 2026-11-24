"""Ray-only execution: node owners reserve pools and supervise leased work.

The framework imports this module only when resources.executor is ray. Paths in
WorkerSpec and log destinations must be on the cluster's shared filesystem.
"""
from dataclasses import replace
import fcntl
from hashlib import sha256
import json
import inspect
import os
from pathlib import Path
from multiprocessing.connection import Client
import secrets
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from uuid import uuid4

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .executor import Executor, WorkerHandle, WorkerSpec
from .errors import WorkerLaunchUncertainError, WorkerLostError
from .workers import worker_environment
from .process_identity import process_identity
from rsicomposer.runtime.storage.files import atomic_write, json_bytes


def _class_environment(cls, environment):
    """Forward the class's verified source package root, not all driver sys.path."""
    environment = dict(environment)
    try:
        path = Path(inspect.getsourcefile(cls)).resolve()
    except (TypeError, OSError):
        return environment
    parts = cls.__module__.split('.')
    suffix = (*parts, '__init__.py') if path.name == '__init__.py' else (*parts[:-1], parts[-1] + '.py')
    if path.is_file() and tuple(path.parts[-len(suffix):]) == suffix:
        entries = [str(path.parents[len(suffix) - 1]), *environment.get('PYTHONPATH', '').split(os.pathsep)]
        environment['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(entry for entry in entries if entry))
    return environment


def _lost():
    return WorkerLostError('Ray worker lost; effects uncertain (效果不确定). Retry with the same call_id.')


def _check_lease(executor, ident):
    status = ray.get(executor.resources.broker.status.remote(ident))
    if not status['valid']:
        raise _lost()


class _NodeOwner:
    def __init__(self, pool, bundle_index, *, lock_timeout=10):
        self.pool, self.bundle_index = pool, bundle_index
        self.jobs, self.groups = {}, {}
        self.lock = self._lock = threading.RLock()
        self._handles = set()
        self.devices = tuple(str(int(v)) if isinstance(v, (int, float)) else str(v) for v in ray.get_gpu_ids())
        self._gpu_locks = []
        root = Path(tempfile.gettempdir()) / 'rsisys-resources'
        root.mkdir(exist_ok=True)
        # These are the same locks as local execution. Supervisors inherit them,
        # so an actor crash cannot free a still executing child's physical GPU.
        deadline = time.monotonic() + lock_timeout
        try:
            for device in self.devices:
                stream = (root / (sha256(('gpu:' + device).encode()).hexdigest() + '.lock')).open('a+b')
                self._gpu_locks.append(stream)
                while True:
                    try:
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError('GPU locks were not available before owner startup deadline')
                        time.sleep(.01)
        except BaseException:
            for stream in self._gpu_locks:
                stream.close()
            raise

    def ready(self):
        return {'node_id': ray.get_runtime_context().get_node_id(), 'devices': self.devices}

    def _environment(self, supplied, assigned):
        if not all(type(i) is int and 0 <= i < len(self.devices) for i in assigned):
            raise PermissionError('Worker devices exceed its Ray bundle allocation')
        devices = ','.join(self.devices[i] for i in assigned)
        return dict(supplied, CUDA_VISIBLE_DEVICES=devices, NVIDIA_VISIBLE_DEVICES=devices)

    def launch(self, job_id, spec, registration, token):
        directory = Path(spec['directory'])
        directory.mkdir(parents=True, exist_ok=True)
        marker = Path(registration['marker_path']).open('a+b')
        fcntl.flock(marker, fcntl.LOCK_SH)
        payload = dict(command=spec['command'], cwd=spec['cwd'], timeout=spec['timeout'],
            limits=spec['limits'], marker_path=registration['marker_path'],
            budget_root=registration['budget_root'], lease_id=registration['lease_id'],
            owner_pid=os.getpid(), owner_identity=process_identity(os.getpid()),
            cancel_signal=spec['cancel_signal'], deadline_file=spec.get('deadline_file'), fds=[s.fileno() for s in self._gpu_locks] + [marker.fileno()])
        request = directory / 'request.json'
        request.write_text(json.dumps(payload))
        stream = open(spec['stdout'], 'ab') if spec['stdout'] is not None else subprocess.DEVNULL
        try:
            process = subprocess.Popen([sys.executable, str(Path(__file__).with_name('ray_worker_guard.py')), str(request)],
                env=self._environment(spec['environment'], spec['devices']),
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=tuple(payload['fds']))
        except BaseException:
            marker.close()
            if stream != subprocess.DEVNULL:
                stream.close()
            raise
        job = {'process': process, 'handle': WorkerHandle(self, process, directory), 'directory': directory, 'marker': marker,
               'broker': registration['broker'], 'lease_id': registration['lease_id'], 'token': token,
               'reference': spec['reference'], 'local_done': threading.Event()}
        with self.lock:
            self.jobs[job_id] = job
        threading.Thread(target=self._reap, args=(job,), daemon=True).start()
        if spec['reference'] is not None:
            atomic_write(directory / 'launched.json', json_bytes(spec['reference']))
        if stream != subprocess.DEVNULL:
            stream.close()
        return self.ready()

    @staticmethod
    def _reap(job):
        job['process'].wait()
        Path(job['marker'].name).touch()
        job['marker'].close()
        job['local_done'].set()
        # Exactly-once completion is enforced by the broker token, including when
        # terminate races normal exit. Node loss is handled by the driver's lost path.
        job['broker'].worker_done.remote(job['lease_id'], job['token'])

    def status(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
        code = job['process'].poll() if job['local_done'].is_set() else None
        path = job['directory'] / 'result.json'
        result = json.loads(path.read_text()) if path.exists() else {}
        return {'returncode': code, 'timed_out': result.get('timed_out', False)}

    def terminate(self, job_id, grace_seconds=10):
        with self.lock:
            job = self.jobs[job_id]
        job['handle'].terminate(grace_seconds)
        if not job['local_done'].wait(timeout=5):
            raise RuntimeError('Worker completion bookkeeping did not finish; retaining resource lease')
        return self.status(job_id)

    def cleanup(self, reference, command):
        if self.jobs[reference['job_id']]['reference'] != reference:
            raise PermissionError('Cleanup reference differs from the submitted Ray job')
        if not command or any(not isinstance(v, str) or not v for v in command):
            raise ValueError('Cleanup command must contain strings')
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=30,
            env=RayExecutor._sanitized_environment(dict(os.environ, CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='')))
        return {'code': result.returncode, 'stdout': result.stdout.decode(errors='replace')}

    def _group_request(self, group, request):
        from ray import cloudpickle
        with group['lock']:
            connection = Client(group['socket'], family='AF_UNIX', authkey=bytes.fromhex(group['auth']))
            try:
                connection.send_bytes(cloudpickle.dumps(request))
                response = cloudpickle.loads(connection.recv_bytes())
            finally:
                connection.close()
        if not response['ok']:
            raise response['error']
        return response['result']

    def group_create(self, key, cls, args, kwargs, allocation, registration, token, environment, directory, timeout):
        identity = (allocation['group'], allocation['rank_index'])
        with self.lock:
            group = self.groups.get(identity)
            if group is None:
                temporary = tempfile.TemporaryDirectory(prefix='rsisys-ray-group-')
                socket = str(Path(temporary.name) / 'rpc')
                auth = secrets.token_hex(32)
                ident = uuid4().hex
                deadline_file = Path(directory) / (ident + '.deadline.json')
                deadline_file.write_text(json.dumps({'expires_at': time.time() + timeout if timeout is not None else None}))
                spec = dict(directory=str(Path(directory) / ident), deadline_file=str(deadline_file),
                    command=[sys.executable, str(Path(__file__).with_name('ray_group_worker.py')), socket, auth],
                    cwd=None, timeout=None, limits={}, cancel_signal=signal.SIGTERM,
                    environment=environment, devices=allocation['devices'],
                    stdout=str(Path(directory) / (ident + '.log')), reference=None)
                self.launch(ident, spec, registration, token)
                group = {'socket': socket, 'auth': auth, 'temporary': temporary, 'objects': {},
                    'ident': ident, 'server_token': token, 'deadline_file': deadline_file, 'lock': threading.RLock()}
                self.groups[identity] = group
                deadline = time.monotonic() + 10
                while not Path(socket).exists():
                    status = self.status(ident)
                    if status['returncode'] is not None or time.monotonic() >= deadline:
                        self.terminate(ident)
                        self.groups.pop(identity, None)
                        temporary.cleanup()
                        raise RuntimeError('Ray worker group failed before RPC publication')
                    time.sleep(.01)
            if timeout is not None:
                deadline = time.time() + timeout
                current = json.loads(group['deadline_file'].read_text())['expires_at']
                if current is None or deadline < current:
                    temporary_deadline = group['deadline_file'].with_suffix('.tmp')
                    temporary_deadline.write_text(json.dumps({'expires_at': deadline}))
                    os.replace(temporary_deadline, group['deadline_file'])
            try:
                result = self._group_request(group, dict(mode='create', key=key, cls=cls,
                    args=args, kwargs=kwargs, marker=registration['marker_path'], environment={k: v for k, v in environment.items()
                        if k not in {'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES'}}))
            except BaseException:
                if not group['objects']:
                    self.terminate(group['ident'])
                    self.groups.pop(identity, None)
                    group['temporary'].cleanup()
                raise
            group['objects'][key] = (registration, token)
        return result

    def group_call(self, key, method, args, kwargs, environment):
        with self.lock:
            group = next(g for g in self.groups.values() if key in g['objects'])
        return self._group_request(group, dict(mode='call', key=key, method=method,
            args=args, kwargs=kwargs, environment={k: v for k, v in environment.items()
                        if k not in {'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES'}}))

    def group_destroy(self, key, *, force=False):
        with self.lock:
            identity, group = next(((identity, g) for identity, g in self.groups.items() if key in g['objects']), (None, None))
            if group is None:
                return
            if force:
                self.terminate(group['ident'])
            alive = self.jobs[group['ident']]['process'].poll() is None
            if alive:
                try:
                    self._group_request(group, dict(mode='destroy', key=key))
                except (EOFError, ConnectionError, OSError):
                    # A killed class process can close its socket just before the
                    # outer supervisor reaps it. Only release after that exit.
                    self.jobs[group['ident']]['process'].wait(timeout=5)
                    alive = False
            registration, token = group['objects'].pop(key)
            if token != group['server_token']:
                registration['broker'].worker_done.remote(registration['lease_id'], token)
            if not group['objects']:
                if alive:
                    self._group_request(group, dict(mode='close'))
                self.jobs[group['ident']]['process'].wait(timeout=5)
                self.groups.pop(identity)
                group['temporary'].cleanup()

    def close(self):
        for group in tuple(self.groups.values()):
            for key in tuple(group['objects']):
                self.group_destroy(key)
        with self.lock:
            jobs = list(self.jobs)
        for ident in jobs:
            self.terminate(ident)
        for stream in self._gpu_locks:
            stream.close()
        self._gpu_locks.clear()
        return True


_NodeOwnerActor = ray.remote(max_concurrency=64, max_restarts=0)(_NodeOwner)


class RayWorkerHandle:
    def __init__(self, executor, owner, ident, lease, token, *, detached=False, reference=None):
        self.executor, self.owner, self.ident = executor, owner, ident
        self.lease, self.token, self.detached = lease, token, detached
        self._result, self._reference, self._lost = None, reference, False

    @property
    def pid(self):
        raise TypeError('Remote workers have Ray identities; local PID protocols require executor=local')

    @property
    def worker_pid(self):
        return self.pid

    def reference(self):
        return dict(self._reference)

    def run_cleanup(self, command):
        try:
            return ray.get(self.owner.cleanup.remote(self.reference(), command))
        except ray.exceptions.RayError as error:
            raise _lost() from error

    def _worker_done(self):
        try:
            ray.get(self.executor.resources.broker.worker_done.remote(
                self._reference['lease_id'], self._reference['worker_token']))
        except ray.exceptions.RayError:
            pass  # Execution is confirmed; broker maintenance may finish bookkeeping.

    def _mark_lost(self):
        self._lost = True
        try:
            self._result = ray.get(self.owner.terminate.remote(self.ident), timeout=10)
        except ray.exceptions.ActorDiedError:
            pass
        except (ray.exceptions.RayError, AttributeError):
            return  # An unconfirmed termination must retain its worker token.
        self._worker_done()
        self.executor._handles.discard(self)

    def _status(self):
        if self._lost:
            raise _lost()
        if self._result is not None:
            return self._result
        try:
            result = ray.get(self.owner.status.remote(self.ident))
            if result['returncode'] is None:
                _check_lease(self.executor, self._reference['lease_id'])
            else:
                self._result = result
                self._worker_done()
            return result
        except (ray.exceptions.RayError, ValueError, WorkerLostError) as error:
            self._mark_lost()
            raise _lost() from error

    @property
    def returncode(self):
        return self._status()['returncode']

    @property
    def timed_out(self):
        return self._status()['timed_out']

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while (code := self.poll()) is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(self.ident, timeout)
            time.sleep(.05)
        return code

    def terminate(self, grace_seconds=10):
        if self._result is None:
            try:
                self._result = ray.get(self.owner.terminate.remote(self.ident, grace_seconds))
            except ray.exceptions.RayTaskError as error:
                raise error.as_instanceof_cause()
            except ray.exceptions.RayError as error:
                self._mark_lost()
                raise _lost() from error
        self._worker_done()
        self.executor._handles.discard(self)


class _ScopedExecutor:
    def __init__(self, owner, component):
        self.owner, self.name = owner, component

    def launch(self, spec, lease):
        env = self.owner._environment(self.name, spec.environment)
        return self.owner.launch(replace(spec, environment=dict(spec.environment, **env)), lease, _trusted_environment=env)

    def run(self, spec, lease):
        return self.owner._run(self.launch(spec, lease))

    def run_workers(self, commands, environments, directory, lease, *, timeout=None):
        from .workers import run_workers
        return run_workers(commands, environments, directory, timeout, executor=self, lease=lease)

    def reattach(self, reference):
        return self.owner.reattach(reference)

    def find_job(self, command):
        return self.owner.find_job(command)

    def worker_group(self, role, cls, ranks=None, *args, **kwargs):
        return self.owner.worker_group(self.name.removeprefix('lab:') + '.' + role, cls, ranks, *args,
                                       component=self.name, **kwargs)


class RayExecutor(Executor):
    def __init__(self, root, resources, budget, *, environment_provider=None):
        super().__init__(root, resources, budget)
        self.environment_provider, self._groups = environment_provider, set()

    def component(self, name):
        return _ScopedExecutor(self, name)

    def _environment(self, component, supplied=None):
        return self.environment_provider(component, supplied) if self.environment_provider else {}

    @staticmethod
    def _sanitized_environment(environment, trusted=None):
        runtime = re.compile(r'^(?:PATH|PYTHONPATH|PYTHONHOME|PYTHONUNBUFFERED|PYTHONDONTWRITEBYTECODE|'
            r'HOME|TMPDIR|TMP|TEMP|LD_LIBRARY_PATH|LANG|LC_ALL|TZ|HF_HOME|HUGGINGFACE_HUB_CACHE|'
            r'HF_HUB_OFFLINE|TRANSFORMERS_CACHE|TRANSFORMERS_OFFLINE|TOKENIZERS_PARALLELISM|OPENBLAS_NUM_THREADS|'
            r'PYTORCH_CUDA_ALLOC_CONF|CONDA_PREFIX|SSL_CERT_FILE|TRITON_CACHE_DIR|'
            r'NUMEXPR_NUM_THREADS|VECLIB_MAXIMUM_THREADS|(?:CUDA|NVIDIA|NCCL|OMP|MKL|TORCH|VLLM)_[A-Z0-9_]+)$')
        sensitive = re.compile(r'(?:^|_)(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|COOKIE|SESSION)(?:_|$)')
        result = {key: value for key, value in environment.items() if runtime.fullmatch(key)
                  and not sensitive.search(key)}
        for key in ('RSISYS_HOST_URL', 'RSISYS_HOST_TOKEN', 'RSISYS_LOG_URL', 'RSISYS_LOG_TOKEN',
                    'RSISYS_OBSERVATION_URL',
                    'RSISYS_MODEL_BASE_URL', 'OPENAI_BASE_URL', 'OPENAI_API_KEY'):
            if key in (trusted or {}):
                result[key] = trusted[key]
        result.update({key: value for key, value in (trusted or {}).items()
                       if re.fullmatch(r'RSISYS_NESTED_MODEL_TOKEN_[0-9A-F]{32}', key)})
        return result

    @staticmethod
    def _owner_name(allocation):
        return 'rsisys-bundle-owner-' + sha256((allocation['pool'] + ':' + str(allocation['bundle_index']) + ':'
            + allocation['placement_group'].id.hex()).encode()).hexdigest()[:24]

    def _owner(self, allocation):
        settings = self.resources.config.get('host', {})
        timeout = settings.get('owner_ready_timeout', 30)
        existing = ray.get(self.resources.broker.owner.remote(allocation['pool'], allocation['bundle_index']))
        if existing is not None:
            try:
                if ray.get(existing.ready.remote(), timeout=timeout)['node_id'] == allocation['node_id']:
                    return existing
            except ray.exceptions.RayError:
                pass
            ray.kill(existing, no_restart=True)
            ray.get(self.resources.broker.unregister_owner.remote(
                allocation['pool'], allocation['bundle_index'], existing))
        name = self._owner_name(allocation)
        strategy = PlacementGroupSchedulingStrategy(placement_group=allocation['placement_group'],
            placement_group_bundle_index=allocation['bundle_index'], placement_group_capture_child_tasks=False)
        actor = _NodeOwnerActor.options(name=name, namespace=self.resources.namespace, lifetime='detached', get_if_exists=True,
            num_cpus=allocation['bundle_cpus'], num_gpus=allocation['bundle_gpus'],
            scheduling_strategy=strategy, runtime_env={'env_vars': {
                'PYTHONPATH': str(Path(__file__).resolve().parents[3]) + os.pathsep + os.environ.get('PYTHONPATH', '')}}).remote(
                    allocation['pool'], allocation['bundle_index'], lock_timeout=settings.get('gpu_lock_timeout', 30))
        try:
            ready = ray.get(actor.ready.remote(), timeout=timeout)
            if ready['node_id'] != allocation['node_id']:
                raise _lost()
        except BaseException:
            ray.kill(actor, no_restart=True)
            raise
        return ray.get(self.resources.broker.register_owner.remote(
            allocation['pool'], allocation['bundle_index'], actor, ready['node_id']))

    def launch(self, spec, lease, *, _trusted_environment=None):
        if not isinstance(spec, WorkerSpec) or lease.manager is not self.resources or lease.released:
            raise TypeError('launch requires a WorkerSpec and a live Ray Lease')
        if self._closed:
            raise RuntimeError('Executor admission is closed')
        if not spec.command or any(not isinstance(v, str) or not v for v in spec.command):
            raise ValueError('Worker command must contain nonempty strings')
        if lease.mode == 'hybrid':
            raise ValueError('Hybrid roles execute in one process through worker_group, not subprocess launch')
        _trusted_environment = dict(_trusted_environment or {}, **self._environment(None, spec.environment))
        allocations = lease.allocations
        requested = tuple(lease.devices if spec.devices is None else spec.devices)
        if any(type(device) is not int for device in requested) or not set(requested) <= set(lease.devices):
            raise PermissionError('Worker devices exceed its Ray lease')
        offset, selected = 0, []
        for allocation in allocations:
            count = allocation['gpu_count']
            if (requested and set(requested) <= set(range(offset, offset + count))
                    or not requested and len(allocations) == 1):
                selected.append((allocation, tuple(device - offset for device in requested)))
            offset += allocation['gpu_count']
        if len(selected) != 1:
            raise ValueError('One command must fit one Ray bundle; use worker_group across bundles')
        with self._lock:
            if spec.detached and getattr(lease, '_launched_count', 0):
                raise ValueError('Detached workers require a lease with no previous launches')
            timeout_values = [v for v in (spec.timeout, lease.timeout) if v is not None]
            timeout = self.budget.timeout(min(timeout_values) if timeout_values else None, gpus=len(lease.devices))
            ident = uuid4().hex
            allocation, devices = selected[0]
            token = lease.retain(ident, bundle_index=allocation['bundle_index'])
            try:
                owner = self._owner(allocation)
                stdout = None
                if spec.stdout is not None:
                    if not isinstance(getattr(spec.stdout, 'name', None), str):
                        raise TypeError('Ray stdout must be a named file on the shared filesystem')
                    spec.stdout.flush()
                    stdout = str(Path(spec.stdout.name).resolve())
                reference = dict(backend='ray', actor=self._owner_name(allocation), namespace=self.resources.namespace,
                    job_id=ident, command_digest=sha256(json.dumps(list(spec.command)).encode()).hexdigest(),
                    lease_id=lease.ident, worker_token=token)
                payload = dict(directory=str(self.root / ident), command=list(spec.command), reference=reference,
                    devices=devices,
                    environment=self._sanitized_environment(spec.environment, _trusted_environment), cwd=str(spec.cwd) if spec.cwd is not None else None,
                    timeout=timeout, limits=spec.limits, cancel_signal=spec.cancel_signal, stdout=stdout)
                submission = owner.launch.remote(ident, payload, lease.worker_registration(), token)
                try:
                    ray.get(submission)
                except BaseException as error:
                    raise WorkerLaunchUncertainError('Ray worker submitted without a confirmed launch outcome') from error
            except WorkerLaunchUncertainError:
                # A remote owner may already retain the marker and run a child.
                # Reconciliation must establish cleanup before quota is returned.
                raise
            except BaseException:
                lease.worker_done(token)
                raise
            lease._launched_count = getattr(lease, '_launched_count', 0) + 1
            handle = RayWorkerHandle(self, owner, ident, lease, token, detached=spec.detached, reference=reference)
            if spec.detached:
                lease.release()
            else:
                self._handles.add(handle)
            return handle

    def find_job(self, command):
        digest = sha256(json.dumps(list(command)).encode()).hexdigest()
        references = [json.loads(path.read_text()) for path in self.root.glob('*/launched.json')]
        matches = [reference for reference in references if reference['command_digest'] == digest]
        if len(matches) > 1:
            raise WorkerLostError('Multiple Ray jobs match this command; effects uncertain')
        return matches[0] if matches else None

    def reattach(self, reference):
        required = {'backend', 'actor', 'namespace', 'job_id', 'command_digest', 'lease_id', 'worker_token'}
        if not isinstance(reference, dict) or set(reference) != required or reference['backend'] != 'ray':
            raise ValueError('Invalid Ray worker reference')
        if (not all(isinstance(value, str) and value for value in reference.values())
                or not re.fullmatch(r'[0-9a-f]{32}', reference['job_id'])):
            raise ValueError('Invalid Ray worker identity')
        if reference['namespace'] != self.resources.namespace:
            raise PermissionError('Ray worker belongs to another resource namespace')
        path = self.root / reference['job_id'] / 'launched.json'
        if not path.is_file() or json.loads(path.read_text()) != reference:
            raise PermissionError('Ray reference differs from the confirmed submitted command')
        request = json.loads(path.with_name('request.json').read_text())
        if sha256(json.dumps(request['command']).encode()).hexdigest() != reference['command_digest']:
            raise PermissionError('Ray worker command differs from its submitted reference')
        try:
            owner = ray.get_actor(reference['actor'], namespace=reference['namespace'])
            handle = RayWorkerHandle(self, owner, reference['job_id'], None, reference['worker_token'],
                                     detached=True, reference=dict(reference))
            handle._status()
            return handle
        except (ValueError, ray.exceptions.RayError) as error:
            raise _lost() from error

    def worker_group(self, role, cls, ranks=None, *args, component=None, **kwargs):
        if self._closed:
            raise RuntimeError('Executor admission is closed')
        group = RayWorkerGroup(self, role, cls, ranks, args, kwargs, component)
        self._groups.add(group)
        return group

    def close(self):
        self._closed = True
        for group in tuple(self._groups):
            group.close()
        for handle in tuple(self._handles):
            try:
                handle.terminate()
            except WorkerLostError:
                pass

    def validate_snapshot(self):
        if self._groups or any(handle.poll() is None for handle in self._handles):
            raise RuntimeError('Snapshot requires all Ray workers and worker groups exited')


class RayWorkerGroup:
    """A leased group of trusted Python objects; hybrid partners share a process."""
    def __init__(self, executor, role, cls, ranks, args, kwargs, component):
        self.executor, self.component = executor, component
        self.lease = executor.resources.acquire(role)
        self.members, self.closed = [], False
        count = len(self.lease.devices) or sum(a['cpus'] for a in self.lease.allocations)
        ranks = count if ranks is None else ranks
        if type(ranks) is not int or not 0 < ranks <= count:
            self.lease.release()
            raise ValueError('Worker group ranks must fit its leased GPUs or CPUs')
        try:
            remaining = ranks
            for allocation in self.lease.allocations:
                local_count = allocation['gpu_count'] or allocation['cpus']
                owner = executor._owner(allocation)
                for index in range(min(local_count, remaining)):
                    key, token = uuid4().hex, self.lease.retain(bundle_index=allocation['bundle_index'])
                    rank_allocation = dict(allocation, devices=(index,) if allocation['gpu_count'] else (), rank_index=index)
                    self.members.append((owner, key, token))
                    trusted = executor._environment(component)
                    environment = _class_environment(cls, dict(worker_environment(), **trusted))
                    ray.get(owner.group_create.remote(key, cls, args, kwargs, rank_allocation,
                        self.lease.worker_registration(), token, executor._sanitized_environment(environment, trusted), str(executor.root),
                        executor.budget.timeout(self.lease.timeout, gpus=len(self.lease.devices))))
                    remaining -= 1
                if remaining == 0:
                    break
        except BaseException:
            self.close()
            raise

    def _call(self, members, method, arguments, kwargs):
        if self.closed:
            raise RuntimeError('Worker group is closed')
        try:
            _check_lease(self.executor, self.lease.ident)
            env = self.executor._environment(self.component) if self.component else {}
            refs = [owner.group_call.remote(key, method, args, kwargs, env)
                    for (owner, key, _), args in zip(members, arguments, strict=True)]
            pending = refs
            while pending:
                _, pending = ray.wait(pending, num_returns=len(pending), timeout=.1)
                _check_lease(self.executor, self.lease.ident)
            return ray.get(refs)
        except ray.exceptions.RayTaskError as exc:
            cause = exc.as_instanceof_cause()
            if not isinstance(cause, (EOFError, ConnectionError, OSError)):
                raise cause
            error = exc
        except (ray.exceptions.RayError, WorkerLostError) as exc:
            error = exc
        try:
            self.close(force=True)
        except Exception as cleanup_error:
            raise _lost() from cleanup_error
        raise _lost() from error

    def call_all(self, method, *args, **kwargs):
        return self._call(self.members, method, [args] * len(self.members), kwargs)

    def call_rank0(self, method, *args, **kwargs):
        return self._call(self.members[:1], method, [args], kwargs)[0]

    def map(self, method, shards):
        shards = tuple(shards)
        if len(shards) != len(self.members):
            raise ValueError('Provide one shard per worker rank')
        return self._call(self.members, method, [(shard,) for shard in shards], {})

    def close(self, *, force=False):
        if self.closed:
            return
        # Actor/node death is terminal. Retrying a lost object cannot help cleanup;
        # identity or user close failures remain retryable and keep their lease.
        while self.members:
            owner, key, token = self.members[-1]
            try:
                ray.get(owner.group_destroy.remote(key, force=force), timeout=30)
                self.lease.worker_done(token)
            except ray.exceptions.ActorDiedError:
                self.lease.worker_done(token)
            except ray.exceptions.RayTaskError as exc:
                raise exc.as_instanceof_cause()
            except ray.exceptions.RayError as exc:
                raise _lost() from exc
            self.members.pop()
        self.lease.release()
        self.closed = True
        self.executor._groups.discard(self)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
