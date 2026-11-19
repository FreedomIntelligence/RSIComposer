"""Host-owned supervised processes; detached jobs keep their resource locks."""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from uuid import uuid4

from .workers import worker_environment
from .process_identity import process_identity, process_group_alive


@dataclass(frozen=True)
class WorkerSpec:
    command: tuple[str, ...] | list[str]
    environment: dict = field(default_factory=dict)
    cwd: Path | str | None = None
    stdout: object = None
    timeout: float | None = None
    devices: tuple[int, ...] | None = None
    limits: dict = field(default_factory=dict)
    detached: bool = False
    cancel_signal: int = signal.SIGTERM

    def __post_init__(self):
        import math
        if self.timeout is not None and (type(self.timeout) not in (int, float)
                or not math.isfinite(self.timeout) or self.timeout <= 0):
            raise ValueError('Worker timeout must be finite and positive')
        if type(self.detached) is not bool or self.cancel_signal not in (signal.SIGTERM, signal.SIGINT):
            raise ValueError('Invalid worker lifecycle options')


class WorkerHandle:
    def __init__(self, executor, process, directory):
        self.executor, self.process, self.directory = executor, process, directory
        self.pid = process.pid

    @property
    def returncode(self):
        return self.process.returncode

    @property
    def timed_out(self):
        path = self.directory / 'result.json'
        return path.exists() and json.loads(path.read_text()).get('timed_out', False)

    def poll(self):
        return self.process.poll()

    def wait(self, timeout=None):
        return self.process.wait(timeout=timeout)

    @property
    def worker_pid(self):
        path = self.directory / 'child.json'
        deadline = time.monotonic() + 5
        while not path.exists():
            if self.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError('Worker supervisor failed before child publication')
            time.sleep(.01)
        return json.loads(path.read_text())['pid']

    def terminate(self, grace_seconds=10):
        if self.poll() is None:
            self.process.send_signal(signal.SIGTERM)
        try:
            self.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            # The child starts a separate session. Killing only its supervisor
            # would release lease FDs while the actual work continues.
            path = self.directory / 'child.json'
            try:
                child = json.loads(path.read_text())
            except (OSError, ValueError) as exc:
                raise RuntimeError('Cannot safely terminate worker without child identity') from exc
            pid, identity = child['pid'], child.get('identity')
            if identity is None or process_identity(pid) != identity:
                raise RuntimeError('Worker child identity changed; retaining supervisor and leases')
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 5
            while process_group_alive(pid):
                if time.monotonic() >= deadline:
                    raise RuntimeError('Worker process group did not exit; retaining supervisor and leases')
                time.sleep(.01)
            # A stopped supervisor can now reap and publish through its finally.
            if self.poll() is None:
                self.process.send_signal(signal.SIGCONT)
            try:
                self.wait(timeout=1)
            except subprocess.TimeoutExpired:
                request = json.loads((self.directory / 'request.json').read_text())
                marker = Path(request['budget_root']) / 'leases' / (request['lease_id'] + '.lock')
                marker.touch()
                self.process.kill()
                self.wait()
        with self.executor._lock:
            self.executor._handles.discard(self)


class Executor:
    def __new__(cls, root, resources, budget, **options):
        if cls is Executor and resources.config.get('executor', 'local') == 'ray':
            from .ray_executor import RayExecutor
            return RayExecutor(root, resources, budget, **options)
        return super().__new__(cls)

    def __init__(self, root, resources, budget, *, environment_provider=None):
        self.root, self.resources, self.budget = Path(root), resources, budget
        self.root.mkdir(parents=True, exist_ok=True)
        self._handles, self._closed = set(), False
        self._lock = threading.RLock()

    def launch(self, spec, lease):
        if not isinstance(spec, WorkerSpec) or lease.manager is not self.resources or lease.released:
            raise TypeError('launch requires a WorkerSpec and a live host Lease')
        if (not spec.command or any(not isinstance(v, str) or not v for v in spec.command)):
            raise ValueError('Worker command must contain nonempty strings')
        devices = tuple(lease.devices if spec.devices is None else spec.devices)
        if any(type(i) is not int or i < 0 for i in devices):
            raise ValueError('Worker devices must be nonnegative physical indices')
        if not set(devices) <= set(lease.devices):
            raise PermissionError('Worker devices exceed its lease')
        inherited = os.environ.get('CUDA_VISIBLE_DEVICES')
        if devices and inherited is not None:
            visible = inherited.split(',') if inherited else []
            if any(not value.isdecimal() for value in visible):
                inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                    '--format=csv,noheader'], text=True, timeout=10)
                mapping = {uuid.strip(): index.strip() for index, uuid in
                           (line.split(',') for line in inventory.splitlines())}
                visible = [mapping.get(value, value) for value in visible]
            if not {str(device) for device in devices} <= set(visible):
                raise PermissionError('Worker devices exceed inherited CUDA_VISIBLE_DEVICES')
        timeouts = [v for v in (spec.timeout, lease.timeout) if v is not None]
        timeout = self.budget.timeout(min(timeouts) if timeouts else None, gpus=len(lease.devices))
        directory = self.root / uuid4().hex
        directory.mkdir()
        payload = dict(command=list(spec.command), cwd=str(spec.cwd) if spec.cwd is not None else None,
                       timeout=timeout, limits=spec.limits, fds=list(lease.fds),
                       budget_root=str(self.budget.root), lease_id=lease.ident,
                       detached=spec.detached, cancel_signal=spec.cancel_signal)
        request = directory / 'request.json'
        request.write_text(json.dumps(payload))
        env = dict(spec.environment, CUDA_VISIBLE_DEVICES=','.join(map(str, devices)),
                   NVIDIA_VISIBLE_DEVICES=','.join(map(str, devices)))
        from rsisys.runtime.observation import observations_suppressed
        if observations_suppressed():
            from rsisys.runtime.observation.client import INBOX_ENV
            env.pop(INBOX_ENV, None)
        with self._lock:
            if self._closed:
                raise RuntimeError('Executor admission is closed')
            if spec.detached and lease._launched_count:
                raise ValueError('Detached workers require a lease with no previous launches')
            process = subprocess.Popen([sys.executable, str(Path(__file__).with_name('worker_guard.py')), str(request)],
                env=env, stdin=subprocess.DEVNULL, stdout=spec.stdout, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=lease.fds)
            lease._launched_count += 1
            handle = WorkerHandle(self, process, directory)
            if spec.detached:
                # A detached supervisor now owns these same open file descriptions.
                # Closing parent descriptors cannot unlock its live worker's GPUs.
                lease.transferred = True
                lease.release()
                threading.Thread(target=process.wait, daemon=True).start()
            else:
                self._handles.add(handle)
            return handle

    def run(self, spec, lease):
        return self._run(self.launch(spec, lease))

    def _run(self, handle):
        try:
            code = handle.wait()
            if handle.timed_out:
                raise TimeoutError('Host worker timeout exceeded')
            return code
        finally:
            handle.terminate()
            with self._lock:
                self._handles.discard(handle)

    def run_workers(self, commands, environments, directory, lease, *, timeout=None):
        """Reuse fail-fast group polling while all children share one host lease."""
        from .workers import run_workers
        return run_workers(commands, environments, directory, timeout,
                           executor=self, lease=lease)

    def close(self):
        with self._lock:
            self._closed = True
            handles = tuple(self._handles)
        for handle in handles:
            handle.terminate()
            self._handles.discard(handle)

    def validate_snapshot(self):
        if any(handle.poll() is None for handle in self._handles):
            raise RuntimeError('Snapshot requires all owned workers exited')


def execution_capabilities(context):
    """Bind real host capabilities; standalone Lab fixtures receive an owned CPU host."""
    resources, executor = getattr(context, 'resources', None), getattr(context, 'executor', None)
    if resources is not None and executor is not None:
        from .resources import ResourceManager
        return (resources.component('lab') if isinstance(resources, ResourceManager) else resources), executor
    # Explicit standalone tests can use this context without an Application.
    # GPU roles still require explicit placement and never guess a device.
    from .budget import HostBudget
    from .resources import ResourceManager
    root = Path(context.data_dir) / 'host_execution'
    budget = HostBudget(root, restoring=(root / 'budget.json').exists())
    manager = ResourceManager(budget=budget)
    executor = Executor(root / 'workers', manager, budget)
    resources = manager.component('standalone')
    context.resources, context.executor = resources, executor
    return resources, executor


def run_cpu(context, role, command, *, environment=None, cwd=None, stdout=None, timeout=None, limits=None):
    from .resources import ResourceRequest
    resources, executor = execution_capabilities(context)
    resources.declare({role: ResourceRequest()})
    with resources.acquire(role) as lease:
        return executor.run(WorkerSpec(command, worker_environment() if environment is None else environment,
            cwd=cwd, stdout=stdout, timeout=timeout, limits=limits or {}), lease)
