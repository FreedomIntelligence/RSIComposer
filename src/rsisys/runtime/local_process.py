"""Local process execution for trusted tasks. This is NOT a security sandbox."""

from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
from uuid import uuid4

from rsisys.loops.contracts.jobs import JobStatus
from .sandbox import SandboxSpec, SandboxRef, ExecRequest, ProcessRef, ProcessState, SandboxCapabilities


@dataclass
class _Process:
    process: subprocess.Popen
    timeout: float
    stdout: bytearray = field(default_factory=bytearray)
    stderr: bytearray = field(default_factory=bytearray)
    done: threading.Event = field(default_factory=threading.Event)
    lock: threading.RLock = field(default_factory=threading.RLock)
    state: ProcessState | None = None
    cancelled: bool = False
    timed_out: bool = False
    truncated: bool = False


class LocalProcessBackend:
    def __init__(self, root, *, max_output_bytes: int = 1024 * 1024):
        if type(max_output_bytes) is not int or max_output_bytes < 1:
            raise ValueError('max_output_bytes must be positive')
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_output_bytes = max_output_bytes
        self._sessions: dict[SandboxRef, tuple[Path, float]] = {}
        self._processes: dict[ProcessRef, _Process] = {}

    @property
    def active_count(self):
        return len(self._sessions)

    def capabilities(self):
        return SandboxCapabilities(simulated=False)

    def create(self, spec: SandboxSpec) -> SandboxRef:
        if spec.image_ref != 'local':
            raise ValueError('LocalProcessBackend only accepts the explicit local execution profile')
        ref = SandboxRef(uuid4().hex)
        self._sessions[ref] = (Path(tempfile.mkdtemp(prefix='session-', dir=self.root)), spec.timeout_seconds)
        return ref

    def workspace(self, sandbox: SandboxRef) -> Path:
        if sandbox not in self._sessions:
            raise ValueError('Unknown or closed local session')
        return self._sessions[sandbox][0]

    def file_path(self, sandbox: SandboxRef, relative: str) -> Path:
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError('Workspace paths must be relative')
        root = self.workspace(sandbox)
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Path escapes the local workspace')
        return path

    def exec(self, sandbox: SandboxRef, request: ExecRequest) -> ProcessRef:
        root = self.workspace(sandbox)
        timeout = min(request.timeout_seconds, self._sessions[sandbox][1])
        # Do not inherit credential environment variables into model-generated commands.
        environment = {'PATH': os.environ.get('PATH', os.defpath), 'HOME': str(root),
                       'TMPDIR': str(root), 'LANG': 'C.UTF-8'}
        process = subprocess.Popen(request.argv, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        ref = ProcessRef(uuid4().hex, sandbox)
        entry = _Process(process, timeout)
        self._processes[ref] = entry
        threading.Thread(target=self._watch, args=(entry,), daemon=True).start()
        return ref

    def _drain(self, stream, target: bytearray, entry: _Process):
        try:
            while block := stream.read(8192):
                with entry.lock:
                    room = max(0, self.max_output_bytes - len(target))
                    target.extend(block[:room])
                    entry.truncated = entry.truncated or len(block) > room
        finally:
            stream.close()

    @staticmethod
    def _kill(entry: _Process):
        try:
            os.killpg(entry.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def _watch(self, entry: _Process):
        readers = [threading.Thread(target=self._drain, args=(stream, target, entry), daemon=True)
                   for stream, target in ((entry.process.stdout, entry.stdout), (entry.process.stderr, entry.stderr))]
        for reader in readers:
            reader.start()
        try:
            try:
                code = entry.process.wait(timeout=entry.timeout)
            except subprocess.TimeoutExpired:
                with entry.lock:
                    entry.timed_out = True
                self._kill(entry)
                code = entry.process.wait()
            # A command owns its child processes; background children are not retained.
            self._kill(entry)
            for reader in readers:
                reader.join(timeout=2)
            with entry.lock:
                status = (JobStatus.CANCELLED if entry.cancelled else
                          JobStatus.FAILED if entry.timed_out or code != 0 else JobStatus.SUCCEEDED)
                stderr = entry.stderr.decode('utf-8', errors='replace')
                if entry.timed_out:
                    stderr += '\nExecution timed out'
                if entry.truncated:
                    stderr += '\nOutput was truncated at the configured byte limit'
                entry.state = ProcessState(status, entry.stdout.decode('utf-8', errors='replace'),
                                           stderr, code, False)
        finally:
            entry.done.set()

    def _get(self, ref: ProcessRef) -> _Process:
        if ref not in self._processes:
            raise ValueError('Unknown process reference')
        return self._processes[ref]

    def poll(self, ref: ProcessRef) -> ProcessState:
        entry = self._get(ref)
        with entry.lock:
            if entry.state is not None:
                return entry.state
            if entry.done.is_set():
                raise RuntimeError('Process watcher failed before reporting an outcome')
            return ProcessState(JobStatus.RUNNING, entry.stdout.decode('utf-8', errors='replace'),
                                entry.stderr.decode('utf-8', errors='replace'), None, False)

    def wait(self, ref: ProcessRef) -> ProcessState:
        entry = self._get(ref)
        if not entry.done.wait(entry.timeout + 10):
            self.cancel(ref)
            raise TimeoutError('Local process cleanup did not finish')
        return self.poll(ref)

    def cancel(self, ref: ProcessRef) -> None:
        entry = self._get(ref)
        with entry.lock:
            if entry.state is not None:
                return
            entry.cancelled = True
            self._kill(entry)

    def close(self, sandbox: SandboxRef) -> None:
        if sandbox not in self._sessions:
            return
        for ref, entry in tuple(self._processes.items()):
            if ref.sandbox == sandbox and not entry.done.is_set():
                self.cancel(ref)
                if not entry.done.wait(10):
                    raise TimeoutError('Local processes did not stop; workspace was retained')
        root, _ = self._sessions.pop(sandbox)
        shutil.rmtree(root)

    def close_all(self):
        for ref in tuple(self._sessions):
            self.close(ref)
