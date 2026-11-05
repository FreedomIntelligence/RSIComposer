"""Local process execution for trusted tasks. This is NOT a security sandbox."""

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import shutil
import select
import signal
import subprocess
import tempfile
import threading
import time
from uuid import uuid4

from rsisys.loops.contracts.jobs import JobStatus
from .sandbox import SandboxSpec, SandboxRef, ExecRequest, ProcessRef, ProcessState, SandboxCapabilities


def _kill_process_group(process):
    """Caller serializes with wait/poll; only an unreaped leader reserves its PGID."""
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        # Darwin reports EPERM when a group contains only zombies. A genuine
        # permission failure with live members must still fail cleanup.
        rows = subprocess.check_output(['ps', '-axo', 'pgid=,stat='], text=True, timeout=5)
        for row in rows.splitlines():
            group, status = row.split()
            if int(group) == process.pid and not status.startswith('Z'):
                raise


def _wait_for_process_exit(process, timeout):
    """Observe exit without reaping, so the owned group cannot be recycled."""
    pid = process.pid
    if hasattr(os, 'waitid'):
        deadline = time.monotonic() + timeout
        while os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, timeout)
            time.sleep(min(.01, remaining))
    else:
        queue = select.kqueue()
        try:
            try:
                queue.control([select.kevent(pid, filter=select.KQ_FILTER_PROC,
                    flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                    fflags=select.KQ_NOTE_EXIT)], 0, 0)
            except ProcessLookupError:
                return  # Already exited, but still unreaped by this owner.
            if not queue.control(None, 1, timeout):
                raise subprocess.TimeoutExpired(process.args, timeout)
        finally:
            queue.close()


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
    error: str | None = None
    readers: list[threading.Thread] = field(default_factory=list)
    logs: dict = field(default_factory=dict)


class LocalProcessBackend:
    def __init__(self, root, *, max_output_bytes: int = 1024 * 1024,
                 max_log_bytes: int = 64 * 1024 * 1024):
        if type(max_output_bytes) is not int or max_output_bytes < 1:
            raise ValueError('max_output_bytes must be positive')
        if type(max_log_bytes) is not int or max_log_bytes < 1:
            raise ValueError('max_log_bytes must be positive')
        if not hasattr(os, 'waitid') and not hasattr(select, 'kqueue'):
            raise NotImplementedError('Local process cleanup requires waitid or kqueue exit observation')
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_output_bytes = max_output_bytes
        self.max_log_bytes = max_log_bytes
        self.log_root = self.root / 'logs'
        self.log_root.mkdir(mode=0o700, exist_ok=True)
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
                       'TMPDIR': str(root), 'LANG': 'C.UTF-8', 'PYTHONUNBUFFERED': '1'}
        ref = ProcessRef(uuid4().hex, sandbox)
        directory = Path(tempfile.mkdtemp(prefix=ref.process_id + '-', dir=self.log_root))
        logs = {}
        try:
            for name in ('stdout', 'stderr'):
                path, tail = directory / (name + '.bin'), directory / (name + '.tail.bin')
                logs[name] = {'path': path, 'tail_path': tail,
                    'file': path.open('x+b', buffering=0), 'tail_file': tail.open('x+b', buffering=0),
                    'observed': 0, 'stored': 0, 'tail': bytearray(), 'sha256': None}
            process = subprocess.Popen(request.argv, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        except BaseException:
            for log in logs.values():
                log['file'].close()
                log['tail_file'].close()
            shutil.rmtree(directory)
            raise
        entry = _Process(process, timeout, logs=logs)
        self._processes[ref] = entry
        try:
            self._process_started(ref, request, entry)
            threading.Thread(target=self._watch, args=(entry,), daemon=True).start()
        except Exception as exc:
            self._record_error(entry, exc)
            try:
                self._finish(entry)
            finally:
                entry.done.set()
            raise
        return ref

    def _process_started(self, ref, request, entry):
        """Optional explicit persistence integration, before starting supervision."""

    def _drain(self, stream, target: bytearray, entry: _Process, name):
        try:
            # Buffered read(size) can wait for size bytes or EOF, hiding short
            # flushed logs throughout a running program. read1 drains what is
            # available from the pipe without waiting to fill the buffer.
            while block := stream.read1(8192):
                with entry.lock:
                    log = entry.logs[name]
                    log['observed'] += len(block)
                    room = max(0, self.max_log_bytes - log['stored'])
                    if room:
                        kept = block[:room]
                        log['file'].write(kept)
                        log['stored'] += len(kept)
                    log['tail'].extend(block)
                    del log['tail'][:-self.max_output_bytes]
                    log['tail_file'].seek(0)
                    log['tail_file'].write(log['tail'])
                    log['tail_file'].truncate()
                    # Bounded head + tail for ordinary receipts. Full byte
                    # windows come from the durable log, not this preview.
                    target.extend(block)
                    if len(target) > self.max_output_bytes:
                        head = self.max_output_bytes // 2
                        del target[head:len(target) - (self.max_output_bytes - head)]
                        entry.truncated = True
        except Exception as exc:
            self._record_error(entry, exc)
        finally:
            stream.close()

    @staticmethod
    def _record_error(entry: _Process, exc: Exception):
        message = f'Process supervision failed ({type(exc).__name__}): {exc}'
        with entry.lock:
            if entry.error is None:
                entry.error = message
            elif message not in entry.error:
                entry.error += '\n' + message

    @staticmethod
    def _kill(entry: _Process):
        with entry.lock:
            _kill_process_group(entry.process)

    def _watch(self, entry: _Process):
        try:
            for name, stream, target in (('stdout', entry.process.stdout, entry.stdout),
                                         ('stderr', entry.process.stderr, entry.stderr)):
                reader = threading.Thread(target=self._drain, args=(stream, target, entry, name), daemon=True)
                reader.start()
                entry.readers.append(reader)
            try:
                _wait_for_process_exit(entry.process, entry.timeout)
            except subprocess.TimeoutExpired:
                with entry.lock:
                    entry.timed_out = True
        except Exception as exc:
            self._record_error(entry, exc)
        finally:
            try:
                self._finish(entry)
            except Exception as exc:
                # A finished watcher is not proof of process exit. Retain the
                # session and unreaped process so close() can retry cleanup.
                self._record_error(entry, exc)
            finally:
                entry.done.set()

    def _finish(self, entry: _Process):
        with entry.lock:
            if entry.state is not None:
                return
            # A command owns its child processes; signal before wait() reaps
            # the group leader, including on normal exit or observer failure.
            self._kill(entry)
            code = entry.process.wait(timeout=10)
        for reader in entry.readers:
            reader.join(timeout=2)
        # A watcher or reader may have failed to start; close pipes for which
        # no reader ever took ownership.
        for stream in (entry.process.stdout, entry.process.stderr)[len(entry.readers):]:
            stream.close()
        with entry.lock:
            if any(reader.is_alive() for reader in entry.readers):
                raise TimeoutError('Owned log readers did not finish; retain ownership for cleanup retry')
            for log in entry.logs.values():
                if not log['file'].closed:
                    os.fsync(log['file'].fileno())
                    os.fsync(log['tail_file'].fileno())
                    log['file'].close()
                    log['tail_file'].close()
                    with log['path'].open('rb') as saved:
                        log['sha256'] = hashlib.file_digest(saved, 'sha256').hexdigest()
            status = (JobStatus.FAILED if entry.error else
                      JobStatus.CANCELLED if entry.cancelled else
                      JobStatus.FAILED if entry.timed_out or code != 0 else JobStatus.SUCCEEDED)
            stderr = entry.stderr.decode('utf-8', errors='replace')
            if entry.timed_out:
                stderr += '\nExecution timed out'
            if entry.truncated:
                stderr += '\nOutput was truncated in the preview; read durable logs for retained byte windows'
            if any(log['observed'] > log['stored'] for log in entry.logs.values()):
                stderr += '\nDurable log quota reached; retained head/tail coverage is explicit in logs'
            if entry.error:
                stderr += '\n' + entry.error
            entry.state = ProcessState(status, entry.stdout.decode('utf-8', errors='replace'),
                                       stderr, code, False,
                                       'infrastructure' if entry.error else
                                       'timeout' if entry.timed_out else
                                       'program' if status == JobStatus.FAILED else None)

    def _get(self, ref: ProcessRef) -> _Process:
        if ref not in self._processes:
            raise ValueError('Unknown process reference')
        return self._processes[ref]

    def log_info(self, ref: ProcessRef):
        entry = self._get(ref)
        with entry.lock:
            return {name: {'path': str(log['path']), 'tail_path': str(log['tail_path']),
                'observed_bytes': log['observed'], 'stored_bytes': log['stored'],
                'tail_offset': log['observed'] - len(log['tail']), 'tail_bytes': len(log['tail']),
                'quota_bytes': self.max_log_bytes, 'quota_exceeded': log['observed'] > log['stored'],
                'tail_sha256': hashlib.sha256(log['tail']).hexdigest(),
                'sha256': log['sha256'], 'complete': entry.state is not None}
                for name, log in entry.logs.items()}

    def read_log(self, ref: ProcessRef, stream, *, offset=None, max_bytes=8192):
        if (stream not in ('stdout', 'stderr') or type(max_bytes) is not int or max_bytes < 1
                or (offset is not None and (type(offset) is not int or offset < 0))):
            raise ValueError('Invalid log stream or byte window')
        entry = self._get(ref)
        with entry.lock:
            log = entry.logs[stream]
            total, stored = log['observed'], log['stored']
            tail_offset = total - len(log['tail'])
            offset = max(0, total - max_bytes) if offset is None else offset
            gap = None
            if offset < stored:
                with log['path'].open('rb') as saved:
                    saved.seek(offset)
                    raw = saved.read(min(max_bytes, stored - offset))
            elif offset >= tail_offset:
                raw = bytes(log['tail'][max(0, offset - tail_offset):max(0, offset - tail_offset) + max_bytes])
            else:
                raw = b''
                gap = {'start': stored, 'end': tail_offset, 'next_retained_offset': tail_offset}
            return {'stream': stream, 'offset': offset,
                'next_offset': tail_offset if gap is not None else offset + len(raw),
                'content': raw.decode('utf-8', errors='replace'), 'bytes': len(raw),
                'observed_bytes': total, 'gap': gap, 'eof': offset + len(raw) >= total,
                'complete': entry.state is not None, 'reference': self.log_info(ref)[stream]}

    def poll(self, ref: ProcessRef) -> ProcessState:
        entry = self._get(ref)
        with entry.lock:
            if entry.state is not None:
                return entry.state
            if entry.done.is_set():
                raise RuntimeError('Process watcher failed; owned process cleanup is incomplete: ' + str(entry.error))
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
            if ref.sandbox == sandbox:
                if not entry.done.is_set():
                    self.cancel(ref)
                    if not entry.done.wait(10):
                        raise TimeoutError('Local processes did not stop; workspace was retained')
                self._finish(entry)
                if entry.process.returncode is None:
                    raise RuntimeError('Owned process exit is unconfirmed; workspace was retained')
        root, _ = self._sessions[sandbox]
        try:
            shutil.rmtree(root)
        except FileNotFoundError:
            if root.exists():
                raise
        # Keep ownership on any cleanup failure so another close can retry.
        del self._sessions[sandbox]

    def close_all(self):
        for ref in tuple(self._sessions):
            self.close(ref)
