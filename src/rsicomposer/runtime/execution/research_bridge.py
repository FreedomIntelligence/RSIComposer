"""Owned worker events on a dedicated pipe, separate from stdout and stderr."""
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

from rsicomposer.runtime.execution.process_identity import process_group_alive

_EVENT_FD_ENV = 'RSICOMPOSER_UPSTREAM_EVENT_FD'
_EMIT_LOCK = threading.Lock()


def _group_alive(pgid):
    if sys.platform == 'darwin':
        # Darwin reports EPERM (including killpg(..., 0)) for an exited group.
        # A reaped launcher alone cannot prove that its descendants also exited.
        rows = subprocess.check_output(['ps', '-axo', 'pgid=,stat='], text=True)
        return any(int(fields[0]) == pgid and not fields[1].startswith('Z')
                   for row in rows.splitlines() if len(fields := row.split()) >= 2)
    return process_group_alive(pgid)


def emit_event(value):
    """Send one complete JSON event through the host-issued event descriptor."""
    try:
        descriptor = int(os.environ[_EVENT_FD_ENV])
    except (KeyError, ValueError) as error:
        raise RuntimeError('Research worker requires a host-issued event descriptor') from error
    data = memoryview((json.dumps(value, allow_nan=False) + '\n').encode('utf-8'))
    with _EMIT_LOCK:
        while data:
            data = data[os.write(descriptor, data):]


class ResearchBridge:
    def __init__(self, command, *, log, env=None, cwd=None):
        self.log_path = Path(log)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.stderr_path = self.log_path.with_name(self.log_path.stem + '.stderr' + self.log_path.suffix)
        self.events = queue.Queue()
        self._closed = False
        self._close_lock = threading.Lock()
        reader, writer = os.pipe()
        self._event_stream = os.fdopen(reader, encoding='utf-8')
        environment = dict(os.environ if env is None else env)
        environment[_EVENT_FD_ENV] = str(writer)
        try:
            with self.log_path.open('wb') as stdout, self.stderr_path.open('wb') as stderr:
                self.process = subprocess.Popen(command, cwd=cwd, env=environment,
                    stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, pass_fds=(writer,),
                    text=True, bufsize=1, start_new_session=True)
        except BaseException:
            self._event_stream.close()
            raise
        finally:
            os.close(writer)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        try:
            with self._event_stream:
                for line in self._event_stream:
                    try:
                        self.events.put(json.loads(line))
                    except ValueError as exc:
                        self.events.put(exc)
        except (OSError, UnicodeError) as error:
            self.events.put(error)
        finally:
            self.events.put(EOFError('Research worker exited; inspect ' + str(self.log_path)))

    def send(self, value):
        try:
            self.process.stdin.write(json.dumps(value, allow_nan=False) + '\n')
            self.process.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise RuntimeError('Research worker is unavailable: ' + str(self.log_path)) from exc

    def receive(self, timeout):
        event = self.poll(timeout)
        if event is None:
            self.close()
            raise TimeoutError('Research worker timed out: ' + str(self.log_path))
        return event

    def poll(self, timeout=0):
        """Return one ready event; an empty poll never closes the worker."""
        if type(timeout) not in (int, float) or timeout < 0:
            raise ValueError('Research bridge poll timeout must be nonnegative')
        try:
            event = self.events.get(timeout=timeout)
        except queue.Empty:
            return None
        if isinstance(event, BaseException):
            raise event
        if not isinstance(event, dict) or not isinstance(event.get('kind'), str):
            raise ValueError('Invalid research worker event')
        if event['kind'] == 'error':
            raise RuntimeError(event.get('error', 'Research worker failed'))
        return event

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            # Keep cleanup retryable until the whole owned group and reader are
            # gone. The launcher may exit before its descendants do.
            for sig in (signal.SIGTERM, signal.SIGKILL):
                if not _group_alive(self.process.pid):
                    break
                try:
                    os.killpg(self.process.pid, sig)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    if _group_alive(self.process.pid):
                        raise
                deadline = time.monotonic() + 5
                while _group_alive(self.process.pid) and time.monotonic() < deadline:
                    self.process.poll()
                    time.sleep(.01)
            if _group_alive(self.process.pid):
                raise RuntimeError('Research worker process group did not exit')
            self.process.wait(timeout=5)
            self.process.stdin.close()
            self._reader.join(timeout=5)
            if self._reader.is_alive():
                raise RuntimeError('Research worker event channel did not drain')
            self._closed = True
