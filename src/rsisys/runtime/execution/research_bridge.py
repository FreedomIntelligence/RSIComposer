"""Owned JSON-lines bridge for optional, dependency-isolated research agents.

The worker emits events with a fixed prefix; ordinary upstream output remains
in the run log. This transport does not choose experiments or interpret metrics.
"""
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading

PREFIX = 'RSISYS_RESEARCH '


def emit_event(value):
    import sys
    sys.__stdout__.write(PREFIX + json.dumps(value, allow_nan=False) + '\n')
    sys.__stdout__.flush()


class ResearchBridge:
    def __init__(self, command, *, log, env=None, cwd=None):
        self.log_path = Path(log)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.events = queue.Queue()
        self._closed = False
        self._close_lock = threading.Lock()
        self.process = subprocess.Popen(command, cwd=cwd, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        try:
            with self.log_path.open('w') as log:
                for line in self.process.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith(PREFIX):
                        try:
                            self.events.put(json.loads(line[len(PREFIX):]))
                        except ValueError as exc:
                            self.events.put(exc)
        finally:
            self.events.put(EOFError('Research worker exited; inspect ' + str(self.log_path)))

    def send(self, value):
        try:
            self.process.stdin.write(json.dumps(value, allow_nan=False) + '\n')
            self.process.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise RuntimeError('Research worker is unavailable: ' + str(self.log_path)) from exc

    def receive(self, timeout):
        try:
            event = self.events.get(timeout=timeout)
        except queue.Empty as exc:
            self.close()
            raise TimeoutError('Research worker timed out: ' + str(self.log_path)) from exc
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
            self._closed = True
        # Descendants may outlive a launcher. Escalate for the owned group even
        # if waiting for the parent succeeds; do not infer group exit from it.
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.wait(timeout=5)
        self.process.stdin.close()
        self._reader.join(timeout=1)
        if not self._reader.is_alive():
            self.process.stdout.close()
