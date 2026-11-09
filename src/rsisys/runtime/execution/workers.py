"""Owned worker-group execution shared by Lab backends."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def run_workers(commands, environments, directory, timeout):
    """Fail fast, terminate whole process groups, and retain useful worker errors."""
    processes, streams = [], []
    started = time.monotonic()
    handlers = {}
    def cancelled(signum, frame):
        raise InterruptedError(f'GPU execution interrupted by signal {signum}')
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, cancelled)
    try:
        for rank, (command, env) in enumerate(zip(commands, environments, strict=True)):
            stream = (directory / f'worker-{rank}.log').open('w')
            streams.append(stream)
            env = dict(env, PATH=str(Path(command[0]).absolute().parent) + os.pathsep + env.get('PATH', ''))
            processes.append(subprocess.Popen(command, env=env, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True))
        while True:
            statuses = [p.poll() for p in processes]
            failed = next((i for i, status in enumerate(statuses) if status not in (None, 0)), None)
            if failed is not None:
                raise RuntimeError(f'GPU worker {failed} failed ({statuses[failed]}):\n' +
                    (directory / f'worker-{failed}.log').read_text()[-12000:])
            if all(status == 0 for status in statuses):
                break
            if time.monotonic() - started > timeout:
                raise TimeoutError(f'GPU workers exceeded {timeout}s; last log:\n' +
                    (directory / 'worker-0.log').read_text()[-4000:])
            time.sleep(0.2)
        for rank in range(len(processes)):
            for line in (directory / f'worker-{rank}.log').read_text().splitlines():
                if line.startswith('RSISYS_WORKER '):
                    print(line, file=sys.stderr, flush=True)
    finally:
        # A launcher can exit before its CUDA descendants; signal its group too.
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            # Reaping the launcher does not prove that its descendants exited.
            # Always escalate for the owned group, including successful workers.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        for stream in streams:
            stream.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
