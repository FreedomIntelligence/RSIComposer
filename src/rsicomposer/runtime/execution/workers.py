"""Owned worker-group execution shared by Lab backends."""
import os
from pathlib import Path
import signal
import sys
import threading
import time

import rsicomposer


def worker_environment(**overrides):
    """Use the imported framework package in child interpreters after moves.

    Anchor at the package rather than a caller's depth in the source tree.
    The parent PYTHONPATH is preserved unless explicitly overridden.
    """
    env = dict(os.environ, **overrides)
    root = str(Path(rsicomposer.__file__).resolve().parent.parent)
    env['PYTHONPATH'] = root + os.pathsep + env.get('PYTHONPATH', '')
    return env


def run_workers(commands, environments, directory, timeout, *, executor, lease):
    """Fail fast, terminate whole process groups, and retain useful worker errors."""
    from .executor import WorkerSpec
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
            devices = tuple(int(i) for i in env.get('CUDA_VISIBLE_DEVICES', '').split(',') if i)
            processes.append(executor.launch(WorkerSpec(command, env, stdout=stream,
                timeout=timeout, devices=devices), lease))
        while True:
            statuses = [p.poll() for p in processes]
            failed = next((i for i, status in enumerate(statuses) if status not in (None, 0)), None)
            if failed is not None:
                raise RuntimeError(f'GPU worker {failed} failed ({statuses[failed]}):\n' +
                    (directory / f'worker-{failed}.log').read_text()[-12000:])
            if all(status == 0 for status in statuses):
                break
            if timeout is not None and time.monotonic() - started > timeout:
                raise TimeoutError(f'GPU workers exceeded {timeout}s; last log:\n' +
                    (directory / 'worker-0.log').read_text()[-4000:])
            time.sleep(0.2)
        for rank in range(len(processes)):
            for line in (directory / f'worker-{rank}.log').read_text().splitlines():
                if line.startswith('RSISYS_WORKER '):
                    print(line, file=sys.stderr, flush=True)
    finally:
        for process in processes:
            process.terminate()
        for stream in streams:
            stream.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
