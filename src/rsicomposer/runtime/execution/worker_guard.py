"""Process supervisor entry: keep lease FDs alive through host exit and reap children."""
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time


def main():
    path = Path(sys.argv[1])
    spec = json.loads(path.read_text())
    # Host runtime path is independent of the candidate's sanitized environment.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from rsicomposer.runtime.execution.budget import HostBudget
    from rsicomposer.runtime.storage.files import atomic_write, json_bytes
    from rsicomposer.runtime.execution.process_identity import process_identity, process_group_alive
    budget = HostBudget(spec['budget_root'], json.loads(
        (Path(spec['budget_root']) / 'budget.json').read_text())['limits'], restoring=True)
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    process, timed_out, code = None, False, 1
    def limits():
        for name, value in spec['limits'].items():
            if name not in {'RLIMIT_AS', 'RLIMIT_CPU', 'RLIMIT_FSIZE', 'RLIMIT_CORE'}:
                raise ValueError('Unsupported worker resource limit')
            resource.setrlimit(getattr(resource, name), tuple(value))
    try:
        started = time.monotonic()
        process = subprocess.Popen(spec['command'], cwd=spec['cwd'], env=os.environ.copy(),
            stdin=subprocess.DEVNULL, start_new_session=True,
            preexec_fn=limits if spec['limits'] else None)
        child = path.parent / 'child.json'
        temporary = child.with_suffix('.tmp')
        temporary.write_text(json.dumps({'pid': process.pid, 'identity': process_identity(process.pid)}))
        os.replace(temporary, child)
        while process.poll() is None:
            timed_out = spec['timeout'] is not None and time.monotonic() - started >= spec['timeout']
            # Shared GPU budget accounts every active lease, including parallel groups.
            remaining = budget.enforcement_remaining()
            timed_out = timed_out or any(v <= 0 for k, v in remaining.items() if k != 'model_calls')
            if stopping or timed_out:
                break
            time.sleep(.05)
        code = process.returncode if process.returncode is not None else 124 if timed_out else 143
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, spec['cancel_signal'])
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            while process_group_alive(process.pid):
                time.sleep(.01)
        if spec['detached']:
            budget.end_lease(spec['lease_id'])
        marker = Path(spec['budget_root']) / 'leases' / (spec['lease_id'] + '.lock')
        marker.touch()
        # Locks close only after the owned process group has exited.
        for fd in spec['fds']:
            os.close(fd)
        # The parent polls this path while the guard is live: the terminal
        # marker must become visible only as a complete JSON document.
        atomic_write(path.parent / 'result.json', json_bytes({'returncode': code, 'timed_out': timed_out}))
    return code if 0 <= code <= 255 else 1


if __name__ == '__main__':
    raise SystemExit(main())
