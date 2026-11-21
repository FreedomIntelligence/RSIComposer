"""Node-local supervisor for a Ray-owned process, independent of the driver."""
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time


def main():
    request = Path(sys.argv[1])
    spec = json.loads(request.read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from rsicomposer.runtime.execution.budget import HostBudget
    from rsicomposer.runtime.execution.process_identity import process_identity, process_group_alive
    budget = HostBudget(spec['budget_root'], json.loads(
        (Path(spec['budget_root']) / 'budget.json').read_text())['limits'], restoring=True)
    marker = Path(spec['marker_path']).open('a+b')
    fcntl.flock(marker, fcntl.LOCK_SH)
    stop, timed_out, process, code = False, False, None, 1

    def stopping(*_):
        nonlocal stop
        stop = True

    def limits():
        for key, value in spec['limits'].items():
            if key not in {'RLIMIT_AS', 'RLIMIT_CPU', 'RLIMIT_FSIZE', 'RLIMIT_CORE'}:
                raise ValueError('Unsupported worker resource limit')
            resource.setrlimit(getattr(resource, key), tuple(value))

    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    try:
        started = time.monotonic()
        process = subprocess.Popen(spec['command'], cwd=spec['cwd'], env=os.environ.copy(),
            stdin=subprocess.DEVNULL, start_new_session=True,
            preexec_fn=limits if spec['limits'] else None)
        child = request.parent / 'child.json'
        temporary = child.with_suffix('.tmp')
        temporary.write_text(json.dumps({'pid': process.pid, 'identity': process_identity(process.pid)}))
        os.replace(temporary, child)
        while process.poll() is None:
            remaining = budget.enforcement_remaining()
            timed_out = ((spec['timeout'] is not None and time.monotonic() - started >= spec['timeout'])
                or any(v <= 0 for k, v in remaining.items() if k != 'model_calls'))
            if spec.get('deadline_file'):
                deadline = json.loads(Path(spec['deadline_file']).read_text())['expires_at']
                timed_out = timed_out or (deadline is not None and time.time() >= deadline)
            orphan = process_identity(spec['owner_pid']) != spec['owner_identity']
            if stop or timed_out or orphan:
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
        Path(spec['marker_path']).touch()
        marker.close()
        for fd in spec['fds']:
            os.close(fd)
        (request.parent / 'result.json').write_text(json.dumps({'returncode': code, 'timed_out': timed_out}))
    return code if 0 <= code <= 255 else 1


if __name__ == '__main__':
    raise SystemExit(main())
