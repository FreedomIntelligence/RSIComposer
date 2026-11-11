"""Execute generated HumanEval code only in the host's OS isolation."""

import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from rsisys.runtime.execution.isolation import ExecutionPolicy, ProcessIsolation, system_runtime_paths
from rsisys.runtime.storage.workspace_files import read_workspace_file


class SandboxedExecutor:
    def __init__(self, source):
        self.source = source
        self.worker = Path(__file__).with_name('case_worker.py')
        # Expose executor code only. The benchmark directory (including other
        # tasks' hidden tests) must not be mounted into a candidate process.
        self.runner = ProcessIsolation(ExecutionPolicy(read_only=(
            *system_runtime_paths(sys.executable), str(source / 'executors'), str(self.worker))))
        self.runner.check()
    def _call(self, payload, timeout):
        if len(json.dumps(payload).encode()) > 256000:
            raise ValueError('Reflexion execution request exceeds source bound')
        with tempfile.TemporaryDirectory(prefix='reflexion-case-') as temporary:
            path = Path(temporary)
            (path / 'input.json').write_text(json.dumps(payload))
            argv, env = self.runner.command(path, (sys.executable, '-I', '-B',
                str(self.worker), str(path), str(self.source)), {})
            def limits():
                resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
                resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout + 1))
                resource.setrlimit(resource.RLIMIT_FSIZE, (4*1024**2, 4*1024**2))
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            with tempfile.TemporaryFile() as out:
                process = subprocess.Popen(argv, env=env, stdout=out, stderr=out,
                    preexec_fn=limits, start_new_session=True)
                try:
                    code = process.wait(timeout=timeout+3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    return None
                finally:
                    # Also remove surviving descendants after a normal exit.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if code != 0:
                    return None
                try:
                    return json.loads(read_workspace_file(path, 'result.json', maximum=256000))
                except (OSError, ValueError):
                    return None
    def execute(self, func, tests, timeout=5):
        if not isinstance(tests, list) or len(tests) > 64 or not tests:
            return SimpleNamespace(is_passing=False, feedback='No bounded internal tests were generated.')
        # The author executor may execute each failing test twice (feedback).
        # Its per-test timeout must not become a single timeout for six tests.
        value = self._call({'mode': 'execute', 'func': func, 'tests': tests, 'timeout': timeout},
                           timeout * max(1, len(tests)) * 2)
        if value is None:
            value = [False, 'Execution did not complete inside the isolated evaluator.', [False]*len(tests)]
        if not isinstance(value, list) or len(value) != 3 or type(value[0]) is not bool or not isinstance(value[1], str):
            return SimpleNamespace(is_passing=False, feedback='Malformed isolated test receipt.')
        return SimpleNamespace(is_passing=value[0], feedback=value[1])
    def evaluate(self, name, func, test, timeout=5):
        value = self._call({'mode': 'evaluate', 'func': func, 'name': name, 'test': test, 'timeout': timeout}, timeout)
        passed = value is True
        # Hidden test content and error details are never returned to the model.
        return passed

