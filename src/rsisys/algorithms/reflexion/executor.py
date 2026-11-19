"""Execute generated HumanEval code only in the host's OS isolation."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

from rsisys.api import ExecutionPolicy, ProcessIsolation, read_workspace_file, system_runtime_paths, run_cpu


class SandboxedExecutor:
    def __init__(self, source, context):
        self.source, self.context = source, context
        self.worker = Path(__file__).with_name('case_worker.py')
        # Expose executor code only. The benchmark directory (including other
        # tasks' hidden tests) must not be mounted into a candidate process.
        self.runner = ProcessIsolation(ExecutionPolicy(read_only=(
            *system_runtime_paths(sys.executable), str(source / 'executors'), str(self.worker))))
        self.runner.check()
    def _call(self, payload, timeout):
        if len(json.dumps(payload).encode()) > 256000:
            raise ValueError('Reflexion execution request exceeds source bound')
        temporary_root = Path(self.context.data_dir) / 'reflexion-workers'
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='case-', dir=temporary_root) as temporary:
            path = Path(temporary)
            (path / 'input.json').write_text(json.dumps(payload))
            argv, env = self.runner.command(path, (sys.executable, '-I', '-B',
                str(self.worker), str(path), str(self.source)), {})
            limits = dict(RLIMIT_AS=(1024**3, 1024**3), RLIMIT_CPU=(timeout, timeout+1),
                          RLIMIT_FSIZE=(4*1024**2, 4*1024**2), RLIMIT_CORE=(0, 0))
            with (path / 'worker.log').open('wb') as out:
                try:
                    code = run_cpu(self.context, 'evaluator', argv, environment=env,
                                   stdout=out, timeout=timeout+3, limits=limits)
                except TimeoutError:
                    return None
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
