"""Run and reconcile one Harbor worker in a single measurement directory."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
from threading import Lock
import time
from uuid import uuid4

from rsisys.loops.contracts.coordination import ExecutionStatus, OperationResult
from rsisys.loops.contracts.execution import LabInfrastructureError
from .harbor_runner import atomic_json


def infrastructure_failure(message):
    return LabInfrastructureError(OperationResult('Coding environment infrastructure failure', (), False,
        status=ExecutionStatus.FAILED, error=message))


class HarborTrialRunner:
    """Own worker launch, submission journals and sandbox cleanup checks."""

    def __init__(self, context, config, agent_spec):
        self._execution_context = context
        self._execution_lock = Lock()
        self.config = config
        self.agent_spec = agent_spec

    def _execution(self):
        from rsisys.runtime.execution.executor import execution_capabilities
        # Standalone contexts initialize a shared host budget on the first trial.
        with self._execution_lock:
            return execution_capabilities(self._execution_context)

    @staticmethod
    def resource_roles():
        from rsisys.runtime.execution.resources import ResourceRequest
        return {'evaluator': ResourceRequest(timeout=7200)}

    def run(self, root, task, artifact, endpoint, seed, protocol):
        pending = self._pending(root)
        if pending is not None:
            name, command = pending
        else:
            name = f"000-{re.sub('[^a-zA-Z0-9_-]', '-', task['id'])[:65]}-{uuid4().hex[:8]}"
            command = [self.config['python'], '-m', 'rsisys.labs.coding.harbor_runner',
                       '--task', task['path'], '--artifact', artifact.path,
                       '--agent-spec', str(self.agent_spec), '--endpoint', endpoint,
                       '--output', str(root / 'trials'), '--trial-name', name,
                       '--receipt', str(root / 'task-000.json'), '--model', self.config['model']]
            for option in ('max_steps', 'max_tokens', 'total_tokens', 'agent_timeout', 'verifier_timeout'):
                command.extend(['--' + option.replace('_', '-'), str(protocol[option])])
            command.extend(['--seed', str(seed)])
            if protocol['thinking']:
                command.append('--thinking')
        return self._run_process(command, root, name), name

    @staticmethod
    def _pending(root):
        journal = root / 'task-000-inflight.json'
        if journal.exists():
            record = json.loads(journal.read_text())
            return record['trial_name'], record['command']
        return None

    def reconcile(self, root):
        pending = self._pending(root)
        if pending is not None:
            name, command = pending
            self._run_process(command, root, name)

    def _run_process(self, cmd, root, trial_name):
        """Reattach a live submitted worker before considering a retry.

        A submission is journaled before spawn. PID discovery covers a parent
        crash between spawn and PID persistence. A live Docker project without
        a live worker is uncertain and requires explicit host reconciliation.
        """
        record_path = root / 'task-000-inflight.json'
        child = None
        if record_path.exists():
            record = json.loads(record_path.read_text())
            trial_name = record['trial_name']
            cmd = record['command']
        else:
            record = {'trial_name': trial_name, 'command': cmd, 'submitted_at': time.time()}
            atomic_json(record_path, record)
            with (root / (trial_name + '.log')).open('wb') as log:
                from rsisys.runtime.execution.executor import WorkerSpec
                from rsisys.runtime.execution.workers import worker_environment
                resources, executor = self._execution()
                resources.declare(self.resource_roles())
                with resources.acquire('evaluator') as lease:
                    child = executor.launch(WorkerSpec(cmd, worker_environment(), stdout=log,
                        detached=True, cancel_signal=signal.SIGINT), lease)
            if callable(getattr(child, 'reference', None)):
                record['worker'] = child.reference()
            else:
                record['pid'] = child.worker_pid
            atomic_json(record_path, record)

        if 'worker' not in record and child is None:
            configured = getattr(self._execution_context, 'resources', None)
            owner = getattr(configured, 'owner', configured)
            if getattr(owner, 'config', {}).get('executor', 'local') == 'ray':
                _, executor = self._execution()
                reference = executor.find_job(cmd)
                if reference is None:
                    raise infrastructure_failure('Remote submission ownership is uncertain; reconcile ' + str(record_path))
                record['worker'] = reference
                atomic_json(record_path, record)

        if 'worker' in record:
            _, executor = self._execution()
            child = child or executor.reattach(record['worker'])
            try:
                code = child.wait(timeout=7200)
            except (TimeoutError, subprocess.TimeoutExpired):
                child.terminate()
                raise infrastructure_failure('Owned remote task exceeded host wall time; reconcile ' + str(record_path))
            result = child.run_cleanup(['docker', 'ps', '-aq', '--filter',
                'name=^/' + re.escape(trial_name.lower()) + '__'])
            if result['code'] != 0 or result['stdout'].strip():
                raise infrastructure_failure('Uncertain live/orphaned remote task container; inspect ' + str(record_path))
            record_path.replace(root / (trial_name + '-submission.json'))
            return code

        def live_pid():
            # A saved PID is a hint, not proof of ownership: match the exact
            # unique trial name and receipt root in the actual command line.
            candidates = ([Path('/proc') / str(record['pid'])] if 'pid' in record else [])
            if not candidates:
                candidates = [p for p in Path('/proc').iterdir() if p.name.isdigit()]
            for p in candidates:
                try:
                    args = (p / 'cmdline').read_bytes().decode(errors='replace').split('\0')
                    if trial_name in args and str(root / 'task-000.json') in args:
                        return int(p.name)
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    pass
            return None

        started = time.monotonic()
        while (child.poll() is None if child is not None else live_pid() is not None):
            if time.monotonic() - started > 7200:
                pid = child.pid if child is not None else live_pid()
                if pid is not None:
                    # SIGINT lets asyncio/Harbor run cancellation cleanup. Do
                    # not start another task until cleanup can be reconciled.
                    try:
                        os.kill(pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass
                raise infrastructure_failure('Owned task exceeded host wall time; reconcile ' + str(record_path))
            time.sleep(2)
        result = subprocess.run(['docker', 'ps', '-aq', '--filter',
                                 'name=^/' + re.escape(trial_name.lower()) + '__'],
                                capture_output=True, text=True, timeout=30)
        if result.returncode != 0 or result.stdout.strip():
            raise infrastructure_failure('Uncertain live/orphaned task container; inspect ' + str(record_path))
        record_path.replace(root / (trial_name + '-submission.json'))
        return child.returncode if child is not None else 0
