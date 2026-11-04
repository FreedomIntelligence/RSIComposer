"""Observe execution contract boundaries with owned CPU processes and model fixtures.

Run from the repository: PYTHONPATH=src python docs/review/repros/execution-lifecycle-contract-probe.py
No external services, model inference, training, package installs or remote jobs.
Only temporary processes created here are signalled, and every case cleans them.
The observations are intentionally usable before and after a subsequent repair.
"""
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import threading
from unittest.mock import patch

from rsisys.adapters.local_lab import LocalProcessLab
from rsisys.application import Application, ApplicationConfig
from rsisys.config import DemoConfig, build_demo
from rsisys.loops.contracts.decision import OperationCall
from rsisys.loops.contracts.model_service import GenerationResponse
from rsisys.runtime.artifacts import InMemoryArtifactStore
from rsisys.runtime.local_process import LocalProcessBackend
from rsisys.runtime import local_process
from rsisys.runtime.run_policy import TimeRunPolicy


ROOT = Path(__file__).resolve().parents[3]


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def create_lab(root, lab_type=LocalProcessLab):
    backend = LocalProcessBackend(root / 'owned-processes', max_output_bytes=4096)
    return lab_type(backend, InMemoryArtifactStore(), timeout_seconds=30), backend


def start(lab, key):
    result = lab.execute(OperationCall('lab_start', {'key': key,
        'argv': [sys.executable, '-c', 'import time; time.sleep(25)']}))
    job = result.data['job_id']
    entry = lab.backend._processes[lab._jobs[job][0]]
    return job, entry


def make_application(root, lab):
    template = ApplicationConfig.load(ROOT / 'configs/local.json')
    config = replace(template, data_dir=root / 'application', execution='mock',
                     roles={'decision_maker': template.roles['decision_maker']})
    return Application(config, lab_factory=lambda context: lab)


def execute_in_application(root, lab, name, arguments):
    app = make_application(root, lab)
    try:
        policy = app.runner.decision_making.harness.policy
        requests = []
        def generate(request):
            requests.append(request)
            return GenerationResponse(json.dumps({'kind': 'invoke', 'operation': name,
                'arguments': arguments}), request.model, simulated=False)
        policy.backend.generate = generate
        raised, kind = None, None
        try:
            kind = app.runner.step('Audit the execution boundary').kind
        except Exception as error:
            raised = type(error).__name__
        history = app.runner.decision_making.harness.memory.history(kind='operation')
        result = history[-1].details['result'] if history else None
        return {'raised_to_host': raised, 'runner_step': kind,
                'recorded_status': result['status'] if result else None,
                'recorded_error': result['error'] if result else None,
                'decision_requests': len(requests)}
    finally:
        app.close()


def close_retry(root, kind):
    class CountingLab(LocalProcessLab):
        close_attempts = 0
        def close(self):
            self.close_attempts += 1
            return super().close()

    lab, backend = create_lab(root, CountingLab)
    owner = None
    try:
        if kind == 'application':
            owner = make_application(root, lab)
        else:
            config = DemoConfig.load(ROOT / 'configs/demo.json')
            config = replace(config, roles={'decision_maker': config.roles['decision_maker']})
            owner, _ = build_demo(config, lab_factory=lambda context: lab)
        job, entry = start(lab, 'owned')
        pid = entry.process.pid
        with patch('rsisys.runtime.local_process._kill_process_group',
                   side_effect=PermissionError('fixture transient signal failure')):
            try:
                owner.close()
                first_error = None
            except Exception as error:
                first_error = type(error).__name__
        # The injected fault has gone away. Retry only through the public owner.
        owner.close()
        observation = {'first_close_error': first_error,
            'lab_close_attempts_after_owner_retry': lab.close_attempts,
            'owned_process_alive_after_owner_retry': alive(pid),
            'active_sessions_after_owner_retry': backend.active_count,
            'assembly_marked_closed': owner.assembly._closed}
        if kind == 'application':
            observation.update(application_marked_closed=owner._closed,
                               workspace_lock_released=owner._lock.closed,
                               shared_store_released=owner.store is None)
        lab.close()
        observation.update(owned_process_alive_after_direct_lab_cleanup=alive(pid),
                           active_sessions_after_direct_lab_cleanup=backend.active_count)
        return observation
    finally:
        lab.close()
        backend.close_all()
        if owner is not None:
            owner.close()


def path_errors(root):
    observations = {}
    for case in ('non_directory_parent', 'directory_destination'):
        case_root = root / case
        lab, backend = create_lab(case_root)
        try:
            if case == 'non_directory_parent':
                lab.execute(OperationCall('lab_write_file', {'path': 'ordinary.txt', 'content': 'text'}))
                name, args = 'lab_read_file', {'path': 'ordinary.txt/child.txt'}
            else:
                (lab.workspace / 'existing-directory').mkdir()
                name, args = 'lab_write_file', {'path': 'existing-directory', 'content': 'text'}
            observations[case] = execute_in_application(case_root, lab, name, args)
        finally:
            lab.close()
            backend.close_all()
    return observations


def supervision_error(root):
    lab, backend = create_lab(root)
    try:
        with patch('rsisys.runtime.local_process._wait_for_process_exit',
                   side_effect=OSError('fixture exit-observer failure')):
            job, entry = start(lab, 'observer-failed')
            assert entry.done.wait(3), 'Probe watcher did not settle'
        observation = execute_in_application(root, lab, 'lab_result', {'job_id': job})
        observation.update(process_exit_confirmed=entry.process.returncode is not None,
                           owned_process_alive=alive(entry.process.pid))
        return observation
    finally:
        lab.close()
        backend.close_all()


def cancellation_after_poll_failure(root):
    lab, backend = create_lab(root)
    app = None
    try:
        gate, failed_pid = threading.Event(), [None]
        wait, kill = local_process._wait_for_process_exit, local_process._kill_process_group
        def fail_first_observer(process, timeout):
            assert gate.wait(3), 'Probe did not release the watchers'
            if process.pid == failed_pid[0]:
                raise RuntimeError('fixture exit-observer failure')
            return wait(process, timeout)
        def fail_first_cleanup(process):
            if process.pid == failed_pid[0]:
                raise PermissionError('fixture first cleanup failure')
            return kill(process)
        with patch('rsisys.runtime.local_process._wait_for_process_exit',
                   side_effect=fail_first_observer), \
             patch('rsisys.runtime.local_process._kill_process_group',
                   side_effect=fail_first_cleanup):
            first_job, first_entry = start(lab, 'first')
            failed_pid[0] = first_entry.process.pid
            later_job, later_entry = start(lab, 'later')
            gate.set()
            assert first_entry.done.wait(3), 'Probe watcher did not settle'
        try:
            lab.shutdown('cancel')
            raised = None
        except Exception as error:
            raised = type(error).__name__
        observation = {'raised_to_host': raised,
            'first_watcher_done': first_entry.done.is_set(),
            'first_terminal_state_exists': first_entry.state is not None,
            'later_cancellation_requested': later_entry.cancelled,
            'later_owned_process_alive': alive(later_entry.process.pid),
            'owned_jobs': len(lab._jobs)}
        # Exercise the same broken shutdown through Runner, past its cleanup limit.
        lab.workspace.joinpath('ready.txt').write_text('ready')
        app = make_application(root, lab)
        runner = app.runner
        now = [0.0]
        runner._clock = lambda: now[0]
        runner.run_policy = TimeRunPolicy(1)
        runner.cleanup_timeout_seconds = 1
        policy = runner.decision_making.harness.policy
        policy.backend.generate = lambda request: GenerationResponse(json.dumps({
            'kind': 'invoke', 'operation': 'lab_read_file', 'arguments': {'path': 'ready.txt'}}),
            request.model, simulated=False)
        assert runner.step('Audit cleanup').kind == 'progressed'
        stop_errors = []
        for instant in (2.0, 10.0):
            now[0] = instant
            try:
                runner.step('Audit cleanup')
                stop_errors.append(None)
            except Exception as error:
                stop_errors.append(type(error).__name__)
        observation.update(stop_tick_errors=stop_errors,
            elapsed_cleanup_seconds=now[0] - runner._stop_started,
            configured_cleanup_seconds=runner.cleanup_timeout_seconds,
            terminal_stop_report_exists=runner._terminal_step is not None)
        try:
            lab.close()
            observation['direct_close_error'] = None
        except Exception as error:
            observation['direct_close_error'] = type(error).__name__
        observation.update(all_owned_processes_exited_after_direct_close=
            not alive(first_entry.process.pid) and not alive(later_entry.process.pid),
            active_sessions_after_direct_close=backend.active_count)
        return observation
    finally:
        lab.close()
        backend.close_all()
        if app is not None:
            app.close()


def main():
    with TemporaryDirectory(prefix='rsisys-execution-contract-') as directory:
        root = Path(directory)
        observations = {
            'close_retry': {kind: close_retry(root / kind, kind) for kind in ('application', 'demo')},
            'correctable_path_errors': path_errors(root / 'paths'),
            'infrastructure_observer_error': supervision_error(root / 'observer'),
            'cancellation_after_poll_failure': cancellation_after_poll_failure(root / 'cancel'),
            'scope': {'real_cpu_processes': True, 'real_model_inference': False,
                      'real_training': False, 'remote_jobs': False},
        }
        print(json.dumps(observations, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
