"""CPU-only review evidence. Run with PYTHONPATH=src; no model/service calls."""
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch

from rsisys.adapters.local_lab import LocalProcessLab
from rsisys.application import Application, ApplicationConfig
from rsisys.loops.contracts.decision import Decision, OperationCall
from rsisys.research import ResearchAssembly
from rsisys.runtime.local_process import LocalProcessBackend
from rsisys.runtime.sandbox import ExecRequest, SandboxSpec


ROOT = Path(__file__).resolve().parents[3]


def runtime_probe(root):
    backend = LocalProcessBackend(root)
    try:
        session = backend.create(SandboxSpec('local', 5))
        ref = backend.exec(session, ExecRequest((sys.executable, '-c',
            'from pathlib import Path; import time; print("READY", flush=True); '
            '\nwhile not Path("release").exists(): time.sleep(.01)'), 5))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            running = backend.poll(ref)
            if running.stdout:
                break
            time.sleep(.01)
        (backend.workspace(session) / 'release').touch()
        streaming = {'running_status': running.status.value, 'running_stdout': running.stdout,
                     'final_stdout': backend.wait(ref).stdout}
        backend.close(session)

        session = backend.create(SandboxSpec('local', 5))
        workspace = backend.workspace(session)
        with patch('rsisys.runtime.local_process.shutil.rmtree', side_effect=PermissionError('fixture deletion')):
            try:
                backend.close(session)
            except PermissionError:
                pass
        retry = {'workspace_still_exists': workspace.exists(),
                 'active_sessions_after_failed_removal': backend.active_count}
        backend.close(session)
        retry['workspace_exists_after_retry'] = workspace.exists()

        session = backend.create(SandboxSpec('local', 5))
        with patch('rsisys.runtime.local_process._wait_for_process_exit', side_effect=RuntimeError('fixture observer')):
            ref = backend.exec(session, ExecRequest((sys.executable, '-c', 'import time; time.sleep(30)'), 5))
            entry = backend._processes[ref]
            if not entry.done.wait(3):
                raise TimeoutError('Probe observer did not finish')
        observer = {'watcher_done': entry.done.is_set(), 'terminal_state_exists': entry.state is not None}
        backend.close(session)
        try:
            os.kill(entry.process.pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        observer.update(process_alive_after_close=alive, active_sessions_after_close=backend.active_count,
                        outcome=backend.poll(ref).status.value, stderr=backend.poll(ref).stderr)
        return {'streaming': streaming, 'cleanup_retry': retry, 'observer_failure': observer}
    finally:
        backend.close_all()


class RecordedProduct:
    def __init__(self):
        self.events = []

    def process(self, event, model):
        self.events.append(event)
        return {'probe': 'no inference'}


def delivery_probe(root):
    config = replace(ApplicationConfig.load(ROOT / 'configs/local.json'), data_dir=root,
                     execution='local_process')
    labs, receipts = [], []
    product = RecordedProduct()
    def factory(context):
        labs.append(LocalProcessLab(context.sandbox, context.artifacts, timeout_seconds=5))
        return labs[-1]
    with Application(config, lab_factory=factory, research=ResearchAssembly(memory_processor=product)) as app:
        lab, runner = labs[0], app.runner
        harness = runner.decision_making.harness
        harness.add_receipt_listener(receipts.append)
        def invoke(name, **args):
            harness.policy.decide = lambda context: Decision('invoke', operation=OperationCall(name, args))
            return runner.step('CPU review fixture')
        error = None
        try:
            invoke('lab_read_file', path='missing.py')
        except FileNotFoundError as exc:
            error = type(exc).__name__
        failed_receipt = receipts[-1]
        recovered = invoke('lab_write_file', path='fixed.py', content='print("one program")\n')
        started = invoke('lab_start', key='once', argv=[sys.executable, 'fixed.py'])
        ident = started.event['operation_result']['data']['job_id']
        lab.backend.wait(lab._jobs[ident][0])
        invoke('lab_result', job_id=ident)
        invoke('lab_result', job_id=ident)
        service = runner.decision_making.feedback_service
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            service.refresh()
            rows = list(service.learning.state['jobs'].values())
            if all(row['status'] in ('completed', 'archived') for row in rows):
                break
            time.sleep(.01)
        collected = [row for row in rows if row['raw_input']['receipt']['operation']['name'] == 'lab_result']
        events = [event for event in product.events if event.request.get('name') == 'lab_result']
        return {
            'tool_error': {'raised_to_host': error, 'recorded_status': failed_receipt.result['status'],
                           'recorded_error': failed_receipt.result['error'].split(':')[0],
                           'next_host_step_if_exception_caught': recovered.kind},
            'repeated_collection': {
                'native_programs': len(lab._jobs), 'collections': 2,
                'distinct_collection_decisions': len({r['raw_input']['receipt']['task']['decision_id'] for r in collected}),
                'feedback_jobs': len(collected), 'feedback_statuses': [r['status'] for r in collected],
                'memory_processor_calls': len(events),
                'distinct_feedback_event_ids': len({event.event_id for event in events}),
                'same_native_job': len({r['raw_input']['receipt']['result']['data']['job_id'] for r in collected}) == 1,
            },
        }


if __name__ == '__main__':
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        print(json.dumps({'runtime': runtime_probe(root / 'processes'),
                          'delivery': delivery_probe(root / 'application'),
                          'real_cpu_processes': True, 'real_model_inference': False,
                          'real_training': False}, ensure_ascii=False, indent=2))
