"""Trainer callback and HTTP job payload checks without models, sockets or GPUs."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.qwen_toy import worker
from examples.qwen_toy.adapters import WorkerJobs
from rsisys.loops.contracts.jobs import JobRef


def worker_fixture():
    # Reuse the existing fake Trainer/HTTP handler without modifying its tests.
    spec = importlib.util.spec_from_file_location('progress_worker_fixture', ROOT / 'tests/test_qwen_worker.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WorkerFixture()


PROGRESS_FIELDS = {'step', 'total_steps', 'elapsed_seconds', 'metrics'}


class PayloadJobs(WorkerJobs):
    def convert(self, request, raw):
        return raw


def polling_adapter(read, ident, *, kind='train'):
    events = []
    adapter = PayloadJobs(SimpleNamespace(call=lambda path: read()), None, kind=kind,
                          notify=lambda event, **details: events.append((event, deepcopy(details))))
    job = JobRef(ident, adapter.backend_id)
    adapter.requests[job] = None
    return adapter, job, events


class TrainingProgressTests(unittest.TestCase):
    def test_callback_observations_reach_job_and_notify_and_survive_success(self):
        fixture = worker_fixture()
        fixture.trainer.state.global_step = 3
        clock = [100.0]

        def scenario():
            code, receipt = fixture.post('/train', {'idempotency_key': 'progress'})
            self.assertEqual(code, 200)
            ident = receipt['job_id']
            self.assertFalse(PROGRESS_FIELDS & fixture.job(ident).keys())
            adapter, job, events = polling_adapter(lambda: fixture.job(ident), ident)
            adapter.poll(job)
            self.assertFalse(PROGRESS_FIELDS & events[-1][1].keys())
            callbacks = []

            def train():
                callback = fixture.transformers.Trainer.call_args.kwargs['callbacks'][1]
                callbacks.append(callback)
                state = SimpleNamespace(global_step=0, max_steps=3)
                control = SimpleNamespace()
                # A step/log before train_begin cannot invent an elapsed origin.
                callback.on_step_end(None, state, control)
                self.assertFalse(PROGRESS_FIELDS & fixture.job(ident).keys())
                callback.on_train_begin(None, state, control)
                first = fixture.job(ident)
                self.assertEqual((first['step'], first['total_steps'], first['elapsed_seconds']), (0, 3, 0.0))
                self.assertNotIn('metrics', first)

                state.global_step, clock[0] = 1, 101.5
                callback.on_step_end(None, state, control)
                logs = {'loss': 0.75, 'learning_rate': 0.0002, 'epoch': 0.3, 'secret': 'omit'}
                callback.on_log(None, state, control, logs=logs)
                logs['loss'] = 99  # Published metrics must not alias Trainer's log dictionary.
                observed = fixture.job(ident)
                self.assertEqual(observed['metrics'], {'loss': 0.75, 'learning_rate': 0.0002})
                observed['metrics']['loss'] = 88  # HTTP snapshots cannot mutate shared jobs.
                adapter.poll(job)
                event, details = events[-1]
                self.assertEqual(event, 'job')
                self.assertEqual(details, {'job_id': ident, 'kind': 'train', 'status': 'running',
                    'step': 1, 'total_steps': 3, 'elapsed_seconds': 1.5,
                    'metrics': {'loss': 0.75, 'learning_rate': 0.0002}})

                state.global_step, clock[0] = 2, 103.0
                callback.on_step_end(None, state, control)
                callback.on_log(None, state, control,
                                logs={'loss': float('nan'), 'learning_rate': 0.0001})
                self.assertEqual(fixture.job(ident)['metrics'], {'loss': 0.75, 'learning_rate': 0.0001})
                state.global_step, clock[0] = 3, 104.0
                callback.on_step_end(None, state, control)
                clock[0] = 104.5
                callback.on_train_end(None, state, control)
                clock[0] = 900.0  # Saving/publication must not fabricate another training observation.
                return SimpleNamespace(metrics={'train_loss': 0.25})

            fixture.trainer.train.side_effect = train
            fixture.drain()
            terminal = fixture.job(ident)
            self.assertEqual(terminal['status'], 'succeeded')
            self.assertEqual((terminal['step'], terminal['total_steps'], terminal['elapsed_seconds']), (3, 3, 4.5))
            self.assertEqual(terminal['metrics']['loss'], 0.75)
            self.assertEqual(terminal['result']['metrics']['train_loss'], 0.25)
            adapter.poll(job)
            self.assertEqual(events[-1][1]['step'], 3)
            self.assertEqual(events[-1][1]['elapsed_seconds'], 4.5)
            # A late callback cannot alter an already terminal job.
            callbacks[0].on_step_end(None, SimpleNamespace(global_step=99, max_steps=99), SimpleNamespace())
            self.assertEqual(fixture.job(ident), terminal)

            _, evaluation = fixture.post('/evaluate', {'idempotency_key': 'evaluate'})
            fixture.drain()
            self.assertEqual(fixture.job(evaluation['job_id'])['status'], 'succeeded')
            self.assertFalse(PROGRESS_FIELDS & fixture.job(evaluation['job_id']).keys())

        with patch.object(worker.time, 'monotonic', side_effect=lambda: clock[0]):
            fixture.run(scenario, '--steps', '3')

    def test_failure_and_cancellation_keep_observed_partial_progress(self):
        for status in ('failed', 'cancelled'):
            with self.subTest(status=status):
                fixture = worker_fixture()
                clock = [10.0]

                def scenario():
                    _, receipt = fixture.post('/train', {'idempotency_key': status})
                    ident = receipt['job_id']

                    def train():
                        cancel, callback = fixture.transformers.Trainer.call_args.kwargs['callbacks']
                        state = SimpleNamespace(global_step=0, max_steps=24)
                        control = SimpleNamespace(should_training_stop=False, should_epoch_stop=False)
                        callback.on_train_begin(None, state, control)
                        state.global_step, clock[0] = 2, 13.0
                        callback.on_log(None, state, control, logs={'loss': 1.5, 'learning_rate': 0.0002})
                        if status == 'failed':
                            raise RuntimeError('fixture failure after observed step 2')
                        fixture.post('/cancel', {'job_id': ident})
                        cancel.on_substep_end(None, state, control)
                        self.assertTrue(control.should_training_stop)
                        self.assertTrue(control.should_epoch_stop)
                        return SimpleNamespace(metrics={})

                    fixture.trainer.train.side_effect = train
                    fixture.drain()
                    terminal = fixture.job(ident)
                    self.assertEqual(terminal['status'], status)
                    self.assertEqual((terminal['step'], terminal['total_steps'], terminal['elapsed_seconds']), (2, 24, 3.0))
                    self.assertEqual(terminal['metrics']['loss'], 1.5)
                    self.assertNotIn('result', terminal)
                    adapter, job, events = polling_adapter(lambda: fixture.job(ident), ident)
                    adapter.poll(job)
                    self.assertEqual(events[-1][1]['status'], status)
                    self.assertEqual(events[-1][1]['step'], 2)

                with patch.object(worker.time, 'monotonic', side_effect=lambda: clock[0]):
                    fixture.run(scenario)

    def test_queued_cancellation_and_no_callback_success_do_not_invent_progress(self):
        for cancel in (True, False):
            with self.subTest(cancel=cancel):
                fixture = worker_fixture()

                def scenario():
                    _, receipt = fixture.post('/train', {'idempotency_key': 'unobserved'})
                    if cancel:
                        fixture.post('/cancel', {'job_id': receipt['job_id']})
                    fixture.drain()
                    terminal = fixture.job(receipt['job_id'])
                    self.assertEqual(terminal['status'], 'cancelled' if cancel else 'succeeded')
                    self.assertFalse(PROGRESS_FIELDS & terminal.keys())

                fixture.run(scenario)

    def test_cancel_check_can_reenter_job_endpoint_without_holding_job_lock(self):
        fixture = worker_fixture()
        real_lock = worker.threading.Lock
        locks = []

        class CheckedLock:
            def __init__(self):
                self.lock = real_lock()
                locks.append(self)

            def acquire(self, *args, **kwargs):
                return self.lock.acquire(*args, **kwargs)

            def release(self):
                self.lock.release()

            def __enter__(self):
                if not self.lock.acquire(timeout=0.5):
                    raise AssertionError('Nested job lock acquisition would deadlock')
                return self

            def __exit__(self, *args):
                self.lock.release()

        def scenario():
            _, receipt = fixture.post('/train', {'idempotency_key': 'lock-check'})
            ident = receipt['job_id']

            def train():
                cancel, callback = fixture.transformers.Trainer.call_args.kwargs['callbacks']
                event = cancel.event

                def cancel_check():
                    self.assertFalse(locks[1].lock.locked())
                    self.assertEqual(fixture.post('/cancel', {'job_id': ident})[0], 200)
                    return event.is_set()

                cancel.event = SimpleNamespace(is_set=cancel_check)
                control = SimpleNamespace(should_training_stop=False, should_epoch_stop=False)
                state = SimpleNamespace(global_step=1, max_steps=24)
                callback.on_train_begin(None, state, control)
                cancel.on_step_end(None, state, control)
                self.assertTrue(control.should_training_stop)
                return SimpleNamespace(metrics={})

            fixture.trainer.train.side_effect = train
            fixture.drain()
            self.assertEqual(fixture.job(ident)['status'], 'cancelled')

        with patch.object(worker.threading, 'Lock', CheckedLock):
            fixture.run(scenario)


class AdapterProgressTests(unittest.TestCase):
    def test_missing_queued_and_evaluation_progress_stays_absent(self):
        for status, kind, fields in (
            ('running', 'train', {}), ('succeeded', 'train', {}),
            ('queued', 'train', {'step': 0, 'total_steps': 24, 'elapsed_seconds': 0}),
            ('running', 'evaluate', {'step': 24, 'total_steps': 24, 'metrics': {'loss': 0.1}}),
        ):
            with self.subTest(status=status, kind=kind):
                raw = {'job_id': 'job', 'status': status, 'result': {}, **fields}
                adapter, job, events = polling_adapter(lambda: raw, 'job', kind=kind)
                adapter.poll(job)
                self.assertEqual(events[-1], ('job', {'job_id': 'job', 'kind': kind, 'status': status}))

    def test_progress_filters_nonfinite_wrong_types_and_unknown_metric_keys(self):
        raw = {'job_id': 'job', 'status': 'running', 'step': True, 'total_steps': -1,
               'elapsed_seconds': float('inf'),
               'metrics': {'loss': float('nan'), 'learning_rate': False, 'secret': 123}}
        adapter, job, events = polling_adapter(lambda: raw, 'job')
        adapter.poll(job)
        self.assertFalse(PROGRESS_FIELDS & events[-1][1].keys())
        raw.update(step=0, total_steps=24, elapsed_seconds=0,
                   metrics={'loss': 0.0, 'learning_rate': 0.0, 'epoch': 1.0})
        adapter.poll(job)
        self.assertEqual(events[-1][1]['step'], 0)
        self.assertEqual(events[-1][1]['elapsed_seconds'], 0)
        self.assertEqual(events[-1][1]['metrics'], {'loss': 0.0, 'learning_rate': 0.0})


if __name__ == '__main__':
    unittest.main()
