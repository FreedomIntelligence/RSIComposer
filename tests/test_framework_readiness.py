"""Local integration readiness: owned cleanup barriers and launcher lifecycle."""
from dataclasses import asdict, replace
import fcntl
import io
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from examples.medical import run as medical
from examples.medical.evaluation import backend_inference
from rsisys.application import Application, ApplicationConfig
from rsisys.runtime.artifacts import FileArtifactStore
from rsisys.runtime.sandbox import ExecRequest, SandboxSpec
from rsisys.runtime.systems import ReferenceSystemActivator
from tests.test_medical_example import DECISION, MODEL, fixture_http, row, write_rows


ROOT = Path(__file__).resolve().parents[1]


class FrameworkReadinessTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def application(self):
        config = replace(ApplicationConfig.load(ROOT / 'configs/local.json'),
                         data_dir=self.root / 'application')
        app = Application(config)
        # Also clean up with the backend directly if the regression is present.
        self.addCleanup(app.sandbox.close_all)
        self.addCleanup(app.close)
        return app

    def assert_owned_cleanup_retained(self, app):
        self.assertIsNotNone(app.store)
        self.assertFalse(app._lock.closed)
        self.assertTrue(app.assembly.cleanup_pending)
        self.assertEqual(tuple(app.experience.retrieve()), ())
        with (app.config.data_dir / '.lock').open('a+b') as lock:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            app.run()

    def test_shared_executor_close_failure_retains_lock_store_and_retry(self):
        app = self.application()
        session = app.sandbox.create(SandboxSpec('local', 30))
        process = app.sandbox.exec(session, ExecRequest(
            (sys.executable, '-c', 'import time; time.sleep(30)'), 30))
        pid = app.sandbox._processes[process].process.pid
        with patch('rsisys.runtime.local_process._kill_process_group',
                   side_effect=PermissionError('fixture owned kill denied')):
            with self.assertRaisesRegex(PermissionError, 'owned kill denied'):
                app.close()
        os.kill(pid, 0)
        self.assert_owned_cleanup_retained(app)
        app.close()
        self.assertEqual(app.sandbox.active_count, 0)
        self.assertIsNone(app.store)
        self.assertTrue(app._lock.closed)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        app.close()

    def test_legacy_environment_close_is_a_retryable_owned_barrier(self):
        app = self.application()
        environment = app.runner.exploration.harness.environment
        session = app.sandbox.create(SandboxSpec('local', 30))
        environment._session = session
        with patch('rsisys.runtime.local_process.shutil.rmtree',
                   side_effect=PermissionError('fixture directory busy')):
            with self.assertRaises(PermissionError):
                app.close()
        self.assert_owned_cleanup_retained(app)
        self.assertEqual(environment._session, session)
        app.close()
        self.assertIsNone(environment._session)
        self.assertEqual(app.sandbox.active_count, 0)
        self.assertIsNone(app.store)

    def medical_cli(self, backend, *, steps=30):
        write_rows(self.root / 'train.jsonl', [row('train', 'Training symbol')])
        write_rows(self.root / 'dev.jsonl', [row('dev', 'Development symbol')])
        values = {
            'task': {'train': 'train.jsonl', 'development': [
                {'benchmark_id': 'medqa', 'role': 'primary', 'path': 'dev.jsonl'}]},
            'backends': {'initial_model': asdict(MODEL), 'decision_model': asdict(DECISION),
                         'decision_binding': asdict(backend.binding),
                         'actor_binding': asdict(backend.binding)},
            'settings': {'objective': 'Verify CLI lifecycle with synthetic symbols',
                         'max_runner_steps': steps, 'poll_interval_seconds': 0},
        }
        for name, value in values.items():
            (self.root / (name + '.json')).write_text(json.dumps(value))
        return ['run', '--config', str(self.root / 'task.json'),
                '--backends', str(self.root / 'backends.json'),
                '--settings', str(self.root / 'settings.json'),
                '--output-directory', str(self.root / 'artifacts')]

    def test_medical_cli_closes_after_success_and_step_exhaustion(self):
        for steps, expected in ((30, 0), (1, 2)):
            with self.subTest(steps=steps), fixture_http() as (backend, calls):
                captured = []
                builder = medical.build
                def build(*args, **kwargs):
                    assembly = builder(*args, **kwargs)
                    captured.append(assembly)
                    self.addCleanup(lambda: assembly.close())
                    return assembly
                with patch.object(medical, 'build', side_effect=build), \
                        patch('sys.stdout', new_callable=io.StringIO):
                    self.assertEqual(medical.main(self.medical_cli(backend, steps=steps)), expected)
                assembly, = captured
                self.assertTrue(calls)
                self.assertTrue(assembly.runner.assembly._closed)
                self.assertFalse(assembly.runner.assembly.cleanup_pending)
                self.assertTrue(any(r.kind == 'medical_run_result' for r in assembly.artifacts.refs()))

    def test_medical_cli_preserves_run_error_and_reports_cleanup_error(self):
        with fixture_http() as (backend, calls):
            captured = []
            builder = medical.build
            def build(*args, **kwargs):
                assembly = builder(*args, **kwargs)
                captured.append(assembly)
                self.addCleanup(lambda: assembly.close())
                return assembly
            with patch.object(medical, 'build', side_effect=build), \
                    patch.object(medical.MedicalAssembly, 'run', side_effect=ValueError('fixture run error')), \
                    patch.object(medical.MedicalAssembly, 'close', side_effect=OSError('fixture close error')):
                with self.assertRaisesRegex(ValueError, 'fixture run error') as caught:
                    medical.main(self.medical_cli(backend))
            self.assertFalse(calls)
            self.assertTrue(any('fixture close error' in note
                                for note in getattr(caught.exception, '__notes__', ())))
            captured[0].close()

    def test_medical_cli_closes_if_result_persistence_fails(self):
        with fixture_http() as (backend, calls):
            captured = []
            builder, put = medical.build, FileArtifactStore.put
            def build(*args, **kwargs):
                assembly = builder(*args, **kwargs)
                captured.append(assembly)
                self.addCleanup(lambda: assembly.close())
                return assembly
            def persist(store, content, *, kind, **options):
                if kind == 'medical_run_result':
                    raise OSError('fixture result persistence error')
                return put(store, content, kind=kind, **options)
            with patch.object(medical, 'build', side_effect=build), \
                    patch.object(FileArtifactStore, 'put', new=persist):
                with self.assertRaisesRegex(OSError, 'result persistence error'):
                    medical.main(self.medical_cli(backend))
            self.assertTrue(calls)
            self.assertTrue(captured[0].runner.assembly._closed)
            self.assertFalse(captured[0].runner.assembly.cleanup_pending)

    def test_medical_closed_admission_precedes_anchor_inference(self):
        for close_runner in (False, True):
            with self.subTest(close_runner=close_runner), fixture_http() as (backend, calls):
                self.medical_cli(backend)
                task = medical.prepare(json.loads((self.root / 'task.json').read_text()),
                                       base_directory=self.root)
                assembly = medical.build(task, artifacts=FileArtifactStore(self.root / 'artifacts'),
                    decision_backend=backend, decision_model=DECISION, initial_model=MODEL,
                    inference=backend_inference(backend),
                    system_activator=ReferenceSystemActivator(MODEL))
                self.addCleanup(assembly.close)
                (assembly.runner.close if close_runner else assembly.close)()
                self.assertTrue(assembly.runner.closed)
                with self.assertRaisesRegex(RuntimeError, 'closed'):
                    assembly.run(medical.RunSettings('must not infer', 1, 0))
                with self.assertRaisesRegex(RuntimeError, 'closed'):
                    with assembly:
                        self.fail('closed assembly admitted')
                self.assertFalse(calls)
