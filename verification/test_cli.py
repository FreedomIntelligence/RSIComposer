"""Real source-entrypoint checks with no model, network, or optional dependencies."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


SRC = Path(__file__).resolve().parents[1] / 'src'


class ExplainTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = self.root / 'run.json'
        self.plugin = self.root / 'fixture.py'
        self.plugin.write_text('''
def build(context, config):
    raise AssertionError("explain constructed a Lab")

def register(registry):
    registry.register_lab('explain_fixture', build, version='13')
''', encoding='utf-8')
        self.raw = {
            'objective': 'PRIVATE-OBJECTIVE', 'data_dir': 'workspace', 'rounds': 3,
            'framework': {'controller': {'name': 'single_loop', 'config': {'operation': 'math.evaluate'}},
                          'labs': {'math': {'name': 'explain_fixture', 'config': {'internal': 'PRIVATE-LAB'}}}},
        }

    def cli(self, *options, exit_code=0):
        self.config.write_text(json.dumps(self.raw), encoding='utf-8')
        result = subprocess.run([sys.executable, '-m', 'rsicomposer', 'explain', '--config', str(self.config),
                                 '--plugin', str(self.plugin), *options], cwd=self.root,
                                env={**os.environ, 'PYTHONPATH': str(SRC)}, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, exit_code, result.stderr)
        self.assertFalse((self.root / 'workspace').exists())
        self.assertFalse((self.root / 'overridden').exists())
        if exit_code:
            return result.stderr
        self.assertNotIn('PRIVATE-', result.stdout)
        return json.loads(result.stdout)

    def test_config_only_and_cli_default(self):
        info = self.cli()
        self.assertEqual(info['validation'], 'configuration_only')
        self.assertEqual(info['framework']['controller']['name'], 'single_loop')
        self.assertEqual(info['framework']['labs'], {'math': {'name': 'explain_fixture', 'version': '13'}})
        self.assertEqual(info['data_dir'], str(self.root.resolve() / 'workspace'))
        self.assertEqual((info['rounds'], info['max_decision_steps'], info['max_decision_steps_source']),
                         (3, 64, 'cli_default'))
        self.assertEqual(info['plugins'][0]['name'], 'fixture.py')
        self.assertEqual(len(info['plugins'][0]['sha256']), 64)

    def test_config_value_then_explicit_flag(self):
        self.raw['max_decision_steps'] = 19
        info = self.cli()
        self.assertEqual((info['max_decision_steps'], info['max_decision_steps_source']), (19, 'config'))
        info = self.cli('--rounds', '4', '--max-decision-steps', '7', '--data-dir', 'overridden')
        self.assertEqual((info['rounds'], info['max_decision_steps'], info['max_decision_steps_source']), (4, 7, 'flag'))
        self.assertEqual(info['data_dir'], str(self.root.resolve() / 'overridden'))

    def test_invalid_limit_and_selection(self):
        self.assertIn('must be positive', self.cli('--max-decision-steps', '0', exit_code=2))
        self.raw['framework']['labs']['math']['name'] = 'missing_fixture'
        self.assertIn('Unknown Lab: missing_fixture', self.cli(exit_code=2))


class DurableStartupTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.workspace = self.root / 'workspace'
        self.config = self.root / 'launch.json'
        self.plugin = self.root / 'fixture.py'
        self.literature = self.root / 'notes.txt'
        self.literature.write_text('Fixture literature', encoding='utf-8')
        self.config.write_text(json.dumps({
            'objective': 'Resume after committing startup', 'data_dir': 'workspace',
            'framework': {'controller': {'name': 'startup_fixture'},
                          'labs': {'fixture': {'name': 'startup_fixture'}}},
        }), encoding='utf-8')
        self.plugin.write_text('''
from rsicomposer.api import Decision, OperationSpec

class State:
    def state(self):
        return {}

    def restore(self, state):
        if state != {}:
            raise ValueError('Unexpected fixture state')

class Policy(State):
    def decide(self, window):
        return Decision('finish', reason='initialized controller')

class Lab(State):
    def describe(self):
        return (OperationSpec('evaluate', 'Fixture', {'type': 'object'}),)

    def execute(self, call):
        raise AssertionError('startup-only fixture must not dispatch a Lab call')

def register(registry):
    registry.register_controller('startup_fixture', lambda context, config: Policy(), version='1')
    registry.register_lab('startup_fixture', lambda context, config: Lab(), version='1')
''', encoding='utf-8')

    def cli(self, *options, invoke=('-m', 'rsicomposer')):
        return subprocess.run([sys.executable, *invoke, *options, '--no-monitor'], cwd=self.root,
                              env={**os.environ, 'PYTHONPATH': str(SRC)}, capture_output=True, text=True, timeout=15)

    def interrupted_launch(self, boundary):
        # Replace only the atomic write at the selected boundary, using a real
        # process exit so no normal CLI or Application cleanup can repair it.
        return self.cli('run', '--durable', '--config', str(self.config), '--plugin', str(self.plugin),
                        '--literature', str(self.literature), invoke=('-c', '''
import fcntl
import os
import sys
from rsicomposer import application, cli
from rsicomposer.runtime.state import recovery

boundary = sys.argv.pop(1)
write = application.atomic_write

def write_launch(path, data):
    if path.name == 'framework.json':
        with (path.parent / '.lock').open('rb') as contender:
            try:
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise AssertionError('workspace unlocked during launch publication')
        if boundary == 'launch-write':
            raise OSError('fixture launch metadata write failed')
    write(path, data)

def write_recovery(path, data):
    if path.name == 'checkpoint.json' and boundary == 'before-checkpoint':
        os._exit(74)
    write(path, data)
    if path.name == 'checkpoint.json' and boundary == 'after-checkpoint':
        os._exit(75)

application.atomic_write = write_launch
recovery.atomic_write = write_recovery
cli.main()
''', boundary))

    def test_failed_launch_metadata_write_publishes_no_checkpoint(self):
        launch = self.interrupted_launch('launch-write')
        self.assertEqual(launch.returncode, 2, launch.stderr)
        self.assertIn('fixture launch metadata write failed', launch.stderr)
        self.assertFalse((self.workspace / 'framework.json').exists())
        self.assertFalse((self.workspace / 'recovery/checkpoint.json').exists())

    def test_uncommitted_startup_is_not_treated_as_a_resumable_run(self):
        launch = self.interrupted_launch('before-checkpoint')
        self.assertEqual(launch.returncode, 74, launch.stderr)
        self.assertTrue((self.workspace / 'framework.json').is_file())
        self.assertTrue((self.workspace / 'recovery/journal.json').is_file())
        self.assertFalse((self.workspace / 'recovery/checkpoint.json').exists())
        resume = self.cli('resume', '--data-dir', str(self.workspace))
        self.assertEqual(resume.returncode, 2, resume.stderr)
        self.assertIn('Durable startup did not commit an initial checkpoint', resume.stderr)

    def test_first_checkpoint_is_resumable_even_after_immediate_process_exit(self):
        launch = self.interrupted_launch('after-checkpoint')
        self.assertEqual(launch.returncode, 75, launch.stderr)
        manifest = self.workspace / 'framework.json'
        saved = manifest.read_bytes()
        self.assertEqual(json.loads(saved)['plugins'], [str(self.plugin.resolve())])
        self.assertEqual(json.loads((self.workspace / 'recovery/journal.json').read_text()),
                         {'generation': 0, 'status': 'committed'})
        self.assertFalse((self.workspace / 'result.json').exists())
        [artifact] = (self.workspace / 'artifacts').iterdir()
        self.assertEqual(json.loads((artifact / 'ref.json').read_text())['kind'], 'literature')
        self.assertEqual((artifact / 'data').read_bytes(), self.literature.read_bytes())

        retry_plugin = self.root / 'another_fixture.py'
        retry_plugin.write_bytes(self.plugin.read_bytes())
        retry = self.cli('run', '--durable', '--config', str(self.config), '--plugin', str(retry_plugin))
        self.assertEqual(retry.returncode, 2, retry.stderr)
        self.assertIn('Workspace is not empty', retry.stderr)
        self.assertEqual(manifest.read_bytes(), saved)
        # Deliberately omit --plugin: resume must use the committed launch metadata.
        resume = self.cli('resume', '--data-dir', str(self.workspace))
        self.assertEqual(resume.returncode, 0, resume.stderr)
        self.assertEqual(json.loads(resume.stdout)['status'], 'finished')


class RunResultCleanupTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.workspace = self.root / 'workspace'
        self.result_path = self.workspace / 'result.json'
        self.config = self.root / 'launch.json'
        self.plugin = self.root / 'fixture.py'
        self.config.write_text(json.dumps({
            'objective': 'Run result survives workspace cleanup', 'data_dir': 'workspace',
            'framework': {'controller': {'name': 'single_loop', 'config': {'operation': 'fixture.evaluate'}},
                          'labs': {'fixture': {'name': 'result_fixture'}}},
        }), encoding='utf-8')

    def launch(self, *, fail_close=False, exit_during_close=False, verify_lock=False,
               report_fault=None, snapshot_fault=False):
        self.plugin.write_text('''
import json
import os
from rsicomposer.api import Decision, OperationCall, OperationResult, OperationSpec

class Policy:
    def __init__(self, context):
        self.root = run_root(context)
        self.invoked = False

    def decide(self, window):
        if not self.invoked:
            self.invoked = True
            return Decision('invoke', operations=(OperationCall('fixture.evaluate', {}),))
        return Decision('finish', reason='fixture completed')

    def report(self):
        pending = json.loads((self.root / 'result.json').read_text(encoding='utf-8'))
        assert pending['status'] == 'finished'
        assert pending['postprocessing']['controller_report'] == {'status': 'pending'}
        if REPORT_FAULT == 'raise':
            raise ValueError('report failed TEST_DIAGNOSTIC')
        if REPORT_FAULT == 'invalid':
            return ['invalid report']
        return {'fixture': 'report produced'}

def run_root(context):
    return next(path for path in (context.data_dir, *context.data_dir.parents)
                if (path / '.lock').is_file())

class Lab:
    def __init__(self, context):
        # Registry scopes Labs below the run directory.
        self.root = run_root(context)

    def describe(self):
        return (OperationSpec('evaluate', 'Fixture', {'type': 'object'}),)

    def execute(self, call):
        return OperationResult('Recorded before cleanup', (), True)

    def close(self):
        pending = json.loads((self.root / 'result.json').read_text(encoding='utf-8'))
        assert pending['cleanup'] == {'status': 'pending'}
        assert pending['status'] == 'finished'
        if EXIT_DURING_CLOSE:
            os._exit(71)
        if FAIL_CLOSE:
            raise RuntimeError('fixture cleanup failed TEST_DIAGNOSTIC')

def register(registry):
    registry.register_lab('result_fixture', lambda context, config: Lab(context), version='1')
    if REPORT_FAULT is not None:
        registry.register_controller('result_fixture', lambda context, config: Policy(context), version='1')
''' + (f'\nFAIL_CLOSE = {fail_close!r}\nEXIT_DURING_CLOSE = {exit_during_close!r}\n'
       f'REPORT_FAULT = {report_fault!r}\n'), encoding='utf-8')
        if report_fault is not None:
            config = json.loads(self.config.read_text(encoding='utf-8'))
            config['framework']['controller'] = {'name': 'result_fixture'}
            self.config.write_text(json.dumps(config), encoding='utf-8')
        if verify_lock:
            invoke = ['-c', '''
import fcntl
import json
import sys
from rsicomposer import cli

write = cli.atomic_write

def locked_write(path, data):
    if path.name == 'result.json' and json.loads(data)['cleanup']['status'] == 'completed':
        with (path.parent / '.lock').open('rb') as contender:
            try:
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise AssertionError('workspace unlocked before final result publication')
    return write(path, data)

cli.atomic_write = locked_write
cli.main()
''']
        elif snapshot_fault:
            invoke = ['-c', '''
import json
import os
from rsicomposer import cli
from rsicomposer.runtime.state import component_state

component_state.snapshot_integration = lambda app: True

def fail_capture(capture, name):
    pending = json.loads((capture.application.config.data_dir / 'result.json').read_text(encoding='utf-8'))
    assert pending['status'] == 'finished'
    assert pending['postprocessing']['snapshot'] == {'status': 'pending'}
    if os.environ.get('RSI_EXIT_INSIDE_SNAPSHOT') == '1':
        os._exit(73)
    raise OSError('fixture snapshot export failed')

cli.LocalSnapshotCapture.capture = fail_capture
cli.main()
''']
        else:
            invoke = ['-m', 'rsicomposer']
        env = {**os.environ, 'PYTHONPATH': str(SRC), 'RSI_EXIT_INSIDE_SNAPSHOT': str(int(snapshot_fault == 'exit'))}
        return subprocess.run([sys.executable, *invoke, 'run', '--config', str(self.config),
                               '--plugin', str(self.plugin), '--no-monitor'], cwd=self.root,
                              env=env, capture_output=True, text=True, timeout=15)

    def read_result(self):
        raw = self.result_path.read_text(encoding='utf-8')
        self.assertNotIn('TEST_DIAGNOSTIC', raw)
        result = json.loads(raw)
        self.assertEqual(result['status'], 'finished')
        self.assertEqual(result['stop_reason'], 'researcher_finished')
        receipt, = [row['operation_result'] for row in result['events'] if row.get('operation_result')]
        self.assertEqual(receipt['summary'], 'Recorded before cleanup')
        from rsicomposer.runtime.storage.codec import decode
        [stored] = (self.workspace / 'lab_receipts').glob('*.receipt')
        self.assertEqual(decode(json.loads(stored.read_text()))['outcome'].call_id, receipt['call_id'])
        return result

    def test_cli_preserves_the_run_and_exits_nonzero_on_cleanup_failure(self):
        cli = self.launch(fail_close=True)
        self.assertNotEqual(cli.returncode, 0)
        self.assertIn('fixture cleanup failed', cli.stderr)
        result = self.read_result()
        self.assertEqual(result['cleanup']['status'], 'failed')
        self.assertIsInstance(result['cleanup']['error_type'], str)

    def test_exit_during_cleanup_leaves_a_readable_pending_summary(self):
        cli = self.launch(exit_during_close=True)
        self.assertEqual(cli.returncode, 71, cli.stderr)
        self.assertEqual(self.read_result()['cleanup'], {'status': 'pending'})

    def test_success_replaces_pending_with_completed_under_workspace_lock(self):
        cli = self.launch(verify_lock=True)
        self.assertEqual(cli.returncode, 0, cli.stderr)
        result = self.read_result()
        self.assertEqual(result['cleanup'], {'status': 'completed'})
        self.assertEqual(json.loads(cli.stdout), result)

    def assert_optional_report_failure(self, fault, error_type):
        cli = self.launch(report_fault=fault)
        self.assertEqual(cli.returncode, 0, cli.stderr)
        self.assertNotIn('TEST_DIAGNOSTIC', cli.stderr)
        self.assertIn('optional Controller report failed', cli.stderr)
        result = self.read_result()
        self.assertNotIn('controller_report', result)
        self.assertEqual(result['cleanup'], {'status': 'completed'})
        self.assertEqual(result['postprocessing']['controller_report'], {'status': 'failed', 'error_type': error_type})
        self.assertEqual(json.loads(cli.stdout), result)

    def test_optional_controller_report_failure_does_not_fail_a_completed_run(self):
        self.assert_optional_report_failure('raise', 'ValueError')

    def test_invalid_controller_report_does_not_fail_a_completed_run(self):
        self.assert_optional_report_failure('invalid', 'TypeError')

    def test_snapshot_failure_does_not_lose_report_or_core_summary(self):
        cli = self.launch(snapshot_fault=True, report_fault='success')
        self.assertNotEqual(cli.returncode, 0)
        self.assertIn('fixture snapshot export failed', cli.stderr)
        result = self.read_result()
        self.assertTrue(result['snapshot_supported'])
        self.assertIsNone(result['snapshot'])
        self.assertEqual(result['cleanup'], {'status': 'completed'})
        self.assertEqual(result['postprocessing']['snapshot'], {'status': 'failed', 'error_type': 'OSError'})
        self.assertEqual(result['postprocessing']['controller_report'], {'status': 'completed'})
        self.assertEqual(result['controller_report'], {'fixture': 'report produced'})

    def test_snapshot_and_cleanup_failures_are_both_recorded(self):
        cli = self.launch(snapshot_fault=True, fail_close=True)
        self.assertNotEqual(cli.returncode, 0)
        self.assertIn('fixture cleanup failed', cli.stderr)
        result = self.read_result()
        self.assertEqual(result['cleanup']['status'], 'failed')
        self.assertEqual(result['postprocessing']['snapshot'], {'status': 'failed', 'error_type': 'OSError'})

    def test_process_exit_inside_snapshot_has_core_summary_on_disk(self):
        cli = self.launch(snapshot_fault='exit')
        self.assertEqual(cli.returncode, 73, cli.stderr)
        result = self.read_result()
        self.assertEqual(result['cleanup'], {'status': 'pending'})
        self.assertEqual(result['postprocessing']['snapshot'], {'status': 'pending'})
