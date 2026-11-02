"""Optional real-runtime test; the ordinary dependency-free suite skips it."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
HAS_SDK = (importlib.util.find_spec('deepseek_harness') is not None
           and importlib.util.find_spec('deepseek_harness_runtime') is not None)


@unittest.skipUnless(HAS_SDK, 'Install rsisys[dsh] to exercise the real DSH runtime')
class DSHRuntimeTests(unittest.TestCase):
    def run_smoke(self, *options):
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join(filter(None, (str(ROOT / 'src'), env.get('PYTHONPATH'))))
        result = subprocess.run(
            [sys.executable, str(ROOT / 'examples/dsh_smoke.py'), '--timeout', '20', *options],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def test_restricted_runtime_resume_and_authorized_decisions(self):
        report = self.run_smoke()
        self.assertTrue(report['real_dsh_runtime'])
        self.assertFalse(report['real_model_used'])
        self.assertTrue(report['sdk']['runtime_restart_resume'])
        self.assertTrue(report['sdk']['restored_user_and_assistant_history'])
        self.assertEqual(report['sdk']['model_requests'], 2)
        self.assertEqual(report['sdk']['native_tools'], [])
        self.assertEqual(report['integration']['authorized_operations'], 1)
        self.assertEqual(report['integration']['directives_completed'], 1)
        self.assertTrue(report['integration']['finished'])
        self.assertTrue(report['integration']['async_wait_observed'])
        self.assertEqual(report['integration']['native_tools'], [])
        self.assertEqual(report['integration']['model_requests'], 3)
        self.assertTrue(report['integration']['actor_and_execution_simulated'])

    def test_native_automatic_compaction_uses_checkpoint(self):
        report = self.run_smoke('--compaction-only')
        self.assertTrue(report['real_dsh_runtime'])
        self.assertFalse(report['real_model_used'])
        compact = report['compaction']
        self.assertTrue(compact['automatic_compaction'])
        self.assertEqual(compact['context_window'], 4096)
        self.assertLessEqual(compact['generation_max_tokens'], 1024)
        self.assertEqual(compact['summary_max_tokens'], 1024)
        self.assertGreaterEqual(compact['compaction_summaries'], 1)
        self.assertEqual(compact['compaction_summaries'], compact['summary_requests'])
        self.assertTrue(compact['checkpoint_used_by_followup'])
        self.assertTrue(compact['continued_structured_output'])
        self.assertLessEqual(compact['history_turns'], 12)
        self.assertTrue(compact['usage_is_synthetic'])
        self.assertEqual(compact['native_tools'], [])

    def test_latest_large_adapter_result_is_user_data_and_is_not_tool_pruned(self):
        report = self.run_smoke('--latest-result-only')
        self.assertTrue(report['real_dsh_runtime'])
        self.assertFalse(report['real_model_used'])
        result = report['latest_result']
        self.assertEqual(result['adapter_used'], '_SessionBackend')
        self.assertGreater(result['latest_result_characters'], 128 * 1024)
        self.assertEqual(result['latest_wire_role'], 'user')
        self.assertTrue(result['latest_data_exactly_preserved'])
        self.assertTrue(result['middle_and_tail_retained'])
        self.assertEqual(result['native_tool_result_events'], 0)
        self.assertEqual(result['native_prune_events'], 0)
        self.assertGreaterEqual(result['summary_requests'], result['compaction_summaries'])
        self.assertGreater(result['final_input_tokens_synthetic'], result['context_window'])
        self.assertTrue(result['oversized_request_reached_fixture'])
        self.assertTrue(result['fixture_accepts_oversized_requests'])
        self.assertTrue(result['usage_is_synthetic'])
        self.assertLessEqual(result['model_requests'], 12)
        self.assertEqual(result['native_tools'], [])
        projected = report['projected_latest']
        self.assertEqual(projected['host_projection'], 'engine._project_context')
        self.assertTrue(projected['identity_compactor_preserved'])
        self.assertGreater(projected['latest_result_raw_bytes'], 128 * 1024)
        self.assertGreater(projected['latest_feedback_raw_bytes'], 128 * 1024)
        self.assertLess(projected['latest_result_view_bytes'], 8192)
        self.assertLess(projected['latest_feedback_view_bytes'], 8192)
        self.assertTrue(projected['exact_metrics_and_ownership_preserved'])
        self.assertTrue(projected['raw_archive_unchanged'])
        self.assertEqual(projected['latest_result_read_path']['operation'], 'research_archive')
        self.assertLess(projected['input_tokens_synthetic'], result['final_input_tokens_synthetic'])
        self.assertLess(projected['projected_packet_bytes'], 8192)
        self.assertEqual(projected['latest_wire_role'], 'user')
        self.assertTrue(projected['structured_response'])
        self.assertEqual(projected['model_requests'], 1)
        self.assertTrue(projected['usage_is_synthetic'])
        self.assertFalse(projected['real_model_used'])

    @unittest.skipUnless(all(importlib.util.find_spec(name) is not None
                             for name in ('pip', 'setuptools', 'wheel')),
                         'Wheel verification additionally requires pip, setuptools>=68 and wheel')
    def test_installed_wheel_contains_and_uses_restricted_patch(self):
        # Build a copy: setuptools must not write build/ or egg-info into the
        # shared checkout. No package downloads or dependency resolution occur.
        with tempfile.TemporaryDirectory(prefix='rsisys-dsh-wheel-') as directory:
            root = Path(directory)
            project, installed = root / 'project', root / 'installed'
            project.mkdir()
            shutil.copy2(ROOT / 'pyproject.toml', project / 'pyproject.toml')
            shutil.copytree(ROOT / 'src', project / 'src',
                            ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
            built = subprocess.run(
                [sys.executable, '-m', 'pip', '--disable-pip-version-check', 'wheel', '--no-index',
                 '--no-deps', '--no-build-isolation', '--wheel-dir', str(root / 'wheels'), str(project)],
                cwd=root, capture_output=True, text=True, timeout=45,
            )
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            wheels = list((root / 'wheels').glob('rsisys-*.whl'))
            self.assertEqual(len(wheels), 1)
            patch_path = 'rsisys/loops/decision_making/harness/dsh.patch.yml'
            with zipfile.ZipFile(wheels[0]) as archive:
                self.assertEqual(archive.read(patch_path), (ROOT / 'src' / patch_path).read_bytes())
            installation = subprocess.run(
                [sys.executable, '-m', 'pip', '--disable-pip-version-check', 'install', '--no-index',
                 '--no-deps', '--target', str(installed), str(wheels[0])],
                cwd=root, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(installation.returncode, 0, installation.stdout + installation.stderr)
            code = '''
from pathlib import Path
import runpy
import sys
from rsisys.loops.decision_making.harness import dsh
if not Path(dsh.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve()):
    raise RuntimeError('Smoke imported the checkout instead of the installed wheel')
script = sys.argv[2]
sys.argv = [script, '--sdk-only']
runpy.run_path(script, run_name='__main__')
'''
            result = subprocess.run(
                [sys.executable, '-c', code, str(installed), str(ROOT / 'examples/dsh_smoke.py')],
                cwd=root, env=dict(os.environ, PYTHONPATH=str(installed)),
                capture_output=True, text=True, timeout=45,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(result.stdout)
            self.assertTrue(report['sdk']['runtime_restart_resume'])
            self.assertEqual(report['sdk']['native_tools'], [])


if __name__ == '__main__':
    unittest.main()
