"""Public imports, packaged resources and the embeddable CLI contract.

Also runnable against an installed wheel from outside the source checkout:
    /path/to/venv/bin/python -I /path/to/tests/test_monitor_package.py
"""
from importlib.resources import files
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rsimonitor import MonitorServer, RunStore, project_snapshot
from rsimonitor import server


class MonitorPackageTests(unittest.TestCase):
    def test_module_cli_has_no_preloaded_module_warning(self):
        result = subprocess.run(
            [sys.executable, '-m', 'rsimonitor.server', '--help'],
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, '')
        self.assertIn('--runs-dir', result.stdout)

    def test_public_api_reads_experiment_snapshots(self):
        self.assertIs(MonitorServer, server.MonitorServer)
        self.assertIs(RunStore, server.RunStore)
        self.assertIs(project_snapshot, server.project_snapshot)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'experiment').mkdir()
            (root / 'experiment/monitor.json').write_text(
                '{"schema_version":1,"project":{"project_id":"research"},'
                '"status":"running","payload":"private"}', encoding='utf-8')
            store = RunStore(root)
            snapshot = store.read('experiment')
            self.assertEqual(snapshot['run_id'], 'experiment')
            self.assertNotIn('payload', snapshot)
            self.assertEqual(store.projects()[0]['project_id'], 'research')
            self.assertEqual(store.list()[0]['status'], 'running')

    def test_runtime_assets_are_complete_without_development_fixtures(self):
        assets = files('rsimonitor').joinpath('static')
        expected = {'index.html', 'style.css', 'i18n.js', 'decisions.js', 'app.js'}
        self.assertEqual({asset.name for asset in assets.iterdir()}, expected)
        self.assertEqual({filename for filename, _ in server.STATIC.values()}, expected)
        for asset in assets.iterdir():
            with self.subTest(asset=asset.name):
                self.assertTrue(asset.read_bytes())
                self.assertEqual(asset.read_bytes(), (server.STATIC_DIR / asset.name).read_bytes())

    def test_main_accepts_explicit_arguments_and_closes_server(self):
        with patch.object(server, 'MonitorServer') as factory, patch('builtins.print'):
            factory.return_value.serve_forever.side_effect = KeyboardInterrupt
            server.main(['--runs-dir', '/tmp/monitor-runs', '--host', '127.0.0.1', '--port', '8877'])
        factory.assert_called_once_with(('127.0.0.1', 8877), Path('/tmp/monitor-runs'))
        factory.return_value.serve_forever.assert_called_once_with()
        factory.return_value.server_close.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
