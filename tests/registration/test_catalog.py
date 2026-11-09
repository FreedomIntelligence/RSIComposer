"""Discovery must not allocate resources or leak mutable registry state."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from rsisys.api import MethodDefinition, Registry
from tests import ROOT


class CatalogTests(unittest.TestCase):
    def test_metadata_without_constructing_any_factory(self):
        def unavailable(*args):
            raise AssertionError('discovery must not construct runtime components')
        registry = Registry(builtins=False)
        registry.register_controller('custom', unavailable, version='c1')
        registry.register_lab('remote', unavailable, version='l2')
        registry.register_method('complete', MethodDefinition(unavailable, 'm3'))
        catalog = registry.describe()
        self.assertEqual(catalog['methods'], {'complete': {'version': 'm3'}})
        catalog['controllers']['custom']['version'] = 'changed'
        self.assertEqual(registry.describe()['controllers']['custom']['version'], 'c1')

    def test_plugin_identity_is_detached_and_catalog_cli_runs_outside_repo(self):
        with TemporaryDirectory() as temporary:
            plugin = Path(temporary) / 'plugin.py'
            plugin.write_text('def register(registry):\n'
                              '    registry.register_lab("external", lambda *a: None, version="7")\n')
            registry = Registry()
            registry.load_plugin(plugin)
            catalog = registry.describe()
            catalog['plugins'][0]['sha256'] = 'bad'
            self.assertNotEqual(registry.describe()['plugins'][0]['sha256'], 'bad')
            result = subprocess.run([sys.executable, '-m', 'rsisys', 'catalog', '--plugin', str(plugin)],
                cwd=temporary, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')},
                capture_output=True, text=True, timeout=20, check=True)
            output = json.loads(result.stdout)
            self.assertEqual(output['labs']['external'], {'version': '7'})
            self.assertIn('ttrl', output['controllers'])
            self.assertEqual(sorted(p.name for p in Path(temporary).iterdir()), ['plugin.py'])

    def test_builtin_discovery_does_not_import_ml_dependencies(self):
        code = ('import sys; from rsisys.api import Registry; Registry().describe(); '
                'assert not any(n.split(".")[0] in {"torch", "transformers", "peft", '
                '"vllm", "math_verify"} for n in sys.modules)')
        subprocess.run([sys.executable, '-c', code], check=True, timeout=20,
                       env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')})
