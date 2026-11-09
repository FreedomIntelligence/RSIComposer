"""Weight Labs keep independent immutable artifact ownership when composed."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from rsisys.api import Application, ApplicationConfig, Registry
from rsisys.labs.math_verifiers import StrictNumericVerifier
from rsisys.labs.seal import SEALLab, validate_config as validate_seal
from rsisys.labs.ttrl import TTRLLab, validate_config as validate_ttrl


def engine():
    return SimpleNamespace(name='fixture-v1', simulated=True,
                           generate=lambda *a, **k: None, adapt=lambda *a, **k: None,
                           train=lambda *a, **k: None)


class WeightArtifactOwnershipTests(unittest.TestCase):
    def test_registered_labs_of_same_method_keep_distinct_artifact_ownership(self):
        for implementation in (SEALLab, TTRLLab):
            with self.subTest(method=implementation.method), TemporaryDirectory() as temporary:
                labs, registry = [], Registry()

                def factory(context, config):
                    kwargs = {'verifier': StrictNumericVerifier()} if implementation is TTRLLab else {}
                    lab = implementation(context, config, engine=engine(), **kwargs)
                    labs.append(lab)
                    return lab

                registry.register_lab('fixture', factory, version='test')
                modules = registry.build({
                    'controller': {'name': 'single_loop', 'config': {'operation': 'a.evaluate'}},
                    'labs': {alias: {'name': 'fixture', 'config': {'output_subdir': f'weights-{alias}'}}
                             for alias in ('a', 'b')},
                })
                config = ApplicationConfig('ownership', Path(temporary) / 'run', {}, {})
                with Application(config, modules=modules):
                    first, second = labs
                    artifact = first.write('test', 1, [{'evidence': 1}])
                    self.assertNotEqual(first.root, second.root)
                    self.assertEqual(first.read(artifact['dataset'], owned=True), [{'evidence': 1}])
                    with self.assertRaisesRegex(ValueError, 'Unowned'):
                        second.read(artifact['dataset'], owned=True)
                    with self.assertRaises(FileExistsError):
                        first.write('test', 1, [])

    def test_invalid_output_paths_rejected_before_lab_allocation(self):
        for validate in (validate_seal, validate_ttrl):
            for value in ('../escape', '/tmp/escape', 'nested/path', '', None, 5, '.', '..'):
                with self.subTest(validate=validate.__module__, value=value):
                    with self.assertRaisesRegex(ValueError, 'output_subdir'):
                        validate({'output_subdir': value})
