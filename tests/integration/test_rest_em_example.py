import json
from pathlib import Path
import tempfile
import unittest
from tests import ROOT
from rsisys.api import Application, ApplicationConfig, Registry

class RestEMTests(unittest.TestCase):
    def test_real_application_fixture_filter_cap_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            config = json.loads((ROOT / 'examples/quickstart/rest_em.json').read_text())
            config['framework']['labs']['math']['config']['data_root'] = str(ROOT / 'examples/quickstart/data')
            registry = Registry()
            with Application(ApplicationConfig('test', Path(directory) / 'run', {}, {}, max_decision_steps=12),
                             modules=registry.build(config['framework'])) as app:
                result = app.run()
                self.assertEqual(result.status, 'finished')
                receipts = app.runner.decision_making.harness.state.operation_results
                self.assertEqual(len(receipts), 9)
                self.assertTrue(all(r.simulated for r in receipts))
                self.assertTrue(all(r.data['training_performed'] is False for r in receipts))
                self.assertEqual([receipts[i].data['count'] for i in (1, 2, 5, 6)], [8, 2, 8, 2])
                trained = [receipts[i].data['manifest'] for i in (3, 7)]
                self.assertTrue(all(m['initial_model'] == 'fixture-initial' for m in trained))
                self.assertNotEqual(trained[0]['dataset'], trained[1]['dataset'])
                rows = [json.loads(line) for line in Path(trained[0]['dataset']).read_text().splitlines()]
                self.assertTrue(all(set(row) == {'id', 'prompt', 'completion'} for row in rows))
