import unittest
from unittest.mock import Mock
from examples.qwen_toy.export_swanlab import export


class SwanLabExportTests(unittest.TestCase):
    def test_optional_offline_export_filters_private_fields(self):
        sdk = Mock()
        count = export({'schema_version': 1, 'run_id': 'test', 'jobs': [
            {'job_id': 'j1', 'step': 3, 'metrics': {'loss': .2, 'api_key': 123, 'prompt': 'secret'}}]},
            project='test', sdk=sdk)
        self.assertEqual(count, 1)
        self.assertEqual(sdk.init.call_args.kwargs['mode'], 'offline')
        sdk.init.return_value.log.assert_called_once_with({'loss': .2, 'optimizer_step': 3}, step=0)
        sdk.finish.assert_called_once()

    def test_finish_on_logging_error(self):
        sdk = Mock()
        sdk.init.return_value.log.side_effect = RuntimeError('disk full')
        with self.assertRaises(RuntimeError):
            export({'schema_version': 1, 'jobs': [{'metrics': {'loss': .2}}]}, project='test', sdk=sdk)
        sdk.finish.assert_called_once()

    def test_no_fabricated_history(self):
        sdk = Mock()
        self.assertEqual(export({'schema_version': 1, 'jobs': [{}]}, project='test', sdk=sdk), 0)
        sdk.init.assert_not_called()

    def test_curve_keeps_optimizer_steps_and_observation_order(self):
        sdk = Mock()
        snapshot = {'schema_version': 1, 'jobs': [{'job_id': 'train', 'metric_history': [
            {'step': 4, 'elapsed_seconds': 1.2, 'metrics': {'loss': .4}},
            {'step': 8, 'elapsed_seconds': 2.5, 'metrics': {'loss': .1}}]}]}
        self.assertEqual(export(snapshot, project='test', sdk=sdk), 1)
        calls = sdk.init.return_value.log.call_args_list
        self.assertEqual([c.kwargs['step'] for c in calls], [0, 1])
        self.assertEqual([c.args[0]['optimizer_step'] for c in calls], [4, 8])
