"""CPU HTTP fixture verifies the review/query bridge, not real-model execution."""
import json
from pathlib import Path
import tempfile
import unittest

from test_qwen_toy import Services, servers
from examples.qwen_toy.run import parse_args, run


class ResearchServices(Services):
    def reply(self, role, path, payload):
        result = super().reply(role, path, payload)
        if path.startswith('/jobs/') and result['status'] == 'running':
            result.update(step=3, total_steps=24, elapsed_seconds=1.5,
                          observed_at=1730000000.0, metrics={'loss': 0.7})
        if (path == '/v1/chat/completions'
                and payload['messages'][0]['content'].startswith('Review the running execution')):
            context = self.reviews[-1]
            directive_id = context['execution']['progress']['directive_id']
            observations = [item for result in context['operation_results']
                            if result['data'].get('name') == 'metrics'
                            for item in result['data']['data']['items']
                            if item['directive_id'] == directive_id]
            if not observations:
                operation = 'query_exploration_metrics'
                assert any(s['name'] == operation and s['read_only'] for s in context['operations'])
                value = {'kind': 'invoke', 'operation': operation,
                         'arguments': {'filters': {'directive_id': directive_id}}}
            else:
                assert observations[-1]['metrics']['loss'] == 0.7
                value = {'action': 'continue', 'reason': 'Queried loss is 0.7', 'next_review_seconds': None}
            result['choices'][0]['message']['content'] = json.dumps(value)
        return result


class ResearcherIntegrationTests(unittest.TestCase):
    def test_http_training_queries_then_continues_and_completes(self):
        service = ResearchServices(poll_delay=0.015, plan=[
            {'kind': 'invoke', 'operation': 'query_exploration_model', 'arguments': {}},
            'train', 'evaluate', 'apply_candidate', 'execute', 'finish'])
        # The scripted model schedules its review; the CLI interval is only a hint.
        service.review_after = 0.005
        with tempfile.TemporaryDirectory() as directory, servers(service) as urls:
            root = Path(directory)
            args = parse_args(['--baseline-url', urls[0] + '/v1', '--worker-url', urls[1],
                '--run-dir', str(root), '--poll-interval', '0.001', '--timeout', '10',
                '--max-trainings', '1', '--review-after-seconds', '100', '--require-training-review'])
            result = run(args)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(len(service.reviews), 2)
            self.assertEqual(service.decisions[1]['review_after_seconds'], 0.005)
            self.assertEqual(service.reviews[0]['operation_results'][0]['data']['name'], 'model')
            self.assertIsNone(service.reviews[0]['execution']['progress']['feedback'])
            self.assertEqual(result['round_results'][0]['training_review_count'], 1)
            self.assertEqual(result['training_reviews'][0]['action'], 'continue')
            self.assertEqual(result['training_submissions'], 1)
            self.assertEqual(result['actor_revision'], 'candidate-1')
            self.assertEqual(service.reviews[0]['execution']['progress']['job_progress']['step'], 3)
            records = json.loads((root / 'decision_memory.json').read_text())['history']
            queries = [row for row in records if row['details'].get('event') == 'review_query']
            self.assertEqual(len(queries), 1)
            self.assertEqual(queries[0]['details']['result']['data']['source'], 'exploration')
            observed = queries[0]['details']['result']['data']['data']['items'][0]
            self.assertEqual(observed['job_progress']['step'], 3)
            self.assertEqual(observed['job_progress']['observed_at'], 1730000000.0)
            final = next(row for row in records if row['details'].get('event') == 'review')
            self.assertEqual(final['details']['review']['reason'], 'Queried loss is 0.7')
            self.assertTrue(final['details']['query_refs'])


if __name__ == '__main__':
    unittest.main()
