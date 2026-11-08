"""Toy assembly research tools with HTTP fixtures; no real model or training."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from tests.experiments.qwen_toy.test_qwen_toy import Services, servers
from experiments.qwen_toy import run as controller
from rsisys.loops.contracts.model_service import GenerationRequest, ModelRef
from rsisys.loops.decision_making.harness.research import ResearchAgenda


class QwenResearchDefaultsTests(unittest.TestCase):
    def args(self, *options):
        return controller.parse_args(['--baseline-url', 'http://localhost:1',
                                      '--worker-url', 'http://localhost:2', *options])

    def test_output_token_cli_default_override_and_validation(self):
        self.assertEqual(self.args().max_output_tokens, 200)
        for value in (1, 512, 1024):
            with self.subTest(value=value):
                self.assertEqual(self.args('--max-output-tokens', str(value)).max_output_tokens, value)
        for value in ('0', '-1', '1.5', 'nan'):
            with self.subTest(value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args('--max-output-tokens', value)

    def test_budgeted_model_allows_larger_outputs_and_preserves_smaller_request_limits(self):
        for limit in (512, 1024):
            for requested in (80, 2048):
                with self.subTest(limit=limit, requested=requested):
                    backend = Mock()
                    budget = controller.ResearchBudget(max_output_tokens=limit)
                    model = controller.BudgetedModel(backend, budget)
                    request = GenerationRequest('decision_maker', ModelRef('test', 'qwen', 'base'),
                                                '{}', max_output_tokens=requested)
                    model.generate(request)
                    self.assertEqual(backend.generate.call_args.args[0].max_output_tokens,
                                     min(limit, requested))
                    self.assertEqual(request.max_output_tokens, requested)

    def test_model_patch_read_and_archives_share_harness_state_and_persist(self):
        patch = {'expected_revision': 0, 'upserts': [
            {'id': 'inspect', 'kind': 'todo', 'text': 'Inspect existing evidence',
             'status': 'open', 'evidence_refs': []}],
            'remove_ids': [], 'reason': 'Choose the next research question', 'evidence_refs': []}
        service = Services(plan=[
            {'kind': 'invoke', 'operation': 'research_patch', 'arguments': patch},
            {'kind': 'invoke', 'operation': 'research_read', 'arguments': {}},
            {'kind': 'invoke', 'operation': 'research_archive',
             'arguments': {'collection': 'history', 'limit': 32}},
            {'kind': 'invoke', 'operation': 'research_archive',
             'arguments': {'collection': 'experience', 'limit': 32}},
            {'kind': 'finish', 'reason': 'Evidence inspection complete'},
        ])
        with tempfile.TemporaryDirectory() as directory, servers(service) as urls:
            root = Path(directory)
            args = controller.parse_args(['--baseline-url', urls[0], '--worker-url', urls[1],
                '--run-dir', str(root), '--timeout', '10', '--max-context-chars', '60000',
                '--max-output-tokens', '128'])
            with redirect_stdout(io.StringIO()):
                result = controller.run(args)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['stop_reason'], 'researcher_finished')
            self.assertEqual(result['training_submissions'], 0)
            self.assertEqual(result['experiments'], [])
            self.assertEqual(result['model_calls'], 5)
            self.assertEqual(result['budget']['max_output_tokens'], 128)
            requests = [payload for _, path, payload in service.calls if path == '/v1/chat/completions']
            self.assertEqual([request['max_tokens'] for request in requests], [128] * 5)

            operations = {spec['name']: spec for spec in service.contexts[0]['operations']}
            self.assertTrue({'research_read', 'research_patch', 'research_archive'} <= operations.keys())
            self.assertIn('workspace_read', operations)
            collections = operations['research_archive']['input_schema']['properties']['collection']['enum']
            self.assertIn('experience', collections)
            self.assertEqual(service.contexts[0]['research_agenda']['revision'], 0)
            for context in service.contexts[1:]:
                self.assertEqual(context['research_agenda']['revision'], 1)
                self.assertEqual(context['research_agenda']['items'][0]['id'], 'inspect')

            read = service.contexts[2]['operation_results'][-1]['data']
            self.assertEqual(read['revision'], 1)
            self.assertEqual(read['items'][0]['text'], patch['upserts'][0]['text'])
            history_page = service.contexts[3]['operation_results'][-1]['data']
            experience_page = service.contexts[4]['operation_results'][-1]['data']
            archived_history = self.archive_rows(history_page)
            archived_experience = self.archive_rows(experience_page)
            receipt = next(row for row in archived_history if row['kind'] == 'operation'
                           and row['details']['operation']['name'] == 'research_patch')
            self.assertEqual(receipt['details']['result']['data']['revision'], 1)
            self.assertTrue(any(row['record_id'] == receipt['record_id']
                                for row in archived_experience))

            memory = json.loads((root / 'decision_memory.json').read_text())
            self.assertIn(receipt, memory['history'])
            agenda = ResearchAgenda(root / 'decision_research.json')
            self.assertEqual(agenda.read()['items'], read['items'])
            self.assertEqual(agenda.export()['revisions'][0], {'revision': 1, **patch})
            self.assertEqual(json.loads((root / 'result.json').read_text()), result)

    def archive_rows(self, page):
        self.assertTrue(page['found'])
        self.assertIsNone(page['next_cursor'])
        records = {}
        for chunk in page['items']:
            records.setdefault(chunk['record_id'], []).append(chunk['text'])
        return [json.loads(''.join(chunks)) for chunks in records.values()]


if __name__ == '__main__':
    unittest.main()
