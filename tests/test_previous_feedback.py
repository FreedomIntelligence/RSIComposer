"""Previous native feedback reaches the real decision request before cards."""
from copy import deepcopy
from dataclasses import asdict
import json
import unittest
from unittest.mock import patch

from rsisys.config import build_demo
from rsisys.loops._memory import HistoryEntry
from rsisys.loops.contracts.decision import OperationCall
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import GenerationResponse
from rsisys.loops.decision_making.action_space import OperationResult
from rsisys.runtime.json_tools import encoded, result_id
from tests.test_feedback_required_outputs import CONFIG


class PreviousFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.runner, _ = build_demo(CONFIG)
        self.addCleanup(self.runner.close)
        self.loop = self.runner.decision_making
        self.harness = self.loop.harness

    def append(self, result, route='context', name='lab_result'):
        self.harness.state.operation_results.append(result)
        self.harness.memory.append(HistoryEntry(str(len(self.harness.state.operation_results)),
            'operation', None, result.summary, False, details={
                'operation': asdict(OperationCall(name, {'feedback_route': route})),
                'result': result.payload()}))

    def request(self):
        context = self.harness._context('Debug the actual program', self.loop.action_space)
        return json.loads(self.harness.policy._decision_request(context).prompt), context

    def test_native_failure_over_card_limit_is_verbatim_in_actual_next_decision(self):
        native = OperationResult('Program failed', (), False, status=ExecutionStatus.FAILED, error='native error',
            data={'job_id': 'j', 'state': 'failed', 'stderr': 'trace line\n' * 900})
        self.append(native)
        before = deepcopy(self.harness.memory.history())
        payload, context = self.request()
        self.assertIn('evidence_card', context.operation_results[-1].data)
        self.assertEqual(payload['research_agenda']['previous_feedback']['result'], json.loads(encoded(native.payload())))
        self.assertTrue(payload['research_agenda']['previous_feedback']['complete'])
        requests = []
        def generate(request):
            requests.append(request)
            return GenerationResponse('{"kind":"finish"}', request.model, True)
        with patch.object(self.harness.policy.backend, 'generate', side_effect=generate):
            self.harness._call_policy(self.harness.policy.decide, context)
        self.assertEqual(json.loads(requests[-1].prompt)['research_agenda']['previous_feedback']['result'], json.loads(encoded(native.payload())))
        self.assertEqual(self.harness.memory.history(), before)

    def test_window_replaces_previous_body_on_the_next_operation(self):
        self.append(OperationResult('Old', (), False, data={'native': 'OLD'}))
        self.assertEqual(self.request()[0]['research_agenda'].get('previous_feedback')['result']['data'], {'native': 'OLD'})
        latest = OperationResult('New', (), False, data={'native': 'NEW'})
        self.append(latest, name='lab_read_file')
        payload, _ = self.request()
        self.assertEqual(payload['research_agenda']['previous_feedback']['result'], json.loads(encoded(latest.payload())))
        self.assertNotIn('OLD', encoded(payload['research_agenda']['previous_feedback']))

    def test_memory_route_does_not_duplicate_body_in_immediate_window(self):
        self.append(OperationResult('Reusable finding', (), False, data={'native': 'memory'}), route='memory')
        self.assertIsNone(self.request()[0]['research_agenda'].get('previous_feedback'))

    def test_large_feedback_is_a_literal_utf8_page_with_exact_original_address(self):
        native = OperationResult('Program output', (), False, data={'stdout': '中文\n' * 20000})
        self.append(native)
        payload, _ = self.request()
        view = payload['research_agenda']['previous_feedback']
        self.assertFalse(view['complete'])
        page = view['raw_json_page']
        raw = encoded(native.payload()).encode('utf-8')
        self.assertEqual(page['content'].encode('utf-8'), raw[:page['end_offset']])
        self.assertLessEqual(page['end_offset'], 16384)
        path = dict(view['read_path'])
        path.pop('operation')
        self.assertEqual(path['record_id'], result_id(native.payload()))
        tools = self.harness.tools.research_tools
        text, cursor = '', None
        while True:
            result = tools._archive({**path, 'limit': 32, 'cursor': cursor})
            text += ''.join(item['text'] for item in result.data['items'])
            cursor = result.data['next_cursor']
            if cursor is None:
                break
        self.assertEqual(json.loads(text), json.loads(encoded(native.payload())))


if __name__ == '__main__':
    unittest.main()
