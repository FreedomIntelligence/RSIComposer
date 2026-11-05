"""Latest observations are bounded views, with exact archive round trips."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import unittest

from rsisys.loops._memory import HistoryEntry
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.decision_making.action_space import OperationResult
from rsisys.loops.decision_making.harness.research_state import bounded_evidence_view
from rsisys.loops.decision_making.harness.context import BasicContextCompactor, encoded, result_id
from tests.test_research_evidence_projection import fixture, METRICS


def archive_text(tools, **args):
    texts, cursor = [], None
    while True:
        page = tools._archive({**args, 'cursor': cursor, 'limit': 32})
        # JSON escaping and multi-byte text are counted, not just characters.
        assert len(encoded(asdict(page)).encode('utf-8')) < 8192
        texts.extend(item['text'] for item in page.data['items'])
        cursor = page.data['next_cursor']
        if cursor is None:
            return json.loads(''.join(texts))


class LatestEvidenceTests(unittest.TestCase):
    def test_latest_tool_rows_and_error_are_bounded_without_losing_metrics_or_archive(self):
        raw, tools, memory, _ = fixture(count=0)
        error = 'BackendTimeout: ' + '诊断信息' * 30000 + '\nlast traceback line'
        result = OperationResult('Evaluation failed after partial measurement. ' * 1000, (), False,
            status=ExecutionStatus.FAILED, error=error, data={
                'quality_metrics': METRICS, 'model': {'backend_id': 'npu', 'model_id': 'target', 'revision': 'v3'},
                'error_type': 'BackendTimeout', 'protocol_ref': 'protocol-v1',
                'items': [{'sample_id': str(i), 'quality_metrics': {'accuracy': 0.0}} for i in range(4000)]})
        history = HistoryEntry('big-tool', 'operation', None, result.summary, False,
                               details={'result': result.payload()})
        memory.append(history)
        raw = replace(raw, history=memory.history(), operation_results=(result,))
        original = deepcopy(raw)
        view = bounded_evidence_view(raw)
        visible = view.operation_results[0]
        self.assertLess(len(encoded(asdict(visible)).encode()), 8192)
        self.assertEqual(visible.status, ExecutionStatus.FAILED)
        self.assertIn('BackendTimeout:', visible.error)
        self.assertIn('last traceback line', visible.error)
        self.assertEqual(visible.data['facts']['/data/quality_metrics'], METRICS)
        self.assertEqual(visible.data['facts']['/data/error_type'], 'BackendTimeout')
        omitted = visible.data['evidence_card']['omissions']
        self.assertIn('/data/items', [item['path'] for item in omitted])
        card = visible.data['evidence_card']
        self.assertEqual(card['sha256'], result_id(result.payload()))
        self.assertEqual(archive_text(tools, collection='operation_results', record_id=card['read_path']['record_id'],
                                     evidence_path='/result/data/items/3999'), result.data['items'][-1])
        self.assertEqual(archive_text(tools, collection='operation_results', record_id=card['read_path']['record_id'],
                                     evidence_path='/result/error'), error)
        self.assertEqual(raw, original)
        self.assertEqual(bounded_evidence_view(view), view)
        self.assertEqual(bounded_evidence_view(replace(raw, history=())).operation_results, (result,))

    def test_large_keyed_table_keeps_aggregate_metrics_and_model_identity(self):
        raw, _, _, _ = fixture(count=0)
        data = {f'row-{i}': {'value': i} for i in range(2000)}
        data.update(quality_metrics=METRICS, error_type='PartialFailure',
                    model={'backend_id': 'fixture', 'model_id': 'actor', 'revision': 'v1'},
                    evidence_card='untrusted domain field, not a host card')
        result = OperationResult('Measured keyed results', (), False, data=data)
        entry = HistoryEntry('keyed', 'operation', None, result.summary, False, details={'result': result.payload()})
        raw = replace(raw, history=(*raw.history, entry), operation_results=(result,))
        view = bounded_evidence_view(raw)
        self.assertEqual(bounded_evidence_view(replace(raw, history=())).operation_results, (result,))
        facts = view.operation_results[0].data['facts']
        self.assertEqual(facts['/data/quality_metrics'], METRICS)
        self.assertEqual(facts['/data/model'], data['model'])
        self.assertEqual(facts['/data/error_type'], 'PartialFailure')
        self.assertLess(len(encoded(asdict(view.operation_results[0])).encode()), 8192)

    def test_latest_feedback_has_exact_identity_metrics_and_retrievable_prose(self):
        raw, tools, memory, _ = fixture(count=0)
        feedback = Feedback('big-feedback', ExecutionStatus.FAILED, 'qualified observation ' * 10000,
            False, quality_metrics=METRICS, error='TimeoutError: ' + 'trace ' * 30000,
            evidence_refs=('evidence-original',))
        entry = HistoryEntry('big-feedback-history', 'feedback', feedback.directive_id, feedback.summary,
                             False, details={'feedback': asdict(feedback)})
        memory.append(entry)
        raw = replace(raw, last_feedback=feedback, history=memory.history(), operation_results=(), experience=())
        view = bounded_evidence_view(raw)
        self.assertLess(len(encoded(asdict(view)).encode()), 16000)
        self.assertEqual(view.last_feedback.quality_metrics, METRICS)
        self.assertEqual(view.last_feedback.evidence_refs, feedback.evidence_refs)
        self.assertEqual(view.last_feedback.directive_id, feedback.directive_id)
        descriptor = json.loads(view.last_feedback.summary)['evidence_view']
        args = {k: v for k, v in descriptor['read_path'].items() if k != 'operation'}
        self.assertEqual(archive_text(tools, **args), json.loads(encoded(asdict(feedback))))
        self.assertEqual(bounded_evidence_view(view), view)
        self.assertEqual(raw.last_feedback, feedback)
        # A live feedback with no exact archived receipt cannot be truncated.
        self.assertEqual(bounded_evidence_view(replace(raw, history=())).last_feedback, feedback)
        self.assertEqual(bounded_evidence_view(replace(raw, operations=())).last_feedback, feedback)

    def test_archive_pages_are_readable_and_cursor_cannot_switch_fields(self):
        raw, tools, memory, _ = fixture(count=0)
        entry = HistoryEntry('unicode', 'operation', None, '😀\\\"' * 10000, False,
                             details={'other': 'different field ' * 10000})
        memory.append(entry)
        args = {'collection': 'history', 'record_id': entry.record_id, 'evidence_path': '/summary'}
        first = tools._archive({**args, 'limit': 32})
        self.assertIsNotNone(first.data['next_cursor'])
        self.assertEqual(archive_text(tools, **args), entry.summary)
        view = bounded_evidence_view(replace(raw, operation_results=(first,)))
        self.assertEqual(view.operation_results[0], first)
        rejected = tools._archive({**args, 'evidence_path': '/details/other', 'cursor': first.data['next_cursor']})
        self.assertEqual(rejected.status, ExecutionStatus.FAILED)
        self.assertNotIn('items', rejected.data)
        self.assertIsNone(rejected.data['read_path']['cursor'])
        with self.assertRaises(ValueError):
            tools._archive({'collection': 'history', 'evidence_path': '/summary'})

    def test_default_and_dsh_share_input_projection_without_replacing_compactors(self):
        from rsisys.config import build_demo
        from rsisys.loops.decision_making.harness.dsh import dsh_decision_factory
        from unittest.mock import patch
        from tests.test_dsh_harness import Runtime, DSH, CONFIG
        raw, _, _, _ = fixture(count=0)
        huge = OperationResult('latest', (), False, data={'text': 'bulk ' * 40000, 'quality_metrics': METRICS})
        recorded = HistoryEntry('huge-op', 'operation', None, huge.summary, False,
                                details={'result': huge.payload()})
        raw = replace(raw, operation_results=(huge,), history=(*raw.history, recorded))
        for factory in (None, dsh_decision_factory(DSH)):
            with patch('rsisys.loops.decision_making.harness.dsh._sdk_runtime', return_value=Runtime([])):
                runner, _ = build_demo(CONFIG, decision_factory=factory)
            try:
                harness = runner.decision_making.harness
                compactor = harness.context_compactor
                view = harness._project_context(raw)
                self.assertIs(harness.context_compactor, compactor)
                self.assertLess(len(encoded(asdict(view.operation_results[0]))), 8192)
                self.assertEqual(view.operation_results[0].data['facts']['/data/quality_metrics'], METRICS)
            finally:
                runner.close()
