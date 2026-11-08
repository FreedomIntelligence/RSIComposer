"""A new Lab submits native content without implementing RSISys business schemas."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rsisys.adapters.trajectory import FeedbackTrajectorySource
from rsisys.config import build_demo
from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.receipts import ExecutionReceipt, receipt_facts, receipt_task
from rsisys.research import ResearchAssembly, assemble_research
from rsisys.runtime.files import json_bytes
from rsisys.runtime.json_tools import result_id
from tests.feedback.test_feedback_learning_integration import ImmediateExecutor
from tests.feedback.test_feedback_required_outputs import CONFIG, EmptyBackend, exact


class NativeLabOutputTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        runner, _ = build_demo(CONFIG)
        self.addCleanup(runner.close)
        self.backend = EmptyBackend()
        self.service = assemble_research(runner.decision_making,
            assembly=ResearchAssembly(), path=self.root / 'feedback.json',
            backend=self.backend, executor=ImmediateExecutor(), max_input_bytes=8000)

    def submit(self, output=None, **options):
        return self.service.observe_output(output, task_id=options.pop('task_id', 'native'),
            status=options.pop('status', ExecutionStatus.COMPLETED), **options)

    def test_native_text_json_arrays_and_null_need_no_business_schema(self):
        values = ['stdout\n错误\n', {'custom': [1, False, None]}, [1, {'answer': 'a'}], 7, None]
        for i, output in enumerate(values):
            ident = self.submit(output, task_id=str(i))
            row = self.service.learning.state['jobs'][ident]
            self.assertEqual(row['raw_input']['receipt']['result']['output'], output)
            self.assertEqual(row['status'], 'queued')
            self.assertIsNone(row.get('request'))
        self.assertEqual(self.backend.requests, [])

    def test_native_field_names_cannot_override_host_status_metrics_or_identity(self):
        output = {'status': 'failed', 'summary': {'free': 'shape'},
                  'quality_metrics': {'accuracy': 'native string'},
                  'candidate_id': 'native candidate', 'task_id': 'another task',
                  'adopted_candidate_ids': ['not-an-adoption']}
        ident = self.submit(output)
        receipt = self.service.learning.state['jobs'][ident]['raw_input']['receipt']
        self.assertEqual(receipt['result']['output'], output)
        facts = receipt_facts(receipt)
        self.assertEqual(facts['status'], ExecutionStatus.COMPLETED)
        self.assertEqual(facts['quality_metrics'], {})
        self.assertEqual(facts['adopted_candidate_ids'], ())
        self.assertEqual(receipt_task(receipt)['task_id'], 'native')
        self.assertIsNone(self.service.learning.state['jobs'][ident]['input'].get('candidate_evidence'))

    def test_failed_native_task_needs_no_separate_error_or_summary_from_lab(self):
        output = 'Traceback from an external worker with its own format'
        ident = self.submit(output, status=ExecutionStatus.FAILED)
        self.service.refresh()
        receipt = self.service.learning.state['jobs'][ident]['raw_input']['receipt']
        self.assertEqual(receipt_facts(receipt)['status'], ExecutionStatus.FAILED)
        self.assertIsNone(receipt_facts(receipt)['error'])
        self.assertEqual(json.loads(self.service.context())['total'], 0)
        self.assertEqual(exact(self.service.learning, job_id=ident,
            evidence_path='/receipt/result/output'), output)
        self.assertEqual(self.backend.requests, [])

    def test_reference_only_output_keeps_native_reference_format(self):
        references = [{'producer_url': 'lab://artifacts/log.bin', 'native_metadata': {'encoding': 'binary'}}]
        ident = self.submit(references=references, lab_id='external-lab')
        raw = self.service.trajectory_snapshot(ident)['capture']['receipt']
        self.assertIsNone(raw['result']['output'])
        self.assertEqual(raw['result']['references'], references)
        self.assertEqual(raw['identity']['lab_id'], 'external-lab')

    def test_related_evidence_and_default_harness_catalog_are_not_required(self):
        def unexpected_provider(receipt):
            raise AssertionError('Native input must not request a domain catalog')
        self.service.adapter.receipt_evidence = unexpected_provider
        ident = self.submit('raw stdout')
        raw = self.service.trajectory_snapshot(ident)['capture']
        self.assertEqual(raw['history'], [])
        self.assertEqual(raw['evaluations'], [])
        self.assertNotIn('grouping_context', raw)

    def test_large_unknown_fields_have_bounded_views_and_exact_readback(self):
        output = {'arbitrary_metric_ids': ['opaque-' + str(i) for i in range(6000)],
                  'producer_specific': '原始日志' * 12000}
        ident = self.submit(output, request={'native_input': 'x' * 20000})
        row = self.service.learning.state['jobs'][ident]
        self.assertEqual(row['status'], 'queued')
        self.assertLessEqual(len(json_bytes(row['input'])), 8000)
        view = row['input']['receipt']['result']['output']
        self.assertEqual(view['representation'], 'json_preview')
        self.assertFalse(view['complete'])
        self.assertEqual(exact(self.service.learning, job_id=ident,
            evidence_path=view['read']['evidence_path']), output)
        self.assertEqual(exact(self.service.learning, job_id=ident,
            evidence_path='/receipt/operation/input'), {'native_input': 'x' * 20000})

    def test_identity_and_repeat_delivery_are_checked_without_reinterpreting_payload(self):
        output = {'native_result': ['ok']}
        ident = self.submit(output, lab_id='a', run_id='run', decision_id='decision')
        source = FeedbackTrajectorySource(self.service, archive_id='run')
        reference = source.reference(ident)
        self.assertEqual(self.submit(deepcopy(output), lab_id='a', run_id='run',
                                    decision_id='decision'), ident)
        self.service.refresh()
        self.assertEqual(source.reference(ident), reference)
        before = result_id(self.service.learning.state)
        with self.assertRaisesRegex(ValueError, 'receipt changed'):
            self.submit({'native_result': 'different'}, lab_id='a', run_id='run', decision_id='decision')
        self.assertEqual(result_id(self.service.learning.state), before)
        other = self.submit(output, lab_id='b', run_id='run')
        self.assertNotEqual(other, ident)

    def test_native_source_does_not_collide_with_existing_operation_task(self):
        native = self.submit('stdout')
        receipt = ExecutionReceipt.from_operation('native', '', None, OperationResult('done', (), False))
        old = self.service.observe(receipt)
        self.assertNotEqual(native, old)
        self.assertEqual(len(self.service.learning.state['jobs']), 2)

    def test_executor_metadata_remains_validated_and_input_is_detached(self):
        output = {'data': ['original']}
        ident = self.submit(output)
        output['data'].clear()
        self.assertEqual(self.service.learning.state['jobs'][ident]
                         ['raw_input']['receipt']['result']['output'], {'data': ['original']})
        for changes in ({'task_id': ''}, {'status': 'running'}, {'simulated': 'false'}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                self.submit('text', **changes)

    def test_old_typed_receipts_still_validate_their_own_business_contract(self):
        with self.assertRaisesRegex(ValueError, 'metrics'):
            ExecutionReceipt('typed', '', 'operation', {}, {
                'status': ExecutionStatus.COMPLETED, 'summary': '', 'simulated': False,
                'error': None, 'data': {'quality_metrics': {'score': 'bad'}}})


if __name__ == '__main__':
    unittest.main()
