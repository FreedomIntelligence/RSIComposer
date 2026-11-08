"""Archived failure residency and explicit Ascend deployment boundaries."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from rsisys.loops._memory import HistoryEntry
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.decision_making.action_space import OperationResult
from rsisys.loops.decision_making.harness.context import BasicContextCompactor, encoded, result_id
from rsisys.loops.decision_making.harness.research_state import (
    bounded_evidence_view, resident_evidence_indices, research_working_view,
)
from tests.decision.test_research_evidence_projection import fixture, measure, read_original


class ArchivedFailureResidencyTests(unittest.TestCase):
    def failures(self):
        context, tools, memory, _ = fixture(count=1)
        receipt = OperationResult('Program failed', (), False, ExecutionStatus.FAILED,
            'exact failure: ' + 'x' * 1900, {'job_id': 'failed-program', 'stderr': 's' * 1500})
        rows = tuple(HistoryEntry('failed-' + str(i), 'operation', None, receipt.summary, False,
            details={'operation': {'name': 'lab_start', 'arguments': {'code': 'c' * 12000}},
                     'result': receipt.payload()}) for i in range(24))
        for row in rows:
            memory.append(row)
        context = replace(context, history=rows, operation_results=(receipt,) * len(rows),
                          experience=(), last_feedback=None, research_state={})
        return context, tools, receipt

    def test_projecting_history_does_not_permanently_pin_repeated_failures(self):
        raw, tools, receipt = self.failures()
        saved = deepcopy(raw)
        projected = bounded_evidence_view(raw)
        self.assertIn('evidence_card', projected.history[0].details)
        self.assertEqual(resident_evidence_indices(projected, 'operation_results'), {22, 23})
        card = projected.history[0].details['evidence_card']
        self.assertEqual(card['operation_result_sha256'], result_id(receipt.payload()))
        original = read_original(tools, card)
        self.assertEqual(original['details']['result'], json.loads(encoded(receipt.payload())))
        working = research_working_view(projected, [], [])
        compactor = BasicContextCompactor(trigger_tokens=80000, target_tokens=24000, summary_tokens=1000)
        view = compactor.compact(working, measure=measure,
                                 summarize=lambda *args: 'Repeated failed receipts remain archived failures.')
        self.assertLessEqual(measure(view), 24000)
        self.assertLessEqual(len(view.operation_results), 2)
        self.assertEqual(view.operation_results[-1], receipt)
        self.assertEqual(raw, saved)

    def test_missing_or_mismatched_archive_identity_keeps_unrecorded_failures(self):
        raw, _, _ = self.failures()
        projected = bounded_evidence_view(raw)
        for mode in ('legacy', 'wrong-record', 'malformed-digest'):
            rows = []
            for row in projected.history:
                details = deepcopy(row.details)
                card = details['evidence_card']
                if mode == 'legacy':
                    card.pop('operation_result_sha256', None)
                elif mode == 'wrong-record':
                    card['read_path']['record_id'] = 'other-record'
                else:
                    card['operation_result_sha256'] = 'not-a-digest'
                rows.append(replace(row, details=details))
            with self.subTest(mode=mode):
                self.assertEqual(resident_evidence_indices(replace(projected, history=tuple(rows)),
                                                          'operation_results'), set(range(24)))


class AscendDeploymentTests(unittest.TestCase):
    def test_generated_child_receives_only_explicit_npu_mapping(self):
        from experiments.medical.open_lab_experiment import AllocatedProcessBackend
        from rsisys.runtime.sandbox import ExecRequest, SandboxSpec
        with TemporaryDirectory() as directory:
            backend = AllocatedProcessBackend(directory, {'ASCEND_RT_VISIBLE_DEVICES': '13'}, device_type='npu')
            try:
                session = backend.create(SandboxSpec('local', 5))
                ref = backend.exec(session, ExecRequest((sys.executable, '-c',
                    'import os,json; print(json.dumps({k:os.environ.get(k) for k in '
                    '["ASCEND_RT_VISIBLE_DEVICES","CUDA_VISIBLE_DEVICES"]}))'), 5))
                state = backend.wait(ref)
                self.assertEqual(json.loads(state.stdout), {'ASCEND_RT_VISIBLE_DEVICES': '13',
                                                           'CUDA_VISIBLE_DEVICES': None})
            finally:
                backend.close_all()

    def test_npu_backend_requires_npu_mapping_even_with_cuda_present(self):
        from experiments.medical.open_lab_experiment import AllocatedProcessBackend
        with TemporaryDirectory() as directory, self.assertRaises(ValueError):
            AllocatedProcessBackend(directory, {'CUDA_VISIBLE_DEVICES': '0'}, device_type='npu')

    def test_provider_token_counter_bypasses_incompatible_local_templates(self):
        from experiments.medical.open_lab_experiment import RecordedBackend
        from tests.experiments.medical.test_medical_open_lab import Transport, Tokenizer, MODEL
        from rsisys.loops.contracts.model_service import GenerationRequest
        class CountedTransport(Transport):
            def count_tokens(self, request):
                return 1234
        with TemporaryDirectory() as directory:
            backend = RecordedBackend(CountedTransport(), Tokenizer(), directory, max_calls=1,
                                      provider_token_counter=True)
            self.assertEqual(backend.count_tokens(GenerationRequest('decision_maker', MODEL, 'observe')), 1234)
            self.assertEqual(backend.calls, 0)
            self.assertEqual(backend.token_count_mode, 'provider_tokenizer')


class FeedbackRoutingTests(unittest.TestCase):
    def test_context_collection_does_not_consume_later_explicit_memory_organization(self):
        from rsisys.adapters.local_lab_feedback import ContextOrMemoryAdmission
        from rsisys.loops.decision_making.feedback.learning import FeedbackLearning
        from rsisys.loops.decision_making.feedback.context_memory import CallbackContextMemory
        from rsisys.loops.contracts.receipts import ExecutionReceipt
        from rsisys.loops.contracts.delivery import FeedbackDelivery
        from tests.feedback.test_feedback_required_outputs import MODEL
        from tests.feedback.test_feedback_learning_integration import ImmediateExecutor
        learning = FeedbackLearning(None, context_memory=CallbackContextMemory(lambda query: ''),
            processor=lambda event, model: {}, admission=ContextOrMemoryAdmission(), executor=ImmediateExecutor())
        self.addCleanup(learning.close)
        ids = []
        for i, route in enumerate(('context', 'memory', 'memory')):
            receipt = ExecutionReceipt.from_output(str(i), {'result': 'one immutable observation'},
                status=ExecutionStatus.COMPLETED, lab_id='lab', run_id='run',
                request={'name': 'lab_result', 'arguments': {'feedback_route': route}},
                feedback_delivery=FeedbackDelivery('one-owned-result'))
            ids.append(learning.capture_receipt(receipt, MODEL))
        self.assertEqual([learning.state['jobs'][i]['status'] for i in ids], ['archived', 'queued', 'archived'])
        self.assertEqual(learning.state['jobs'][ids[2]]['duplicate_of'], ids[1])
        self.assertNotIn('duplicate_of', learning.state['jobs'][ids[1]])
        self.assertEqual(len(learning.trajectories()), 3)

    def test_context_route_archives_without_organizing_and_memory_route_queues(self):
        from rsisys.adapters.local_lab_feedback import ContextOrMemoryAdmission
        from rsisys.loops.decision_making.feedback.learning import FeedbackLearning
        from rsisys.loops.decision_making.feedback.context_memory import CallbackContextMemory
        from tests.feedback.test_feedback_required_outputs import MODEL, observation
        from tests.feedback.test_feedback_learning_integration import ImmediateExecutor
        learning = FeedbackLearning(None, context_memory=CallbackContextMemory(lambda query: 'existing memory'),
            processor=lambda event, model: {}, admission=ContextOrMemoryAdmission(), executor=ImmediateExecutor())
        self.addCleanup(learning.close)
        for route in ('context', 'memory'):
            event = replace(observation(route), request={'name': 'lab_result',
                'arguments': {'job_id': route, 'feedback_route': route}})
            ident = learning.capture_receipt(event, MODEL)
            self.assertEqual(learning.state['jobs'][ident]['status'], 'archived' if route == 'context' else 'queued')
        self.assertEqual(len(learning.trajectories()), 2)
        self.assertEqual(learning.context(), 'existing memory')
        self.assertEqual(learning.state['attempted'], 0)

    def test_instant_history_expires_and_memory_body_is_replaced_only_in_view(self):
        from rsisys.adapters.local_lab_feedback import ImmediateFeedbackWindow
        raw, tools, receipt = ArchivedFailureResidencyTests().failures()
        memory_receipt = replace(receipt, data={'job_id': 'memory-program', 'content': 'long-term finding'})
        memory_row = HistoryEntry('remember', 'operation', None, memory_receipt.summary, False,
            details={'operation': {'name': 'lab_result', 'arguments': {'feedback_route': 'memory'}},
                     'result': memory_receipt.payload()})
        raw = replace(raw, history=(*raw.history, memory_row),
                      operation_results=(*raw.operation_results, memory_receipt))
        saved = deepcopy(raw)
        view = ImmediateFeedbackWindow(recent=2)(bounded_evidence_view(raw))
        self.assertEqual([h.record_id for h in view.history], ['failed-23', 'remember'])
        self.assertEqual(len(view.operation_results), 2)
        self.assertEqual(view.operation_results[-1].data['feedback_route'], 'memory')
        self.assertNotIn('long-term finding', encoded(view.operation_results[-1].payload()))
        self.assertEqual(raw, saved)
        # Without an archive the lifecycle adapter must preserve exact evidence.
        self.assertEqual(ImmediateFeedbackWindow()(replace(raw, operations=())), replace(raw, operations=()))

    def test_routing_hint_never_reaches_native_lab_or_changes_result(self):
        from rsisys.adapters.local_lab_feedback import RoutedLab
        from rsisys.loops.contracts.decision import OperationCall
        from rsisys.loops.contracts.execution import LabToolError
        class Native:
            def execute(self, call):
                self.call = call
                return OperationResult('exact native result', (), False, data={'content': 'observation'})
        native = Native()
        routed = RoutedLab(native)
        call = OperationCall('lab_result', {'job_id': 'one', 'feedback_route': 'memory'})
        receipt = routed.execute(call)
        self.assertEqual(native.call.arguments, {'job_id': 'one'})
        self.assertEqual(receipt.data, {'content': 'observation'})
        self.assertEqual(call.arguments['feedback_route'], 'memory')
        with self.assertRaises(LabToolError):
            routed.execute(OperationCall('lab_result', {'job_id': 'one', 'feedback_route': 'invalid'}))
