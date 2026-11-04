"""Invalid model-selected evidence paths are observations, not fatal loop errors."""
from copy import deepcopy
from dataclasses import replace
import json
import unittest
from unittest.mock import Mock

from rsisys.loops._memory import HistoryEntry
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.decision import OperationArgumentError
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.decision_making.action_space import ModularDecisionActionSpace, OperationCall, OperationResult
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.learning_harness import LearningTools
from rsisys.loops.decision_making.harness.feedback_learning import FeedbackLearning, SerializedLearningBackend
from rsisys.loops.decision_making.harness.research_state import bounded_evidence_view
from rsisys.loops.decision_making.harness.tools import BoundDecisionTools
from rsisys.runtime.json_tools import EvidencePathError, evidence_value, result_id
from tests.test_feedback_learning_evidence import FixtureBackend, ImmediateExecutor
from tests.test_latest_evidence_view import archive_text
from tests.test_research_evidence_projection import fixture


class EvidencePathRecoveryTests(unittest.TestCase):
    def test_json_pointer_exact_keys_and_array_bounds(self):
        value = {'a/b': {'~': [{'': 0}]}, '01': 'object key'}
        self.assertIs(evidence_value(value, ''), value)
        self.assertEqual(evidence_value(value, '/a~1b/~0/0/'), 0)
        self.assertEqual(evidence_value(value, '/01'), 'object key')
        for path in ('/missing', '/a~1b/~0/1', '/a~1b/~0/-1', '/a~1b/~0/01',
                     '/a~1b/~0/0//deeper', '/~2', '/trailing~', 'not/a/pointer'):
            with self.subTest(path=path), self.assertRaises(EvidencePathError):
                evidence_value(value, path)

    def test_invalid_archive_path_is_recorded_and_next_decision_can_correct_it(self):
        raw, provider, memory, experience = fixture(count=0)
        original = OperationResult('Training page', (), False, data={'rows': [{'id': 'row-zero'}]})
        memory.append(HistoryEntry('preview', 'operation', None, original.summary, False,
                                   details={'result': original.payload()}))
        args = {'collection': 'operation_results', 'record_id': result_id(original.payload()),
                'evidence_path': '/data/rows/0', 'limit': 1}
        policy = Mock()
        policy.decide.side_effect = [Decision('invoke', operation=OperationCall('research_archive', args)),
            Decision('invoke', operation=OperationCall('research_archive', {**args, 'evidence_path': '/result/data/rows/0'}))]
        harness = DefaultDecisionHarness(policy, BoundDecisionTools(provider.bindings), experience, memory=memory)
        space = ModularDecisionActionSpace(provider.definitions)
        first = harness.step('Inspect training evidence', space)
        self.assertEqual(first.kind, 'progressed')
        self.assertEqual(first.operation_result.status, ExecutionStatus.FAILED)
        self.assertEqual(first.operation_result.data['root_fields'], ['record_id', 'result', 'history_refs'])
        self.assertIn('Evidence path', first.operation_result.error)
        second = harness.step('Inspect training evidence', space)
        self.assertEqual(second.operation_result.status, ExecutionStatus.COMPLETED)
        self.assertEqual(json.loads(second.operation_result.data['items'][0]['text']), {'id': 'row-zero'})
        receipts = memory.history(kind='operation')[-2:]
        self.assertEqual([r.details['result']['status'] for r in receipts], ['failed', 'completed'])
        self.assertEqual(memory.history(kind='operation')[2].details['result'], original.payload())

    def test_projected_tool_card_has_exact_receipt_root_and_rejects_missing_elements(self):
        raw, provider, memory, _ = fixture(count=0)
        result = OperationResult('Training page', (), False,
            data={'rows': [{'id': str(i), 'text': 'bulk ' * 100} for i in range(32)]})
        memory.append(HistoryEntry('oversized', 'operation', None, result.summary, False,
                                   details={'result': result.payload()}))
        view = bounded_evidence_view(replace(raw, history=memory.history(), operation_results=(result,)))
        path = view.operation_results[0].data['evidence_card']['read_path']
        self.assertEqual(path['evidence_path'], '/result')
        args = {k: v for k, v in path.items() if k != 'operation'}
        self.assertEqual(archive_text(provider, **args)['data'], result.data)
        before = memory.history()
        for pointer in ('/result/data/rows/32', '/result/data/rows/-1', '/result/data/missing'):
            with self.subTest(pointer=pointer):
                receipt = provider._archive({**args, 'evidence_path': pointer})
                self.assertEqual(receipt.status, ExecutionStatus.FAILED)
        self.assertEqual(memory.history(), before)

    def test_feedback_bad_path_and_unknown_job_do_not_poison_next_read(self):
        learning = FeedbackLearning(SerializedLearningBackend(FixtureBackend()), executor=ImmediateExecutor())
        self.addCleanup(learning.close)
        ident = learning.capture(Directive('receipt', 'objective', 'test'),
            Feedback('receipt', ExecutionStatus.COMPLETED, 'Measured', False, quality_metrics={'accuracy': 0.0}),
            ModelRef('fixture', 'model', 'v1'))
        before = deepcopy(learning.state)
        tools = LearningTools(None, learning)
        for job, pointer in ((ident, '/receipt/missing'), (ident, '/history/123'), ('missing', '')):
            with self.subTest(job=job, pointer=pointer):
                result = tools.execute(OperationCall('feedback_learning_read', {'job_id': job, 'evidence_path': pointer}))
                self.assertEqual(result.status, ExecutionStatus.FAILED)
        result = tools.execute(OperationCall('feedback_learning_read', {
            'job_id': ident, 'evidence_path': '/receipt/feedback/quality_metrics'}))
        self.assertEqual(result.status, ExecutionStatus.COMPLETED)
        self.assertEqual(json.loads(result.data['text']), {'accuracy': 0.0})
        self.assertEqual(learning.state, before)

    def test_feedback_internal_fault_is_not_disguised_as_a_bad_path(self):
        learning = Mock()
        for error in (KeyError('broken internal state'), ValueError('identity'), PermissionError('denied'),
                      OSError('storage'), OperationArgumentError('unexpected after validation')):
            learning.read.side_effect = error
            with self.subTest(error=error), self.assertRaises(type(error)):
                LearningTools(None, learning).execute(OperationCall('feedback_learning_read', {'job_id': 'known'}))

    def test_direct_feedback_read_rejects_bad_arguments_before_accessing_storage(self):
        learning = Mock()
        tools = LearningTools(None, learning)
        for arguments in ({'view': 'jobs', 'target': 'context_memory'},
                          {'view': 'trajectories', 'job_id': 'known'},
                          {'limit': 0}, {'offset': -1}, {'evidence_path': '/receipt'}):
            with self.subTest(arguments=arguments):
                result = tools.execute(OperationCall('feedback_learning_read', arguments))
                self.assertEqual(result.status, ExecutionStatus.FAILED)
        learning.read.assert_not_called()
