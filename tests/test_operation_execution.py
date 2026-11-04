"""Decision/review share execution semantics while retaining different scheduling choices."""
from dataclasses import asdict
import unittest
from unittest.mock import Mock

import test_researcher_review as fixtures
from rsisys.loops.contracts.coordination import ReviewDecision
from rsisys.loops.contracts.jobs import JobRef
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.decision import OperationArgumentError
from rsisys.loops.decision_making.action_space import (
    ModularDecisionActionSpace, OperationCall, OperationResult, OperationSpec,
)
from rsisys.loops.decision_making.action_space.access import ActionPermissions, authorized_space
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.tools import execute_operation


class Normalize:
    spec = OperationSpec('normalize', 'Normalize a string', {'type': 'object'})

    def validate(self, arguments):
        return OperationCall(self.spec.name, {'text': arguments['text'].strip()})


class SharedOperationExecutionTests(unittest.TestCase):
    def test_only_explicit_validator_argument_rejections_are_recoverable_without_execution(self):
        call = OperationCall('inspect', {'view': 'jobs', 'target': 'context_memory'})
        tools = Mock()
        for error in (OperationArgumentError('target requires a proposals view'),
                      ValueError('identity conflict'), PermissionError('denied'), OSError('store failed')):
            with self.subTest(error=error):
                space = Mock(validate=Mock(side_effect=error))
                result = execute_operation(tools, space, call, simulated=True)
                self.assertEqual(result.observation().status, ExecutionStatus.FAILED)
                self.assertTrue(result.observation().simulated)
                self.assertEqual(result.call, call)
                if isinstance(error, OperationArgumentError):
                    self.assertIsNone(result.error)
                else:
                    self.assertIs(result.error, error)
        tools.execute.assert_not_called()

    def test_argument_exception_from_executor_does_not_hide_an_internal_fault(self):
        call = OperationCall('inspect', {})
        error = OperationArgumentError('unexpected after validation')
        result = execute_operation(Mock(execute=Mock(side_effect=error)),
                                   Mock(validate=Mock(return_value=call)), call)
        self.assertIs(result.error, error)
        self.assertEqual(result.observation().status, ExecutionStatus.FAILED)

    def test_both_entry_points_record_the_same_normalized_result_and_simulation(self):
        saved = []
        for phase in ('decision', 'review'):
            f = fixtures.ResearcherReviewTests()
            f.setUp()
            f.loop.action_space = ModularDecisionActionSpace((Normalize(),),
                permissions=ActionPermissions.unrestricted())
            f.harness.tools = Mock()
            def execute(call):
                self.assertEqual(call.arguments, {'text': 'clean'})
                call.arguments['text'] = 'tool mutation'
                return OperationResult('clean result', (), False, data={'nested': [1]})
            f.harness.tools.execute.side_effect = execute
            decision = Decision('invoke', operation=OperationCall('normalize', {'text': ' clean '}), simulated=True)
            if phase == 'decision':
                f.harness.state.pending_directive = None
                f.harness.state.pending_directive_id = None
                f.policy.decide = Mock(return_value=decision)
                f.loop.step('task')
            else:
                f.policy.review = Mock(side_effect=[decision, ReviewDecision()])
                self.assertTrue(f.loop.review('task', f.progress).simulated)
            entry = f.harness.memory.history(kind='operation')[0]
            self.assertEqual(entry.details['operation']['arguments'], {'text': 'clean'})
            self.assertTrue(entry.details['result']['simulated'])
            saved.append(entry.details['result'])
            f.harness.tools.execute.assert_called_once()
        self.assertEqual(saved[0], saved[1])

    def test_invalid_custom_tool_result_is_failed_evidence_in_both_phases(self):
        saved = []
        for phase in ('decision', 'review'):
            f = fixtures.ResearcherReviewTests()
            f.setUp()
            f.harness.tools = Mock(execute=Mock(return_value=object()))
            decision = Decision('invoke', operation=OperationCall('inspect', {}))
            if phase == 'decision':
                f.harness.state.pending_directive = None
                f.harness.state.pending_directive_id = None
                f.policy.decide = Mock(return_value=decision)
                with self.assertRaisesRegex(TypeError, 'OperationResult or JobRef'):
                    f.loop.step('task')
            else:
                f.policy.review = Mock(side_effect=[decision, ReviewDecision()])
                f.loop.review('task', f.progress)
            result = f.harness.memory.history(kind='operation')[0].details['result']
            self.assertEqual(result['status'], ExecutionStatus.FAILED)
            saved.append(result)
        self.assertEqual(saved[0], saved[1])

    def test_job_receipt_retains_job_and_reports_submission_not_completion(self):
        space = authorized_space(ModularDecisionActionSpace((Normalize(),)))
        job = JobRef('external-job', 'provider')
        receipt = execute_operation(Mock(execute=Mock(return_value=job)), space,
                                    OperationCall('normalize', {'text': ' next '}), simulated=True)
        self.assertEqual(receipt.outcome, job)
        self.assertIsNone(receipt.error)
        self.assertEqual(receipt.call.arguments, {'text': 'next'})
        self.assertEqual(receipt.observation().data, {'job': asdict(job), 'status': 'submitted'})
        self.assertTrue(receipt.observation().simulated)


if __name__ == '__main__':
    unittest.main()
