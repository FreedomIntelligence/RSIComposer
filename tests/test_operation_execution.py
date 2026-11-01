"""Decision/review share execution semantics while retaining different scheduling choices."""
from dataclasses import asdict
import unittest
from unittest.mock import Mock

import test_researcher_review as fixtures
from rsisys.loops.contracts.coordination import ReviewDecision
from rsisys.loops.contracts.jobs import JobRef
from rsisys.loops.contracts.lifecycle import ExecutionStatus
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
