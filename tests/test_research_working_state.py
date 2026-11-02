"""Research compact contracts using fixtures, never model-quality claims."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from rsisys.loops._memory import HistoryEntry
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef
from rsisys.loops.decision_making.action_space import OperationSpec
from rsisys.loops.decision_making.harness.context import BasicContextCompactor, ContextBudgetError, encoded, result_id
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.learning_harness import FeedbackLearningHarness
from rsisys.loops.decision_making.harness.learning_policy import MemoryPrefixDecisionPolicy
from rsisys.loops.decision_making.harness.research_state import research_working_view
from rsisys.loops.decision_making.harness.state import DecisionContext
from tests import test_feedback_learning_integration as integration

MODEL = ModelRef('fixture', 'decision', 'v1')
ARCHIVE = OperationSpec('research_archive', 'Read original records', {'type': 'object', 'properties': {
    'collection': {'enum': ['history', 'memories', 'decision_graph']}}}, read_only=True)
READ = OperationSpec('feedback_learning_read', 'Read learning records', {'type': 'object'}, read_only=True)


def job(index=0):
    row = {'job_id': 'job-' + str(index), 'capture_index': index, 'model': asdict(MODEL),
           'status': 'completed', 'simulated': True,
           'input': {'receipt': {'directive': asdict(Directive('run-' + str(index), 'objective', 'Test one change')),
                                'feedback': asdict(Feedback('run-' + str(index), ExecutionStatus.COMPLETED, 'Observed', True))},
                     'source_refs': ['feedback:fixture'], 'research_context': {}},
           'proposals': [{'kind': 'memory', 'content': 'Fixture observation', 'applicability': 'fixture only',
                          'validation_plan': 'independent check', 'source_refs': ['feedback:fixture']}]}
    memory = FeedbackLearningHarness._memory(row)
    row.update(published=True, memory_digest=result_id(asdict(memory)))
    return row, memory


def context():
    history = tuple(HistoryEntry(str(i), 'decision', None, 'old concrete public plan ' * 20, True)
                    for i in range(24))
    return DecisionContext('objective', (), (ARCHIVE, READ), (), history=history,
        decision_graph=tuple({'decision_id': str(i), 'status': 'completed'} for i in range(24)))


class ResearchWorkingStateTests(unittest.TestCase):
    def test_only_exact_learning_memory_duplicates_leave_view_and_archive_is_required(self):
        row, memory = job()
        raw = replace(context(), memories=(memory,))
        saved = deepcopy(raw)
        view = research_working_view(raw, [row], [])
        self.assertEqual(view.memories, ())
        self.assertEqual(view.research_state['findings'][0]['proposals'], row['proposals'])
        self.assertEqual(raw, saved)
        edited = replace(memory, content=memory.content + ' independent correction')
        self.assertEqual(research_working_view(replace(raw, memories=(edited,)), [row], []).memories, (edited,))
        no_archive = replace(raw, operations=(READ,))
        unchanged = research_working_view(no_archive, [row], [])
        self.assertEqual(unchanged.memories, raw.memories)
        self.assertEqual(unchanged.history, raw.history)

    def test_old_decision_chatter_archived_but_evidence_and_live_dependencies_remain(self):
        row, _ = job()
        raw = context()
        feedback = HistoryEntry('negative', 'feedback', 'prior', 'Contradictory measured result', True,
            details={'feedback': {'status': 'completed', 'quality_metrics': {'accuracy': 0.0}}})
        error = HistoryEntry('error', 'operation', None, 'Earlier tool error', True,
            details={'result': {'status': 'failed', 'error': 'validation failed'}})
        graph = (*raw.decision_graph, {'decision_id': 'active', 'status': 'waiting', 'depends_on_decision_ids': ['0']})
        raw = replace(raw, history=(feedback, error, *raw.history), decision_graph=graph,
                      outstanding=(Directive('pending', 'objective', 'Await evidence'),),
                      last_feedback=Feedback('latest', ExecutionStatus.COMPLETED, 'Full latest feedback', True))
        view = research_working_view(raw, [row], [])
        self.assertLess(len(view.history), len(raw.history))
        self.assertEqual(view.history[:3], raw.history[:3])
        self.assertEqual(view.history[-2:], raw.history[-2:])
        self.assertEqual(view.last_feedback, raw.last_feedback)
        self.assertEqual(view.outstanding, raw.outstanding)
        self.assertTrue({'negative', 'error', '0'}.issubset(view.research_state['protected_history_ids']))
        self.assertGreater(view.research_agenda['working_projection']['history']['omitted_count'], 0)

    def test_semantic_prefix_stays_stable_across_unrelated_waits(self):
        row, memory = job()
        raw = replace(context(), memories=(memory,))
        first = research_working_view(raw, [row], [])
        changed = replace(raw, history=(*raw.history, HistoryEntry('tick', 'decision', None, 'Wait', True)),
                          run_control={'seconds_remaining': 10})
        second = research_working_view(changed, [row], [])
        self.assertEqual(first.research_state, second.research_state)
        policy = MemoryPrefixDecisionPolicy(integration.Backend(), MODEL)
        a, b = policy._decision_request(first).prompt, policy._decision_request(second).prompt
        self.assertEqual(a.split(',"experience":')[0], b.split(',"experience":')[0])
        self.assertNotEqual(a, b)

    def test_recent_attempts_follow_capture_order_not_uuid_order(self):
        rows = [job(i)[0] for i in range(12)]
        attempts = [{'attempt_id': 'attempt-' + str(99-i), 'title': 'Idea', 'summary': str(i),
                     'revision': 1, 'member_decision_ids': [], 'source_refs': [], 'simulated': True,
                     'provenance': {'job_id': row['job_id'], 'model': asdict(MODEL), 'input_digest': 'fixture'}}
                    for i, row in enumerate(rows)]
        raw = context()
        a = research_working_view(raw, rows, attempts).research_state
        b = research_working_view(raw, list(reversed(rows)), list(reversed(attempts))).research_state
        self.assertEqual(a, b)
        self.assertEqual([int(r['summary']) for r in a['attempts']], list(range(4, 12)))

    def test_token_compaction_keeps_small_measured_counterevidence_verbatim(self):
        row, _ = job()
        evidence = HistoryEntry('counterexample', 'feedback', 'old', 'Measured negative', True,
            details={'feedback': {'quality_metrics': {'accuracy': 0.0}, 'status': 'completed'}})
        history = (evidence, *(HistoryEntry(str(i), 'operation', None, 'x' * 800, True) for i in range(25)))
        raw = replace(context(), history=history)
        view = research_working_view(raw, [row], [])
        compactor = BasicContextCompactor(trigger_tokens=14000, target_tokens=7000, summary_tokens=200)
        projected = compactor.compact(view, summarize=lambda *args: 'Unverified navigation only')
        self.assertEqual(compactor.compactions, 1)
        self.assertIn(evidence, projected.history)
        self.assertEqual(projected.history[-2:], raw.history[-2:])
        self.assertEqual(projected.research_state, view.research_state)
        self.assertEqual(raw.history[0], evidence)

    def test_oversized_protected_evidence_fails_without_committing_compaction(self):
        row, _ = job()
        evidence = HistoryEntry('huge', 'feedback', None, 'Earlier failure', True,
                                details={'feedback': {'error': 'important failure ' * 1500}})
        view = research_working_view(replace(context(), history=(evidence, *context().history)), [row], [])
        compactor = BasicContextCompactor(trigger_tokens=10000, target_tokens=5000, summary_tokens=200)
        with self.assertRaises(ContextBudgetError):
            compactor.compact(view)
        self.assertEqual(compactor.compactions, 0)

    def test_real_policy_refresh_preserves_state_navigation_and_latest_tool_result(self):
        with TemporaryDirectory() as directory:
            ctx, _, loop = integration.FeedbackLearningIntegrationTests.assembly(self, directory)
            harness = loop.harness
            directive = Directive('fixture', 'objective', 'Concrete action')
            with patch.object(harness.policy, 'decide', return_value=Decision('explore', directive=directive)):
                harness.step('objective', loop.action_space)
            harness.receive_feedback(directive, Feedback('fixture', ExecutionStatus.COMPLETED, 'Raw feedback', True))
            initial = harness._context('objective', loop.action_space)
            requests = []
            def generate(request):
                requests.append(request)
                return GenerationResponse('{"kind":"finish","reason":"fixture done"}', request.model, simulated=True)
            with patch.object(ctx.backends['fixture'], 'generate', side_effect=generate):
                harness._call_policy(harness.policy.decide, initial)
            payload = json.loads(requests[0].prompt)
            self.assertTrue(payload['research_state']['findings'])
            self.assertIn('feedback_learning', payload['research_agenda'])
            self.assertEqual(payload['last_feedback']['summary'], 'Raw feedback')
            self.assertEqual(len(requests), 1)
            self.assertNotIn('Fixture observation', requests[0].system)
            self.assertIsNotNone(harness.last_working_view)
            ctx.close()
