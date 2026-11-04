"""Research knowledge is bounded, journal-derived and usable by any harness."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rsisys.loops.decision_making.feedback import WorkingMemoryPolicy
from rsisys.loops.decision_making.feedback.knowledge import proposal_records, working_memory
from rsisys.loops.decision_making.feedback.learning import FeedbackLearning, SerializedLearningBackend
from rsisys.loops.decision_making.feedback.learning import validate_proposals
from rsisys.runtime.json_tools import encoded
from tests.test_research_working_state import job
from tests import test_attempt_learning as attempt_tests
from tests.test_feedback_learning_integration import ImmediateExecutor
from tests.test_feedback_decoupling import AlienHarness, AlienAdapter
from rsisys.loops.decision_making.harness.learning_harness import enable_feedback_learning, LearningTools
from rsisys.loops.decision_making.action_space import OperationCall
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from tests.test_feedback_learning_integration import Backend as SimpleBackend


def attempt(row, ident, members=(), summary='Latest observation'):
    return {'attempt_id': ident, 'title': 'Same title', 'summary': summary,
        'revision': 2, 'member_decision_ids': list(members), 'source_refs': ['decision:' + m for m in members],
        'simulated': True, 'provenance': {'job_id': row['job_id']}}


class FeedbackKnowledgeTests(unittest.TestCase):
    def test_growing_sources_and_prose_have_bounded_residency_without_rewriting_journal(self):
        jobs = [job(i)[0] for i in range(100)]
        for row in jobs:
            row['proposals'][0]['content'] = '失败证据不能被说成成功。' * 3000
        attempts = [attempt(jobs[i], 'idea-' + str(i), [str(j) for j in range(2000)],
                            summary='未验证的解释。' * 2000) for i in range(20)]
        saved = deepcopy((jobs, attempts))
        for cap in (2048, 4096, 12000):
            with self.subTest(cap=cap):
                view = working_memory(jobs, attempts, policy=WorkingMemoryPolicy(max_bytes=cap))
                self.assertLessEqual(len(encoded(view).encode('utf-8')), cap)
                self.assertEqual(view['totals']['learning_jobs'], 100)
                self.assertTrue(all('source_refs' not in a for a in view['attempts']))
                self.assertIn('read_path', view)
        self.assertEqual((jobs, attempts), saved)

    def test_latest_attempt_replaces_related_job_findings_and_active_attempt_gets_priority(self):
        jobs = [job(i)[0] for i in range(12)]
        attempts = [attempt(row, 'idea-' + str(i), ['member-' + str(i)], summary='revision 2')
                    for i, row in enumerate(jobs)]
        for i, row in enumerate(jobs):
            row['input']['research_context']['decision_id'] = 'member-' + str(i)
        jobs[0]['attempt_revisions'] = [attempts[0]]
        view = working_memory(jobs, attempts, active_decision_ids=['member-0'],
                              policy=WorkingMemoryPolicy(recent=2))
        self.assertEqual({a['attempt_id'] for a in view['attempts']}, {'idea-0', 'idea-11'})
        self.assertEqual(view['findings'], [])
        # A title is never a grouping key. Explicitly unrelated observations survive.
        ungrouped = job(20)[0]
        view = working_memory([*jobs, ungrouped], attempts)
        self.assertEqual(view['findings'][0]['job_id'], ungrouped['job_id'])

    def test_three_targets_are_separate_and_proposals_never_become_applied_training(self):
        row = job()[0]
        for kind in ('harness_change', 'training_data', 'system_change'):
            row['proposals'].append({**row['proposals'][0], 'kind': kind, 'content': 'PRIVATE_' + kind})
        view = working_memory([row], [])
        self.assertNotIn('PRIVATE_', encoded(view))
        self.assertEqual({k: v['proposals'] for k, v in view['targets'].items()},
                         {'context_memory': 1, 'harness': 1, 'weights': 1, 'system': 1})
        for target in view['targets']:
            record, = proposal_records([row], target=target)
            self.assertEqual(record['target'], target)
            self.assertFalse(record['training_ready'])
            self.assertFalse(record['applied'])
            self.assertEqual(record['status'], 'proposed')
        failed = deepcopy(row)
        failed['status'] = 'failed'
        self.assertEqual(proposal_records([failed]), [])

    def test_lab_and_meta_agent_are_explicit_subjects_and_legacy_is_not_guessed(self):
        row = job()[0]
        legacy = deepcopy(row['proposals'][0])
        row['proposals'] = [{**legacy, 'subject': 'lab', 'content': 'LAB_ONLY_MEMORY'},
                            {**legacy, 'subject': 'meta_agent', 'content': 'Meta-agent observation'}, legacy]
        self.assertEqual(validate_proposals(encoded({'proposals': row['proposals']}), ['feedback:fixture']),
                         row['proposals'])
        records = proposal_records([row])
        self.assertEqual([p['subject'] for p in records], ['lab', 'meta_agent', 'unspecified'])
        self.assertTrue(all(p['source_snapshot']['job_id'] == row['job_id'] for p in records))
        self.assertNotIn('LAB_ONLY_MEMORY', encoded(working_memory([row], [])))
        bad = {**legacy, 'subject': 'unknown'}
        with self.assertRaisesRegex(ValueError, 'subject'):
            validate_proposals(encoded({'proposals': [bad]}), ['feedback:fixture'])

    def test_knowledge_pages_reconstruct_exact_interpretations_and_detect_changed_catalog(self):
        backend = attempt_tests.Backend()
        learning = FeedbackLearning(SerializedLearningBackend(backend), executor=ImmediateExecutor())
        self.addCleanup(learning.close)
        first = attempt_tests.AttemptLearningTests.capture(self, learning, 'one', ['one'])
        learning.poll(); learning.poll()
        digest, parts, offset = None, [], 0
        while True:
            page = learning.read(view='attempts', offset=offset, limit=1, expected_digest=digest)
            digest = page['sha256']
            parts.append(page['text'])
            self.assertEqual(page['authority'], 'unverified_interpretations')
            offset = page['next_offset']
            if offset is None:
                break
        self.assertEqual(json.loads(''.join(parts)), learning.attempts()['items'])
        original = learning.read(first, evidence_path='/attempt_revisions', limit=16)
        self.assertEqual(json.loads(original['text']), learning.state['jobs'][first]['attempt_revisions'])
        attempt_tests.AttemptLearningTests.capture(self, learning, 'two', ['one', 'two'])
        learning.poll(); learning.poll()
        with self.assertRaisesRegex(ValueError, 'changed'):
            learning.read(view='attempts', expected_digest=digest)
        # A normal concurrent catalog update is a recoverable tool receipt,
        # never an exception that aborts the research run or triggers a model call.
        rejected = LearningTools(object(), learning).execute(OperationCall('feedback_learning_read',
            {'view': 'attempts', 'expected_digest': digest}))
        self.assertEqual(rejected.status, ExecutionStatus.FAILED)
        self.assertIn('restart pagination', rejected.error)
        self.assertEqual(rejected.data['view'], 'attempts')
        self.assertEqual(len(backend.requests), 2)

    def test_proposal_navigation_and_reopen_preserve_exact_records_without_more_inference(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'learning.json'
            backend = SimpleBackend()
            learning = FeedbackLearning(SerializedLearningBackend(backend), path=path,
                                        executor=ImmediateExecutor())
            ident = learning.capture(Directive('one', 'objective', 'test'),
                Feedback('one', ExecutionStatus.COMPLETED, 'original execution fact', True),
                AlienAdapter(AlienHarness()).model())
            learning.poll(); learning.poll()
            view = learning.working_memory()
            page = learning.read(view='proposals', target='context_memory', limit=16)
            record, = json.loads(page['text'])
            exact = learning.read(**{k: v for k, v in record['read_path'].items() if k != 'operation'}, limit=16)
            self.assertEqual(json.loads(exact['text']), learning.state['jobs'][ident]['proposals'][0])
            raw = learning.read(ident, evidence_path='/receipt/feedback/summary', limit=16)
            self.assertEqual(json.loads(raw['text']), 'original execution fact')
            self.assertEqual(raw['authority'], 'untrusted_evidence')
            learning.close()
            restored = FeedbackLearning(SerializedLearningBackend(backend), path=path,
                                        executor=ImmediateExecutor())
            try:
                self.assertEqual(restored.working_memory(), view)
                self.assertEqual(restored.read(view='proposals', target='context_memory', limit=16), page)
                self.assertEqual(len(backend.requests), 1)
            finally:
                restored.close()

    def test_custom_harness_can_consume_knowledge_without_default_state(self):
        harness = AlienHarness()
        loop = DecisionMakingLoop(harness, object())
        enable_feedback_learning(loop, adapter=AlienAdapter(harness), backend=SimpleBackend(),
                                 executor=ImmediateExecutor())
        try:
            loop.receive_feedback(Directive('one', 'objective', 'test'),
                Feedback('one', ExecutionStatus.COMPLETED, 'observed', True))
            loop.step('objective'); loop.step('objective')
            view = loop.feedback_service.working_memory()
            self.assertTrue(view['findings'])
            self.assertFalse(hasattr(harness, 'state'))
            self.assertFalse(hasattr(harness, 'context_compactor'))
        finally:
            loop.feedback_service.close()
