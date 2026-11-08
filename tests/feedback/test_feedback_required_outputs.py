"""Active context and default optional archival do not depend on model proposals."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from rsisys.config import DemoConfig, build_demo
from rsisys.loops._memory import HistoryEntry
from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef
from rsisys.loops.contracts.receipts import ExecutionReceipt
from rsisys.loops.decision_making.action_space import OperationCall
from rsisys.loops.decision_making.feedback import FeedbackEvidence, WorkingMemoryPolicy
from rsisys.loops.decision_making.feedback.learning import FeedbackLearning, SerializedLearningBackend
from rsisys.loops.decision_making.harness.learning_harness import enable_feedback_learning
from rsisys.runtime.json_tools import EvidencePathError, result_id
from tests.feedback.test_feedback_learning_integration import Backend, ImmediateExecutor


CONFIG = DemoConfig.load(Path('configs/demo.json'))
MODEL = ModelRef('fixture', 'meta-agent', 'v1')


class EmptyBackend(Backend):
    def generate(self, request):
        self.requests.append(request)
        return GenerationResponse('{"proposals":[]}', request.model, simulated=True)


def observation(ident='one', *, summary='Executed once', stdout='output'):
    return ExecutionReceipt.from_operation(ident, 'objective',
        OperationCall('execute_code', {'code': 'public fixture code'}),
        OperationResult(summary, (), True, data={'stdout': stdout}),
        decision_id='decision-' + ident, issuing_model=MODEL)


def exact(learning, *, job_id=None, evidence_path=None, view='jobs'):
    parts, offset, digest = [], 0, None
    while True:
        options = {'view': view, 'offset': offset, 'limit': 16, 'expected_digest': digest}
        if evidence_path is not None:
            options['evidence_path'] = evidence_path
        page = learning.read(job_id, **options)
        parts.append(page['text'])
        digest = page['sha256']
        offset = page['next_offset']
        if offset is None:
            return json.loads(''.join(parts))


class RequiredFeedbackOutputsTests(unittest.TestCase):
    def install(self, runner, backend=None, **options):
        loop = runner.decision_making
        backend = backend or EmptyBackend()
        enable_feedback_learning(loop, backend=backend, executor=ImmediateExecutor(), **options)
        return loop.feedback_service, backend

    def test_empty_proposals_still_publish_context_and_archive_public_trajectory(self):
        runner, _ = build_demo(CONFIG)
        with runner:
            service, backend = self.install(runner)
            history = (HistoryEntry('decision-one', 'decision', None, 'Public plan', True),)
            ident = service.observe(observation(), evidence=FeedbackEvidence(history=history))
            service.refresh()
            row = service.learning.state['jobs'][ident]
            self.assertEqual(row['proposals'], [])
            self.assertTrue(row['published'])
            memory, = runner.decision_making.harness.memory.recall()
            self.assertEqual(json.loads(memory.content)['receipt_facts']['status'], 'completed')
            context = exact(service.learning, job_id=ident, evidence_path='/context_memory')
            self.assertEqual(context['facts']['summary'], 'Executed once')
            self.assertEqual(context['interpretations'], [])
            view = service.working_memory()
            self.assertEqual(view['findings'][0]['observation']['status'], 'completed')
            self.assertEqual(view['findings'][0]['authority'], 'execution_observation')
            trace, = service.trajectories()
            self.assertEqual(trace['issuing_model'], asdict(MODEL))
            self.assertEqual(trace['captured_decision_records'], 1)
            self.assertFalse(trace['training_ready'])
            self.assertFalse(trace['complete_conversation'])
            self.assertEqual(exact(service.learning, view='trajectories'), [trace])
            self.assertEqual(exact(service.learning, job_id=ident, evidence_path='/feedback_request'),
                             json.loads(json.dumps(asdict(backend.requests[0]))))
            self.assertEqual(exact(service.learning, job_id=ident, evidence_path='/feedback_response'),
                             json.loads(json.dumps(row['response'])))
            self.assertEqual(len(backend.requests), 1)

    def test_invalid_interpretation_preserves_facts_and_failed_response_without_prose_leak(self):
        runner, _ = build_demo(CONFIG)
        with runner:
            backend = Backend()
            backend.bad_source = True
            service, _ = self.install(runner, backend)
            ident = service.observe(observation())
            service.refresh()
            row = service.learning.state['jobs'][ident]
            self.assertEqual(row['status'], 'failed')
            self.assertEqual(row['proposals'], [])
            context = exact(service.learning, job_id=ident, evidence_path='/context_memory')
            self.assertEqual(context['facts']['status'], 'completed')
            self.assertEqual(context['interpretation_status'], 'failed')
            self.assertEqual(context['interpretations'], [])
            self.assertEqual(service.working_memory()['findings'][0]['observation']['status'], 'completed')
            self.assertIn('unknown', exact(service.learning, job_id=ident,
                                          evidence_path='/feedback_response')['text'])
            self.assertEqual(service.trajectories()[0]['feedback_status'], 'failed')
            self.assertEqual(service.trajectories()[0]['execution_status'], 'completed')

    def test_default_feedback_follows_active_weights_in_a_separate_context(self):
        runner, _ = build_demo(CONFIG)
        with runner:
            loop = runner.decision_making
            current = loop.harness.policy.model
            service, backend = self.install(runner, model=current)
            first = service.observe(observation('first'))
            service.refresh()
            updated = replace(current, revision=current.revision + '-new')
            loop.harness.policy.update_model(updated)
            second = service.observe(observation('second'))
            service.refresh()
            self.assertEqual([r.model for r in backend.requests], [current, updated])
            self.assertEqual(service.learning.state['jobs'][first]['model'], asdict(current))
            self.assertEqual(service.learning.state['jobs'][second]['model'], asdict(updated))
            self.assertEqual(service.observe(observation('first')), first)
            self.assertEqual(len(backend.requests), 2)
            for request in backend.requests:
                self.assertIn('weights are shared', request.system)
                self.assertIn('receipt', json.loads(request.prompt))
                self.assertNotIn('operations', json.loads(request.prompt))
        runner, _ = build_demo(CONFIG)
        with runner:
            with self.assertRaisesRegex(ValueError, 'active meta-agent model'):
                enable_feedback_learning(runner.decision_making, backend=EmptyBackend(), model=MODEL)

    def test_trajectory_and_context_survive_admission_rejection_and_reopen(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'feedback.json'
            backend = EmptyBackend()
            learning = FeedbackLearning(SerializedLearningBackend(backend), path=path,
                executor=ImmediateExecutor(), max_calls=1)
            first = learning.capture_receipt(observation('one'), MODEL)
            learning.poll(); learning.poll()
            second = learning.capture_receipt(observation('two'), MODEL)
            learning.poll()
            self.assertEqual(learning.state['jobs'][second]['status'], 'budget_exhausted')
            self.assertEqual(len(learning.trajectories()), 2)
            self.assertEqual(exact(learning, job_id=second, evidence_path='/context_memory')['facts']['status'],
                             'completed')
            original_request = exact(learning, job_id=first, evidence_path='/feedback_request')
            learning.close()
            restored = FeedbackLearning(SerializedLearningBackend(backend), path=path,
                executor=ImmediateExecutor(), max_calls=1)
            try:
                with patch.object(restored, '_request', side_effect=AssertionError('Must not reconstruct history')):
                    self.assertEqual(exact(restored, job_id=first, evidence_path='/feedback_request'), original_request)
                    self.assertEqual(restored.trajectories()[1]['feedback_status'], 'budget_exhausted')
                with self.assertRaises(EvidencePathError):
                    restored.read(second, evidence_path='/feedback_request')
                self.assertEqual(len(backend.requests), 1)
            finally:
                restored.close()

    def test_large_archive_uses_bounded_context_without_truncating_the_original(self):
        runner, _ = build_demo(CONFIG)
        with runner:
            service, _ = self.install(runner)
            receipt = observation(summary='long observation ' * 3000, stdout='exact-output' * 5000)
            ident = service.observe(receipt)
            service.refresh()
            raw = service.learning.state['jobs'][ident]['raw_input']
            before = result_id(raw)
            view = service.working_memory(policy=WorkingMemoryPolicy(max_bytes=2048))
            self.assertLessEqual(len(json.dumps(view, ensure_ascii=False).encode()), 2048)
            self.assertEqual(service.trajectories()[0]['snapshot_digest'], before)
            self.assertEqual(exact(service.learning, job_id=ident, evidence_path='/receipt/result/data/stdout'),
                             receipt.result['data']['stdout'])
            self.assertEqual(result_id(raw), before)

    def test_unknown_issuer_can_be_filled_but_conflicting_provenance_is_rejected(self):
        learning = FeedbackLearning(SerializedLearningBackend(EmptyBackend()), executor=ImmediateExecutor())
        try:
            learning.capture_receipt(observation('unknown'), MODEL,
                research_context={'issuing_decision_model': None})
            with self.assertRaisesRegex(ValueError, 'issuing model differs'):
                learning.capture_receipt(observation('conflict'), MODEL,
                    research_context={'issuing_decision_model': asdict(replace(MODEL, revision='other'))})
        finally:
            learning.close()


if __name__ == '__main__':
    unittest.main()
