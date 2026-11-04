"""Context is active but replaceable; public archival is a default optional product."""
import json
import unittest

from rsisys.config import build_demo
from rsisys.loops._memory import Memory
from rsisys.loops.decision_making.feedback import JournalContextMemory, WorkingMemoryPolicy
from rsisys.loops.decision_making.feedback.learning import FeedbackLearning, SerializedLearningBackend
from rsisys.loops.decision_making.harness.learning_harness import enable_feedback_learning
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.runtime.json_tools import EvidencePathError, result_id
from tests.test_feedback_decoupling import AlienHarness, AlienAdapter
from tests.test_feedback_learning_integration import Backend, ImmediateExecutor
from tests.test_feedback_required_outputs import CONFIG, EmptyBackend, exact, observation


class ExternalContext:
    """Own format and publication, with no call to the built-in organizer."""
    def __init__(self):
        self.calls = []

    def record(self, job):
        self.calls.append(('record', job['job_id']))
        value = {'provider': 'external', 'observed': job['raw_input']['receipt']['result']['summary']}
        job['raw_input'].clear()  # Providers must only receive detached snapshots.
        return value

    def working_memory(self, jobs, attempts, *, active_decision_ids=(), policy=None):
        self.calls.append(('view', tuple(active_decision_ids), policy))
        value = {'provider': 'external', 'observed_jobs': len(jobs)}
        jobs.clear()
        attempts.clear()
        return value

    def memory(self, job):
        self.calls.append(('publication', job['job_id']))
        memory = Memory('external:' + job['job_id'], 'episodic', 'External organized context',
                        tuple(job['input']['source_refs']))
        job['proposals'].clear()
        job['raw_input'].clear()
        return memory


class ExternalArchive:
    def records(self, jobs, *, research_attempt_id=None):
        result = [{'provider': 'external-archive', 'job_id': job['job_id'],
                   'snapshot_digest': result_id(job['raw_input'])} for job in jobs
                  if research_attempt_id is None or research_attempt_id ==
                  job['input'].get('research_context', {}).get('research_attempt_id')]
        for job in jobs:
            job['raw_input'].clear()
        return result


class FeedbackComponentTests(unittest.TestCase):
    def install(self, *, backend=None, **options):
        harness = AlienHarness()
        loop = DecisionMakingLoop(harness, object())
        backend = backend or EmptyBackend()
        enable_feedback_learning(loop, adapter=AlienAdapter(harness), backend=backend,
                                 executor=ImmediateExecutor(), **options)
        self.addCleanup(loop.feedback_service.close)
        return loop.feedback_service, harness, backend

    def test_disabled_or_incomplete_context_is_rejected_before_inference(self):
        backend = EmptyBackend()
        for provider in (None, False, object()):
            with self.subTest(provider=provider), self.assertRaisesRegex(TypeError, 'active context/memory'):
                FeedbackLearning(SerializedLearningBackend(backend), context_memory=provider)
        self.assertEqual(backend.requests, [])

    def test_external_context_replaces_record_view_and_publication_without_default_harness(self):
        provider = ExternalContext()
        service, harness, backend = self.install(context_memory=provider, trajectory_archive=None)
        ident = service.observe(observation())
        service.refresh()
        before = result_id(service.learning.state)
        self.assertIs(service.learning.context_memory, provider)
        self.assertEqual(exact(service.learning, job_id=ident, evidence_path='/context_memory'),
                         {'provider': 'external', 'observed': 'Executed once'})
        policy = WorkingMemoryPolicy(max_bytes=2048)
        self.assertEqual(service.working_memory(active_decision_ids=['active'], policy=policy),
                         {'provider': 'external', 'observed_jobs': 1})
        self.assertIn(('view', ('active',), policy), provider.calls)
        self.assertEqual([memory.memory_id for memory in harness.summaries], ['external:' + ident])
        self.assertEqual(result_id(service.learning.state), before)
        self.assertFalse(hasattr(harness, 'state'))
        self.assertEqual(len(backend.requests), 1)
        service.refresh()
        self.assertEqual(len(harness.summaries), 1)

    def test_external_context_reaches_default_harness_working_view(self):
        runner, _ = build_demo(CONFIG)
        with runner:
            loop = runner.decision_making
            enable_feedback_learning(loop, backend=EmptyBackend(), executor=ImmediateExecutor(),
                                     context_memory=ExternalContext(), trajectory_archive=None)
            loop.feedback_service.observe(observation())
            context = loop.harness._context('objective', loop.action_space)
            self.assertEqual(context.research_state['provider'], 'external')
            self.assertEqual(context.research_state['observed_jobs'], 1)

    def test_disabled_trajectory_does_not_disable_context_or_delete_execution_evidence(self):
        service, harness, backend = self.install(trajectory_archive=None)
        ident = service.observe(observation())
        service.refresh()
        self.assertIsInstance(service.learning.context_memory, JournalContextMemory)
        self.assertEqual(service.learning.read(ident)['trajectory'], {'enabled': False})
        self.assertFalse(service.learning.read(view='trajectories')['enabled'])
        self.assertEqual(service.trajectories(), [])
        self.assertEqual(exact(service.learning, view='trajectories'), [])
        with self.assertRaisesRegex(EvidencePathError, 'disabled'):
            service.learning.read(ident, evidence_path='/trajectory')
        self.assertEqual(exact(service.learning, job_id=ident, evidence_path='/context_memory')
                         ['facts']['status'], 'completed')
        self.assertTrue(service.working_memory()['findings'])
        self.assertEqual(len(harness.summaries), 1)
        self.assertEqual(exact(service.learning, job_id=ident, evidence_path='/feedback_request')['system'],
                         backend.requests[0].system)
        self.assertNotIn('always archives', backend.requests[0].system)

    def test_external_trajectory_replaces_both_index_and_individual_read(self):
        service, _, _ = self.install(trajectory_archive=ExternalArchive())
        ident = service.observe(observation())
        service.refresh()
        before = result_id(service.learning.state)
        segment, = service.trajectories()
        self.assertEqual(segment['provider'], 'external-archive')
        self.assertTrue(service.learning.read(view='trajectories')['enabled'])
        self.assertEqual(exact(service.learning, view='trajectories'), [segment])
        self.assertEqual(exact(service.learning, job_id=ident, evidence_path='/trajectory'), segment)
        self.assertEqual(service.trajectories(research_attempt_id='not-captured'), [])
        self.assertEqual(result_id(service.learning.state), before)

    def test_external_publication_cannot_bypass_model_output_validation(self):
        provider = ExternalContext()
        service, harness, _ = self.install(backend=Backend(), context_memory=provider)
        ident = service.observe(observation())
        service.learning.poll()
        row = service.learning.state['jobs'][ident]
        row['proposals'][0]['source_refs'] = ['unknown']
        service.refresh()
        self.assertEqual(service.learning.state['jobs'][ident]['status'], 'failed')
        self.assertEqual(harness.summaries, [])
        self.assertEqual(provider.calls, [])
        self.assertEqual(service.learning.state['jobs'][ident]['raw_input']['receipt']['result']['summary'],
                         'Executed once')

    def test_provider_delivery_failure_retries_publication_without_repeating_model_call(self):
        class FlakyContext(ExternalContext):
            fail = True

            def memory(self, job):
                if self.fail:
                    self.fail = False
                    raise RuntimeError('External context store unavailable')
                return super().memory(job)

        service, harness, backend = self.install(context_memory=FlakyContext())
        ident = service.observe(observation())
        with self.assertRaisesRegex(RuntimeError, 'store unavailable'):
            service.refresh()
        self.assertEqual(service.learning.state['jobs'][ident]['status'], 'completed')
        self.assertFalse(service.learning.state['jobs'][ident].get('published', False))
        service.refresh()
        self.assertTrue(service.learning.state['jobs'][ident]['published'])
        self.assertEqual(len(backend.requests), 1)
        self.assertEqual(len(harness.summaries), 1)

    def test_context_can_publish_through_its_view_without_a_duplicate_memory_object(self):
        class ViewContext(ExternalContext):
            def memory(self, job):
                return None

        service, harness, _ = self.install(context_memory=ViewContext(), trajectory_archive=None)
        ident = service.observe(observation())
        service.refresh()
        self.assertEqual(service.working_memory()['observed_jobs'], 1)
        self.assertEqual(harness.summaries, [])
        self.assertTrue(service.learning.state['jobs'][ident]['published'])
        self.assertIsNone(service.learning.state['jobs'][ident]['memory_digest'])


if __name__ == '__main__':
    unittest.main()
