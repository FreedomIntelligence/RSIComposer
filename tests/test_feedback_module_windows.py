"""Replace products without learning job schemas; offline integration checks."""
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from rsisys.config import build_demo
from rsisys.loops.contracts.model_service import GenerationRequest
from rsisys.loops.decision_making.feedback import (
    ArchiveEvent, CallbackContextMemory, CallbackFeedbackProcessor, FeedbackFailurePolicy,
    FeedbackPipeline, FeedbackContextEvent,
)
from rsisys.loops.decision_making.feedback.learning import FeedbackLearning, SerializedLearningBackend
from rsisys.loops.decision_making.feedback.legacy import LegacyFeedbackInterpreter
from rsisys.research import ResearchAssembly, assemble_research
from rsisys.runtime.json_tools import EvidencePathError
from tests.test_feedback_components import ExternalContext
from tests.test_feedback_learning_integration import ImmediateExecutor
from tests.test_feedback_required_outputs import CONFIG, MODEL, EmptyBackend, exact, observation


class NativeArchive:
    def __init__(self):
        self.events, self.writes = {}, []

    def write(self, event):
        self.writes.append(event.event_id)
        self.events[event.event_id] = deepcopy(event)
        return {'key': event.event_id}

    def read(self, reference):
        return deepcopy(self.events[reference['key']])


class ManualExecutor:
    def __init__(self):
        self.tasks = []

    def submit(self, function, *args):
        future = Future()
        self.tasks.append((future, function, args))
        return future

    def finish(self):
        for future, function, args in self.tasks:
            if not future.done():
                future.set_result(function(*args))


class FeedbackModuleWindowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.backend = EmptyBackend()

    def install(self, assembly=None, **options):
        runner, _ = build_demo(CONFIG)
        self.addCleanup(runner.close)
        service = assemble_research(runner.decision_making,
            assembly=ResearchAssembly(context_memory=lambda query: 'current') if assembly is None else assembly,
            backend=self.backend, path=self.root / 'feedback.json', executor=ImmediateExecutor(),
            owner=runner.assembly, **options)
        return runner, service

    def test_single_method_processor_returns_native_payload_not_host_fields(self):
        class Processor:
            def process(self, event, model):
                self.event = event
                event.identity['untrusted'] = True
                return {'context': 'useful', 'status': 'running', 'job_id': 'forged', 'charged_calls': 999}

        processor = Processor()
        assembly = ResearchAssembly(context_memory=lambda q: '', memory_processor=processor)
        self.assertFalse(assembly.requires_model)
        _, service = self.install(assembly)
        ident = service.observe_output({'native': [1, 2]}, task_id='native', status='completed')
        self.assertFalse(hasattr(processor, 'event'))
        service.refresh()
        row = service.learning.state['jobs'][ident]
        self.assertIsInstance(processor.event, FeedbackContextEvent)
        self.assertEqual(processor.event.output, {'native': [1, 2]})
        self.assertEqual(row['job_id'], ident)
        self.assertEqual(row['status'], 'completed')
        self.assertEqual(service.learning.state['attempted'], 0)
        self.assertNotIn('untrusted', row['raw_input']['receipt'].get('identity', {}))
        output, = service.outputs(product='context_memory')
        self.assertEqual(output['output']['charged_calls'], 999)
        self.assertEqual(exact(service.learning, job_id=ident, evidence_path='/outputs/context_memory'), output['output'])
        self.assertFalse(self.backend.requests)

    def test_plain_function_and_sdk_callback_need_no_recovery_or_configuration(self):
        store = {}
        def update(event, model):
            store['context'] = 'Observed ' + event.status
            return ['native', event.task_id]
        memory = CallbackContextMemory(lambda q: store.get('context', ''))
        _, service = self.install(ResearchAssembly(context_memory=memory, memory_processor=update))
        service.observe(observation())
        service.refresh()
        self.assertEqual(service.context(), 'Observed completed')
        self.assertEqual(service.outputs()[0]['output'], ['native', 'one'])

    def test_proposer_needs_only_public_inputs_and_native_output_reaches_decision_navigation(self):
        class Proposer:
            def propose(self, event, model):
                return {'ideas': ['check parser'], 'producer_format': 7}
        runner, service = self.install(ResearchAssembly(context_memory=lambda q: '',
                                                        improvement_proposals=Proposer()))
        service.observe(observation())
        context = runner.decision_making.harness._context('objective', runner.decision_making.action_space)
        record, = service.outputs(product='improvement_proposals')
        self.assertEqual(record['output'], {'ideas': ['check parser'], 'producer_format': 7})
        self.assertFalse(record['applied'])
        navigation = context.research_agenda['feedback_learning']['product_outputs']
        self.assertEqual(navigation['recent'][0]['read_path'], record['read_path'])
        self.assertEqual(service.resolve_improvement_proposals([]), ())
        with self.assertRaises(ValueError):
            service.resolve_improvement_proposals([record['event_id']])

    def test_unserializable_native_output_fails_without_touching_execution(self):
        _, service = self.install(ResearchAssembly(context_memory=lambda q: '',
            memory_processor=lambda event, model: object()))
        ident = service.observe(observation())
        service.refresh()
        row = service.learning.state['jobs'][ident]
        self.assertEqual(row['status'], 'failed')
        self.assertEqual(row['raw_input']['receipt']['result']['status'], 'completed')
        self.assertEqual(service.outputs(), [])

    def test_zero_call_processor_cannot_invoke_a_model(self):
        def process(event, model):
            return model.generate(GenerationRequest('decision_maker', model.model, 'fixture', max_output_tokens=8))
        _, service = self.install(ResearchAssembly(context_memory=lambda q: '', memory_processor=process))
        ident = service.observe(observation())
        service.refresh()
        row = service.learning.state['jobs'][ident]
        self.assertEqual(row['products']['context_memory']['status'], 'failed')
        self.assertIn('call limit', row['response']['products']['context_memory']['error'])
        self.assertEqual(service.learning.state['attempted'], 0)
        self.assertFalse(self.backend.requests)

    def test_model_access_is_accounted_and_each_exchange_is_saved_by_host(self):
        def process(event, model):
            request = GenerationRequest('decision_maker', model.model, 'fixture', max_output_tokens=8)
            model.generate(request)
            model.generate(request)
            return 'unreachable'
        processor = CallbackFeedbackProcessor(process, max_model_calls=1)
        _, service = self.install(ResearchAssembly(context_memory=lambda q: '', memory_processor=processor))
        ident = service.observe(observation())
        service.refresh()
        self.assertEqual(service.learning.state['attempted'], 1)
        self.assertEqual(len(self.backend.requests), 1)
        stage = service.learning.state['jobs'][ident]['response']['products']['context_memory']
        self.assertEqual(stage['status'], 'failed')
        self.assertEqual(len(stage['response']['model_exchanges']), 1)
        self.assertTrue((self.root / 'feedback-exchanges' / ident / 'context_memory.json').exists())

    def test_native_stage_cannot_switch_model_or_exceed_output_limit(self):
        for wrong_model in (True, False):
            with self.subTest(wrong_model=wrong_model):
                root = self.root / str(wrong_model)
                root.mkdir()
                self.root = root
                def process(event, model):
                    identity = replace(model.model, revision='wrong') if wrong_model else model.model
                    return model.generate(GenerationRequest('decision_maker', identity, 'fixture',
                        max_output_tokens=8 if wrong_model else 4097)).text
                processor = CallbackFeedbackProcessor(process, max_model_calls=1)
                _, service = self.install(ResearchAssembly(context_memory=lambda q: '', memory_processor=processor))
                ident = service.observe(observation())
                service.refresh()
                self.assertEqual(service.learning.state['jobs'][ident]['status'], 'failed')
                self.assertFalse(self.backend.requests)

    def test_native_model_gateway_admits_the_complete_request(self):
        def process(event, model):
            return model.generate(GenerationRequest('decision_maker', model.model, '',
                messages=[{'role': 'user', 'content': 'full fixture input'}], max_output_tokens=8)).text
        processor = CallbackFeedbackProcessor(process, max_model_calls=1)
        _, service = self.install(ResearchAssembly(context_memory=lambda q: '', memory_processor=processor))
        self.backend.count_tokens = lambda request: 1000000
        ident = service.observe(observation())
        service.refresh()
        self.assertEqual(service.learning.state['jobs'][ident]['status'], 'failed')
        self.assertEqual(service.learning.state['attempted'], 1)
        self.assertFalse(self.backend.requests)
        stage = service.learning.state['jobs'][ident]['response']['products']['context_memory']
        self.assertEqual(stage['response']['model_exchanges'][0]['status'], 'input_too_large')

    def test_native_gateway_rejects_response_identity_mismatch(self):
        from rsisys.loops.contracts.model_service import GenerationResponse
        def generate(request):
            return GenerationResponse('{}', replace(request.model, revision='wrong'), simulated=True)
        self.backend.generate = generate
        processor = CallbackFeedbackProcessor(lambda event, model: model.generate(
            GenerationRequest('decision_maker', model.model, 'fixture', max_output_tokens=8)).text,
            max_model_calls=1)
        _, service = self.install(ResearchAssembly(context_memory=lambda q: '', memory_processor=processor))
        ident = service.observe(observation())
        service.refresh()
        self.assertEqual(service.learning.state['jobs'][ident]['status'], 'failed')
        self.assertEqual(service.learning.state['attempted'], 1)
        self.assertEqual(service.outputs(), [])

    def test_pure_products_still_run_after_model_call_allowance_is_exhausted(self):
        _, service = self.install(ResearchAssembly(context_memory=lambda q: '',
            memory_processor=lambda event, model: 'no inference needed'))
        state = deepcopy(service.learning.state)
        state['attempted'] = state['limits']['max_calls']
        service.learning._commit(state)
        ident = service.observe(observation())
        service.refresh()
        self.assertEqual(service.learning.state['jobs'][ident]['status'], 'completed')
        self.assertEqual(service.outputs()[0]['output'], 'no inference needed')
        self.assertFalse(self.backend.requests)

    def test_archive_write_read_window_deduplicates_and_preserves_native_input(self):
        archive = NativeArchive()
        _, service = self.install(ResearchAssembly(context_memory=lambda q: '', trajectory_archive=archive))
        ident = service.observe_output({'raw': ['log', {'metric': 0.2}]}, task_id='lab-task', status='completed')
        self.assertEqual(archive.writes, [])
        service.refresh()
        snapshot = service.trajectory_snapshot(ident)
        self.assertEqual(snapshot['capture']['receipt']['result']['output'], {'raw': ['log', {'metric': 0.2}]})
        self.assertFalse(snapshot['complete_conversation'])
        snapshot['capture'].clear()
        self.assertTrue(service.trajectory_snapshot(ident)['capture'])
        service.observe_output({'raw': ['log', {'metric': 0.2}]}, task_id='lab-task', status='completed')
        service.refresh()
        self.assertEqual(archive.writes, [ident])
        self.assertEqual(service.trajectories()[0]['archive_delivery']['status'], 'delivered')

    def test_archive_read_checks_identity_and_original_payload(self):
        archive = NativeArchive()
        _, service = self.install(ResearchAssembly(context_memory=lambda q: '', trajectory_archive=archive))
        ident = service.observe(observation())
        service.refresh()
        event = archive.events[ident]
        archive.events[ident] = replace(event, event_id='wrong')
        with self.assertRaisesRegex(EvidencePathError, 'differs'):
            service.trajectory_snapshot(ident)
        archive.events[ident] = replace(event, payload={'fabricated': 'contents'})
        with self.assertRaisesRegex(EvidencePathError, 'differs'):
            service.extract_trajectory(ident)

    def test_archive_failure_is_visible_and_does_not_block_context_or_replay(self):
        class FailingArchive(NativeArchive):
            def write(self, event):
                self.writes.append(event.event_id)
                raise OSError('external store unavailable')
        archive = FailingArchive()
        runner, service = self.install(ResearchAssembly(context_memory=lambda q: 'valid', trajectory_archive=archive))
        ident = service.observe(observation())
        context = runner.decision_making.harness._context('objective', runner.decision_making.action_space)
        self.assertEqual(context.memory_context, 'valid')
        self.assertEqual(service.learning.state['jobs'][ident]['archive_delivery']['status'], 'failed')
        service.refresh()
        self.assertEqual(archive.writes, [ident])

    def test_archive_delivery_survives_reopen_without_writing_again(self):
        archive = NativeArchive()
        assembly = ResearchAssembly(context_memory=lambda q: '', trajectory_archive=archive)
        runner, service = self.install(assembly)
        ident = service.observe(observation())
        service.refresh()
        original = service.trajectory_snapshot(ident)
        runner.close()
        _, restored = self.install(assembly)
        restored.refresh()
        self.assertEqual(archive.writes, [ident])
        self.assertEqual(restored.trajectory_snapshot(ident), original)

    def test_uncertain_archive_write_is_not_replayed_after_reopen(self):
        archive = NativeArchive()
        assembly = ResearchAssembly(context_memory=lambda q: '', trajectory_archive=archive)
        runner, service = self.install(assembly)
        ident = service.observe(observation())
        runner.close()
        path = self.root / 'feedback.json'
        saved = json.loads(path.read_text())
        saved['jobs'][ident]['archive_delivery']['status'] = 'running'
        path.write_text(json.dumps(saved))
        _, restored = self.install(assembly)
        restored.refresh()
        self.assertEqual(restored.learning.state['jobs'][ident]['archive_delivery']['status'], 'interrupted')
        self.assertEqual(archive.writes, [ident])

    def test_scheduler_selects_public_events_without_access_to_jobs(self):
        class Newest:
            def select(self, pending, running):
                self.pending = pending
                return pending[-1].event.event_id
        scheduler = Newest()
        _, service = self.install(ResearchAssembly(context_memory=lambda q: '', scheduler=scheduler))
        first = service.observe(observation('first'))
        second = service.observe(observation('second'))
        service.refresh()
        self.assertEqual(service.learning.state['jobs'][first]['status'], 'queued')
        self.assertEqual(service.learning.state['jobs'][second]['status'], 'completed')
        self.assertIsInstance(scheduler.pending[0].event, FeedbackContextEvent)
        self.assertFalse(hasattr(scheduler.pending[0], 'reserved_calls'))
        service.refresh()
        self.assertEqual(service.learning.state['jobs'][first]['status'], 'completed')

    def test_scheduler_can_defer_without_consuming_calls(self):
        class Deferred:
            def select(self, pending, running):
                return None
        _, service = self.install(ResearchAssembly(context_memory=lambda q: '', scheduler=Deferred()))
        ident = service.observe(observation())
        service.refresh()
        self.assertEqual(service.learning.state['jobs'][ident]['status'], 'queued')
        self.assertEqual(service.learning.state['attempted'], 0)

    def test_scheduler_cannot_select_an_unknown_or_completed_event(self):
        class Invalid:
            def select(self, pending, running):
                return 'not-pending'
        runner, service = self.install(ResearchAssembly(context_memory=lambda q: 'valid', scheduler=Invalid()))
        ident = service.observe(observation())
        context = runner.decision_making.harness._context('objective', runner.decision_making.action_space)
        self.assertEqual(context.memory_context, 'valid')
        self.assertIn('not pending', service.learning.state['scheduler_error'])
        self.assertEqual(service.learning.state['jobs'][ident]['status'], 'queued')
        self.assertEqual(service.learning.state['attempted'], 0)

    def test_executor_parallelism_is_admitted_only_by_host(self):
        executor = ManualExecutor()
        learning = FeedbackLearning(SerializedLearningBackend(self.backend), executor=executor,
            processor=FeedbackPipeline(), max_concurrent_jobs=2)
        try:
            ids = [learning.capture_receipt(observation(str(i)), MODEL) for i in range(3)]
            learning.poll()
            learning.poll()
            learning.poll()
            self.assertEqual(len(executor.tasks), 2)
            self.assertEqual(learning.state['jobs'][ids[-1]]['status'], 'queued')
            executor.finish()
            learning.poll(dispatch=False)
        finally:
            executor.finish()
            learning.close()

    def test_independent_proposal_survives_memory_failure(self):
        def fail(event, model):
            raise ValueError('memory failed')
        _, service = self.install(ResearchAssembly(context_memory=lambda q: 'factual view',
            memory_processor=fail, improvement_proposals=lambda event, model: {'idea': 'inspect output'}))
        ident = service.observe(observation())
        service.refresh()
        row = service.learning.state['jobs'][ident]
        self.assertEqual(row['products']['context_memory']['status'], 'failed')
        self.assertEqual(row['products']['improvement_proposals']['status'], 'completed')
        self.assertEqual(service.outputs(product='improvement_proposals')[0]['output'], {'idea': 'inspect output'})

    def test_declared_proposal_dependency_is_respected(self):
        calls = []
        def fail(event, model):
            raise ValueError('memory failed')
        def propose(event, model):
            calls.append(event.event_id)
        _, service = self.install(ResearchAssembly(context_memory=lambda q: 'factual view',
            memory_processor=fail, improvement_proposals=propose, proposal_requires_memory=True))
        ident = service.observe(observation())
        service.refresh()
        self.assertEqual(service.learning.state['jobs'][ident]['products']['improvement_proposals']['status'], 'not_started')
        self.assertEqual(calls, [])

    def test_optional_publication_failure_does_not_break_actual_decision_context(self):
        class Unavailable(ExternalContext):
            def memory(self, row):
                raise OSError('legacy publication unavailable')
        runner, service = self.install(ResearchAssembly(context_memory=Unavailable()))
        ident = service.observe(observation())
        context = runner.decision_making.harness._context('objective', runner.decision_making.action_space)
        self.assertEqual(json.loads(context.memory_context)['provider'], 'external')
        self.assertEqual(service.learning.state['jobs'][ident]['publication_delivery']['status'], 'failed')

    def test_publication_can_be_configured_to_raise(self):
        class Unavailable(ExternalContext):
            def memory(self, row):
                raise OSError('legacy publication unavailable')
        _, service = self.install(ResearchAssembly(context_memory=Unavailable(),
            failure_policy=FeedbackFailurePolicy(publication='raise')))
        ident = service.observe(observation())
        with self.assertRaisesRegex(OSError, 'publication unavailable'):
            service.refresh()
        self.assertEqual(service.learning.state['jobs'][ident]['publication_delivery']['status'], 'failed')

    def test_commit_failure_after_publication_does_not_automatically_repeat_external_effects(self):
        _, service = self.install(ResearchAssembly(context_memory=ExternalContext()))
        ident = service.observe(observation())
        service.learning.poll()
        service.learning.poll(dispatch=False)
        delivered = []
        service.adapter.publish = delivered.append
        original = service.learning._commit
        def commit(state):
            if state['jobs'][ident].get('published'):
                raise OSError('post-delivery commit unavailable')
            original(state)
        with patch.object(service.learning, '_commit', side_effect=commit):
            with self.assertRaisesRegex(OSError, 'post-delivery'):
                service.refresh()
        self.assertEqual(len(delivered), 1)
        service.refresh()
        self.assertEqual(len(delivered), 1)
        service.retry_publication(ident)
        service.refresh()
        self.assertEqual(len(delivered), 2)
        self.assertEqual(delivered[0].memory_id, delivered[1].memory_id)
        self.assertFalse(self.backend.requests)

    def test_context_errors_are_strict_by_default(self):
        def fail(query):
            raise OSError('retrieval unavailable')
        _, service = self.install(ResearchAssembly(context_memory=fail))
        with self.assertRaisesRegex(OSError, 'retrieval unavailable'):
            service.context()
        self.assertFalse(service.learning.state['context_error']['used_previous'])

    def test_last_valid_context_is_explicit_and_never_reused_for_a_different_objective(self):
        fail = False
        def render(query):
            if fail:
                raise OSError('retrieval unavailable')
            return 'valid view'
        from rsisys.loops.contracts.feedback_context import ContextQuery
        _, service = self.install(ResearchAssembly(context_memory=render,
            failure_policy=FeedbackFailurePolicy(context='last_valid')))
        self.assertEqual(service.context(ContextQuery(objective='one')), 'valid view')
        fail = True
        self.assertEqual(service.context(ContextQuery(objective='one')), 'valid view')
        self.assertTrue(service.learning.state['context_error']['used_previous'])
        with self.assertRaisesRegex(OSError, 'retrieval unavailable'):
            service.context(ContextQuery(objective='two'))

    def test_default_prompt_is_a_compatibility_component_and_override_is_recorded(self):
        interpreter = LegacyFeedbackInterpreter(instructions='Fixture compatibility prompt')
        learning = FeedbackLearning(SerializedLearningBackend(self.backend), executor=ImmediateExecutor(),
            legacy_interpreter=interpreter)
        try:
            ident = learning.capture_receipt(observation(), MODEL)
            learning.poll()
            learning.poll()
            self.assertEqual(self.backend.requests[0].system, 'Fixture compatibility prompt')
            self.assertEqual(learning.state['runtime']['legacy_interpreter']['configuration'], interpreter.configuration())
        finally:
            learning.close()

    def test_running_pure_product_is_interrupted_without_automatic_reexecution(self):
        calls = []
        processor = lambda event, model: calls.append(event.event_id) or 'observed'
        assembly = ResearchAssembly(context_memory=lambda q: '', memory_processor=processor)
        runner, service = self.install(assembly)
        ident = service.observe(observation())
        service.refresh()
        runner.close()
        path = self.root / 'feedback.json'
        saved = json.loads(path.read_text())
        row = saved['jobs'][ident]
        row.update(status='running', published=False)
        row.pop('outputs', None)
        path.write_text(json.dumps(saved))
        _, restored = self.install(assembly)
        restored.refresh()
        self.assertEqual(calls, [ident])
        self.assertEqual(restored.learning.state['jobs'][ident]['products']['context_memory']['status'], 'interrupted')

    def test_products_have_independent_content_limits(self):
        class Six:
            max_calls_per_job = 0
            def __init__(self, kind):
                self.kind = kind
            def process(self, row, backend, limits):
                return {'status': 'completed', 'charged_calls': 0, 'proposals': [
                    {'kind': self.kind, 'subject': 'meta_agent', 'content': str(i),
                     'applicability': 'fixture', 'validation_plan': 'fixture',
                     'source_refs': [row['input']['source_refs'][0]]} for i in range(6)]}
        _, service = self.install(ResearchAssembly(context_memory=lambda q: '',
            memory_processor=Six('memory'), harness_proposals=Six('harness_change')))
        ident = service.observe(observation())
        service.refresh()
        self.assertEqual(service.learning.state['jobs'][ident]['status'], 'completed')
        self.assertEqual(len(service.learning.state['jobs'][ident]['proposals']), 12)
        self.assertEqual(len(service.proposals()), 6)


if __name__ == '__main__':
    unittest.main()
