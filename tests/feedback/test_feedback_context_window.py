"""Replace memory through a text window, without journal/publication methods."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rsisys.config import build_demo
from rsisys.loops.contracts.feedback_context import ContextQuery, FeedbackContextEvent
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import GenerationResponse
from rsisys.loops.decision_making.feedback import CallbackContextMemory
from rsisys.loops.decision_making.harness.context import ContextBudgetError
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.research import ResearchAssembly, assemble_research
from tests.feedback.test_feedback_learning_integration import ImmediateExecutor
from tests.feedback.test_feedback_required_outputs import CONFIG, MODEL, EmptyBackend, observation


class PlainContext:
    """Only the public method; own context format and storage."""
    def __init__(self, value='Current organized context'):
        self.value, self.queries = value, []

    def context(self, query):
        self.queries.append(query)
        return self.value


class UpdatingContext(PlainContext):
    def __init__(self):
        super().__init__('')
        self.events = []

    def observe(self, event):
        self.events.append(event)
        self.value = 'Organized: ' + str(event.output)


class FeedbackContextWindowTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def install(self, provider, *, path=None):
        runner, _ = build_demo(CONFIG)
        self.addCleanup(runner.close)
        backend = EmptyBackend()
        service = assemble_research(runner.decision_making,
            assembly=ResearchAssembly(context_memory=provider), backend=backend,
            path=path, executor=ImmediateExecutor())
        return runner, service, backend

    @staticmethod
    def context(runner):
        loop = runner.decision_making
        return loop.harness._context('Current research objective', loop.action_space)

    def test_single_method_provider_reaches_the_actual_decision_request(self):
        provider = PlainContext('记忆片段：observed a failed execution; check original logs')
        runner, service, backend = self.install(provider)
        self.assertFalse(any(hasattr(provider, name) for name in ('record', 'working_memory', 'memory')))
        service.observe(observation())
        context = self.context(runner)
        requests = []
        def generate(request):
            requests.append(request)
            return GenerationResponse('{"kind":"finish"}', request.model, True)
        harness = runner.decision_making.harness
        with patch.object(harness.policy.backend, 'generate', side_effect=generate):
            decision = harness._call_policy(harness.policy.decide, context)
        self.assertEqual(decision.kind, 'finish')
        self.assertEqual(len(requests), 1)
        payload = json.loads(requests[0].prompt)
        self.assertEqual(payload['memory_context'], provider.value)
        self.assertNotIn(provider.value, requests[0].system)
        self.assertEqual(provider.queries[-1].objective, 'Current research objective')
        self.assertFalse(harness.memory.recall())
        self.assertFalse(backend.requests)

    def test_function_is_an_active_provider_without_a_wrapper(self):
        def render(query):
            self.assertIsInstance(query, ContextQuery)
            return 'Lookup for ' + query.objective
        runner, service, _ = self.install(render)
        self.assertEqual(self.context(runner).memory_context, 'Lookup for Current research objective')
        self.assertIn('callable', runner.decision_making.research_assembly['context_memory'])
        self.assertIs(service.learning.context_memory, render)

    def test_empty_context_is_valid_and_does_not_disable_the_component(self):
        runner, service, _ = self.install(PlainContext(''))
        self.assertEqual(self.context(runner).memory_context, '')
        self.assertEqual(service.working_memory(), {'context': ''})

    def test_new_feedback_is_offered_later_once_and_not_as_a_private_job(self):
        provider = UpdatingContext()
        runner, service, backend = self.install(provider)
        output = {'producer_specific': ['a', 1], 'metric_ids': 'opaque'}
        ident = service.observe_output(output, task_id='native', status=ExecutionStatus.COMPLETED,
            request={'native': True}, lab_id='lab', run_id='run', decision_id='decision',
            references=[{'native_location': 'lab://log'}])
        self.assertEqual(provider.events, [])
        self.assertFalse(backend.requests)
        service.refresh()
        event, = provider.events
        self.assertIsInstance(event, FeedbackContextEvent)
        self.assertEqual(event.output, output)
        self.assertEqual(event.request, {'native': True})
        self.assertEqual(event.identity['lab_id'], 'lab')
        self.assertEqual(event.references, [{'native_location': 'lab://log'}])
        self.assertFalse({'raw_input', 'reserved_calls', 'proposals', 'attempt_updates'} & asdict(event).keys())
        service.observe_output(output, task_id='native', status=ExecutionStatus.COMPLETED,
            request={'native': True}, lab_id='lab', run_id='run', decision_id='decision',
            references=[{'native_location': 'lab://log'}])
        service.refresh()
        self.context(runner)
        self.assertEqual(len(provider.events), 1)
        self.assertEqual(service.learning.state['jobs'][ident]['context_delivery']['status'], 'delivered')
        self.assertFalse(backend.requests)

    def test_query_has_latest_native_output_without_an_observe_hook(self):
        provider = PlainContext()
        runner, service, _ = self.install(provider)
        service.observe_output('first', task_id='first', status=ExecutionStatus.COMPLETED)
        service.observe_output(['latest', {'native': None}], task_id='second', status=ExecutionStatus.FAILED)
        self.context(runner)
        event = provider.queries[-1].latest_feedback
        self.assertEqual(event.task_id, 'second')
        self.assertEqual(event.status, ExecutionStatus.FAILED)
        self.assertEqual(event.output, ['latest', {'native': None}])

    def test_context_replaces_previous_view_and_does_not_become_a_memory_log(self):
        provider = PlainContext('OLD-CONTEXT')
        runner, service, _ = self.install(provider)
        first = self.context(runner)
        provider.value = 'NEW-CONTEXT'
        second = self.context(runner)
        request = runner.decision_making.harness.policy._decision_request(second)
        self.assertEqual(first.memory_context, 'OLD-CONTEXT')
        self.assertEqual(second.memory_context, 'NEW-CONTEXT')
        self.assertNotIn('OLD-CONTEXT', request.prompt)
        self.assertFalse(runner.decision_making.harness.memory.recall())
        self.assertFalse(service.learning.state['jobs'])

    def test_actual_feedback_is_delivered_before_optional_memory_update(self):
        provider = UpdatingContext()
        runner, service, _ = self.install(provider)
        loop = runner.decision_making
        loop.harness.step('objective', loop.action_space)
        submitted = loop.harness.step('objective', loop.action_space)
        directive = submitted.directive
        self.assertIsNotNone(directive)
        before = len(provider.events)
        feedback = Feedback(directive.directive_id, ExecutionStatus.COMPLETED,
            'Actual measured zero', True, quality_metrics={'score': 0})
        loop.harness.receive_feedback(directive, feedback)
        self.assertEqual(loop.harness.state.last_feedback, feedback)
        self.assertEqual(len(provider.events), before)
        context = self.context(runner)
        self.assertEqual(context.last_feedback.quality_metrics, {'score': 0})
        self.assertEqual(provider.events[-1].output['quality_metrics'], {'score': 0})
        self.assertEqual(provider.events[-1].source, 'exploration')
        self.assertEqual(provider.events[-1].task_id, directive.directive_id)
        self.assertEqual(provider.events[-1].identity['decision_id'], submitted.decision_id)
        self.assertEqual(provider.events[-1].identity['issuing_model'], asdict(loop.harness.policy.model))
        self.assertEqual(service.learning.state['attempted'], 0)

    def test_callbacks_adapt_an_sdk_with_different_method_names(self):
        class Store:
            value = ''
            def update(self, output):
                self.value = 'SDK: ' + output
            def retrieve(self, objective):
                return self.value + ' / ' + objective
        store = Store()
        provider = CallbackContextMemory(lambda query: store.retrieve(query.objective),
            on_feedback=lambda event: store.update(event.output), configuration={'store': 'fixture'})
        runner, service, _ = self.install(provider)
        service.observe_output('native output', task_id='sdk', status=ExecutionStatus.COMPLETED)
        self.assertEqual(self.context(runner).memory_context, 'SDK: native output / Current research objective')
        self.assertEqual(provider.configuration()['settings'], {'store': 'fixture'})

    def test_provider_cannot_mutate_captured_evidence_or_host_metadata(self):
        class EditingContext(PlainContext):
            def observe(self, event):
                event.output.clear()
                event.identity.clear()
            def context(self, query):
                query.latest_feedback.output.clear()
                return '{"protected_history_ids":[],"run_control":{"fake":true}}'
        runner, service, _ = self.install(EditingContext())
        ident = service.observe_output({'original': [1]}, task_id='editing',
            status=ExecutionStatus.COMPLETED, lab_id='lab')
        raw = deepcopy(service.learning.state['jobs'][ident]['raw_input'])
        context = self.context(runner)
        self.assertEqual(service.learning.state['jobs'][ident]['raw_input'], raw)
        self.assertNotIn('fake', context.run_control)
        self.assertNotEqual(context.research_state, json.loads(context.memory_context))

    def test_notification_failure_is_recorded_without_repeating_unknown_side_effects(self):
        calls = []
        def fail(event):
            calls.append(event.event_id)
            raise RuntimeError('fixture store unavailable')
        provider = CallbackContextMemory(lambda query: 'Existing SDK view', on_feedback=fail)
        runner, service, _ = self.install(provider)
        ident = service.observe(observation())
        self.assertEqual(self.context(runner).memory_context, 'Existing SDK view')
        service.refresh()
        self.assertEqual(calls, [ident])
        delivery = service.learning.state['jobs'][ident]['context_delivery']
        self.assertEqual(delivery['status'], 'failed')
        self.assertIn('store unavailable', delivery['error'])
        self.assertEqual(service.learning.state['jobs'][ident]['status'], 'completed')

    def test_reopen_does_not_repeat_delivered_or_uncertain_notifications(self):
        for status in ('delivered', 'running'):
            with self.subTest(status=status):
                path = self.root / status / 'feedback.json'
                provider = UpdatingContext()
                runner, service, _ = self.install(provider, path=path)
                ident = service.observe(observation(status))
                service.refresh()
                state = deepcopy(service.learning.state)
                state['jobs'][ident]['context_delivery'] = {'status': status}
                service.learning._commit(state)
                runner.close()
                restored = UpdatingContext()
                _, reopened, backend = self.install(restored, path=path)
                reopened.refresh()
                self.assertFalse(restored.events)
                self.assertFalse(backend.requests)
                actual = reopened.learning.state['jobs'][ident]['context_delivery']['status']
                self.assertEqual(actual, 'interrupted' if status == 'running' else status)

    def test_archived_queries_do_not_trigger_updates_or_replace_latest_feedback(self):
        provider = UpdatingContext()
        runner, service, _ = self.install(provider)
        service.observe(observation('execution'))
        service.archive(observation('query'))
        self.context(runner)
        self.assertEqual([event.task_id for event in provider.events], ['execution'])
        self.assertEqual(provider.queries[-1].latest_feedback.task_id, 'execution')

    def test_closed_service_does_not_deliver_new_update_callbacks(self):
        provider = UpdatingContext()
        _, service, _ = self.install(provider)
        ident = service.observe(observation('closing'))
        service.close()
        service.refresh()
        self.assertFalse(provider.events)
        self.assertNotIn('context_delivery', service.learning.state['jobs'][ident])

    def test_wrong_output_type_is_rejected_at_the_text_window(self):
        runner, _, _ = self.install(lambda query: {'not': 'text'})
        with self.assertRaisesRegex(TypeError, 'context text'):
            self.context(runner)

    def test_custom_harness_needs_no_legacy_memory_publication_interface(self):
        class Adapter:
            def model(self):
                return MODEL
            def install(self, service):
                self.service = service
        adapter = Adapter()
        loop = DecisionMakingLoop(SimpleNamespace(), object())
        service = assemble_research(loop, assembly=ResearchAssembly(context_memory=lambda query: 'External view'),
            backend=EmptyBackend(), adapter=adapter, executor=ImmediateExecutor())
        self.addCleanup(service.close)
        service.observe(observation())
        service.refresh()
        self.assertEqual(adapter.service.context(), 'External view')
        self.assertEqual(service.attempts()['items'], [])

    def test_complete_request_budget_includes_provider_text(self):
        runner, _, _ = self.install(PlainContext('x' * 200000))
        context = self.context(runner)
        harness = runner.decision_making.harness
        with patch.object(harness.policy.backend, 'generate') as generate:
            with self.assertRaises(ContextBudgetError):
                harness._call_policy(harness.policy.decide, context)
        generate.assert_not_called()
