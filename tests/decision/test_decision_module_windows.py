"""Replace decision harness/policy independently without losing host boundaries."""
from tests import ROOT as REPO_ROOT
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from rsisys.adapters.decision import ContextDecisionPolicy, DecisionInputAdapter
from rsisys.config import DemoConfig, build_demo
from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops._memory import Memory
from rsisys.loops.contracts.coordination import (
    ExplorationProgress, LoopStepResult, OperationResult, ReviewDecision,
)
from rsisys.loops.contracts.decision import Decision, DecisionInput, OperationCall, OperationSpec
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef, TokenUsage
from rsisys.loops.decision_making.action_space import DemoDecisionActionSpace, ModularDecisionActionSpace
from rsisys.loops.decision_making.action_space.access import ActionPermissions
from rsisys.loops.decision_making.feedback.learning import SerializedLearningBackend
from rsisys.loops.decision_making.harness.context import IdentityContextCompactor
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.learning_harness import enable_feedback_learning
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.harness.tools import BoundDecisionTools
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.research import ResearchAssembly, assemble_research
from rsisys.runtime.run_policy import TimeRunPolicy
from rsisys.snapshot.codec import decode, encode
from tests.feedback.test_feedback_learning_integration import ImmediateExecutor


CONFIG = DemoConfig.load(REPO_ROOT / 'configs/demo.json')
MODEL = ModelRef('fixture', 'meta-agent', 'v1')


class TransparentPolicy:
    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def decide(self, context):
        return self.inner.decide(context)


class Responses:
    supports_response_schema = False

    def __init__(self, *payloads, wrong_identity=False):
        self.payloads = list(payloads or ('{"kind":"finish"}',))
        self.requests = []
        self.wrong_identity = wrong_identity

    def generate(self, request):
        self.requests.append(request)
        model = replace(request.model, revision='wrong') if self.wrong_identity else request.model
        return GenerationResponse(self.payloads.pop(0), model, True, TokenUsage(11, 3))


class RecordingCompactor:
    def __init__(self):
        self.limits = []

    def compact(self, context, *, recent_limit=32):
        self.limits.append(recent_limit)
        return context


class MinimalHarness:
    def __init__(self, *, validate=True, revoke=None):
        self.validate, self.revoke = validate, revoke
        self.directive = None
        self.done = False

    def step(self, objective, space):
        if self.done:
            return LoopStepResult('finished')
        self.directive = Directive('minimal', objective, 'fixture')
        if self.validate:
            space.validate_directive(self.directive)
        if self.revoke:
            self.revoke()
        return LoopStepResult('dispatch', directive=self.directive)

    def pending(self):
        return self.directive if not self.done else None

    def receive_feedback(self, directive, feedback):
        assert directive == self.directive and feedback.directive_id == directive.directive_id
        self.done = True


class Definition:
    spec = OperationSpec('inspect', 'Read fixture', {'type': 'object'}, read_only=True)

    def validate(self, arguments):
        if arguments:
            raise ValueError('No arguments')
        return OperationCall('inspect', {})


def simple_loop(policy, **options):
    tools = BoundDecisionTools({'inspect': lambda arguments: OperationResult('measured zero', (), True,
        data={'quality_metrics': {'accuracy': 0.0}})})
    harness = DefaultDecisionHarness(policy, tools, BasicExperienceSystem(InMemoryExperienceStore()),
        context_compactor=IdentityContextCompactor(), **options)
    return DecisionMakingLoop(harness, ModularDecisionActionSpace((Definition(),)))


class DecisionHarnessWindowTests(unittest.TestCase):
    def test_minimal_harness_authorization_is_same_in_researcher_and_time_modes(self):
        for run_policy in (None, TimeRunPolicy(60)):
            for allowed in (False, True):
                with self.subTest(mode=type(run_policy).__name__, allowed=allowed):
                    def factory(context, exploration):
                        return DecisionMakingLoop(MinimalHarness(), DemoDecisionActionSpace(
                            permissions=context.action_permissions))
                    with build_demo(CONFIG, decision_factory=factory, run_policy=run_policy,
                            action_permissions=ActionPermissions(directives=None if allowed else ()))[0] as runner:
                        if allowed:
                            self.assertEqual(runner.run('fixture', 1).rounds_completed, 1)
                            self.assertFalse(hasattr(runner.decision_making.harness, 'policy'))
                        else:
                            with self.assertRaises(PermissionError):
                                runner.run('fixture', 1)
                            self.assertFalse(runner.exploration.pending_all())

    def test_host_checks_dispatch_if_plugin_skips_validation_or_permission_is_revoked(self):
        for validate in (False, True):
            with self.subTest(validate=validate):
                space = DemoDecisionActionSpace(permissions=ActionPermissions(
                    directives=None if validate else ()))
                harness = MinimalHarness(validate=validate, revoke=(lambda: setattr(space, 'permissions',
                    ActionPermissions(directives=()))) if validate else None)
                with self.assertRaises(PermissionError):
                    DecisionMakingLoop(harness, space).step('fixture')

    def test_host_checks_review_result_for_both_optional_review_shapes(self):
        for with_tools in (False, True):
            space = DemoDecisionActionSpace(permissions=ActionPermissions(allow_cancel=False))
            harness = MinimalHarness()
            if with_tools:
                harness.review_with_tools = lambda objective, progress, space: ReviewDecision('cancel')
            else:
                harness.review = lambda objective, progress: ReviewDecision('cancel')
            loop = DecisionMakingLoop(harness, space)
            for guarded in (False, True):
                with self.subTest(with_tools=with_tools, guarded=guarded), self.assertRaises(PermissionError):
                    if guarded:
                        loop.review_guarded('fixture', None, lambda: None)
                    else:
                        loop.review('fixture', None)

    def test_guarded_dispatch_checks_boundary_even_if_plugin_skips_it(self):
        loop = DecisionMakingLoop(MinimalHarness(validate=False), DemoDecisionActionSpace())
        def deadline():
            raise TimeoutError('host deadline')
        with self.assertRaisesRegex(TimeoutError, 'host deadline'):
            loop.step_guarded('fixture', deadline)

    def test_malformed_harness_result_never_crosses_endpoint(self):
        harness = MinimalHarness()
        harness.step = lambda objective, space: ('progressed',)
        loop = DecisionMakingLoop(harness, DemoDecisionActionSpace())
        with self.assertRaisesRegex(TypeError, 'LoopStepResult'):
            loop.step('fixture')
        bad = LoopStepResult('progressed')
        object.__setattr__(bad, 'kind', 'dispatch')
        harness.step = lambda objective, space: bad
        with self.assertRaisesRegex(ValueError, 'directive'):
            loop.step('fixture')


class DecisionPolicyCapabilityTests(unittest.TestCase):
    def test_wrapped_policy_accounts_retries_and_failure_responses_equally(self):
        for wrapped in (False, True):
            for invalid_final in (False, True):
                with self.subTest(wrapped=wrapped, invalid_final=invalid_final):
                    backend = Responses('invalid', 'invalid' if invalid_final else '{"kind":"finish"}')
                    inner = ModelDecisionPolicy(backend, MODEL, max_validation_attempts=2,
                        context_compactor=IdentityContextCompactor())
                    loop = simple_loop(TransparentPolicy(inner) if wrapped else inner)
                    if invalid_final:
                        with self.assertRaises(json.JSONDecodeError):
                            loop.step('fixture')
                    else:
                        self.assertEqual(loop.step('fixture').kind, 'finished')
                    self.assertEqual(len(backend.requests), 2)
                    self.assertEqual((loop.harness.state.measured_input_tokens,
                        loop.harness.state.measured_output_tokens), (22, 6))
                    self.assertIsNone(inner._usage_recorder)

    def test_identity_failure_is_counted_and_not_retried_through_wrapper(self):
        backend = Responses(wrong_identity=True)
        inner = ModelDecisionPolicy(backend, MODEL, max_validation_attempts=2)
        loop = simple_loop(TransparentPolicy(inner))
        with self.assertRaisesRegex(ValueError, 'identity'):
            loop.step('fixture')
        self.assertEqual(len(backend.requests), 1)
        self.assertEqual(loop.harness.state.measured_input_tokens, 11)
        self.assertIsNone(inner._usage_recorder)
        self.assertIsNone(inner._context_runtime)

    def test_final_result_usage_is_fallback_for_policy_without_observer(self):
        loop = simple_loop(ContextDecisionPolicy(lambda window: Decision('finish', usage=TokenUsage(7, 4))))
        loop.step('fixture')
        self.assertEqual((loop.harness.state.measured_input_tokens,
            loop.harness.state.measured_output_tokens), (7, 4))

    def test_explicit_compactor_preserved_and_host_default_used_once_otherwise(self):
        for wrapped in (False, True):
            for explicit in (False, True):
                with self.subTest(wrapped=wrapped, explicit=explicit):
                    own, host = RecordingCompactor(), RecordingCompactor()
                    policy = ModelDecisionPolicy(Responses(), MODEL,
                        context_compactor=own if explicit else None)
                    original = policy.context_compactor
                    loop = simple_loop(TransparentPolicy(policy) if wrapped else policy, memory_limit=7)
                    loop.harness.context_compactor = host
                    loop.step('fixture')
                    self.assertIs(policy.context_compactor, original)
                    self.assertEqual(own.limits, [7] if explicit else [])
                    self.assertEqual(host.limits, [] if explicit else [7])
                    self.assertIsNone(policy._context_runtime)

    def test_policy_scopes_restore_previous_values_after_transport_error(self):
        class FailingBackend(Responses):
            def generate(self, request):
                raise OSError('transport')
        policy = ModelDecisionPolicy(FailingBackend(), MODEL)
        loop = simple_loop(TransparentPolicy(policy))
        previous = RecordingCompactor()
        seen = []
        with policy.context_scope(previous, recent_limit=5), policy.track_usage(seen.append):
            with self.assertRaisesRegex(OSError, 'transport'):
                loop.step('fixture')
            self.assertEqual(policy._context_runtime, (previous, 5))
            self.assertEqual(policy._usage_recorder, seen.append)
        self.assertIsNone(policy._context_runtime)
        self.assertIsNone(policy._usage_recorder)

    def test_policy_without_final_request_capability_gets_one_host_projection(self):
        seen = []
        loop = simple_loop(ContextDecisionPolicy(lambda window: seen.append(window) or Decision('finish')))
        host = RecordingCompactor()
        loop.harness.context_compactor = host
        loop.step('fixture')
        self.assertEqual(host.limits, [32])
        self.assertIsInstance(seen[0], DecisionInput)

    def test_usage_observation_also_covers_context_scope_setup_and_teardown(self):
        class ScopedPolicy:
            @contextmanager
            def track_usage(self, record):
                self.record = record
                try:
                    yield
                finally:
                    self.record = None

            @contextmanager
            def context_scope(self, compactor, *, recent_limit):
                self.record(TokenUsage(2, 1))
                try:
                    yield
                finally:
                    self.record(TokenUsage(5, 3))

            def decide(self, window):
                self.record(TokenUsage(7, 4))
                return Decision('finish', usage=TokenUsage(7, 4))

        policy = ScopedPolicy()
        loop = simple_loop(ContextDecisionPolicy(policy))
        loop.step('fixture')
        self.assertEqual((loop.harness.state.measured_input_tokens,
            loop.harness.state.measured_output_tokens), (14, 8))
        self.assertIsNone(policy.record)

    def test_shared_backend_updates_actual_wrapped_policy_without_shadowing(self):
        backend = Responses()
        inner = ModelDecisionPolicy(backend, MODEL)
        wrapper = TransparentPolicy(inner)
        loop = simple_loop(wrapper)
        enable_feedback_learning(loop)
        self.addCleanup(loop.harness.close)
        self.assertNotIn('backend', vars(wrapper))
        self.assertIsInstance(inner.backend, SerializedLearningBackend)
        self.assertIs(loop.feedback_service.learning.backend, inner.backend)
        self.assertIs(inner.backend.backend, backend)


class PublicDecisionInputTests(unittest.TestCase):
    def test_text_window_executes_tools_and_receives_memory_and_actual_results(self):
        windows = []
        def decide(window):
            windows.append(window)
            if len(windows) == 1:
                self.assertIsInstance(window, DecisionInput)
                self.assertIn('known failure', window.context)
                self.assertIn('inspect', [row['name'] for row in window.capabilities if row['kind'] == 'invoke'])
                return Decision('invoke', operation=OperationCall('inspect', {}))
            self.assertIn('measured zero', window.context)
            self.assertIn('accuracy', window.context)
            return Decision('finish')
        loop = simple_loop(ContextDecisionPolicy(decide), model=MODEL)
        loop.harness.memory.remember(Memory('note', 'episodic', 'known failure'))
        first = loop.step('objective')
        self.assertEqual(first.operation_result.data['quality_metrics'], {'accuracy': 0.0})
        self.assertEqual(loop.step('objective').kind, 'finished')
        self.assertEqual(windows[0].objective, 'objective')

    def test_window_mutation_does_not_change_host_or_renderer_owned_values(self):
        references = {'ids': ['original']}
        seen = []
        adapter = DecisionInputAdapter(context=lambda view: 'custom context', references=lambda view: references)
        def decide(window):
            seen.append(window)
            window.capabilities[0]['input_schema']['type'] = 'forged'
            window.references['ids'].append('forged')
            return Decision('finish')
        loop = simple_loop(ContextDecisionPolicy(decide, input_adapter=adapter))
        loop.step('fixture')
        self.assertEqual(seen[0].context, 'custom context')
        self.assertEqual(references, {'ids': ['original']})
        self.assertEqual(loop.action_space.describe()[0].input_schema, {'type': 'object'})

    def test_review_window_advertises_controls_and_permitted_tools_without_exploration(self):
        class PublicPolicy:
            def decide(self, window):
                return Decision('finish')
            def review(self, window):
                self.window = window
                return ReviewDecision('continue')
        policy = PublicPolicy()
        loop = simple_loop(ContextDecisionPolicy(policy))
        directive = Directive('active', 'objective', 'fixture')
        loop.harness.state.pending_directive = directive
        loop.harness.state.pending_directive_id = directive.directive_id
        loop.review('objective', ExplorationProgress('active', 'running', directive))
        self.assertEqual({row['kind'] for row in policy.window.capabilities}, {'invoke', 'review'})
        self.assertEqual([row['name'] for row in policy.window.capabilities if row['kind'] == 'review'],
                         ['continue', 'cancel'])

    def test_policy_without_review_capability_continues(self):
        adapter = ContextDecisionPolicy(lambda window: Decision('finish'))
        self.assertEqual(adapter.review(simple_loop(adapter).harness._context(
            'fixture', ModularDecisionActionSpace(()))).action, 'continue')

    def test_invalid_input_or_adapter_output_fails_before_policy(self):
        for kwargs in ({'objective': 1}, {'objective': 'ok', 'context': {}},
                       {'objective': 'ok', 'references': float('nan')},
                       {'objective': 'ok', 'capabilities': ({'bad': object()},)}):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                DecisionInput(**kwargs)
        called = []
        loop = simple_loop(ContextDecisionPolicy(lambda window: called.append(window),
            input_adapter=lambda view: 'not an input'))
        with self.assertRaisesRegex(TypeError, 'DecisionInput'):
            loop.step('fixture')
        self.assertFalse(called)

    def test_public_types_keep_legacy_aliases_and_snapshot_wire_shape(self):
        from rsisys.loops.decision_making.harness.decision import Decision as LegacyDecision
        from rsisys.loops.decision_making.action_space import OperationCall as LegacyCall
        self.assertIs(LegacyDecision, Decision)
        self.assertIs(LegacyCall, OperationCall)
        decision = Decision('invoke', operation=OperationCall('inspect', {}), decision_id='stable')
        encoded = encode(decision)
        self.assertEqual(encoded['$type'], 'Decision')
        self.assertEqual(encoded['fields']['operation']['$type'], 'OperationCall')
        self.assertEqual(decode(encoded), decision)

    def test_snapshot_refuses_unintegrated_identity_or_policy_owned_compactor(self):
        from tempfile import TemporaryDirectory
        from rsisys.application import Application, ApplicationConfig
        from rsisys.snapshot.local import LocalSnapshotCapture
        config = ApplicationConfig.load(REPO_ROOT / 'configs/local.json')
        for customized in ('identity', 'compactor'):
            with self.subTest(customized=customized), TemporaryDirectory() as directory:
                with Application(replace(config, data_dir=Path(directory))) as app:
                    harness = app.runner.decision_making.harness
                    if customized == 'identity':
                        harness.model_provider = lambda: harness.policy.model
                        expected = 'identity providers'
                    else:
                        harness.policy.context_compactor = IdentityContextCompactor()
                        expected = 'Policy-owned context compactors'
                    with self.assertRaisesRegex(TypeError, expected):
                        LocalSnapshotCapture(app).capture('unintegrated')
                    self.assertFalse((Path(directory) / 'snapshots').exists())


class PlainPolicyFeedbackTests(unittest.TestCase):
    def test_zero_inference_feedback_has_no_backend_and_context_reaches_next_decision(self):
        windows, memory = [], {}
        def policy(window):
            windows.append(window)
            return Decision('invoke', operation=OperationCall('inspect', {}))
        def process(event, model):
            memory['text'] = 'Current observed result: ' + str(event.output)
            return {'native': 'stored'}
        loop = simple_loop(ContextDecisionPolicy(policy), model=MODEL)
        service = assemble_research(loop, assembly=ResearchAssembly(
            context_memory=lambda query: memory.get('text', ''), memory_processor=process),
            executor=ImmediateExecutor())
        self.addCleanup(loop.harness.close)
        self.assertIsNone(service.learning.backend)
        service.observe_output('quality=zero', task_id='first', status='completed')
        loop.step('fixture')
        service.refresh()
        loop.step('fixture')
        self.assertIn('quality=zero', windows[-1].context)
        self.assertEqual(service.learning.state['attempted'], 0)

    def test_host_identity_provider_is_shared_and_refreshes_for_each_capture(self):
        active = [MODEL]
        loop = simple_loop(ContextDecisionPolicy(lambda window: Decision('finish')),
            model_provider=lambda: active[0])
        service = assemble_research(loop, assembly=ResearchAssembly(context_memory=lambda query: 'memory'))
        self.addCleanup(loop.harness.close)
        for version in ('v1', 'v2'):
            active[0] = replace(MODEL, revision=version)
            self.assertEqual(loop.harness._context('fixture', loop.action_space).decision_model, active[0])
            ident = service.observe_output('observed', task_id=version, status='completed')
            self.assertEqual(service.learning.state['jobs'][ident]['model']['revision'], version)

    def test_missing_identity_or_inference_backend_rejected_before_install(self):
        for with_identity in (False, True):
            with self.subTest(with_identity=with_identity):
                loop = simple_loop(ContextDecisionPolicy(lambda window: Decision('finish')),
                    model=MODEL if with_identity else None)
                with self.assertRaisesRegex(ValueError, 'backend' if with_identity else 'model binding'):
                    enable_feedback_learning(loop)
                self.assertFalse(hasattr(loop, 'feedback_service'))
                self.assertFalse(loop.harness._feedback_listeners)

    def test_feedback_identity_cannot_diverge_from_host_identity(self):
        loop = simple_loop(ContextDecisionPolicy(lambda window: Decision('finish')), model=MODEL)
        with self.assertRaisesRegex(ValueError, 'active meta-agent'):
            assemble_research(loop, assembly=ResearchAssembly(context_memory=lambda query: ''),
                model_provider=lambda: replace(MODEL, revision='other'))
        self.assertFalse(hasattr(loop, 'feedback_service'))

    def test_feedback_identity_cannot_fill_in_a_missing_host_binding(self):
        for options in ({'model': MODEL}, {'model_provider': lambda: MODEL}):
            with self.subTest(option=next(iter(options))):
                loop = simple_loop(ContextDecisionPolicy(lambda window: Decision('finish')))
                with self.assertRaisesRegex(ValueError, 'host meta-agent model binding'):
                    assemble_research(loop, assembly=ResearchAssembly(context_memory=lambda query: ''), **options)
                self.assertIsNone(loop.harness.current_model())
                self.assertFalse(hasattr(loop, 'feedback_service'))

    def test_authoritative_identity_provider_cannot_fall_back_to_stale_binding(self):
        loop = simple_loop(ContextDecisionPolicy(lambda window: Decision('finish')),
            model=MODEL, model_provider=lambda: None)
        with self.assertRaisesRegex(TypeError, 'provider must return'):
            loop.step('fixture')

    def test_dynamic_proxy_does_not_fabricate_an_identity_or_optional_capabilities(self):
        policy = Mock()
        policy.decide.return_value = Decision('finish')
        loop = simple_loop(policy)
        self.assertEqual(loop.step('fixture').kind, 'finished')
        self.assertIsNone(loop.harness.current_model())
        policy.model = 'invalid declared identity'
        with self.assertRaisesRegex(TypeError, 'ModelRef'):
            loop.harness.current_model()

    def test_host_identity_provider_cannot_relabel_a_different_policy_model(self):
        backend = Responses()
        loop = simple_loop(ModelDecisionPolicy(backend, MODEL),
            model_provider=lambda: replace(MODEL, revision='other'))
        with self.assertRaisesRegex(ValueError, 'current host binding'):
            loop.step('fixture')
        self.assertFalse(backend.requests)


if __name__ == '__main__':
    unittest.main()
