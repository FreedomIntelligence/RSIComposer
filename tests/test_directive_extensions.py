"""Custom system instructions use public definitions/handlers and the normal runner."""
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import get_type_hints
import unittest
from unittest.mock import Mock

from rsisys.application import Application, ApplicationConfig
from rsisys.assembly import build_default_exploration
from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.coordination import DecisionEndpoint, LoopStepResult, OperationResult
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import BackendBinding, GenerationResponse, ModelRef
from rsisys.loops.decision_making.action_space import DemoDecisionActionSpace
from rsisys.loops.decision_making.action_space.directives import (
    DirectiveDefinition, DirectiveDefinitions, DirectiveSpec, builtin_directives,
)
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.harness.state import DecisionContext
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.loops.exploration.action_space import EchoActionSpace
from rsisys.loops.exploration.harness.commands import CommandHandler, ExplorationCommands
from rsisys.loops.exploration.harness.engine import DefaultExplorationHarness
from rsisys.loops.exploration.loop import ExplorationLoop
from rsisys.runtime.runner import Runner
from rsisys.snapshot.local import LocalSnapshotCapture

MODEL = ModelRef('test', 'model', 'v1')


@dataclass(frozen=True)
class TuningSpec:
    temperature: float


def parse_tuning(arguments, candidates):
    if set(arguments) != {'temperature'} or type(arguments['temperature']) not in (int, float):
        raise ValueError('temperature must be a number')
    if not 0 <= arguments['temperature'] <= 2:
        raise ValueError('temperature is outside the supported range')
    return TuningSpec(arguments['temperature'])


def tuning_definition():
    return DirectiveDefinition(DirectiveSpec('tune', 'Tune a serving parameter', {
        'type': 'object', 'properties': {'temperature': {'type': 'number', 'minimum': 0, 'maximum': 2}},
        'required': ['temperature'], 'additionalProperties': False,
    }), parse_tuning)


class TuningHandler(CommandHandler):
    def __init__(self, *, lose_response=False):
        self.lose_response = lose_response
        self.effects = {}
        self.keys = []
        self.calls = 0
        self.cancelled = False

    def prepare(self, harness, state):
        if not isinstance(state.directive.payload, TuningSpec):
            raise TypeError('tune requires TuningSpec')
        state.request = state.directive.payload

    def advance(self, harness, state, action_space):
        self.calls += 1
        if state.phase in ('accepted', 'submitting'):
            state.phase = 'submitting'
            self.keys.append(state.idempotency_key)
            self.effects.setdefault(state.idempotency_key, state.request.temperature)
            if self.lose_response:
                self.lose_response = False
                raise OSError('submission response lost')
            state.phase = 'running'
            return None
        self.cancelled = state.cancel_requested
        status = ExecutionStatus.CANCELLED if self.cancelled else ExecutionStatus.COMPLETED
        return Feedback(state.directive.directive_id, status, 'serving parameter handled', True)


class DirectiveExtensionTests(unittest.TestCase):
    def build(self, handler=None):
        handler = handler if handler is not None else TuningHandler()
        definitions = DirectiveDefinitions((*builtin_directives(), tuning_definition()))
        backend = Mock(generate=Mock(return_value=GenerationResponse(json.dumps({
            'kind': 'explore', 'mode': 'tune', 'strategy': 'adjust serving',
            'arguments': {'temperature': 0.4},
        }), MODEL, True)))
        experience = BasicExperienceSystem(InMemoryExperienceStore())
        decision = DecisionMakingLoop(DefaultDecisionHarness(
            ModelDecisionPolicy(backend, MODEL, directives=definitions), Mock(), experience),
            DemoDecisionActionSpace(directive_modes=('tune',), directives=definitions))
        actor = Mock(act=Mock(side_effect=AssertionError('System instruction reached Actor')))
        environment = Mock()
        action_space = Mock(spec=EchoActionSpace)
        exploration = ExplorationLoop(DefaultExplorationHarness(
            actor, environment, experience, command_handlers={'tune': handler}), action_space)
        return Runner(decision, exploration), handler, backend

    def test_custom_instruction_passes_schema_parser_handler_and_feedback(self):
        runner, handler, backend = self.build()
        self.assertEqual(runner.step('task').kind, 'waiting')
        context = json.loads(backend.generate.call_args.args[0].prompt)
        self.assertEqual(context['directives'][0]['mode'], 'tune')
        self.assertIn('temperature', context['directives'][0]['input_schema']['properties'])
        self.assertEqual(runner.step('task').kind, 'completed')
        self.assertEqual(list(handler.effects.values()), [0.4])
        self.assertFalse(runner.busy)
        runner.exploration.harness.policy.act.assert_not_called()
        self.assertEqual(runner.exploration.harness.environment.mock_calls, [])
        self.assertEqual(runner.exploration.action_space.mock_calls, [])
        self.assertEqual(runner.decision_making.harness.state.completed, 1)
        self.assertTrue(runner.decision_making.harness.experience.retrieve())
        backend.generate.assert_called_once()

    def test_uncertain_submission_and_feedback_delivery_do_not_repeat_effects(self):
        runner, handler, backend = self.build(TuningHandler(lose_response=True))
        with self.assertRaisesRegex(OSError, 'submission'):
            runner.step('task')
        self.assertEqual(runner.step('task').kind, 'waiting')
        receive = runner.decision_making.receive_feedback
        runner.decision_making.receive_feedback = Mock(side_effect=OSError('feedback unavailable'))
        with self.assertRaisesRegex(OSError, 'feedback'):
            runner.step('task')
        calls = handler.calls
        runner.decision_making.receive_feedback = receive
        self.assertEqual(runner.step('task').kind, 'completed')
        self.assertEqual(handler.calls, calls)
        self.assertEqual(len(handler.effects), 1)
        self.assertEqual(len(handler.keys), 2)
        self.assertEqual(handler.keys[0], handler.keys[1])
        backend.generate.assert_called_once()

    def test_cancel_before_submission_has_no_effect(self):
        runner, handler, _ = self.build()
        directive = runner.decision_making.step('task').directive
        runner.cancel()
        result = runner.step('task')
        self.assertEqual(result.event['feedback']['status'], ExecutionStatus.CANCELLED)
        self.assertEqual(handler.calls, 0)
        self.assertEqual(handler.effects, {})
        self.assertEqual(runner.exploration.inspect(directive.directive_id).phase, 'cancelled')

    def test_cancel_after_uncertain_submission_resolves_same_work(self):
        runner, handler, _ = self.build(TuningHandler(lose_response=True))
        with self.assertRaises(OSError):
            runner.step('task')
        runner.cancel()
        self.assertEqual(runner.step('task').kind, 'waiting')
        self.assertEqual(runner.step('task').event['feedback']['status'], ExecutionStatus.CANCELLED)
        self.assertTrue(handler.cancelled)
        self.assertEqual(len(handler.effects), 1)

    def test_cancel_after_phase_unchanged_submission_still_reaches_handler(self):
        class DataOnlyHandler(TuningHandler):
            def advance(self, harness, state, action_space):
                self.calls += 1
                if not state.data.get('submitted'):
                    state.data['submitted'] = True
                    self.effects[state.idempotency_key] = state.request.temperature
                    raise OSError('submission response lost')
                self.cancelled = state.cancel_requested
                return Feedback(state.directive.directive_id, ExecutionStatus.CANCELLED, 'cancelled', False)

        runner, handler, _ = self.build(DataOnlyHandler())
        with self.assertRaises(OSError):
            runner.step('task')
        runner.cancel()
        result = runner.step('task')
        self.assertEqual(handler.calls, 2)
        self.assertTrue(handler.cancelled)
        self.assertTrue(result.event['feedback']['simulated'])
        self.assertTrue(runner.decision_making.harness.state.last_feedback.simulated)

    def test_payload_is_validated_before_dispatch(self):
        @dataclass(eq=False)
        class IdentityPayload:
            text: str

        for payload in (object(), IdentityPayload('x'), {'value': float('nan')}, {'value': b'bytes'}):
            with self.subTest(payload=type(payload)), self.assertRaisesRegex(TypeError, 'payload must'):
                Directive('bad', 'task', 'strategy', mode='custom', payload=payload)
        payload = {'values': [1, True, None, 'text'], 'nested': TuningSpec(0.2)}
        directive = Directive('valid', 'task', 'strategy', mode='custom', payload=payload)
        self.assertEqual(directive.payload, payload)

    def test_missing_handler_fails_and_feedback_is_delivered(self):
        runner, _, _ = self.build()
        harness = runner.exploration.harness
        harness.commands = ExplorationCommands(harness)
        with self.assertRaisesRegex(NotImplementedError, 'Unsupported exploration mode'):
            runner.step('task')
        self.assertFalse(runner.busy)
        self.assertEqual(runner.decision_making.harness.state.last_feedback.status, ExecutionStatus.FAILED)

    def test_invalid_feedback_identity_keeps_receipt_pending(self):
        handler = TuningHandler()
        handler.advance = Mock(return_value=Feedback('wrong', ExecutionStatus.COMPLETED, '', True))
        runner, _, _ = self.build(handler)
        with self.assertRaisesRegex(ValueError, 'match its directive'):
            runner.step('task')
        self.assertTrue(runner.busy)
        directive = runner.decision_making.pending()
        self.assertFalse(runner.exploration.inspect(directive.directive_id).terminal)

    def test_catalog_isolation_and_parser_validation(self):
        definition = tuning_definition()
        catalog = DirectiveDefinitions([definition])
        definition.spec.input_schema.clear()
        described = catalog.describe()[0]
        self.assertIn('properties', described.input_schema)
        described.input_schema.clear()
        self.assertIn('properties', catalog.describe()[0].input_schema)
        self.assertNotIn('tune', DirectiveDefinitions().modes())
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            DirectiveDefinitions([definition, definition])
        with self.assertRaisesRegex(ValueError, 'temperature'):
            catalog.parse('tune', {'temperature': 'high'})
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            catalog.parse('missing', {})
        with self.assertRaisesRegex(TypeError, 'object'):
            catalog.parse('tune', [])
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            DemoDecisionActionSpace(directive_modes=('tune',))

    def test_mixed_argument_forms_are_rejected(self):
        backend = Mock(generate=Mock(return_value=GenerationResponse(json.dumps({
            'kind': 'explore', 'mode': 'configure_prompt', 'strategy': '',
            'arguments': {'instructions': 'one'}, 'instructions': 'two',
        }), MODEL, False)))
        with self.assertRaisesRegex(ValueError, 'not both'):
            ModelDecisionPolicy(backend, MODEL).decide(DecisionContext(
                'task', (), (), (), directive_modes=('configure_prompt',)))

    def test_public_endpoint_type_hints_resolve_without_concrete_harness(self):
        self.assertIs(get_type_hints(DecisionEndpoint.step)['return'], LoopStepResult)
        self.assertEqual(OperationResult.__module__, 'rsisys.loops.contracts.coordination')
        from rsisys.loops.decision_making.harness.decision import LoopStepResult as legacy_result
        from rsisys.loops.decision_making.action_space import OperationResult as legacy_operation
        self.assertIs(legacy_result, LoopStepResult)
        self.assertIs(legacy_operation, OperationResult)

    def test_snapshot_rejects_custom_assembly_catalogs_and_handlers_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, change in enumerate(('factory', 'space', 'policy', 'handler')):
                with self.subTest(change=change):
                    config = ApplicationConfig('task', Path(directory) / str(index),
                        {'test': BackendBinding('test', 'mock')}, {'actor': MODEL, 'decision_maker': MODEL})
                    kwargs = {'exploration_factory': build_default_exploration} if change == 'factory' else {}
                    with Application(config, **kwargs) as app:
                        dm, actor = app.runner.decision_making, app.runner.exploration.harness
                        if change == 'space':
                            dm.action_space.directives = DirectiveDefinitions(builtin_directives())
                        elif change == 'policy':
                            dm.harness.policy.directives = DirectiveDefinitions(builtin_directives())
                        elif change == 'handler':
                            actor.commands = ExplorationCommands(actor, {'tune': TuningHandler()})
                        with self.assertRaisesRegex(TypeError, 'explicit snapshot integration'):
                            LocalSnapshotCapture(app).capture('custom')
                        self.assertFalse((config.data_dir / 'snapshots').exists())


if __name__ == '__main__':
    unittest.main()
