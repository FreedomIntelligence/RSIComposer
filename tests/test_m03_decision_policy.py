"""Prompt replacement and proposal metadata across real policy boundaries; CPU only."""
from dataclasses import FrozenInstanceError, asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from examples.medical.experiment import ResearchPolicy
from rsisys.application import Application, ApplicationConfig
from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.coordination import ExplorationProgress
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef, TokenUsage
from rsisys.loops.decision_making.action_space import (
    ModularDecisionActionSpace, OperationCall, OperationResult, OperationSpec,
)
from rsisys.loops.decision_making.harness.attempts import AttemptContract
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.learning_policy import MemoryPrefixDecisionPolicy
from rsisys.loops.decision_making.harness.policy import DecisionPrompts, ModelDecisionPolicy
from rsisys.loops.decision_making.harness.state import DecisionContext
from rsisys.loops.decision_making.harness.tools import BoundDecisionTools
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.snapshot.local import LocalSnapshotCapture


MODEL = ModelRef('fixture', 'meta-agent', 'v1')


class Backend:
    supports_response_schema = True
    max_output_tokens = 8192
    token_count_mode = 'fixture'

    def __init__(self, *outputs):
        self.outputs = iter(outputs)
        self.requests, self.measured = [], []

    def count_tokens(self, request):
        self.measured.append(request)
        return 100  # Deterministic fixture; not a tokenizer measurement.

    def generate(self, request):
        self.requests.append(request)
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return GenerationResponse(json.dumps(output), request.model, True, TokenUsage(5, 3))


class InspectDefinition:
    spec = OperationSpec('inspect', 'Inspect fixture', {'type': 'object'}, read_only=True)

    def validate(self, arguments):
        if arguments:
            raise ValueError('Expected empty fixture arguments')
        return OperationCall('inspect', {})


class DecisionPromptTests(unittest.TestCase):
    def setUp(self):
        self.context = DecisionContext('Fixture objective', (), (InspectDefinition.spec,), (),
            directive_modes=('configure_prompt',), review_actions=('continue',))

    def medical(self, backend, *, mode='time', prompts=None):
        return ResearchPolicy(backend, MODEL, ModelDecisionPolicy(backend, MODEL).directives,
            ModularDecisionActionSpace((InspectDefinition(),)), Mock(), SimpleNamespace(ledger=[]),
            ArtifactRef('train', 'training_data', 'fixture://train'), run_mode=mode, prompts=prompts)

    def test_replacement_reaches_decision_and_review_without_changing_schema_or_user_data(self):
        for policy_type in (ModelDecisionPolicy, MemoryPrefixDecisionPolicy):
            with self.subTest(policy=policy_type.__name__):
                backend = Backend({'kind': 'finish'}, {'action': 'continue'})
                prompts = DecisionPrompts({'research': 'Fixture research role override.'})
                policy = policy_type(backend, MODEL, prompts=prompts)
                original = ModelDecisionPolicy(Backend(), MODEL)
                policy.decide(self.context)
                policy.review(self.context)
                self.assertEqual(len(backend.requests), 2)
                for request, builder in zip(backend.requests, ('_decision_request', '_review_request')):
                    baseline = getattr(original, builder)(self.context)
                    self.assertIn('Fixture research role override.', request.system)
                    self.assertEqual(request.response_schema, baseline.response_schema)
                    self.assertEqual(json.loads(request.prompt), json.loads(baseline.prompt))
                    self.assertIn(request, backend.measured)
                    self.assertEqual(request.model, MODEL)

    def test_bundle_and_manifest_do_not_expose_mutable_configuration(self):
        overrides = {'research': 'Captured role.'}
        bundle = DecisionPrompts(overrides)
        manifest = bundle.manifest()
        overrides['research'] = 'Later mutation.'
        manifest['sha256'].clear()
        self.assertEqual(bundle.render('research'), 'Captured role.')
        self.assertTrue(bundle.manifest()['sha256'])
        self.assertFalse(bundle.is_default)
        with self.assertRaises(FrozenInstanceError):
            bundle.is_default = True
        default = DecisionPrompts()
        self.assertTrue(DecisionPrompts({'research': default.render('research')}).is_default)

    def test_invalid_templates_fail_at_assembly_before_any_model_call(self):
        for override in ({'unknown': 'text'}, {'research': ''}, {'research': None},
                         {'decision': '${unsupported}'}, {'decision': '${unclosed'}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                DecisionPrompts(override)
        with self.assertRaises(TypeError):
            DecisionPrompts('not a mapping')
        with self.assertRaises(TypeError):
            ModelDecisionPolicy(Backend(), MODEL, prompts={})

    def test_custom_summary_and_correction_use_existing_accounting_and_identity(self):
        prompts = DecisionPrompts({'summary': 'Fixture summary instruction.',
            'validation_correction': 'Fixture rejected output (data): ${error}'})
        backend = Backend({'summary': 'Fixture archive pointer'}, {'kind': 'invalid'}, {'kind': 'finish'})
        policy = ModelDecisionPolicy(backend, MODEL, prompts=prompts, max_validation_attempts=2)
        usage = []
        with policy.track_usage(usage.append):
            policy._summarize_context([], '', 100)
            self.assertEqual(policy.decide(self.context).kind, 'finish')
        self.assertEqual(len(backend.requests), 3)
        self.assertEqual(usage, [TokenUsage(5, 3)] * 3)
        self.assertEqual(backend.requests[0].system, 'Fixture summary instruction.')
        correction = backend.requests[-1]
        self.assertIn('Fixture rejected output (data)', json.loads(correction.prompt)['objective'])
        self.assertNotIn('Unsupported decision kind', correction.system)
        self.assertEqual(correction.response_schema, backend.requests[-2].response_schema)
        self.assertEqual(self.context.objective, 'Fixture objective')

    def test_prompt_replacement_cannot_disable_attempt_admission(self):
        backend = Backend({'kind': 'explore', 'mode': 'configure_prompt', 'strategy': 'fixture',
                           'arguments': {'instructions': 'fixture'}})
        policy = ModelDecisionPolicy(backend, MODEL, attempt_contract=AttemptContract(required=True),
            prompts=DecisionPrompts({'attempt_required': 'Fixture alternate instructions.'}))
        with self.assertRaisesRegex(ValueError, 'requires an Attempt'):
            policy.decide(self.context)
        self.assertEqual(len(backend.requests), 1)

    def test_transport_failures_do_not_enter_prompt_correction(self):
        backend = Backend(ValueError('fixture transport failure'))
        policy = ModelDecisionPolicy(backend, MODEL, max_validation_attempts=3)
        with self.assertRaisesRegex(ValueError, 'transport failure'):
            policy.decide(self.context)
        self.assertEqual(len(backend.requests), 1)

    def test_default_snapshot_refuses_to_lose_custom_prompts(self):
        config = ApplicationConfig.load(Path(__file__).resolve().parents[1] / 'configs/local.json')
        with TemporaryDirectory() as directory:
            with Application(replace(config, data_dir=Path(directory) / 'run')) as app:
                app.runner.decision_making.harness.policy.prompts = DecisionPrompts({'research': 'Fixture role.'})
                with self.assertRaisesRegex(ValueError, 'Custom decision prompts'):
                    LocalSnapshotCapture(app).capture('custom-prompts')
                self.assertFalse((app.config.data_dir / 'snapshots').exists())

    def test_medical_preserves_proposal_ids_in_canonical_and_normalized_envelopes(self):
        envelopes = (
            {'kind': 'invoke', 'operation': 'inspect', 'arguments': {}},
            {'kind': 'inspect', 'arguments': {}},
            {'kind': 'explore', 'mode': 'configure_prompt', 'strategy': 'fixture',
             'arguments': {'instructions': 'fixture'}},
            {'kind': 'configure_prompt', 'arguments': {'instructions': 'fixture'}},
        )
        for envelope in envelopes:
            with self.subTest(envelope=envelope):
                payload = {**envelope, 'feedback_proposal_ids': ['feedback:proposal']}
                backend = Backend(payload)
                policy = self.medical(backend)
                decision = policy.decide(self.context)
                self.assertEqual(decision.feedback_proposal_ids, ('feedback:proposal',))
                args = decision.operation.arguments if decision.operation else asdict(decision.directive.payload)
                self.assertNotIn('feedback_proposal_ids', args)
                self.assertEqual(len(backend.requests), 1)

    def test_medical_invalid_proposal_ids_are_rejected_within_existing_retry_bound(self):
        for refs in (['x', 'x'], [' '], ['x'] * 17):
            with self.subTest(refs=refs):
                payload = {'kind': 'invoke', 'operation': 'inspect', 'arguments': {}, 'feedback_proposal_ids': refs}
                backend = Backend(payload, payload, payload)
                policy = self.medical(backend)
                with self.assertRaisesRegex(ValueError, 'three bounded'):
                    policy.decide(self.context)
                self.assertEqual(len(backend.requests), 3)
                self.assertEqual(policy.monitor.event.call_count, 3)

    def test_medical_prompt_schema_and_parser_agree_about_serial_waiting(self):
        for mode in ('time', 'researcher'):
            with self.subTest(mode=mode):
                policy = self.medical(Backend(), mode=mode,
                    prompts=DecisionPrompts({'research': 'Fixture medical role.'}))
                request = policy._decision_request(self.context)
                self.assertIn('Fixture medical role.', request.system)
                self.assertNotIn('"kind":"wait"', request.system)
                kinds = {branch['properties']['kind']['const'] for branch in request.response_schema['oneOf']}
                self.assertNotIn('wait', kinds)
                self.assertEqual('finish' in kinds, mode == 'researcher')

    def test_review_proposal_ids_resolve_before_execution_and_remain_in_ledger(self):
        for valid in (True, False):
            with self.subTest(valid=valid):
                payload = {'kind': 'invoke', 'operation': 'inspect', 'arguments': {},
                           'feedback_proposal_ids': ['feedback:proposal']}
                backend = Backend(payload, {'action': 'continue'})
                tool = Mock(return_value=OperationResult('Fixture observation', (), True))
                harness = DefaultDecisionHarness(ModelDecisionPolicy(backend, MODEL),
                    BoundDecisionTools({'inspect': tool}), BasicExperienceSystem(InMemoryExperienceStore()))
                harness.feedback_proposal_resolver = Mock(
                    return_value=[{'proposal_id': 'feedback:proposal', 'applied': False}],
                    side_effect=None if valid else ValueError('Unknown harness proposal'))
                loop = DecisionMakingLoop(harness, ModularDecisionActionSpace((InspectDefinition(),)))
                directive = Directive('active', 'Fixture objective', 'fixture', simulated=True)
                harness.state.pending_directive = directive
                harness.state.pending_directive_id = 'active'
                progress = ExplorationProgress('active', 'running', directive)
                if valid:
                    self.assertEqual(loop.review('Fixture objective', progress).action, 'continue')
                    tool.assert_called_once_with({})
                    record, = harness.state.decision_records.values()
                    self.assertEqual(record['feedback_proposal_ids'], ('feedback:proposal',))
                    self.assertEqual(record['status'], 'completed')
                    self.assertFalse(record['feedback_proposals'][0]['applied'])
                else:
                    with self.assertRaisesRegex(ValueError, 'Unknown harness proposal'):
                        loop.review('Fixture objective', progress)
                    tool.assert_not_called()
                harness.feedback_proposal_resolver.assert_called_once_with(('feedback:proposal',))


if __name__ == '__main__':
    unittest.main()
