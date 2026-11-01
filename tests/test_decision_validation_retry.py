import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock
from rsisys.config import DemoConfig, build_demo
from rsisys.loops.contracts.model_service import GenerationResponse
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy


class DecisionValidationRetryTests(TestCase):
    def test_only_parsing_is_retried_with_error_feedback(self):
        with build_demo(DemoConfig.load(Path(__file__).resolve().parents[1] / 'configs/demo.json'))[0] as runner:
            harness=runner.decision_making.harness
            context=harness._context('research', runner.decision_making.action_space)
            backend=Mock()
            backend.generate.side_effect=[GenerationResponse('{"kind":"agenda"}',harness.policy.model,False),
                GenerationResponse('{"kind":"finish","reason":"done"}',harness.policy.model,False)]
            policy=ModelDecisionPolicy(backend,harness.policy.model,max_validation_attempts=2)
            self.assertEqual(policy.decide(context).kind,'finish')
            self.assertEqual(backend.generate.call_count,2)
            self.assertIn('Unsupported decision kind',backend.generate.call_args.args[0].prompt)
            self.assertNotIn('Previous decision',context.objective)
            backend.reset_mock();backend.generate.side_effect=ValueError('transport or config')
            with self.assertRaisesRegex(ValueError,'transport'):policy.decide(context)
            self.assertEqual(backend.generate.call_count,1)

    def test_retry_budget_exhaustion_does_not_run_tools(self):
        with build_demo(DemoConfig.load(Path(__file__).resolve().parents[1] / 'configs/demo.json'))[0] as runner:
            harness=runner.decision_making.harness
            context=harness._context('research',runner.decision_making.action_space)
            backend=Mock();backend.generate.return_value=GenerationResponse('bad-json',harness.policy.model,False)
            policy=ModelDecisionPolicy(backend,harness.policy.model,max_validation_attempts=2)
            with self.assertRaises(json.JSONDecodeError):policy.decide(context)
            self.assertEqual(backend.generate.call_count,2)
            self.assertIsNone(harness.pending())

    def test_no_input_directive_does_not_inherit_unrelated_artifacts(self):
        from dataclasses import replace
        from rsisys.loops.decision_making.action_space.directives import DirectiveDefinitions, DirectiveDefinition, builtin_directives
        from rsisys.loops.contracts.artifacts import ArtifactRef
        from rsisys.loops.contracts.coordination import OperationResult
        with build_demo(DemoConfig.load(Path(__file__).resolve().parents[1] / 'configs/demo.json'))[0] as runner:
            harness = runner.decision_making.harness
            definition = next(d for d in builtin_directives() if d.spec.mode == 'configure_prompt')
            catalog = DirectiveDefinitions((DirectiveDefinition(replace(definition.spec, accepts_input_refs=False), definition.parser),))
            context = replace(harness._context('research', runner.decision_making.action_space),
                directive_modes=('configure_prompt',), operation_results=(OperationResult('unrelated data',
                    (ArtifactRef('data', 'training_data', 'artifact://data'),), True),))
            backend = Mock()
            payload = {'kind': 'explore', 'mode': 'configure_prompt', 'strategy': 'prompt change',
                       'arguments': {'instructions': 'answer concisely'}}
            backend.generate.return_value = GenerationResponse(json.dumps(payload), harness.policy.model, False)
            policy = ModelDecisionPolicy(backend, harness.policy.model, directives=catalog)
            decision = policy.decide(context)
            self.assertEqual(decision.directive.input_refs, ())
            self.assertFalse(decision.directive.simulated)
            from rsisys.loops.decision_making.action_space import ModularDecisionActionSpace
            space = ModularDecisionActionSpace((), directives=catalog)
            space.validate_directive(decision.directive)
            with self.assertRaisesRegex(ValueError, 'does not accept input references'):
                space.validate_directive(replace(decision.directive, input_refs=context.operation_results[0].artifacts))
            payload['input_ids'] = ['data']
            backend.generate.return_value = GenerationResponse(json.dumps(payload), harness.policy.model, False)
            with self.assertRaisesRegex(ValueError, 'does not accept input_ids'):
                policy.decide(context)

    def test_snapshot_does_not_silently_reset_custom_retry_budget(self):
        from dataclasses import replace
        from tempfile import TemporaryDirectory
        from rsisys.application import Application, ApplicationConfig
        from rsisys.snapshot.local import LocalSnapshotCapture
        config = ApplicationConfig.load(Path(__file__).resolve().parents[1] / 'configs/local.json')
        with TemporaryDirectory() as directory:
            with Application(replace(config, data_dir=Path(directory) / 'run')) as app:
                app.runner.decision_making.harness.policy.max_validation_attempts = 3
                with self.assertRaisesRegex(ValueError, 'retry budgets'):
                    LocalSnapshotCapture(app).capture('custom-retries')
                self.assertFalse((app.config.data_dir / 'snapshots').exists())
