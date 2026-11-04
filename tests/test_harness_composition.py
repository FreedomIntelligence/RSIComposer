"""Loop composition works without default policies, tools, or memory interfaces."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from rsisys.application import Application, ApplicationConfig
from rsisys.config import DemoConfig, build_demo
from rsisys.loops.contracts.coordination import ExplorationProgress
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.requests import PromptSpec, SFTSpec
from rsisys.loops.decision_making.action_space import DemoDecisionActionSpace
from rsisys.loops.decision_making.harness.decision import Decision, LoopStepResult
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.loops.exploration.action_space import EchoActionSpace
from rsisys.loops.exploration.loop import ExplorationLoop
from rsisys.runtime.runner import Runner
from rsisys.snapshot.local import LocalSnapshotCapture, restore_snapshot

CONFIG = Path(__file__).resolve().parents[1]/'configs/demo.json'


class ScheduledDecision:
    """An external schedule can occupy the decision position without an LLM."""
    def __init__(self, directive):
        self.directive, self.sent, self.finished = directive, False, False
        self.seen_space = None

    def step(self, objective, action_space):
        self.seen_space = action_space
        if self.finished:
            return LoopStepResult('finished')
        action_space.validate_directive(self.directive)
        self.sent = True
        return LoopStepResult('dispatch', directive=self.directive)

    def pending(self):
        return self.directive if self.sent and not self.finished else None

    def receive_feedback(self, directive, feedback):
        if directive != self.directive or feedback.directive_id != directive.directive_id:
            raise ValueError('Wrong scheduled request')
        self.finished = True


class ExternalExecution:
    """Fixture for a job-only execution harness with no Actor or memory."""
    def accept(self, directive):
        self.directive = directive
        return self.inspect(directive.directive_id)

    def advance(self, directive_id, action_space):
        self.seen_space = action_space
        self.feedback = Feedback(directive_id, ExecutionStatus.COMPLETED, 'Fixture job finished', True)
        return self.inspect(directive_id)

    def inspect(self, directive_id):
        feedback = getattr(self, 'feedback', None)
        return ExplorationProgress(directive_id, 'completed' if feedback else 'accepted', self.directive,
                                   feedback=feedback)

    def cancel(self, directive_id):
        raise NotImplementedError

    def acknowledge(self, directive_id):
        self.acknowledged = directive_id


class HarnessCompositionTests(unittest.TestCase):
    def test_custom_harnesses_need_no_memory_or_policy_contract(self):
        directive = Directive('scheduled-train', 'task', 'train the model', mode='sft', payload=SFTSpec('recipe'))
        decision, execution = ScheduledDecision(directive), ExternalExecution()
        decision_actions, actor_actions = DemoDecisionActionSpace(), EchoActionSpace()
        actor_actions.validate = Mock(side_effect=AssertionError('System instruction is not an Actor action'))
        runner = Runner(DecisionMakingLoop(decision, decision_actions), ExplorationLoop(execution, actor_actions))
        self.assertEqual(runner.run('task', 1).status, 'completed')
        self.assertIs(decision.seen_space.source, decision_actions)
        self.assertIs(execution.seen_space, actor_actions)
        self.assertEqual(execution.acknowledged, directive.directive_id)
        self.assertFalse(hasattr(decision, 'memory'))
        self.assertFalse(hasattr(execution, 'policy'))
        actor_actions.validate.assert_not_called()

    def test_action_space_can_change_without_replacing_harness(self):
        runner, _ = build_demo(DemoConfig.load(CONFIG))
        actor = runner.exploration
        original_harness = actor.harness
        constrained = EchoActionSpace()
        constrained.validate = Mock(side_effect=ValueError('Task disallows echo'))
        actor.action_space = constrained
        with patch.object(actor.harness.environment, 'step', wraps=actor.harness.environment.step) as execute:
            with self.assertRaisesRegex(ValueError, 'disallows echo'):
                runner.run('task', 1)
        self.assertIs(actor.harness, original_harness)
        execute.assert_not_called()
        self.assertEqual(runner.decision_making.harness.state.last_feedback.status, ExecutionStatus.FAILED)

    def test_prompt_directive_and_instructions_survive_snapshot_and_reach_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = ApplicationConfig.load(CONFIG.parent/'local.json')
            config = replace(config, data_dir=root/'source')
            directive = Directive('prompt', 'task', 'improve instructions', mode='configure_prompt',
                                  payload=PromptSpec('Verify command outputs before concluding.'))
            with Application(config) as app:
                with patch.object(app.runner.decision_making.harness.policy, 'decide',
                                  return_value=Decision('explore', directive=directive)):
                    self.assertEqual(app.run().status, 'completed')
                manifest = LocalSnapshotCapture(app).capture('prompt-changed')
                source = config.data_dir/'snapshots'/manifest.snapshot_id
            with restore_snapshot(source, root/'restored') as app:
                dm, actor = app.runner.decision_making.harness, app.runner.exploration.harness
                self.assertEqual(dm.state.last_directive.payload, directive.payload)
                self.assertEqual(actor.state.instructions, directive.payload.instructions)
                backend = actor.policy.backend
                with patch.object(backend, 'generate', wraps=backend.generate) as generate:
                    self.assertEqual(app.run().status, 'completed')
                actor_requests = [c.args[0] for c in generate.call_args_list if c.args[0].role == 'actor']
                self.assertTrue(actor_requests)
                self.assertIn(directive.payload.instructions, actor_requests[-1].system)

    def test_custom_harness_snapshot_fails_explicitly_before_inspecting_internal_state(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(ApplicationConfig.load(CONFIG.parent/'local.json'), data_dir=Path(directory)/'run')
            with Application(config) as app:
                app.runner.exploration.harness = ExternalExecution()
                with self.assertRaisesRegex(TypeError, 'Custom harnesses'):
                    LocalSnapshotCapture(app).capture('custom')


if __name__ == '__main__':
    unittest.main()
