"""Training output availability is distinct from a role's current binding."""
from dataclasses import replace
import unittest
from unittest.mock import Mock

from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.contracts.requests import TrainSpec
from rsisys.loops.contracts.training import TrainingResult, ModelActivationPending
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.learner import DecisionTrainingRequest
from rsisys.loops.exploration.action_space import EchoActionSpace
from rsisys.loops.exploration.harness.engine import DefaultExplorationHarness
from rsisys.loops.exploration.harness.learner import ActorTrainingRequest
from rsisys.loops.exploration.harness.learner.command import TrainingCommandHandler
from rsisys.loops.exploration.loop import ExplorationLoop
from rsisys.runtime.jobs import MockJobQueue

BASE = ModelRef('service', 'model', 'base')
NEW = replace(BASE, revision='trained')


class Policy:
    def __init__(self):
        self.model = BASE
        self.binds = 0

    def update_model(self, model):
        self.model = model
        self.binds += 1


class Learner:
    def __init__(self, role):
        self.queue = MockJobQueue('fixture', pending_polls=0)
        self.output = TrainingResult(BASE, NEW, Candidate('candidate', BASE.revision, {}, (role,)))

    def submit(self, request, *, idempotency_key):
        return self.queue.submit(request, lambda: self.output, idempotency_key=idempotency_key)

    def poll(self, job):
        return self.queue.poll(job)

    def cancel(self, job):
        self.queue.cancel(job)


class ActivationTests(unittest.TestCase):
    def build(self, role, activator=None):
        experience = BasicExperienceSystem(InMemoryExperienceStore())
        learner = Learner(role)
        if role == 'actor':
            harness = DefaultExplorationHarness(Policy(), Mock(), experience, learner=learner,
                model_activator=activator, command_handlers={'train': TrainingCommandHandler()})
            request = ActorTrainingRequest(BASE, (), 'backend-config', method='custom')
        else:
            harness = DefaultDecisionHarness(Policy(), Mock(), experience, learner=learner,
                                             model_activator=activator)
            request = DecisionTrainingRequest(BASE, (), 'backend-config')
        return harness, request

    def test_real_checkpoint_requires_confirmation_before_binding_for_both_roles(self):
        for role in ('actor', 'decision_maker'):
            with self.subTest(role=role):
                harness, request = self.build(role)
                job = harness.submit_learning(request, idempotency_key='training')
                with self.assertRaisesRegex(RuntimeError, 'model_activator'):
                    harness.apply_learning(job)
                self.assertEqual(harness.policy.model, BASE)
                self.assertNotIn(job, harness.state.applied_training)

    def test_pending_activation_retries_same_key_then_binds_once(self):
        for role in ('actor', 'decision_maker'):
            with self.subTest(role=role):
                activator = Mock(activate=Mock(side_effect=[ModelActivationPending('loading'), NEW]))
                harness, request = self.build(role, activator)
                job = harness.submit_learning(request, idempotency_key='training')
                with self.assertRaises(ModelActivationPending):
                    harness.apply_learning(job)
                self.assertEqual(harness.policy.model, BASE)
                self.assertEqual(harness.apply_learning(job), NEW)
                self.assertEqual(harness.apply_learning(job), NEW)
                calls = activator.activate.call_args_list
                self.assertEqual(len(calls), 2)
                self.assertEqual(calls[0], calls[1])
                self.assertEqual(harness.policy.binds, 1)

    def test_wrong_version_and_activation_failure_leave_binding_intact(self):
        for outcome in (BASE, None, ValueError('deployment rejected')):
            activator = Mock(activate=Mock(side_effect=outcome) if isinstance(outcome, Exception)
                             else Mock(return_value=outcome))
            harness, request = self.build('actor', activator)
            job = harness.submit_learning(request, idempotency_key='training')
            with self.assertRaises(ValueError):
                harness.apply_learning(job)
            self.assertEqual(harness.policy.model, BASE)
            self.assertNotIn(job, harness.state.applied_training)

    def test_command_waits_for_activation_and_can_cancel_without_switching(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                activator = Mock(activate=Mock(side_effect=[ModelActivationPending('loading'), NEW]))
                harness, _ = self.build('actor', activator)
                loop = ExplorationLoop(harness, EchoActionSpace())
                directive = Directive('train', 'task', 'strategy', mode='train',
                                      payload=TrainSpec('config', 'custom', apply_result=True))
                loop.accept(directive)
                loop.advance('train')
                with self.assertRaises(ModelActivationPending):
                    loop.advance('train')
                self.assertFalse(loop.inspect('train').terminal)
                self.assertEqual(harness.policy.model, BASE)
                if cancel:
                    loop.cancel('train')
                feedback = loop.advance('train').feedback
                self.assertEqual(feedback.status, 'cancelled' if cancel else 'completed')
                self.assertEqual(harness.policy.model, BASE if cancel else NEW)
                self.assertEqual(activator.activate.call_count, 1 if cancel else 2)

    def test_snapshot_rejects_activation_configuration(self):
        from pathlib import Path
        import tempfile
        from rsisys.application import Application, ApplicationConfig
        from rsisys.snapshot.local import LocalSnapshotCapture
        with tempfile.TemporaryDirectory() as directory:
            config = ApplicationConfig.load(Path(__file__).resolve().parents[1] / 'configs/local.json')
            with Application(replace(config, data_dir=Path(directory) / 'run')) as app:
                app.runner.exploration.harness.model_activator = Mock()
                with self.assertRaisesRegex(TypeError, 'activators'):
                    LocalSnapshotCapture(app).capture('unsupported')
                self.assertFalse((app.config.data_dir / 'snapshots').exists())

    def test_stale_base_is_rejected_before_activation(self):
        activator = Mock()
        harness, request = self.build('actor', activator)
        job = harness.submit_learning(request, idempotency_key='training')
        harness.policy.model = replace(BASE, revision='other')
        with self.assertRaisesRegex(ValueError, 'stale'):
            harness.apply_learning(job)
        activator.activate.assert_not_called()
