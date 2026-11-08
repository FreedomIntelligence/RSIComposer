"""Data construction and policy training use ordinary Lab tasks, never feedback execution."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import unittest

from rsisys.adapters.improvement import improvement_exploration_factory, improvement_handlers, improvement_directives
from rsisys.adapters.lab import ExplorationLab
from rsisys.adapters.trajectory import FeedbackTrajectorySource
from rsisys.config import build_demo
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.improvement import TrajectoryRef, DecisionDatasetBuildRequest, DecisionDatasetBuildResult
from rsisys.loops.contracts.jobs import JobOutcome, JobStatus
from rsisys.loops.contracts.lab import LabSpec, LabTask
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.learner import DecisionTrainingRequest
from rsisys.loops.decision_making.harness.learner.backends.mock import MockDecisionLearner
from rsisys.research import ResearchAssembly, assemble_research
from rsisys.runtime.artifacts import InMemoryArtifactStore
from rsisys.runtime.jobs import MockJobQueue
from tests.feedback.test_feedback_required_outputs import CONFIG, EmptyBackend, observation
from tests.feedback.test_feedback_learning_integration import ImmediateExecutor
from tests.regressions.test_m01_contract_boundaries import Script


class FixtureDatasetBuilder:
    """Explicit fixture recipe. No actual policy examples or model inference."""
    def __init__(self, artifacts, *, source=None, pending_polls=0):
        self.artifacts, self.source = artifacts, source
        self.jobs = MockJobQueue('fixture-policy-data', pending_polls=pending_polls)
        self.requests, self.snapshots = [], []
        self.wrong_recipe = False

    def submit(self, request, *, idempotency_key):
        self.requests.append(deepcopy(request))

        def construct():
            if self.source is not None:
                self.snapshots = [self.source.read(ref) for ref in request.trajectory_refs]
            data = self.artifacts.put(b'{"fixture":true,"not_real_training_data":true}', kind='decision_dataset_fixture')
            report = self.artifacts.put(json.dumps({'simulated': True, 'request': asdict(request),
                'example_count': 1, 'recipe_execution': 'fixed_fixture_only'}).encode(), kind='dataset_build_report')
            return DecisionDatasetBuildResult((data,), report, request.trajectory_refs,
                'wrong' if self.wrong_recipe else request.recipe_ref, 1, simulated=True)

        return self.jobs.submit(request, construct, idempotency_key=idempotency_key)

    def poll(self, job):
        return self.jobs.poll(job)

    def cancel(self, job):
        self.jobs.cancel(job)


class ImprovementLabTests(unittest.TestCase):
    def setUp(self):
        self.artifacts = InMemoryArtifactStore()
        self.builder = FixtureDatasetBuilder(self.artifacts)
        self.learner = MockDecisionLearner(self.artifacts, pending_polls=0)

    def lab(self, *, builder=True, learner=True):
        modes = tuple(improvement_handlers(dataset_builder=self.builder if builder else None,
                                           decision_learner=self.learner if learner else None))
        runner, _ = build_demo(CONFIG, exploration_factory=improvement_exploration_factory(
            dataset_builder=self.builder if builder else None, decision_learner=self.learner if learner else None))
        self.addCleanup(runner.close)
        return runner, ExplorationLab(LabSpec('meta-improvement', 'Improve the decision policy', modes), runner.exploration)

    def task(self, payload, *, mode='build_decision_dataset', ident='build'):
        return LabTask(ident, 'decision-' + ident, 'meta-improvement', 'task-key-' + ident,
            Directive(ident, 'Improve the decision policy', 'Explicit fixture task', mode=mode, payload=payload))

    def dataset_request(self):
        return DecisionDatasetBuildRequest((TrajectoryRef('archive', 'trajectory:one', 'a' * 64),), 'recipe:v1')

    def complete(self, lab, task):
        progress = lab.submit(task)
        for _ in range(8):
            if progress.terminal:
                return lab.result(task.task_id)
            progress = lab.advance(task.task_id)
        self.fail('Fixture Lab task did not complete')

    def test_data_construction_and_training_are_separate_lab_tasks_without_activation(self):
        runner, lab = self.lab()
        base = runner.decision_making.harness.policy.model
        task = self.task(self.dataset_request())
        built = self.complete(lab, task)
        self.assertEqual(built.result['status'], 'completed')
        self.assertFalse(built.result['candidates'])
        self.assertFalse(self.learner.jobs._jobs)
        data_ref = next(ref for ref in built.result['artifact_refs'] if ref['kind'] == 'decision_dataset_fixture')
        from rsisys.loops.contracts.artifacts import ArtifactRef
        request = DecisionTrainingRequest(base, (), 'train-recipe:v1', (ArtifactRef(**data_ref),))
        lab.acknowledge(task.task_id)
        trained = self.complete(lab, self.task(request, mode='train_decision_policy', ident='train'))
        self.assertEqual(trained.result['status'], 'completed', trained)
        self.assertEqual(trained.result['candidates'][0]['affected_roles'], ('decision_maker',))
        self.assertFalse(trained.result['adopted_candidate_ids'])
        self.assertEqual(runner.decision_making.harness.policy.model, base)

    def test_main_decision_can_dispatch_data_construction_through_the_normal_runner(self):
        runner, _ = self.lab(learner=False)
        request = self.dataset_request()
        runner.decision_making.harness.policy = Script(Decision('explore', directive=Directive(
            'build-through-decision', 'Build a fixture dataset', 'Use a pinned recipe',
            mode='build_decision_dataset', payload=request)), Decision('finish'))
        for _ in range(8):
            result = runner.step('Build a fixture dataset')
            if result.kind == 'finished':
                break
        else:
            self.fail('Runner did not finish')
        feedback = runner.decision_making.harness.state.last_feedback
        self.assertEqual(feedback.status, ExecutionStatus.COMPLETED)
        self.assertEqual(self.builder.requests, [request])
        self.assertTrue(feedback.artifact_refs)
        self.assertFalse(feedback.candidates)

    def test_raw_feedback_archive_flows_into_lab_builder_without_becoming_a_label(self):
        runner, lab = self.lab(learner=False)
        service = assemble_research(runner.decision_making, assembly=ResearchAssembly(),
            backend=EmptyBackend(), executor=ImmediateExecutor())
        ident = service.observe(observation())
        self.builder.source = FeedbackTrajectorySource(service, archive_id='campaign/run')
        ref = self.builder.source.reference(ident)
        request = DecisionDatasetBuildRequest((ref,), 'fixture-recipe:v1')
        receipt = self.complete(lab, self.task(request))
        self.assertEqual(receipt.result['status'], 'completed')
        snapshot, = self.builder.snapshots
        self.assertEqual(snapshot['capture']['receipt'], json.loads(json.dumps(observation().payload())))
        self.assertFalse(snapshot['training_ready'])
        self.assertEqual(service.learning.state['attempted'], 0)
        self.assertFalse(self.learner.jobs._jobs)

    def test_duplicate_lab_submission_does_not_repeat_dataset_construction(self):
        _, lab = self.lab(learner=False)
        task = self.task(self.dataset_request())
        first = self.complete(lab, task)
        self.assertEqual(lab.submit(task).feedback.status, ExecutionStatus.COMPLETED)
        self.assertEqual(lab.result(task.task_id), first)
        self.assertEqual(len(self.builder.requests), 1)

    def test_wrong_result_recipe_becomes_a_failure_without_retraining_or_resubmission(self):
        self.builder.wrong_recipe = True
        runner, lab = self.lab(learner=False)
        task = self.task(self.dataset_request())
        result = self.complete(lab, task)
        self.assertEqual(result.result['status'], 'failed')
        self.assertIn('recipe', result.result['error'])
        self.assertFalse(result.result['artifact_refs'])
        command = runner.exploration.harness.commands.receipts[task.task_id]
        self.assertIn('backend_outcome', command.data)
        lab.advance(task.task_id)
        self.assertEqual(len(self.builder.requests), 1)

    def test_cancel_before_submission_performs_no_dataset_or_training_work(self):
        _, lab = self.lab()
        task = self.task(self.dataset_request())
        lab.submit(task)
        lab.cancel(task.task_id)
        self.assertTrue(lab.advance(task.task_id).terminal)
        self.assertEqual(lab.result(task.task_id).result['status'], 'cancelled')
        self.assertFalse(self.builder.requests)
        self.assertFalse(self.learner.jobs._jobs)

    def test_cancel_running_backend_waits_for_confirmed_terminal_status(self):
        self.builder.jobs = MockJobQueue('fixture-policy-data', pending_polls=10)
        _, lab = self.lab(learner=False)
        task = self.task(self.dataset_request())
        lab.submit(task)
        lab.advance(task.task_id)
        self.assertFalse(lab.cancel(task.task_id).terminal)
        self.assertEqual(lab.advance(task.task_id).feedback.status, ExecutionStatus.CANCELLED)
        self.assertEqual(len(self.builder.requests), 1)

    def test_missing_plugins_are_not_advertised_or_silently_substituted(self):
        with self.assertRaises(ValueError):
            improvement_handlers()
        runner, _ = self.lab(learner=False)
        context = runner.decision_making.harness._context('objective', runner.decision_making.action_space)
        self.assertEqual(context.directive_modes, ('build_decision_dataset',))
        with self.assertRaises(ValueError):
            improvement_directives(('unknown',))

    def test_public_directive_schema_parses_typed_sources_and_rejects_training_targets(self):
        request = self.dataset_request()
        definitions = improvement_directives()
        self.assertEqual(definitions.parse('build_decision_dataset', asdict(request)), request)
        with self.assertRaises(ValueError):
            definitions.parse('build_decision_dataset', {**asdict(request), 'training_example': {'input': 'x'}})
        with self.assertRaises(ValueError):
            DecisionDatasetBuildRequest((), 'recipe')

    def test_lost_submission_response_reuses_the_same_request_and_backend_key(self):
        _, lab = self.lab(learner=False)
        task = self.task(self.dataset_request())
        submit = self.builder.submit
        calls = []

        def lose_response(request, *, idempotency_key):
            calls.append((deepcopy(request), idempotency_key))
            job = submit(request, idempotency_key=idempotency_key)
            if len(calls) == 1:
                raise OSError('Submission response lost')
            return job

        self.builder.submit = lose_response
        lab.submit(task)
        with self.assertRaisesRegex(OSError, 'response lost'):
            lab.advance(task.task_id)
        self.assertEqual(lab.advance(task.task_id).feedback.status, ExecutionStatus.COMPLETED)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(len(self.builder.jobs._jobs), 1)

    def test_poll_identity_mismatch_never_resubmits_or_accepts_another_job(self):
        _, lab = self.lab(learner=False)
        task = self.task(self.dataset_request())
        poll = self.builder.poll
        first = [True]

        def once_wrong(job):
            if first[0]:
                first[0] = False
                return JobOutcome(replace(job, job_id='other'), JobStatus.RUNNING)
            return poll(job)

        self.builder.poll = once_wrong
        lab.submit(task)
        with self.assertRaisesRegex(ValueError, 'does not match'):
            lab.advance(task.task_id)
        self.assertEqual(lab.advance(task.task_id).feedback.status, ExecutionStatus.COMPLETED)
        self.assertEqual(len(self.builder.requests), 1)

    def test_wrong_training_base_returns_a_failure_without_rebinding_the_live_policy(self):
        runner, lab = self.lab(builder=False)
        base = runner.decision_making.harness.policy.model
        data = self.artifacts.put(b'fixture', kind='dataset_fixture')
        request = DecisionTrainingRequest(base, (), 'recipe', (data,))
        poll = self.learner.poll

        def wrong_base(job):
            outcome = poll(job)
            if outcome.result is None:
                return outcome
            result = outcome.result
            return replace(outcome, result=replace(result,
                base=replace(result.base, revision='other-base'),
                candidate=replace(result.candidate, base_version='other-base')))

        self.learner.poll = wrong_base
        result = self.complete(lab, self.task(request, mode='train_decision_policy', ident='train'))
        self.assertEqual(result.result['status'], 'failed')
        self.assertIn('base', result.result['error'])
        self.assertFalse(result.result['candidates'])
        self.assertEqual(runner.decision_making.harness.policy.model, base)


if __name__ == '__main__':
    unittest.main()
