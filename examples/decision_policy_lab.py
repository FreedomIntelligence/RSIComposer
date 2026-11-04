"""CPU workflow fixture: raw archive -> Lab dataset task -> Lab training task.

Run with PYTHONPATH=src python examples/decision_policy_lab.py. The data recipe
and training backend are explicit fixtures. No real training/model calls occur.
"""
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from rsisys.adapters.improvement import improvement_exploration_factory
from rsisys.adapters.trajectory import FeedbackTrajectorySource
from rsisys.config import DemoConfig, build_demo
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.improvement import DecisionDatasetBuildRequest, DecisionDatasetBuildResult
from rsisys.loops.decision_making.action_space import OperationCall
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.learner import DecisionTrainingRequest
from rsisys.loops.decision_making.harness.learner.backends.mock import MockDecisionLearner
from rsisys.research import ResearchAssembly, assemble_research
from rsisys.runtime.artifacts import InMemoryArtifactStore
from rsisys.runtime.jobs import MockJobQueue


class FixtureDatasetBuilder:
    def __init__(self, artifacts):
        self.artifacts, self.source = artifacts, None
        self.jobs = MockJobQueue('fixture-policy-dataset', pending_polls=0)

    def submit(self, request, *, idempotency_key):
        def construct():
            captures = [self.source.read(ref) for ref in request.trajectory_refs]
            # This fixture validates routing/provenance, not supervision quality.
            data = self.artifacts.put(b'{"fixture":true,"not_real_training_data":true}', kind='dataset_fixture')
            report = self.artifacts.put(json.dumps({'simulated': True, 'recipe_ref': request.recipe_ref,
                'source_refs': [asdict(ref) for ref in request.trajectory_refs],
                'source_coverage': [row['coverage'] for row in captures], 'example_count': 1}).encode(),
                kind='dataset_build_report')
            return DecisionDatasetBuildResult((data,), report, request.trajectory_refs,
                                               request.recipe_ref, 1, simulated=True)
        return self.jobs.submit(request, construct, idempotency_key=idempotency_key)

    def poll(self, job):
        return self.jobs.poll(job)

    def cancel(self, job):
        self.jobs.cancel(job)


class FixturePolicy:
    def __init__(self, model, service, source):
        self.model, self.service, self.source, self.stage = model, service, source, 0

    def decide(self, context):
        self.stage += 1
        if self.stage == 1:
            return Decision('invoke', operation=OperationCall('prepare_data', {'text': 'Public fixture observation'}))
        if self.stage == 2:
            job = self.service.learning.ordered_jobs()[0]['job_id']
            return Decision('explore', directive=Directive('build-policy-data', context.objective,
                'Construct data under an explicit fixture recipe', mode='build_decision_dataset',
                payload=DecisionDatasetBuildRequest((self.source.reference(job),), 'fixture-data-recipe:v1')))
        if self.stage == 3:
            data = tuple(ref for ref in context.last_feedback.artifact_refs if ref.kind == 'dataset_fixture')
            return Decision('explore', directive=Directive('train-policy', context.objective,
                'Train a separate fixture candidate', mode='train_decision_policy',
                payload=DecisionTrainingRequest(self.model, (), 'fixture-train-recipe:v1', data)))
        return Decision('finish')


def main():
    config = DemoConfig.load(Path(__file__).resolve().parents[1] / 'configs/demo.json')
    artifacts = InMemoryArtifactStore()
    builder = FixtureDatasetBuilder(artifacts)
    learner = MockDecisionLearner(artifacts, pending_polls=0)
    runner, _ = build_demo(config, exploration_factory=improvement_exploration_factory(
        dataset_builder=builder, decision_learner=learner))
    with TemporaryDirectory(prefix='rsisys-policy-lab-fixture-') as directory, runner:
        loop = runner.decision_making
        service = assemble_research(loop, assembly=ResearchAssembly(),
            path=Path(directory) / 'feedback.json', owner=runner.assembly)
        source = FeedbackTrajectorySource(service, archive_id='fixture-run')
        builder.source = source
        base = loop.harness.policy.model
        loop.harness.policy = FixturePolicy(base, service, source)
        for _ in range(16):
            if runner.step('Exercise the decision-policy improvement Lab').kind == 'finished':
                break
        else:
            raise RuntimeError('Fixture did not complete')
        assert loop.harness.policy.model == base
        assert len(loop.harness.state.last_feedback.candidates) == 1
        assert not loop.harness.state.last_feedback.adopted_candidate_ids
        print(json.dumps({'status': 'fixture_passed', 'archive_records': len(service.trajectories()),
            'dataset_task': 'completed', 'training_task': 'mock_candidate_returned',
            'real_training_performed': False, 'live_policy_changed': False}, indent=2))


if __name__ == '__main__':
    main()
