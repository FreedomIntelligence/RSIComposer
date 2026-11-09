"""Deterministic interface fixture; does not optimize a real decision model."""

import json
from dataclasses import asdict, replace
from uuid import uuid4

from rsisys.harnesses.decision.learner import DecisionTrainingRequest
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.training import TrainingResult
from rsisys.runtime.orchestration.jobs import MockJobQueue
from rsisys.runtime.storage.artifacts import ArtifactStore


class MockDecisionLearner:
    def __init__(self, artifacts: ArtifactStore, *, pending_polls: int = 1):
        self.artifacts = artifacts
        self.jobs = MockJobQueue[TrainingResult]('mock-decision-training', pending_polls=pending_polls)

    def submit(self, request: DecisionTrainingRequest, *, idempotency_key: str):
        def train():
            for ref in request.data_refs:
                self.artifacts.get(ref)
            model = replace(request.base, revision=f'mock-{uuid4()}')
            manifest = self.artifacts.put(json.dumps({
                'simulated': True, 'request': asdict(request), 'model': asdict(model),
            }).encode(), kind='mock_model_manifest')
            candidate = Candidate(str(uuid4()), request.base.revision, {'model': manifest}, ('decision_maker',))
            return TrainingResult(request.base, model, candidate, simulated=True)
        return self.jobs.submit(request, train, idempotency_key=idempotency_key)

    def poll(self, job):
        return self.jobs.poll(job)

    def cancel(self, job):
        self.jobs.cancel(job)

    def result(self, job):
        return self.jobs.result(job)
