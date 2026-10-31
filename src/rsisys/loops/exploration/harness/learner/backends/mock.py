"""Mock training produces version manifests, never real weights or quality metrics."""

from dataclasses import asdict, replace
import json
from uuid import uuid4

from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.training import TrainingResult
from rsisys.runtime.artifacts import ArtifactStore
from rsisys.runtime.jobs import MockJobQueue

from .. import ActorTrainingRequest


class MockActorLearner:
    def __init__(self, artifacts: ArtifactStore, *, pending_polls: int = 1):
        self.artifacts = artifacts
        self.jobs = MockJobQueue[TrainingResult]('mock-actor-training', pending_polls=pending_polls)

    def submit(self, request: ActorTrainingRequest, *, idempotency_key: str):
        # These are this mock's capabilities, not restrictions on external learners.
        if request.method not in ("sft", "rl"):
            raise ValueError("MockActorLearner only supports sft or rl")
        if request.method == "sft" and not request.data_refs:
            raise ValueError("MockActorLearner SFT requires data artifacts")
        if request.method == "rl" and not request.trajectory_refs:
            raise ValueError("MockActorLearner RL requires trajectory records")

        def train():
            for ref in request.data_refs:
                self.artifacts.get(ref)
            model = replace(request.base, revision=f'mock-{uuid4()}')
            manifest = self.artifacts.put(json.dumps({
                'simulated': True, 'request': asdict(request), 'model': asdict(model),
            }).encode(), kind='mock_model_manifest')
            candidate = Candidate(str(uuid4()), request.base.revision, {'model': manifest}, ('actor',))
            return TrainingResult(request.base, model, candidate, simulated=True)
        return self.jobs.submit(request, train, idempotency_key=idempotency_key)

    def poll(self, job):
        return self.jobs.poll(job)

    def cancel(self, job):
        self.jobs.cancel(job)

    def result(self, job):
        return self.jobs.result(job)
