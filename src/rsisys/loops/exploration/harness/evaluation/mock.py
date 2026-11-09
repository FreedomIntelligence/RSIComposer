from dataclasses import asdict
import json

from rsisys.runtime.storage.artifacts import ArtifactStore
from rsisys.runtime.orchestration.jobs import MockJobQueue

from . import EvaluationRequest, EvaluationResult


class MockEvaluator:
    """Checks fixture artifacts and emits a report, never a fabricated quality score."""

    def __init__(self, artifacts: ArtifactStore, *, pending_polls: int = 1):
        self.artifacts = artifacts
        self.jobs = MockJobQueue[EvaluationResult]('mock-evaluation', pending_polls=pending_polls)

    def submit(self, request: EvaluationRequest, *, idempotency_key: str):
        def evaluate():
            for ref in request.candidate.changes.values():
                self.artifacts.get(ref)
            for ref in request.data_refs:
                self.artifacts.get(ref)
            report = self.artifacts.put(json.dumps({
                'simulated': True, 'request': asdict(request), 'quality_evaluated': False,
            }).encode(), kind='mock_evaluation_report')
            return EvaluationResult(request.candidate.candidate_id, request.protocol_ref,
                                    {}, (report.artifact_id,), simulated=True)
        return self.jobs.submit(request, evaluate, idempotency_key=idempotency_key)

    def poll(self, job):
        return self.jobs.poll(job)

    def cancel(self, job):
        self.jobs.cancel(job)

    def result(self, job):
        return self.jobs.result(job)
