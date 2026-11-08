"""CPU fixture: feedback -> optional proposal -> main decision -> execution -> trace.

Run with PYTHONPATH=src python examples/components/feedback_pipeline.py. No real model,
training, code deployment or harness activation is performed.
"""
from concurrent.futures import Future
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from rsisys.config import DemoConfig, build_demo
from rsisys.loops.contracts.model_service import GenerationResponse
from rsisys.loops.decision_making.feedback import ModelImprovementProposer
from rsisys.research import ResearchAssembly, assemble_research


class InlineExecutor:
    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future


class FixtureBackend:
    supports_response_schema = True

    def count_tokens(self, request):
        return len(request.prompt.encode()) + len(request.system.encode())

    def generate(self, request):
        value = json.loads(request.prompt)
        if 'receipt' in value:
            payload = {'proposals': [{'kind': 'harness_change', 'subject': 'meta_agent',
                'content': 'Create and test a fixture parser candidate.',
                'applicability': 'Only this CPU fixture.',
                'validation_plan': 'Check the generated artifact before a separate activation decision.',
                'source_refs': [value['source_refs'][0]]}]}
            if 'candidate_evidence' in value:
                payload['proposals'][0]['evaluation_claims'] = []
        elif not value['decision_graph']:
            payload = {'kind': 'invoke', 'operation': 'prepare_data', 'arguments': {'text': 'Initial fixture receipt'}}
        else:
            inbox = value['research_agenda']['feedback_learning']['improvement_proposals']['recent']
            if not any(row.get('feedback_proposal_ids') for row in value['decision_graph']):
                payload = {'kind': 'invoke', 'operation': 'prepare_data',
                    'arguments': {'text': 'Fixture construction output; not an installed harness'},
                    'feedback_proposal_ids': [inbox[0]['proposal_id']]}
            else:
                payload = {'kind': 'finish'}
        return GenerationResponse(json.dumps(payload), request.model, simulated=True)


def main():
    runner, _ = build_demo(DemoConfig.load(Path(__file__).resolve().parents[2] / 'configs/demo.json'))
    with TemporaryDirectory(prefix='rsisys-pipeline-fixture-') as directory, runner:
        root = Path(directory)
        loop = runner.decision_making
        backend = FixtureBackend()
        loop.harness.policy.backend = backend
        assemble_research(loop, assembly=ResearchAssembly(
            improvement_proposals=ModelImprovementProposer(root / 'proposals')),
            backend=backend, path=root / 'feedback.json', executor=InlineExecutor(), owner=runner.assembly)
        for _ in range(3):
            loop.step('Exercise feedback product routing')
        service = loop.feedback_service
        service.refresh()
        linked = [p for p in service.proposals() if p['decision_links']]
        segments = [service.extract_trajectory(r['job_id']) for r in service.trajectories()]
        assert len(linked) == 1 and len(segments) == 2
        assert all(not s['training_ready'] and not s['applied'] for s in segments)
        print(json.dumps({'simulated': True, 'execution_receipts': len(segments),
            'proposal_linked_decisions': len(linked), 'trajectory_mode': 'archive_only',
            'harness_activation_verified': False, 'status': 'fixture_passed'}, indent=2))


if __name__ == '__main__':
    main()
