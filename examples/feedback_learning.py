"""CPU fixture: completed feedback -> four proposal kinds -> Decision memory.

Run with PYTHONPATH=src python examples/feedback_learning.py.
No real model, training, code edits, or quality/cache claims.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from rsisys.assembly import AssemblyContext, AssemblyRunner, build_default_exploration
from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef
from rsisys.loops.decision_making.harness.feedback_learning import KINDS
from rsisys.loops.decision_making.harness.learning_harness import feedback_learning_decision_factory
from rsisys.runtime.artifacts import FileArtifactStore
from rsisys.runtime.model import MockModelBackend
from rsisys.runtime.sandbox import MockSandboxBackend


class LearningFixtureBackend(MockModelBackend):
    def generate(self, request):
        packet = json.loads(request.prompt)
        if 'receipt' not in packet:
            return super().generate(request)
        proposals = []
        for kind in KINDS:
            row = dict(kind=kind, content='Fixture proposal; validate against the original receipt.',
                       applicability='Only this simulated echo experiment.',
                       source_refs=[packet['source_refs'][0]],
                       validation_plan='Replay independently before accepting the proposed conclusion.')
            if kind == 'training_data':
                row['training_example'] = dict(input='A simulated experiment returned a receipt.',
                    target='Inspect its evidence before making a quality claim.', target_status='unverified')
            proposals.append(row)
        return GenerationResponse(json.dumps({'proposals': proposals}), request.model, simulated=True)


def main():
    with TemporaryDirectory(prefix='rsisys-feedback-fixture-') as directory:
        root = Path(directory)
        context = AssemblyContext(
            BasicExperienceSystem(InMemoryExperienceStore()), FileArtifactStore(root / 'artifacts'),
            {'fixture': LearningFixtureBackend()}, MockSandboxBackend(),
            {role: ModelRef('fixture', role, 'fixture-v1') for role in ('decision_maker', 'actor')},
            data_dir=root)
        try:
            exploration = build_default_exploration(context)
            decision = feedback_learning_decision_factory(max_calls=4)(context, exploration)
            runner = AssemblyRunner(decision, exploration, context)
            runner.run('Inspect simulated feedback without claiming real improvement', rounds=2,
                       max_decision_steps=8)
        finally:
            context.close()
        learning = decision.harness.feedback_learning
        rows = list(learning.state['jobs'].values())
        assert len(rows) == 2 and all(row['status'] == 'completed' for row in rows), learning.read()
        assert all({p['kind'] for p in row['proposals']} == set(KINDS) for row in rows)
        print(json.dumps({'simulated': True, 'learning_jobs': len(rows),
                          'proposal_kinds': list(KINDS), 'training_or_changes_applied': False,
                          'status': 'fixture_passed'}, indent=2))


if __name__ == '__main__':
    main()
