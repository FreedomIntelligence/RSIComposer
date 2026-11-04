"""Export effective requests from CPU fixtures, with no external model or execution."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from examples.medical.experiment import ResearchPolicy
from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef
from rsisys.loops.decision_making.action_space import ModularDecisionActionSpace
from rsisys.loops.decision_making.harness.attempts import AttemptContract
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.harness.state import DecisionContext


OUTPUT = Path(__file__).with_suffix('')
MODEL = ModelRef('fixture', 'meta-agent', 'review-v1')


class CaptureBackend:
    supports_response_schema = True
    max_output_tokens = 8192
    token_count_mode = 'fixture_constant_not_a_tokenizer'

    def __init__(self, *outputs):
        self.outputs, self.requests = iter(outputs), []

    def count_tokens(self, request):
        return 100

    def generate(self, request):
        self.requests.append(request)
        return GenerationResponse(json.dumps(next(self.outputs)), request.model, True)


def save(name, request, **metadata):
    value = {'scope': 'CPU fixture; no real inference or task execution', **metadata,
             'request': asdict(request)}
    path = OUTPUT / (name + '.json')
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    OUTPUT.mkdir(exist_ok=True)
    # Project-supplied data stays in user content. The host metadata below makes
    # the medical projection visible; its time mode does not expose these limits.
    context = DecisionContext('Synthetic module-review objective', (), (), (),
        directive_modes=('configure_prompt',), review_actions=('continue',),
        run_control={'seconds_remaining': 1234, 'training_remaining': 3, 'model_calls_remaining': 999})
    project = {'lab_id': 'review-fixture', 'objective': 'Synthetic contract review',
               'evidence_scope': 'No medical samples or real metrics'}
    baseline = ModelDecisionPolicy(CaptureBackend(), MODEL)
    for required in (False, True):
        backend = CaptureBackend({'kind': 'finish'}, {'action': 'continue'})
        policy = ModelDecisionPolicy(backend, MODEL, project_context=project,
                                     attempt_contract=AttemptContract(required=required))
        policy.decide(context)
        policy.review(context)
        for kind, request in zip(('decision', 'review'), backend.requests):
            save(kind + ('-required-attempt' if required else ''), request,
                 prompt_manifest=policy.prompts.manifest())

    backend = CaptureBackend({'summary': 'Fixture summary: read original evidence.'})
    policy = ModelDecisionPolicy(backend, MODEL)
    policy._summarize_context([], '', 100, current_state=policy._summary_current_state(context))
    save('summary', backend.requests[0], prompt_manifest=policy.prompts.manifest())

    backend = CaptureBackend({'kind': 'invalid'}, {'kind': 'finish'})
    policy = ModelDecisionPolicy(backend, MODEL, max_validation_attempts=2)
    policy.decide(context)
    save('validation-correction', backend.requests[-1], prompt_manifest=policy.prompts.manifest())

    for mode in ('time', 'researcher'):
        backend = CaptureBackend({'kind': 'explore', 'mode': 'configure_prompt',
            'strategy': 'Exercise parser only; do not execute', 'arguments': {'instructions': 'Fixture'},
            'feedback_proposal_ids': ['fixture:proposal']})
        policy = ResearchPolicy(backend, MODEL, baseline.directives, ModularDecisionActionSpace(()),
            Mock(), SimpleNamespace(ledger=[]), ArtifactRef('fixture-train', 'training_data', 'fixture://train'),
            run_mode=mode, project_context=project)
        decision = policy.decide(context)
        assert decision.feedback_proposal_ids == ('fixture:proposal',)
        request, = backend.requests
        assert '"kind":"wait"' not in request.system
        save('medical-' + mode, request, prompt_manifest=policy.prompts.manifest(),
             note='Metadata shape parsed; proposal existence is checked by the harness before execution.')

    files = {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
             for path in sorted(OUTPUT.glob('*.json')) if path.name != 'manifest.json'}
    source_paths = [
        'src/rsisys/loops/decision_making/harness/policy/__init__.py',
        'src/rsisys/loops/decision_making/harness/policy/prompts.py',
        'examples/medical/experiment.py', 'src/rsisys/snapshot/local.py',
        'tests/test_m03_decision_policy.py', 'docs/review/repros/m03.py',
    ]
    source_paths.extend(str(p.relative_to(ROOT)) for p in sorted(
        (ROOT / 'src/rsisys/loops/decision_making/harness/policy/prompts').glob('*.txt')))
    manifest = {'scope': 'CPU fixture only; generated requests are not evidence of real model behavior',
                'files': files, 'source_sha256': {
                    name: sha256((ROOT / name).read_bytes()).hexdigest() for name in source_paths}}
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(OUTPUT), 'requests': len(files)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
