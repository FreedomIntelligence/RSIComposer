"""Optional proposal products; execution and model accounting belong to the host."""
import json
from copy import deepcopy
from dataclasses import asdict
from importlib import resources
from pathlib import Path

from rsisys.harnesses.decision.feedback.proposal_validation import proposal_schema, validate_proposals
from rsisys.loops.contracts.model_service import GenerationRequest
from rsisys.runtime.storage.files import atomic_write, json_bytes
from rsisys.runtime.storage.json_tools import encoded, result_id


class ModelHarnessProposer:
    """Generate one bounded proposal product without construction tools."""
    max_model_calls = 1
    kinds = ('harness_change',)
    provider = 'model_harness_proposer'
    prompt_resource = 'harness_proposals.txt'
    tool_name = 'propose_harness_changes'

    def __init__(self, path, *, instructions=None):
        self.path = Path(path).resolve()
        self.instructions = (resources.files(__package__).joinpath('prompts/' + self.prompt_resource).read_text()
                             if instructions is None else instructions)
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError('Harness proposal instructions must be nonempty text')

    def configuration(self):
        return {'provider': self.provider, 'path': str(self.path),
                'instructions_digest': result_id(self.instructions), 'max_model_calls': 1}

    def _result_path(self, event):
        return self.path / (result_id(event.event_id) + '.result.json')

    def recover(self, event):
        path = self._result_path(event)
        if not path.exists():
            return None
        saved = json.loads(path.read_text())
        if (saved.get('version') != 1 or saved.get('event_digest') != result_id(asdict(event))
                or saved.get('model') != event.identity.get('feedback_model')
                or saved.get('configuration') != self.configuration()):
            raise ValueError('Saved proposal product differs from captured evidence, model or configuration')
        output = saved['output']
        self._validate(output, event)
        return deepcopy(output)

    def _validate(self, output, event):
        proposals = validate_proposals(encoded(output), event.source_refs,
                                       candidate_evidence=event.candidate_evidence)
        if any(p['kind'] not in self.kinds or p.get('subject') not in ('meta_agent', 'lab')
               or (p['kind'] == 'policy_training' and p.get('subject') != 'meta_agent') for p in proposals):
            raise ValueError('This component only produces explicitly targeted improvement proposals')
        return proposals

    def process(self, event, model):
        if event.identity.get('feedback_model') != asdict(model.model):
            raise ValueError('Proposal model differs from the captured processing identity')
        if self.recover(event) is not None:
            raise ValueError('Proposal already recorded; recover instead of replaying inference')
        schema = proposal_schema(event.source_refs, candidate_evidence=event.candidate_evidence,
                                 kinds=self.kinds)
        transport = {}
        if model.supports_tool_calling:
            transport = {'tools': ({'type': 'function', 'function': {
                'name': self.tool_name, 'description': 'Submit optional improvement task proposals.',
                'parameters': schema}},),
                'tool_choice': {'type': 'function', 'function': {'name': self.tool_name}}}
        elif model.supports_response_schema:
            transport = {'response_schema': schema}
        request = GenerationRequest('decision_maker', model.model, encoded(event.evidence_packet),
            system=self.instructions, max_output_tokens=model.max_output_tokens, **transport)
        response = model.generate(request)
        if response.finish_reason == 'length':
            raise ValueError('Harness proposal output was truncated')
        text = response.text
        if request.tools:
            if (len(response.tool_calls) != 1 or
                    response.tool_calls[0].get('function', {}).get('name') != self.tool_name):
                raise ValueError('Expected one harness proposal function call')
            text = response.tool_calls[0]['function']['arguments']
        output = {'proposals': validate_proposals(text, event.source_refs,
                                                candidate_evidence=event.candidate_evidence)}
        self._validate(output, event)
        atomic_write(self._result_path(event), json_bytes({'version': 1,
            'event_digest': result_id(asdict(event)), 'model': asdict(model.model),
            'configuration': self.configuration(), 'output': output}))
        return output


class ModelImprovementProposer(ModelHarnessProposer):
    """Optional task proposals only; datasets and training stay inside a Lab."""
    kinds = ('harness_change', 'policy_training')
    provider = 'model_improvement_proposer'
    prompt_resource = 'improvement_proposals.txt'
    tool_name = 'propose_improvements'
