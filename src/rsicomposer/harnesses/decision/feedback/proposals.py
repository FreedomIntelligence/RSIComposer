"""Optional proposal products; execution and model accounting belong to the host."""
from copy import deepcopy
from dataclasses import asdict
from importlib import resources
from pathlib import Path

from .proposal_validation import BUILTIN_IMPROVEMENTS, bind_targets, proposal_schema, validate_proposals
from rsicomposer.loops.contracts.model_service import GenerationRequest
from rsicomposer.runtime.storage.files import atomic_write, json_bytes
from rsicomposer.runtime.storage.json_tools import encoded, load_object, result_id


class ModelImprovementProposer:
    """Generate one bounded proposal product without construction tools."""
    max_model_calls = 1
    tool_name = 'propose_improvements'

    def __init__(self, path, *, proposal_targets=BUILTIN_IMPROVEMENTS, instructions=None):
        self.path = Path(path).resolve()
        self._targets = bind_targets(proposal_targets)
        self.instructions = (resources.files(__package__).joinpath('prompts/improvement_proposals.txt').read_text()
                             if instructions is None else instructions)
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError('Improvement proposal instructions must be nonempty text')

    @property
    def proposal_targets(self):
        return tuple(self._targets.values())

    def configuration(self):
        return {'provider': 'model_improvement_proposer', 'path': str(self.path),
                'proposal_targets': [rule.configuration() for rule in self.proposal_targets],
                'instructions_digest': result_id(self.instructions), 'max_model_calls': 1}

    def _result_path(self, event):
        return self.path / (result_id(event.event_id) + '.result.json')

    def recover(self, event):
        path = self._result_path(event)
        if not path.exists():
            return None
        saved = load_object(path.read_text())
        if (saved.get('version') != 2 or saved.get('event_digest') != result_id(asdict(event))
                or saved.get('model') != event.identity.get('feedback_model')
                or saved.get('configuration') != self.configuration()):
            raise ValueError('Saved proposal product differs from captured evidence, model or configuration')
        output = saved['output']
        validate_proposals(output, event.source_refs, targets=self._targets,
                           candidate_evidence=event.candidate_evidence)
        return deepcopy(output)

    def process(self, event, model):
        if event.identity.get('feedback_model') != asdict(model.model):
            raise ValueError('Proposal model differs from the captured processing identity')
        if self.recover(event) is not None:
            raise ValueError('Proposal already recorded; recover instead of replaying inference')
        schema = proposal_schema(event.source_refs, candidate_evidence=event.candidate_evidence, targets=self._targets)
        transport = {}
        if model.supports_tool_calling:
            transport = {'tools': ({'type': 'function', 'function': {
                'name': self.tool_name, 'description': 'Submit optional improvement task proposals.',
                'parameters': schema}},),
                'tool_choice': {'type': 'function', 'function': {'name': self.tool_name}}}
        elif model.supports_response_schema:
            transport = {'response_schema': schema}
        instructions = self.instructions + '\nInstalled proposal targets: ' + encoded(
            [rule.configuration() for rule in self.proposal_targets])
        request = GenerationRequest('decision_maker', model.model, encoded(event.evidence_packet),
            system=instructions, max_output_tokens=model.max_output_tokens, **transport)
        response = model.generate(request)
        if response.finish_reason == 'length':
            raise ValueError('Improvement proposal output was truncated')
        text = response.text
        if request.tools:
            if (len(response.tool_calls) != 1 or
                    response.tool_calls[0].get('function', {}).get('name') != self.tool_name):
                raise ValueError('Expected one improvement proposal function call')
            text = response.tool_calls[0]['function']['arguments']
        output = load_object(text)
        validate_proposals(output, event.source_refs, targets=self._targets, candidate_evidence=event.candidate_evidence)
        atomic_write(self._result_path(event), json_bytes({'version': 2,
            'event_digest': result_id(asdict(event)), 'model': asdict(model.model),
            'configuration': self.configuration(), 'output': output}))
        return output
