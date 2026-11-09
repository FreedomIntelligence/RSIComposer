"""Optional proposal generation; construction belongs to an ordinary decision."""
import json
from copy import deepcopy
from dataclasses import asdict
from importlib import resources
from pathlib import Path

from rsisys.harnesses.decision.feedback.evidence import admitted_candidate_evidence
from rsisys.harnesses.decision.feedback.learning import _schema, validate_proposals
from rsisys.harnesses.decision.feedback.model_gateway import FeedbackModelGateway
from rsisys.loops.contracts.model_service import GenerationRequest, ModelRef
from rsisys.runtime.storage.files import atomic_write, json_bytes
from rsisys.runtime.storage.json_tools import encoded, result_id


class ModelHarnessProposer:
    """One accounted call in a separate context with no construction tools."""
    max_calls_per_job = 1
    actual_exchanges_read_path = '/feedback_response/model_exchanges'
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
                'instructions_digest': result_id(self.instructions), 'max_calls_per_job': 1}

    def _result_path(self, job):
        return self.path / (job['job_id'] + '.result.json')

    def recover(self, job):
        path = self._result_path(job)
        if not path.exists():
            return None
        saved = json.loads(path.read_text())
        if saved['receipt_digest'] != job['receipt_digest'] or saved['model'] != job['model']:
            raise ValueError('Saved harness proposal identity differs from captured feedback')
        if saved.get('configuration') != self.configuration():
            raise ValueError('Saved harness proposal configuration differs from the active component')
        if saved.get('input_digest') != result_id(job['input']):
            raise ValueError('Saved harness proposal evidence snapshot differs from the captured input')
        return deepcopy(saved['result'])

    def process(self, job, backend, limits):
        if self.recover(job) is not None:
            raise ValueError('Proposal already recorded; recover instead of replaying inference')
        gateway = None
        try:
            gateway = FeedbackModelGateway(backend, ModelRef(**job['model']),
                max_calls=min(1, limits['max_calls']), max_input_tokens=limits['max_input_tokens'],
                path=self.path / (job['job_id'] + '.exchanges.json'))
            schema = _schema(job['input']['source_refs'], candidate_evidence=admitted_candidate_evidence(job),
                             kinds=self.kinds)
            transport = {}
            if getattr(backend, 'supports_tool_calling', False) is True:
                transport = {'tools': ({'type': 'function', 'function': {
                    'name': self.tool_name, 'description': 'Submit optional improvement task proposals.',
                    'parameters': schema}},),
                    'tool_choice': {'type': 'function', 'function': {'name': self.tool_name}}}
            elif backend.supports_response_schema:
                transport = {'response_schema': schema}
            request = GenerationRequest('decision_maker', ModelRef(**job['model']), encoded(job['input']),
                system=self.instructions, max_output_tokens=limits['max_output_tokens'], **transport)
            response = gateway.generate(request)
            if response.finish_reason == 'length':
                raise ValueError('Harness proposal output was truncated')
            text = response.text
            if request.tools:
                if (len(response.tool_calls) != 1 or
                        response.tool_calls[0].get('function', {}).get('name') != self.tool_name):
                    raise ValueError('Expected one harness proposal function call')
                text = response.tool_calls[0]['function']['arguments']
            proposals = validate_proposals(text, job['input']['source_refs'],
                                           candidate_evidence=admitted_candidate_evidence(job))
            if any(p['kind'] not in self.kinds or p.get('subject') not in ('meta_agent', 'lab')
                   or (p['kind'] == 'policy_training' and p.get('subject') != 'meta_agent')
                   for p in proposals):
                raise ValueError('This component only produces explicitly targeted harness proposals')
            result = {'status': 'completed', 'error': None, 'charged_calls': gateway.attempted,
                'proposals': proposals, 'simulated': job['simulated'] or response.simulated,
                'response': {'exchange': asdict(response), 'exchanges_ref': str(gateway.path)}}
        except Exception as exc:
            # Before gateway construction succeeds an existing exchange can
            # represent an unknown call. Retain this stage's reservation.
            result = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}',
                'charged_calls': gateway.attempted if gateway is not None else min(1, limits['max_calls']),
                'proposals': [], 'response': {'exchanges_ref': str(gateway.path) if gateway else None}}
        result['response']['model_exchanges'] = gateway.snapshot() if gateway is not None else []
        atomic_write(self._result_path(job), json_bytes({'receipt_digest': job['receipt_digest'],
                     'model': job['model'], 'configuration': self.configuration(),
                     'input_digest': result_id(job['input']), 'result': result}))
        return result


class ModelImprovementProposer(ModelHarnessProposer):
    """Optional task proposals only; datasets and training stay inside a Lab."""
    kinds = ('harness_change', 'policy_training')
    provider = 'model_improvement_proposer'
    prompt_resource = 'improvement_proposals.txt'
    tool_name = 'propose_improvements'
