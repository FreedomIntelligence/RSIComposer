"""Explicit compatibility prompt/schema; new products do not inherit these."""
import json
from dataclasses import asdict
from importlib import resources

from .attempts import validate_attempt_updates
from .evidence import admitted_candidate_evidence
from .proposal_validation import CANDIDATE_INSTRUCTIONS, proposal_schema, validate_proposals
from rsisys.runtime.storage.json_tools import load_object as _load_object

from rsisys.loops.contracts.model_service import GenerationRequest, ModelRef
from rsisys.runtime.storage.json_tools import encoded, result_id


class LegacyFeedbackInterpreter:
    def __init__(self, *, instructions=None):
        self.instructions = (resources.files(__package__).joinpath('prompts/legacy_feedback.txt').read_text()
                             if instructions is None else instructions)
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError('Legacy feedback instructions must be nonempty text')

    def configuration(self):
        return {'provider': 'legacy_feedback_interpreter', 'instructions_digest': result_id(self.instructions)}

    def request(self, row, backend, limits):
        evidence = row['input'].get('candidate_evidence')
        return GenerationRequest('decision_maker', ModelRef(**row['model']), encoded(row['input']),
            system=self.instructions + (CANDIDATE_INSTRUCTIONS if evidence is not None else ''),
            max_output_tokens=limits['max_output_tokens'],
            response_schema=proposal_schema(row['input']['source_refs'], row['input'].get('grouping_context'), evidence)
                            if backend.supports_response_schema else None)


def interpret_feedback(row, backend, limits, request):
    """Return an unverified interpretation; never mutate the owning journal."""
    # No stores, policy mutation, or memory writes in this thread.
    if backend.count_tokens(request) > limits['max_input_tokens']:
        return {'status': 'input_too_large', 'error': 'Complete request exceeds token budget', 'response': None}
    try:
        response = backend.generate(request)
    except Exception as exc:
        return {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'response': None}
    receipt = asdict(response)
    try:
        if response.model != request.model:
            raise ValueError('Learning model identity mismatch')
        if response.finish_reason == 'length':
            raise ValueError('Feedback interpretation output was truncated')
        payload = _load_object(response.text)
        grouping = row['input'].get('grouping_context')
        updates = payload.pop('attempt_updates', []) if grouping is not None else []
        evidence = admitted_candidate_evidence(row)
        proposals = validate_proposals(json.dumps(payload), row['input']['source_refs'],
                                       candidate_evidence=evidence)
        # Grouping failure is isolated from valid feedback and execution.
        grouping_error = None
        try:
            updates = validate_attempt_updates(updates, grouping or {}, row['input']['source_refs'], evidence)
        except (ValueError, TypeError, KeyError) as exc:
            updates, grouping_error = [], str(exc)
    except (ValueError, TypeError, KeyError) as exc:
        return {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'response': receipt}
    return {'status': 'completed', 'error': None, 'proposals': proposals,
            'attempt_updates': updates, 'grouping_error': grouping_error,
            'response': receipt, 'simulated': row['simulated'] or response.simulated}
