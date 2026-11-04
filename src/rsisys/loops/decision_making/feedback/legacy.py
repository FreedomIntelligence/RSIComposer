"""Explicit compatibility prompt/schema; new products do not inherit these."""
from importlib import resources

from rsisys.loops.contracts.model_service import GenerationRequest, ModelRef
from rsisys.runtime.json_tools import encoded, result_id


class LegacyFeedbackInterpreter:
    def __init__(self, *, instructions=None):
        self.instructions = (resources.files(__package__).joinpath('prompts/legacy_feedback.txt').read_text()
                             if instructions is None else instructions)
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError('Legacy feedback instructions must be nonempty text')

    def configuration(self):
        return {'provider': 'legacy_feedback_interpreter', 'instructions_digest': result_id(self.instructions)}

    def request(self, row, backend, limits):
        from .learning import _schema, _CANDIDATE_INSTRUCTIONS
        evidence = row['input'].get('candidate_evidence')
        return GenerationRequest('decision_maker', ModelRef(**row['model']), encoded(row['input']),
            system=self.instructions + (_CANDIDATE_INSTRUCTIONS if evidence is not None else ''),
            max_output_tokens=limits['max_output_tokens'],
            response_schema=_schema(row['input']['source_refs'], row['input'].get('grouping_context'), evidence)
                            if backend.supports_response_schema else None)
