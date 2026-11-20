"""GEPA's Lab: MathLab's data, verifiers and scoring, measuring one registered instruction.

Mutation belongs to the Controller. The Lab renders GEPA's scorer prompt and asks
the actor; the dataset manifest selects the answer verifier (the presets use
marked_numeric, GEPA's GSM8K grader).
"""
from ..shared.math_search import PromptSearchLab
from .controller import candidate_id
from .protocol import valid_instruction


ACTOR_SYSTEM = ('Solve each math word problem. End with one line containing #### followed by the numeric answer only.')


class GEPAGSM8KLab(PromptSearchLab):
    receipt_kind = 'gepa'
    template_name = 'gepa-instruction/v1'
    system = ACTOR_SYSTEM

    def _candidate_id(self, candidate):
        if not isinstance(candidate, dict) or set(candidate) != {'instruction'} or not valid_instruction(candidate['instruction']):
            raise ValueError('Expected one valid GEPA instruction')
        return candidate_id(candidate['instruction'])

    def render(self, problem, instruction, request):
        return f'Problem:\n{problem.prompt}\n\nInstruction:\n{instruction}', ACTOR_SYSTEM
