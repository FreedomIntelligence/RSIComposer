"""OPRO's Lab: MathLab's data and scoring, with the pinned author prompt and grader.

Proposing instructions belongs to the Controller. The Lab renders each problem
with the author's gen_prompt and asks the actor; answers are graded by the
verifier the dataset manifest selects. The catalog adds "opro_official", the
author's GSM8K grader (verifier.py), which the presets select. The read-only
exemplars operation shows the optimizer search-split problems.
"""
from dataclasses import replace
from functools import partial

from rsisys.api import MATH_VERIFIERS

from ..shared.math_search import MathSearchLab
from ..shared.search_registry import operation
from .protocol import SCORER_SYSTEM_V1, candidate_id, upstream
from .verifier import OPROVerifier


class OPROGSM8KLab(MathSearchLab):
    receipt_kind = 'opro'

    def __init__(self, context, config):
        self.opt = upstream(config['source_dir'])
        self.verifiers = {**MATH_VERIFIERS, 'opro_official': partial(OPROVerifier, self.opt.eval_utils)}
        super().__init__(context, config)

    def _candidate_id(self, candidate):
        if (not isinstance(candidate, dict) or set(candidate) != {'instruction'}
                or not isinstance(candidate['instruction'], str) or not 0 < len(candidate['instruction']) <= 500):
            raise ValueError('Expected one nonempty OPRO instruction')
        return candidate_id(candidate['instruction'])

    def describe(self):
        exemplars = operation('exemplars', indices='array')
        return (*super().describe(), replace(exemplars, description='Search-split problems shown to the optimizer',
                                             read_only=True))

    def execute(self, call):
        if call.name == 'exemplars':
            rows = self.datasets['search']
            return self.receipt(call, {'exemplars': [{'question': rows[i].prompt, 'answer': rows[i].answer}
                                                     for i in self._indices(call.arguments, len(rows))]})
        return super().execute(call)

    def feedback(self, record):
        return f"Accuracy: {float(record['correct'])}"

    def solve(self, problems, arguments):
        import pandas as pd

        instruction = arguments['candidate']['instruction']
        adapted = self.config['parameters'].get('local_adapter') == 'qwen-instructions-v1'
        attempts = []
        for problem in problems:
            prompt = self.opt.eval_utils.gen_prompt(pd.DataFrame([(problem.prompt, problem.answer)]),
                instruction=instruction, idx=0, include_qa=False,
                instruction_pos=self.config['parameters']['instruction_pos'], dataset_name='gsm8k')
            attempts.append(self.ask(prompt, SCORER_SYSTEM_V1 if adapted else '',
                                     seed=self.config['seed'] if adapted else None))
        return attempts
