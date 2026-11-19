"""GSM8K evaluation and reflection for the native multi-component GEPA method."""
from dataclasses import replace

from rsisys.api import Invariant, OperationResult

from ..shared.search_registry import operation
from .lab import GEPAGSM8KLab
from .merge_controller import candidate_id


class GEPAMergeLab(GEPAGSM8KLab):
    def invariants(self):
        return (Invariant.baseline_before('reflect'),)

    def _candidate_id(self, candidate):
        return candidate_id(candidate)

    def describe(self):
        register, evaluate, select, _ = super().describe()
        return (register, replace(evaluate, idempotent=False), select,
                replace(operation('reflect', prompt='string', round='integer'), idempotent=False))

    def execute(self, call):
        if call.name == 'reflect':
            request = call.arguments
            text = self.complete('decision_maker', request['prompt'], round_index=request['round'])
            artifact = self.artifacts.put(text.encode(), kind='gepa-reflect')
            return OperationResult('GEPA reflection', (artifact,), False, data={'text': text})
        return super().execute(call)

    def evaluate(self, request):
        split = request['split']
        indices = request.get('indices')
        if indices is None:
            indices = range(len(self.groups[split]))
        elif (not indices or any(type(i) is not int or not 0 <= i < len(self.groups[split]) for i in indices)):
            raise ValueError('Invalid GEPA example offsets')
        # The pinned sampler can pad an epoch with repeated training indices.
        values, evidence = self._evaluate_rows(request, indices, lambda row:
            f"Problem:\n{row['question']}\n\nStrategy:\n{request['strategy']}\n\nVerification:\n{request['verification']}")
        values['candidate_id'] = request['candidate_id']
        if request.get('capture_traces'):
            values['trajectories'] = [_reflection_trace(row) for row in evidence['records']]
        return values, evidence


def _reflection_trace(row):
    output = row['response']
    if len(output) > 2400:
        output = output[:1600] + '\n[middle excerpt omitted]\n' + output[-800:]
    feedback = ('Score 1. The final answer matches the reference answer.' if row['correct'] else
                f"Score 0. Reference answer: {row['expected']}.")
    return {'Inputs': row['question'], 'Generated Outputs': output, 'Feedback': feedback}
