"""GSM8K evaluation for the native multi-component GEPA method; reflection belongs to the Controller."""
from dataclasses import replace


from .lab import GEPAGSM8KLab
from .merge_controller import candidate_id


class GEPAMergeLab(GEPAGSM8KLab):
    def _candidate_id(self, candidate):
        return candidate_id(candidate)

    def describe(self):
        register, evaluate, select = super().describe()
        return (register, replace(evaluate, idempotent=False), select)

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
