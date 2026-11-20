"""GEPA-Merge's Lab: scores two-component candidates; reflection belongs to the Controller."""
from dataclasses import replace

from .lab import ACTOR_SYSTEM, GEPAGSM8KLab
from .merge_controller import candidate_id


class GEPAMergeLab(GEPAGSM8KLab):
    template_name = 'gepa-merge-components/v1'
    def _candidate_id(self, candidate):
        return candidate_id(candidate)

    def describe(self):
        register, evaluate, select = super().describe()
        return (register, replace(evaluate, idempotent=False), select)

    def offsets(self, request, size):
        indices = request.get('indices')
        if indices is None:
            return list(range(size))
        # The pinned sampler can pad an epoch with repeated training indices.
        if not isinstance(indices, list) or not indices or any(type(i) is not int or not 0 <= i < size for i in indices):
            raise ValueError('Invalid GEPA example offsets')
        return indices

    def domain_agent(self, arguments):
        # Two prompt components rather than one instruction; the runtime sees them as one agent.
        candidate = arguments['candidate']
        return {'instruction': f"{candidate['strategy']}\x00{candidate['verification']}"}

    def render(self, problem, instruction, request):
        strategy, verification = instruction.split('\x00', 1)
        return (f"Problem:\n{problem.prompt}\n\nStrategy:\n{strategy}"
                f"\n\nVerification:\n{verification}"), ACTOR_SYSTEM

    def result_data(self, records, ref, arguments):
        data = super().result_data(records, ref, arguments)
        request = arguments['request']
        data['candidate_id'] = request['candidate_id']
        if request.get('capture_traces'):
            data['trajectories'] = [_reflection_trace(row) for row in records]
        return data


def _reflection_trace(row):
    output = row['completion'] or ''
    if len(output) > 2400:
        output = output[:1600] + '\n[middle excerpt omitted]\n' + output[-800:]
    feedback = ('Score 1. The final answer matches the reference answer.' if row['correct'] else
                f"Score 0. Reference answer: {row['expected']}.")
    return {'Inputs': row['question'], 'Generated Outputs': output, 'Feedback': feedback}
