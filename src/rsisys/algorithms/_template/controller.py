"""Search prompt and few-shot choices using measured development scores."""
from rsisys.api import Decision, ExecutionStatus, OperationCall


CANDIDATES = (
    {'instruction': 'Solve the arithmetic problem.', 'few_shot': 0},
    {'instruction': 'Choose the arithmetic operation from the question.', 'few_shot': 1},
    {'instruction': 'Check the calculation before giving the final answer.', 'few_shot': 2},
)


def validate_config(config):
    limit = config.get('max_evaluations', len(CANDIDATES))
    if set(config) - {'max_evaluations'} or type(limit) is not int or not 1 <= limit <= len(CANDIDATES):
        raise ValueError(f'max_evaluations must be between 1 and {len(CANDIDATES)}')


class PromptSearchController:
    def __init__(self, context, config):
        validate_config(config)
        self.limit = config.get('max_evaluations', len(CANDIDATES))
        self.metric = context.lab_catalog.primary_metric('math')
        self.pending = None
        self.results = []
        self.status = 'running'

    def required_operations(self):
        return ('math.evaluate',)

    def decide(self, window):
        if self.pending is not None:
            receipt = next((item for item in window.feedbacks if item.call_id == self.pending.call_id), None)
            if receipt is None:
                raise ValueError('Expected a matching Lab receipt')
            self.pending = None
            if receipt.status != ExecutionStatus.COMPLETED:
                self.status = receipt.status.value
                return Decision('finish', reason='Evaluation failed; inspect the Lab receipt')
            self.results.append(self.metric.value(receipt.data))
        if len(self.results) == self.limit:
            self.status = 'completed'
            return Decision('finish', reason='All prompt candidates evaluated')
        self.pending = OperationCall('math.evaluate', {'split': 'dev', **CANDIDATES[len(self.results)]})
        return Decision.invoke(self.pending)

    def report(self):
        best = None
        if self.results:
            choose = max if self.metric.direction == 'maximize' else min
            best = choose(range(len(self.results)), key=self.results.__getitem__)
        return {'status': self.status, 'metric': self.metric.name,
                'baseline_score': self.results[0] if self.results else None,
                'candidate_scores': list(self.results),
                'selected': dict(CANDIDATES[best]) if best is not None else None}

    def state(self):
        return {'pending': self.pending, 'results': list(self.results), 'status': self.status}

    def restore(self, state):
        self.pending, self.results, self.status = state['pending'], state['results'], state['status']
