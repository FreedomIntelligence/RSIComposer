"""Controller prototype: copy this file to build your own Controller.

A Controller decides what to try next and which result to keep. Its Policy is an
object with decide(window) -> Decision. This one searches prompt and few-shot
settings using measured development scores, and shows each part of the Policy
contract:

  Required
    decide(window)          Read new receipts in window.feedbacks and return a
                            Decision: invoke operations, wait, or finish.

  Optional, recognized by the host when present
    required_operations()   Lab operations this Controller needs; checked at startup.
    report()                Summary written to result.json as controller_report.
    state(), restore()      Save and reload Policy state for durable runs.

The factory receives (context, config). context.lab_catalog describes the Labs;
config is this Controller's section of the run config.

The first candidate is evaluated synchronously and the rest as background jobs,
so one run shows both receipt paths. While a job runs, its call ID is in
window.pending and this Controller returns Decision('wait').
"""
from rsicomposer.api import Decision, ExecutionStatus, OperationCall


CANDIDATES = (
    {'instruction': 'Solve the arithmetic problem.', 'few_shot': 0},
    {'instruction': 'Choose the arithmetic operation from the question.', 'few_shot': 1},
    {'instruction': 'Check the calculation before giving the final answer.', 'few_shot': 2},
)


def validate_config(config):
    # harness and memory select the optional prototypes registered in __init__.py.
    limit = config.get('max_evaluations', len(CANDIDATES))
    if set(config) - {'max_evaluations', 'harness', 'memory', 'scoreboard_rows'}:
        raise ValueError('Unknown template Controller option')
    if type(limit) is not int or not 1 <= limit <= len(CANDIDATES):
        raise ValueError(f'max_evaluations must be between 1 and {len(CANDIDATES)}')
    if config.get('harness', 'default') not in ('default', 'scoreboard'):
        raise ValueError('harness must be default or scoreboard')
    if config.get('memory', 'none') not in ('none', 'best_score'):
        raise ValueError('memory must be none or best_score')


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
                # A background job is still running; its call ID is in window.pending.
                return Decision('wait', reason='Waiting for the evaluation job')
            self.pending = None
            if receipt.status != ExecutionStatus.COMPLETED:
                self.status = receipt.status.value
                return Decision('finish', reason='Evaluation failed; inspect the Lab receipt')
            self.results.append(self.metric.value(receipt.data))
        if len(self.results) == self.limit:
            self.status = 'completed'
            return Decision('finish', reason='All prompt candidates evaluated')
        background = len(self.results) > 0  # Baseline runs synchronously, later candidates as jobs.
        self.pending = OperationCall('math.evaluate', {
            'split': 'dev', **CANDIDATES[len(self.results)], 'background': background})
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
