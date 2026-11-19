"""Example policy: revise a prompt from Lab feedback until all checks pass."""
from rsisys.api import Decision, ExecutionStatus, OperationCall


def validate_config(config):
    if set(config) - {'max_evaluations'}:
        raise ValueError('Unknown custom_prompt option')
    limit = config.get('max_evaluations', 3)
    if type(limit) is not int or limit < 1:
        raise ValueError('max_evaluations must be a positive integer')


class PromptController:
    def __init__(self, context, config):
        validate_config(config)
        self.limit = config.get('max_evaluations', 3)
        self.prompt = 'Solve the problem.'
        self.evaluations = 0
        self.pending = None
        self.score = None
        self.status = 'running'

    def required_operations(self):
        # "checker" is the Lab alias bound by this method's registration entry.
        return ('checker.evaluate',)

    def decide(self, window):
        feedback = window.feedbacks[0] if window.feedbacks else None
        if self.pending is not None:
            # Advance only after receiving the matching terminal receipt.
            if (feedback is None or feedback.call_id != self.pending.call_id):
                raise ValueError('Matching Lab feedback is required')
            self.pending = None
            if feedback.status != ExecutionStatus.COMPLETED:
                self.status = feedback.status.value
                return Decision('finish', reason='Lab failed; inspect its receipt')
            self.evaluations += 1
            self.score = feedback.data['quality_metrics']['instruction_coverage']
            missing = feedback.data['missing_instructions']
            if not missing:
                self.status = 'completed'
                return Decision('finish', reason='All prompt checks passed')
            if self.evaluations >= self.limit:
                self.status = 'budget_exhausted'
                return Decision('finish', reason='Evaluation budget exhausted')
            # This policy belongs to the controller. The host only routes calls.
            self.prompt += ' ' + missing[0]
        self.pending = OperationCall('checker.evaluate', {'prompt': self.prompt})
        return Decision.invoke(self.pending, reason='Evaluate the current prompt')

    # state()/restore() make the method durable: Application(durable=True),
    # Application.resume and snapshots capture exactly this data.
    def state(self):
        return {'prompt': self.prompt, 'evaluations': self.evaluations, 'pending': self.pending,
                'score': self.score, 'status': self.status}

    def restore(self, state):
        self.prompt, self.evaluations = state['prompt'], state['evaluations']
        self.pending, self.score, self.status = state['pending'], state['score'], state['status']

    def report(self):
        return {'method': 'custom_prompt', 'status': self.status, 'evaluations': self.evaluations,
                'prompt': self.prompt, 'instruction_coverage': self.score,
                'training_performed': False}
