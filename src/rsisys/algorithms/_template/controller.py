"""Example policy: revise a prompt from Lab feedback until all checks pass."""
from rsisys.api import Decision, ExecutionStatus, LabRequest


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
        # "checker" is the Lab alias in configs/run.json, not its registry name.
        return ('checker.evaluate',)

    def decide(self, window):
        if self.pending is not None:
            # Advance only after receiving the matching terminal receipt.
            if window.last_operation != self.pending or window.feedback is None:
                raise ValueError('Matching Lab feedback is required')
            feedback = window.feedback
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
        self.pending = LabRequest('checker.evaluate', {'prompt': self.prompt})
        return Decision('invoke', operation=self.pending, reason='Evaluate the current prompt')

    def report(self):
        return {'method': 'custom_prompt', 'status': self.status, 'evaluations': self.evaluations,
                'prompt': self.prompt, 'instruction_coverage': self.score,
                'training_performed': False}
