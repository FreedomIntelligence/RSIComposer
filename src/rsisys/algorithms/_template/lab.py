"""CPU-only wiring example: literal prompt checks, not LLM quality measurement.

Replace execute() with your own benchmark, inference/training engine or harness.
Keep domain checks and execution here; the controller chooses the next action.
"""
from rsisys.api import LabFeedback, OperationSpec


class PromptCheckLab:
    def __init__(self, context, config):
        if config:
            raise ValueError('custom_prompt_check accepts no configuration')
        self.instructions = ('Explain your reasoning.', 'Check your answer.')

    def describe(self):
        return (OperationSpec('evaluate', 'Check required instructions in a prompt', {
            'type': 'object', 'properties': {'prompt': {'type': 'string', 'minLength': 1}},
            'required': ['prompt'], 'additionalProperties': False,
        }),)

    def execute(self, call):
        # Each Lab receives its native operation name, without "checker.".
        if (call.name != 'evaluate' or set(call.arguments) != {'prompt'}
                or not isinstance(call.arguments['prompt'], str) or not call.arguments['prompt'].strip()):
            raise ValueError('Expected evaluate with a nonempty prompt')
        missing = [text for text in self.instructions if text not in call.arguments['prompt']]
        coverage = (len(self.instructions) - len(missing)) / len(self.instructions)
        return LabFeedback('Literal prompt checks completed; no model was called', (), True,
            data={'quality_metrics': {'instruction_coverage': coverage},
                  'missing_instructions': missing})
