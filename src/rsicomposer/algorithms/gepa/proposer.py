"""Route the paper artifact's reflection request to the decision model."""
from rsicomposer.api import GenerationRequest

from ..shared.experiment import ModelCalls


class GEPAReflector:
    def __init__(self, context, config, *, max_calls):
        self.seed = config['seed']
        self.max_output_tokens = config['parameters']['reflection_max_output_tokens']
        self.temperature = config['model']['optimizer_temperature']
        self.top_p = config['model']['optimizer_top_p']
        self.model = context.roles['decision_maker']
        self.artifacts = context.artifacts
        self.calls = ModelCalls(context, max_calls)

    def reflect(self, prompt, *, round_index):
        text = self.calls.text(GenerationRequest('decision_maker', self.model, prompt,
            temperature=self.temperature, top_p=self.top_p, max_output_tokens=self.max_output_tokens,
            seed=(self.seed + round_index) % 2**32))
        self.artifacts.put(text.encode(), kind='gepa-reflect')
        return text
