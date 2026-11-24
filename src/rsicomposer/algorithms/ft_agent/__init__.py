"""ft_agent method package and single registration entry point."""
from rsicomposer.algorithms.ft_agent.controller import FTAgentController
from rsicomposer.api import MethodDefinition, ObservationProbe

def register(registry):
    from rsicomposer.algorithms.ft_agent.controller import validate_config
    from rsicomposer.algorithms.ft_agent.lab import LLMFinetuneLab, validate_config as validate_lab
    registry.register_method('ft_agent', MethodDefinition.from_factories(
        controller=FTAgentController, labs={'finetune': LLMFinetuneLab}, version='2',
        observation_probes=(ObservationProbe('ft_agent.recipe', 'decision',
            (('iteration', 'number'), ('stage', 'id'), ('ok', 'bool')), 'table'),),
        validate_controller=validate_config, validate_labs={'finetune': validate_lab}))
