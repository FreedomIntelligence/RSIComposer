"""ft_agent method package and single registration entry point."""
from rsisys.algorithms.ft_agent.controller import FTAgentController
from rsisys.methods import MethodDefinition

def register(registry):
    from rsisys.algorithms.ft_agent.controller import validate_config
    from rsisys.algorithms.ft_agent.lab import LLMFinetuneLab, validate_config as validate_lab
    registry.register_method('ft_agent', MethodDefinition.from_factories(
        controller=FTAgentController, labs={'finetune': LLMFinetuneLab}, version='1',
        validate_controller=validate_config, validate_labs={'finetune': validate_lab}))
    # Compatibility for existing configurations that select the two sides.
    registry.register_controller('ft_agent', FTAgentController, version='1', validate_config=validate_config)
    registry.register_lab('llm_finetune', LLMFinetuneLab, version='1', validate_config=validate_lab)
