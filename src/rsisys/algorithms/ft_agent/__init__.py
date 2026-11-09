"""ft_agent method package and single registration entry point."""
from .controller import FTAgentController

def register(registry):
    from .controller import validate_config
    from .lab import LLMFinetuneLab, validate_config as validate_lab
    registry.register_controller('ft_agent', FTAgentController, version='1', validate_config=validate_config)
    registry.register_lab('llm_finetune', LLMFinetuneLab, version='1', validate_config=validate_lab)
