"""Concrete training and serving backends; optional dependencies stay lazy."""


def _chat_pool(context, config):
    from .serving.chat import ChatPool
    return ChatPool(**config)


def _lora(context, config):
    from .training.lora.engine import LoRAEngine
    return LoRAEngine(config)


def _trl_sft(context, config):
    from .training.lora.trl_sft import TRLSFTBackend
    return TRLSFTBackend(config)


def _verl_sft(context, config):
    from .training.lora.verl_sft import VeRLSFTBackend
    return VeRLSFTBackend(config)


def _verl_grpo(context, config):
    from .training.verl_grpo import VeRLGRPOBackend
    return VeRLGRPOBackend(config)


def register_components(registry):
    """Expose shared engines without importing or initializing ML runtimes."""
    registry.register_component('backend', 'chat_pool', _chat_pool, version='1')
    registry.register_component('backend', 'lora', _lora, version='1')
    registry.register_component('backend', 'trl_sft', _trl_sft, version='1')
    registry.register_component('backend', 'verl_sft', _verl_sft, version='1')
    registry.register_component('backend', 'verl_grpo', _verl_grpo, version='1')
