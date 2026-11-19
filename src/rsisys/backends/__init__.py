"""Concrete training and serving backends; optional dependencies stay lazy."""


def _chat_pool(context, config):
    from .serving.chat import ChatPool
    return ChatPool(**config)


def _lora(context, config):
    from .training.lora.engine import LoRAEngine
    return LoRAEngine(config)


def register_components(registry):
    """Expose shared engines without importing or initializing ML runtimes."""
    registry.register_component('backend', 'chat_pool', _chat_pool, version='1')
    registry.register_component('backend', 'lora', _lora, version='1')
