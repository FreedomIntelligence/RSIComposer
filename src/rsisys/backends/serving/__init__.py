"""Central selection of serving adapters; optional engines run out of process."""

from importlib import import_module

from .chat_completions import MockModelBackend, RemoteModelBackend


_NATIVE_ADAPTERS = {
    'vllm': ('.vllm', 'VLLMBackend', False),
    'sglang': ('.sglang', 'SGLangBackend', True),
}


def is_native_serving_kind(kind):
    return kind in _NATIVE_ADAPTERS


def is_serving_kind(kind):
    return kind in ('chat_completions', 'http', 'mock', 'openai_chat_completions', 'anthropic_messages') or is_native_serving_kind(kind)


def build_serving_backend(binding, *, models=(), native=False, gateway=False, **options):
    """Bind an adapter to the supplied model pins without contacting the service.

    The host gateway has its own wire protocol and forwards requests to the
    upstream adapter; a nested run never opens the original engine endpoint.
    """
    kind = binding.kind
    adapter = _NATIVE_ADAPTERS.get(kind)
    if binding.engine_version is not None and not (adapter and adapter[2]):
        raise ValueError(f'{kind} does not support engine_version')
    if gateway or (native and kind in ('chat_completions', 'http')):
        adapter = _NATIVE_ADAPTERS['vllm']
    if adapter is not None:
        module, name, _ = adapter
        factory = getattr(import_module(module, __package__), name)
        return factory(binding, models=dict(models), **options)
    if kind == 'mock':
        return MockModelBackend()
    if kind == 'anthropic_messages':
        from .anthropic import AnthropicMessagesBackend
        return AnthropicMessagesBackend(binding)
    return RemoteModelBackend(binding)
