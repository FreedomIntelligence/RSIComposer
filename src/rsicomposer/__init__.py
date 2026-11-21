"""RSIComposer: pluggable RSI controllers and Labs. External integrations use rsicomposer.api."""

from rsicomposer.runtime.observation.client import log  # noqa: E402  rsicomposer.log({...}, step=...) from any process

__all__ = ['log']
