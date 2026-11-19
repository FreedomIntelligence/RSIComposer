"""RSISys: pluggable RSI controllers and Labs. External integrations use rsisys.api."""

from rsisys.runtime.observation.client import log  # noqa: E402  rsisys.log({...}, step=...) from any process

__all__ = ['log']
