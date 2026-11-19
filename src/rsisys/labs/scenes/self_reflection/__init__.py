"""Opt-in Lab for scoped evidence, immutable candidates and nested validation."""
from .lab import SelfReflectionLab, VERSION, validate_config


def register(registry):
    registry.register_lab('self_reflection', SelfReflectionLab,
                          version=VERSION, validate_config=validate_config)


__all__ = ['SelfReflectionLab', 'register']
