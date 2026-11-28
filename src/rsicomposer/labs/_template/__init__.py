"""Register one public Lab for explicit pairing and reuse inside methods."""
from .lab import ArithmeticLab, validate_config


def register(registry):
    registry.register_lab('template_math', ArithmeticLab, version='1',
                          validate_config=validate_config)
