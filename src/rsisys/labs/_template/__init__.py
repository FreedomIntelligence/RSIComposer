"""Register the same Lab for explicit assembly and as a reusable scene factory."""
from .lab import ArithmeticLab, validate_config


def register(registry):
    registry.register_lab('template_math', ArithmeticLab, version='1',
                          validate_config=validate_config)
    registry.register_component('scene', 'template_math', ArithmeticLab, version='1',
                                validate_config=validate_config)
