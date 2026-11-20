"""Math scene. Answer verifiers are shared components in rsisys.labs.verifiers."""


def register_components(registry):
    from .lab import MathLab
    registry.register_lab('math', MathLab, version='1', validate_config=MathLab.validate_config)
