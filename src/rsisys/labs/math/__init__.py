"""Math Lab: task splits, answer verifiers and the official competition bundle."""


def register_components(registry):
    from .lab import MathLab
    registry.register_lab('math', MathLab, version='1', validate_config=MathLab.validate_config)
