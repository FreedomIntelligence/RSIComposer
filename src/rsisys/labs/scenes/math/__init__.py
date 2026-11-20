"""Math scene: EvaluationLab defaults for math plus the official AIME/AIMO3 bundle."""


def register_components(registry):
    from .lab import MathLab
    registry.register_lab('math', MathLab, version='1', validate_config=MathLab.validate_config)
