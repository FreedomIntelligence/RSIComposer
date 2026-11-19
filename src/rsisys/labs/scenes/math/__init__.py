"""Math scene. Answer verifiers live in verifiers.py; a Lab imports the one matching its answer format."""


def register_components(registry):
    from .lab import MathLab
    registry.register_lab('math', MathLab, version='0-placeholder', validate_config=MathLab.validate_config)
