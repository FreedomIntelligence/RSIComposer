"""Math scene and its reusable answer verifiers."""


def _verifier(kind, config):
    if config:
        raise ValueError('Math verifiers accept no configuration options')
    from .verifiers import MathVerifyVerifier, StrictNumericVerifier
    return StrictNumericVerifier() if kind == 'strict_numeric' else MathVerifyVerifier()


def register_components(registry):
    """Optional symbolic dependencies load only when the verifier is resolved."""
    from .lab import MathLab
    registry.register_lab('math', MathLab, version='0-placeholder', validate_config=MathLab.validate_config)
    registry.register_component('verifier', 'strict_numeric',
                                lambda context, config: _verifier('strict_numeric', config), version='1')
    registry.register_component('verifier', 'math_verify',
                                lambda context, config: _verifier('math_verify', config), version='1')
