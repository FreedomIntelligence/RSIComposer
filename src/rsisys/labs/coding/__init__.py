"""Reusable containerized coding task environment, independent of RSI algorithms."""


def register_components(registry):
    from .lab import CodingLab
    registry.register_lab('coding', CodingLab, version='1', validate_config=CodingLab.validate_config)
