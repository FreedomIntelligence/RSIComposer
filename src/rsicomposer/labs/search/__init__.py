"""Retrieval and evidence benchmarks."""


def register_components(registry):
    from .lab import SearchLab
    registry.register_lab('search', SearchLab, version='1', validate_config=SearchLab.validate_config)
