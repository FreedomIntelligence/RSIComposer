"""Reusable Self-Reflection Lab architecture; opt-in, non-executing scaffold.

Methods may call register(registry) inside their own single registration entry.
The package can also be explicitly loaded with --plugin for catalog inspection.
It is not an automatically registered research algorithm or a working backend.
"""
from .contracts import (
    Candidate, MetaAgentVersion, ResearchEpisode, SelfReflectionBackend,
    ValidationEvidence, ValidationRequest,
)
from .lab import SelfReflectionLab, VERSION, validate_config


def register(registry):
    registry.register_lab('self_reflection', SelfReflectionLab,
                          version=VERSION, validate_config=validate_config)


__all__ = ['SelfReflectionLab', 'SelfReflectionBackend', 'MetaAgentVersion',
           'ResearchEpisode', 'Candidate', 'ValidationRequest',
           'ValidationEvidence', 'register']
