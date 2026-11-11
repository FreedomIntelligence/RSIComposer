"""GEPA method registration."""

from rsisys.api import MethodDefinition


def register(registry):
    from .controller import GEPAController
    from .lab import GEPAGSM8KLab

    registry.register_method('gepa', MethodDefinition.from_factories(
        controller=GEPAController, labs={'gepa': GEPAGSM8KLab}, version='1',
    ))
