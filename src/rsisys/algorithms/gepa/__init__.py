"""GEPA method registration."""

from rsisys.api import MethodDefinition


def register(registry):
    from .controller import GEPAController
    from .lab import GEPAGSM8KLab
    from .merge_controller import GEPAMergeController
    from .merge_lab import GEPAMergeLab

    registry.register_method('gepa', MethodDefinition.from_factories(
        controller=GEPAController, labs={'gepa': GEPAGSM8KLab}, version='2',
    ))
    registry.register_method('gepa_merge', MethodDefinition.from_factories(
        controller=GEPAMergeController, labs={'gepa': GEPAMergeLab}, version='1',
    ))
