"""aide method package and single registration entry point."""
from rsisys.algorithms.aide.controller import AIDEController as AIDEController
from rsisys.api import MethodDefinition


def register(registry):
    from rsisys.algorithms.aide.controller import build_aide, validate_config
    from rsisys.algorithms.aide.lab import AIDEProgramLab
    from rsisys.algorithms.aide.lab import validate_config as validate_lab
    registry.register_method('aide', MethodDefinition.from_factories(
        controller=build_aide, labs={'aide': AIDEProgramLab}, version='3',
        validate_controller=validate_config, validate_labs={'aide': validate_lab}))
