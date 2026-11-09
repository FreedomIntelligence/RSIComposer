"""aide method package and single registration entry point."""
from rsisys.controllers.aide.controller import AIDEController

def register(registry):
    from rsisys.controllers.aide.controller import build_aide, validate_config
    from rsisys.labs.aide.lab import AIDEProgramLab, validate_config as validate_lab
    registry.register_controller('aide', build_aide, version='1', validate_config=validate_config)
    registry.register_lab('aide_program', AIDEProgramLab, version='1', validate_config=validate_lab)
