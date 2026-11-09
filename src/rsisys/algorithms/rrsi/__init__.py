"""rrsi method package and single registration entry point."""
from .controller import RRSIController, build_rrsi

def register(registry):
    registry.register_controller('rrsi', build_rrsi, version='1', validate_config=RRSIController.validate_config)
    from rsisys.labs.harness_math import HarnessMathLab, validate_config
    registry.register_lab('harness_math', HarnessMathLab, version='1', validate_config=validate_config)
