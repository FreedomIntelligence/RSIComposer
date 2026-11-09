"""rrsi method package and single registration entry point."""
from rsisys.algorithms.rrsi.controller import RRSIController, build_rrsi
from rsisys.methods import MethodDefinition

def register(registry):
    from rsisys.labs.scenes.math.harness import HarnessMathLab, validate_config
    registry.register_method('rrsi', MethodDefinition.from_factories(
        controller=build_rrsi, labs={'harness': HarnessMathLab}, version='1',
        validate_controller=RRSIController.validate_config,
        validate_labs={'harness': validate_config}))
    # Compatibility for existing configurations that select the two sides.
    registry.register_controller('rrsi', build_rrsi, version='1', validate_config=RRSIController.validate_config)
    registry.register_lab('harness_math', HarnessMathLab, version='1', validate_config=validate_config)
