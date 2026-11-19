"""seal method package and single registration entry point."""
from rsisys.algorithms.seal.controller import SEALController
from rsisys.api import MethodDefinition

def register(registry):
    from rsisys.algorithms.seal.lab import SEALLab, validate_config
    factory = SEALController.from_context
    registry.register_method('seal', MethodDefinition.from_factories(
        controller=factory, labs={'seal': SEALLab}, version='2',
        validate_controller=SEALController.validate_config,
        validate_labs={'seal': validate_config}))
