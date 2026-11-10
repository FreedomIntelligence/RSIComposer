"""seal method package and single registration entry point."""
from rsisys.algorithms.seal.controller import SEALController
from rsisys.methods import MethodDefinition

def register(registry):
    from rsisys.algorithms.seal.lab import SEALLab, validate_config
    factory = lambda context, config: SEALController(**config)
    registry.register_method('seal', MethodDefinition.from_factories(
        controller=factory, labs={'seal': SEALLab}, version='1',
        validate_controller=SEALController.validate_config,
        validate_labs={'seal': validate_config}))
