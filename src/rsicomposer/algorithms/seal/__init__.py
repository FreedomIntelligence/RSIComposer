"""seal method package and single registration entry point."""
from rsicomposer.algorithms.seal.controller import SEALController
from rsicomposer.api import MethodDefinition

def register(registry):
    from rsicomposer.algorithms.seal.lab import SEALLab, validate_config
    factory = SEALController.from_context
    registry.register_method('seal', MethodDefinition.from_factories(
        controller=factory, labs={'seal': SEALLab}, version='2',
        validate_controller=SEALController.validate_config,
        validate_labs={'seal': validate_config}))
