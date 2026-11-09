"""rest_em method package and single registration entry point."""
from rsisys.algorithms.rest_em.controller import RestEMController
from rsisys.methods import MethodDefinition

def register(registry):
    from rsisys.labs.scenes.math import MathLab, validate_config
    factory = lambda context, config: RestEMController(**config)
    registry.register_method('rest_em', MethodDefinition.from_factories(
        controller=factory, labs={'math': MathLab}, version='1',
        validate_controller=RestEMController.validate_config,
        validate_labs={'math': validate_config}))
    # Compatibility for existing configurations that select the two sides.
    registry.register_controller('rest_em', factory,
        version='1', validate_config=RestEMController.validate_config)
    registry.register_lab('math', MathLab, version='1', validate_config=validate_config)
