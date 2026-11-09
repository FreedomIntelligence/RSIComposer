"""rest_em method package and single registration entry point."""
from rsisys.algorithms.rest_em.controller import RestEMController

def register(registry):
    registry.register_controller('rest_em', lambda context, config: RestEMController(**config),
        version='1', validate_config=RestEMController.validate_config)
    from rsisys.labs.math import MathLab, validate_config
    registry.register_lab('math', MathLab, version='1', validate_config=validate_config)
