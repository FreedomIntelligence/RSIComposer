"""rest_em method package and single registration entry point."""
from rsisys.algorithms.rest_em.controller import RestEMController
from rsisys.api import MethodDefinition


def register(registry):
    from rsisys.algorithms.rest_em.lab import RestEMLab, validate_config
    factory = lambda context, config: RestEMController(**config)
    registry.register_controller('rest_em', factory, version='2',
                                 validate_config=RestEMController.validate_config)
    registry.register_lab('rest_em', RestEMLab, version='2', validate_config=validate_config)
    registry.register_method('rest_em', MethodDefinition.from_factories(
        controller=factory, labs={'math': RestEMLab}, version='2',
        validate_controller=RestEMController.validate_config,
        validate_labs={'math': validate_config}))
