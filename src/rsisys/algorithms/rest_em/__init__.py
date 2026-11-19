"""Register ReST^EM. Its Lab is method-private until the shared math Lab is implemented."""
from rsisys.algorithms.rest_em.controller import RestEMController
from rsisys.api import MethodDefinition


def register(registry):
    from rsisys.algorithms.rest_em.lab import RestEMLab, validate_config
    factory = lambda context, config: RestEMController(**config)
    registry.register_method('rest_em', MethodDefinition.from_factories(
        controller=factory, labs={'math': RestEMLab}, version='2',
        validate_controller=RestEMController.validate_config,
        validate_labs={'math': validate_config}))
