"""Register ReST^EM; its Lab is a MathLab subclass."""
from rsisys.algorithms.rest_em.controller import RestEMController
from rsisys.api import MethodDefinition


def register(registry):
    from rsisys.algorithms.rest_em.lab import RestEMLab, validate_config
    factory = lambda context, config: RestEMController(**config)
    def build_lab(context, config):
        return RestEMLab(context, config, resolve_trainer=lambda name, options:
            registry.resolve_component('backend', name, context, options))
    registry.register_method('rest_em', MethodDefinition.from_factories(
        controller=factory, labs={'math': build_lab}, version='2',
        validate_controller=RestEMController.validate_config,
        validate_labs={'math': validate_config}))
