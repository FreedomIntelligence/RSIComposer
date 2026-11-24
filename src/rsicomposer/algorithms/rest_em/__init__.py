"""Register ReST^EM; its Lab is a MathLab subclass."""
from rsicomposer.algorithms.rest_em.controller import RestEMController
from rsicomposer.api import MethodDefinition
from ..shared.weight_observation import weight_probes


def register(registry):
    from rsicomposer.algorithms.rest_em.lab import RestEMLab, validate_config
    factory = lambda context, config: RestEMController(**config)
    def build_lab(context, config):
        return RestEMLab(context, config, resolve_trainer=lambda name, options:
            registry.resolve_component('backend', name, context, options))
    registry.register_method('rest_em', MethodDefinition.from_factories(
        controller=factory, labs={'math': build_lab}, version='2',
        observation_probes=weight_probes('rest_em'),
        validate_controller=RestEMController.validate_config,
        validate_labs={'math': validate_config}))
