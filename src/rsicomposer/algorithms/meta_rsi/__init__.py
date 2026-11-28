"""MetaRSI-derived reference implementation on the public framework ports."""
from rsicomposer.api import ControllerDefinition, MethodDefinition

from .controller import MetaRSIController, NAME, validate_config
from .harness import SchedulerHarness
from .lab import MetaRSILab, validate_config as validate_lab
from .model_controller import ModelMetaRSIController, NAME as MODEL_NAME, validate_model_config


VERSION = '2'


def register(registry):
    controller = ControllerDefinition(MetaRSIController, harness=SchedulerHarness)
    registry.register_controller(NAME, controller, version=VERSION, validate_config=validate_config)
    registry.register_lab(NAME, MetaRSILab, version=VERSION, validate_config=validate_lab)
    labs = {'math': registry.lab(NAME), 'reflect': registry.lab('self_reflection')}
    registry.register_method(NAME, MethodDefinition.from_factories(
        controller=controller, labs=labs, version=VERSION, validate_controller=validate_config))

    # Use the stock Harness so the Host can adopt scheduler revisions between terms.
    model_controller = ControllerDefinition(ModelMetaRSIController)
    registry.register_controller(MODEL_NAME, model_controller, version=VERSION, validate_config=validate_model_config)
    registry.register_method(MODEL_NAME, MethodDefinition.from_factories(
        controller=model_controller, labs=labs, version=VERSION, validate_controller=validate_model_config))
