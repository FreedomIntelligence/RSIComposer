"""MetaRSI-derived reference implementation on the public framework ports."""
from dataclasses import replace

from rsicomposer.api import ControllerDefinition, MethodDefinition
from rsicomposer.labs.self_reflection.lab import SelfReflectionLab, validate_config as validate_reflection

from .controller import MetaRSIController, NAME, validate_config
from .harness import SchedulerHarness
from .lab import MetaRSILab, validate_config as validate_lab, validate_trial_cohorts
from .model_controller import ModelMetaRSIController, NAME as MODEL_NAME, validate_model_config


VERSION = '1'


def register(registry):
    controller = ControllerDefinition(MetaRSIController, harness=SchedulerHarness)
    registry.register_controller(NAME, controller, version=VERSION, validate_config=validate_config)
    registry.register_lab(NAME, MetaRSILab, version=VERSION, validate_config=validate_lab)
    method = MethodDefinition.from_factories(
        controller=controller, labs={'math': MetaRSILab, 'reflect': SelfReflectionLab}, version=VERSION,
        validate_controller=validate_config, validate_labs={'math': validate_lab, 'reflect': validate_reflection})

    def build(config):
        modules = method.factory(config)
        policy = config.get('controller', {})
        if not policy.get('meta_targets'):
            return modules
        parent = {key: value for key, value in config['labs']['math'].items() if key != 'max_concurrent'}

        def checked_lab(context):
            validate_trial_cohorts(parent, policy['trial_lab'])
            return modules.lab_factory(context)

        return replace(modules, lab_factory=checked_lab)

    registry.register_method(NAME, replace(method, factory=build))

    # Use the stock Harness so the Host can adopt scheduler revisions between terms.
    model_controller = ControllerDefinition(ModelMetaRSIController)
    registry.register_controller(MODEL_NAME, model_controller, version=VERSION, validate_config=validate_model_config)
    model_method = MethodDefinition.from_factories(
        controller=model_controller, labs={'math': MetaRSILab, 'reflect': SelfReflectionLab}, version=VERSION,
        validate_controller=validate_model_config, validate_labs={'math': validate_lab, 'reflect': validate_reflection})

    def build_model(config):
        lab = config['labs']['math']
        if lab.get('model_driven') is not True:
            raise ValueError('MetaRSI model controller needs the explicit model-driven Lab')
        if ('paper_kernel' in lab) != config['controller'].get('paper_kernel', False):
            raise ValueError('Controller and Lab must use the same paper kernel')
        if 'paper_kernel' in lab:
            from .aime_release import validate_release_options
            if validate_release_options(lab['paper_kernel'])['min_gain'] < config['controller'].get('margin', 0.):
                raise ValueError('Fixed paired release cannot be weaker than the controller margin')
        return model_method.factory(config)

    registry.register_method(MODEL_NAME, replace(model_method, factory=build_model))
