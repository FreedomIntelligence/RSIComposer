"""Register a whole method and the Controller independently for Lab authors."""
from rsisys.api import MethodDefinition

from .controller import PromptSearchController, validate_config


def register(registry):
    def lab_factory(context, config):
        return registry.resolve_component('scene', 'template_math', context, config)

    registry.register_controller('template_prompt_search', PromptSearchController, version='1',
                                 validate_config=validate_config)
    registry.register_method('template_prompt_search', MethodDefinition.from_factories(
        controller=PromptSearchController,
        labs={'math': lab_factory},
        version='1',
        validate_controller=validate_config,
    ))
