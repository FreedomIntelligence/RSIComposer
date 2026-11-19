"""Copyable method template; explicitly loaded, excluded from built-in discovery.

Copy this directory to algorithms/my_method (without a leading underscore) for
built-in discovery, or anywhere outside the library and load it with --plugin.
Change the method name and the matching selection in configs/run.json.
"""
from .controller import PromptController, validate_config
from .lab import PromptCheckLab, validate_config as validate_lab
from rsisys.api import MethodDefinition


def register(registry):
    # Private factories close the loop; they need no separate registration.
    registry.register_method('custom_prompt', MethodDefinition.from_factories(
        controller=PromptController,
        labs={'checker': PromptCheckLab},
        version='1',
        validate_controller=validate_config,
        validate_labs={'checker': validate_lab},
    ))
