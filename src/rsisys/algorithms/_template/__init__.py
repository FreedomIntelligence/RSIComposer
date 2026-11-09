"""Copyable method template; explicitly loaded, excluded from built-in discovery.

Copy this directory to algorithms/my_method (without a leading underscore) for
built-in discovery, or anywhere outside the library and load it with --plugin.
Change the two registration names and the matching selections in configs/run.json.
"""
from .controller import PromptController, validate_config
from .lab import PromptCheckLab


def register(registry):
    # One entry point owns both sides of the method. No core registry edits.
    registry.register_controller('custom_prompt', PromptController, version='1',
                                 validate_config=validate_config)
    registry.register_lab('custom_prompt_check', PromptCheckLab, version='1')
