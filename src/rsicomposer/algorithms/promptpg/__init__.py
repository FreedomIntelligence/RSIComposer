"""Register the PromptPG selector and its TabMWP Lab."""

from rsicomposer.api import MethodDefinition

from .controller import PromptPGController
from .lab import PromptPGLab


def register(registry):
    registry.register_method('promptpg', MethodDefinition.from_factories(
        controller=PromptPGController, labs={'tabmwp': PromptPGLab}, version='1',
    ))
