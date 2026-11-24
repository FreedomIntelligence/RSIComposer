"""Register STaR; its Lab is a MathLab subclass."""

from rsicomposer.api import MethodDefinition

from .controller import StarController
from .lab import StarLab
from ..shared.weight_observation import weight_probes


def register(registry):
    registry.register_method('star', MethodDefinition.from_factories(
        controller=StarController, labs={'star': StarLab}, version='2',
        observation_probes=weight_probes('star'),
    ))
