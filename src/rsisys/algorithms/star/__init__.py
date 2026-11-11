"""Register STaR's round policy and the local GSM8K Lab."""

from rsisys.api import MethodDefinition

from .controller import StarController
from .lab import StarLab


def register(registry):
    registry.register_method('star', MethodDefinition.from_factories(
        controller=StarController, labs={'star': StarLab}, version='1',
    ))
