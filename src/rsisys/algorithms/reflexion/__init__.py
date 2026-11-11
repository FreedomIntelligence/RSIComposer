"""Register Reflexion's retry policy and isolated HumanEval operations."""

from rsisys.api import MethodDefinition

from .controller import ReflexionController
from .lab import HumanEvalLab


def register(registry):
    registry.register_method('reflexion', MethodDefinition.from_factories(
        controller=ReflexionController, labs={'reflexion': HumanEvalLab}, version='1',
    ))
