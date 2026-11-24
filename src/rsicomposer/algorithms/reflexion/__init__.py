"""Register Reflexion's retry policy and isolated HumanEval operations."""

from rsicomposer.api import MethodDefinition, ObservationProbe

from .controller import ReflexionController
from .lab import HumanEvalLab


def register(registry):
    registry.register_method('reflexion', MethodDefinition.from_factories(
        controller=ReflexionController, labs={'reflexion': HumanEvalLab}, version='1',
        observation_probes=(
            ObservationProbe('reflexion.action', 'decision', (('attempt', 'number'), ('operation', 'id'))),
            ObservationProbe('reflexion.test', 'quality', (('attempt', 'number'), ('suite', 'id'), ('passed', 'bool')), 'table'),
            ObservationProbe('reflexion.summary', 'quality', (('correct', 'number'), ('total', 'number')), 'table'),
        ),
    ))
