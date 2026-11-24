"""GEPA method registration."""

from rsicomposer.api import MethodDefinition, ObservationProbe


def register(registry):
    from .controller import GEPAController
    from .lab import GEPAMathLab

    registry.register_method('gepa', MethodDefinition.from_factories(
        controller=GEPAController, labs={'gepa': GEPAMathLab}, version='6',
        observation_probes=(
            ObservationProbe('gepa.prompt', 'decision', (('component', 'id'), ('baseline', 'bool')), 'diff', content=True),
            ObservationProbe('gepa.proposal', 'decision', (('round', 'number'), ('merge', 'bool')), 'lineage'),
            ObservationProbe('gepa.population', 'decision', (('index', 'number'),), 'lineage'),
            ObservationProbe('gepa.screening', 'decision', (
                ('round', 'number'), ('status', 'id'), ('strategy', 'id'), ('split', 'id'),
                ('batch_size', 'number'), ('usable_feedback', 'number'), ('parent_correct', 'number'),
                ('second_parent_correct', 'number'), ('child_correct', 'number'), ('pool_index', 'number'),
                ('validation_correct', 'number'), ('validation_size', 'number'),
            ), 'table'),
        ),
    ))
