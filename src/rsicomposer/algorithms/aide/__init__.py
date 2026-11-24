"""aide method package and single registration entry point."""
from rsicomposer.algorithms.aide.controller import AIDEController as AIDEController
from rsicomposer.api import MethodDefinition, ObservationProbe


def register(registry):
    from rsicomposer.algorithms.aide.controller import build_aide, validate_config
    from rsicomposer.algorithms.aide.lab import AIDEProgramLab
    from rsicomposer.algorithms.aide.lab import validate_config as validate_lab
    registry.register_method('aide', MethodDefinition.from_factories(
        controller=build_aide, labs={'aide': AIDEProgramLab}, version='3',
        observation_probes=(
            ObservationProbe('aide.node', 'decision', (('step', 'number'), ('stage', 'id'), ('score', 'number')), 'lineage'),
            ObservationProbe('aide.selection', 'decision', (('step', 'number'), ('score', 'number'))),
        ),
        validate_controller=validate_config, validate_labs={'aide': validate_lab}))
