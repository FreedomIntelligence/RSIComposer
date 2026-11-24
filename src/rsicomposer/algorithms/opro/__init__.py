"""OPRO method registration."""

from rsicomposer.api import MethodDefinition, ObservationProbe


def register(registry):
    from .controller import OPROController
    from .lab import OPROGSM8KLab

    registry.register_method('opro', MethodDefinition.from_factories(
        controller=OPROController, labs={'opro': OPROGSM8KLab}, version='2',
        observation_probes=(ObservationProbe('opro.proposal', 'decision', (('round', 'number'),), 'diff', content=True),),
    ))
