"""OPRO method registration."""

from rsisys.api import MethodDefinition


def register(registry):
    from .controller import OPROController
    from .lab import OPROGSM8KLab

    registry.register_method('opro', MethodDefinition.from_factories(
        controller=OPROController, labs={'opro': OPROGSM8KLab}, version='2',
    ))
