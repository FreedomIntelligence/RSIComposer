"""rrsi method package and single registration entry point."""
from rsicomposer.algorithms.rrsi.controller import RRSIController, build_rrsi
from rsicomposer.api import MethodDefinition


def register(registry):
    from rsicomposer.algorithms.shared.harness_lab import FixedModelHarnessLab, validate_config
    registry.register_method('rrsi', MethodDefinition.from_factories(
        controller=build_rrsi, labs={'harness': FixedModelHarnessLab}, version='2',
        validate_controller=RRSIController.validate_config,
        validate_labs={'harness': validate_config}))
