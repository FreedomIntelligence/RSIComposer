"""rrsi method package and single registration entry point."""
from rsisys.algorithms.rrsi.controller import RRSIController, build_rrsi
from rsisys.api import MethodDefinition


def register(registry):
    from rsisys.algorithms.shared.harness_lab import FixedModelHarnessLab, validate_config
    registry.register_method('rrsi', MethodDefinition.from_factories(
        controller=build_rrsi, labs={'harness': FixedModelHarnessLab}, version='2',
        validate_controller=RRSIController.validate_config,
        validate_labs={'harness': validate_config}))
