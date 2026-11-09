"""Fixed-iteration controller template with its own registration entry."""
from rsisys.algorithms.single_loop.controller import SingleLoopController


def register(registry):
    registry.register_controller('single_loop', lambda context, config: SingleLoopController(**config),
                                 version='1', validate_config=SingleLoopController.validate_config)
