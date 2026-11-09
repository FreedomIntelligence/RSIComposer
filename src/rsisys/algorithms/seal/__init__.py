"""seal method package and single registration entry point."""
from rsisys.controllers.seal.controller import SEALController

def register(registry):
    from rsisys.labs.seal.lab import SEALLab, validate_config
    registry.register_controller('seal', lambda context, config: SEALController(**config),
        version='1', validate_config=SEALController.validate_config)
    registry.register_lab('seal', SEALLab, version='1', validate_config=validate_config)
