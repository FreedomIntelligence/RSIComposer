"""ttrl method package and single registration entry point."""
from rsisys.algorithms.ttrl.controller import TTRLController, TTRLVerlController

def register(registry):
    from rsisys.algorithms.ttrl.lab import TTRLLab, validate_config
    from rsisys.algorithms.ttrl.verl import TTRLVerlLab, validate_config as validate_verl
    registry.register_controller('ttrl', lambda context, config: TTRLController(**config),
        version='1', validate_config=TTRLController.validate_config)
    registry.register_lab('ttrl', TTRLLab, version='1', validate_config=validate_config)
    registry.register_controller('ttrl_verl', lambda context, config: TTRLVerlController(**config),
        version='1', validate_config=TTRLVerlController.validate_config)
    registry.register_lab('ttrl_verl', TTRLVerlLab, version='1', validate_config=validate_verl)
