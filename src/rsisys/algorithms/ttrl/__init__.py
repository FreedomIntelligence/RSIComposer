"""ttrl method package and single registration entry point."""
from rsisys.algorithms.ttrl.controller import TTRLController, TTRLVerlController
from rsisys.methods import MethodDefinition

def register(registry):
    from rsisys.algorithms.ttrl.lab import TTRLLab, validate_config
    from rsisys.algorithms.ttrl.verl import TTRLVerlLab, validate_config as validate_verl
    reference = lambda context, config: TTRLController(**config)
    native = lambda context, config: TTRLVerlController(**config)
    registry.register_method('ttrl', MethodDefinition.from_factories(
        controller=reference, labs={'ttrl': TTRLLab}, version='1',
        validate_controller=TTRLController.validate_config,
        validate_labs={'ttrl': validate_config}))
    registry.register_method('ttrl_verl', MethodDefinition.from_factories(
        controller=native, labs={'ttrl': TTRLVerlLab}, version='1',
        validate_controller=TTRLVerlController.validate_config,
        validate_labs={'ttrl': validate_verl}))
    # Compatibility for existing configurations that select the two sides.
    registry.register_controller('ttrl', reference,
        version='1', validate_config=TTRLController.validate_config)
    registry.register_lab('ttrl', TTRLLab, version='1', validate_config=validate_config)
    registry.register_controller('ttrl_verl', native,
        version='1', validate_config=TTRLVerlController.validate_config)
    registry.register_lab('ttrl_verl', TTRLVerlLab, version='1', validate_config=validate_verl)
