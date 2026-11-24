"""ttrl method package and single registration entry point."""
from rsicomposer.algorithms.ttrl.controller import TTRLController, TTRLVerlController
from rsicomposer.api import MethodDefinition, ObservationProbe

def register(registry):
    from rsicomposer.algorithms.ttrl.lab import TTRLLab, validate_config
    from rsicomposer.algorithms.ttrl.verl import TTRLVerlLab, validate_config as validate_verl
    reference = TTRLController.from_context
    native = TTRLVerlController.from_context
    registry.register_method('ttrl', MethodDefinition.from_factories(
        controller=reference, labs={'ttrl': TTRLLab}, version='2',
        validate_controller=TTRLController.validate_config,
        validate_labs={'ttrl': validate_config}))
    registry.register_method('ttrl_verl', MethodDefinition.from_factories(
        controller=native, labs={'ttrl': TTRLVerlLab}, version='3',
        observation_probes=(
            ObservationProbe('ttrl_verl.checkpoint', 'execution', (('step', 'number'), ('sequence', 'number')), 'table'),
            ObservationProbe('ttrl_verl.control', 'decision', (('action', 'id'), ('step', 'number'))),
        ),
        validate_controller=TTRLVerlController.validate_config,
        validate_labs={'ttrl': validate_verl}))
