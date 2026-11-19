"""SEAL reference and Trainer backends, with explicit paper-profile defaults."""
import math

from rsisys.api import WeightEngine, validate_engine

PAPER_DEFAULTS = dict(split_newlines=True, adaptation_seeds=3, evaluation_seeds=1,
    inner_max_length=2048, inner_rank=32, inner_epochs=10, inner_lr=1e-3,
    lora_rank=64, outer_epochs=2, outer_lr=3e-4, batch_size=1,
    outer_gradient_accumulation_steps=5,
    temperature=1.0, max_tokens=8192, max_length=16384)


def validate_reference_engine(config):
    validate_engine(config)
    for key, default in (('inner_rank', 32), ('inner_epochs', 10)):
        if type(config.get(key, default)) is not int or config.get(key, default) < 1:
            raise ValueError(f'{key} must be a positive integer')
    value = config.get('inner_lr', 1e-3)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('inner_lr must be finite and positive')


class SEALReferenceEngine(WeightEngine):
    operation_workers = {**WeightEngine.operation_workers,
                         'adapt': 'rsisys.algorithms.seal.reference_worker'}

    def __init__(self, config):
        validate_reference_engine(config)
        super().__init__(config)

    def adapt(self, model, episodes, directory, *, seed_offset=0):
        return self._run('adapt', model, episodes, directory, seed_offset=seed_offset)


def validate_paper_engine(config):
    merged = {**PAPER_DEFAULTS, **config}
    validate_reference_engine(merged)

class SEALTrainerEngine(SEALReferenceEngine):
    name = 'seal-transformers-trainer-trl-sft-v1'
    worker_module = 'rsisys.algorithms.seal.worker'

    operation_workers = WeightEngine.operation_workers

    def _env(self):
        return dict(super()._env(), VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_USE_V1='1')

    def resource_roles(self):
        from rsisys.api import ResourceRequest
        # Published outer batch: two GPUs × microbatch one × accumulation five.
        return {**super().resource_roles(), 'trainer': ResourceRequest(gpus=2)}
