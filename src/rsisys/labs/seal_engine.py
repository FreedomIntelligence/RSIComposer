"""SEAL Trainer/SFTTrainer backend, with explicit paper-profile defaults."""
from .weight_engine import WeightEngine, validate_engine

PAPER_DEFAULTS = dict(split_newlines=True, adaptation_seeds=3, evaluation_seeds=1,
    inner_max_length=2048, inner_rank=32, inner_epochs=10, inner_lr=1e-3,
    lora_rank=64, outer_epochs=2, outer_lr=3e-4, batch_size=1,
    outer_gpu_ids=[0, 1], outer_gradient_accumulation_steps=5,
    temperature=1.0, max_tokens=8192, max_length=16384)


def validate_paper_engine(config):
    merged = {**PAPER_DEFAULTS, **config}
    ids = merged['outer_gpu_ids']
    validate_engine({**merged, 'gpu_ids': ids})
    if not set(ids) <= set(merged.get('gpu_ids', [0])):
        raise ValueError('outer_gpu_ids must be a subset of gpu_ids')


class SEALTrainerEngine(WeightEngine):
    name = 'seal-transformers-trainer-trl-sft-v1'
    worker_module = 'rsisys.labs.seal_worker'

    def _env(self):
        return dict(super()._env(), VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_USE_V1='1')

    def train(self, model, rows, directory, checkpoint, **kwargs):
        # The published outer recipe uses 2 GPUs × 1 × accumulation 5 = 10.
        # All configured GPUs remain available for independent inner episodes.
        engine = SEALTrainerEngine({**self.config, 'gpu_ids': self.config['outer_gpu_ids']})
        return WeightEngine.train(engine, model, rows, directory, checkpoint, **kwargs)
