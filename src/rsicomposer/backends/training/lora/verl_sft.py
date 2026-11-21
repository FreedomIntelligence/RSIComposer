"""Run veRL's SFT job and export a PEFT adapter in the leased worker."""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from rsicomposer.runtime.storage.files import file_hash

from .engine import LoRAEngine
from .hosted import HostedLoRAEngine, run_request


class VeRLSFTBackend(HostedLoRAEngine):
    name = 'verl-sft-lora/v1'
    train_worker = 'rsicomposer.backends.training.lora.verl_sft'

    def __init__(self, config):
        super().__init__(config)
        pin = self.config.get('verl_version')
        if not isinstance(pin, str) or not pin.strip():
            raise ValueError('veRL SFT requires an exact verl_version')
        if self.config.get('device', 'cuda') != 'cuda':
            raise ValueError('veRL SFT requires CUDA')

    def resource_roles(self):
        return {'trainer': super().resource_roles()['trainer']}

    def bind_execution(self, context):
        super().bind_execution(context)
        if self.resources.request('trainer').gpus != 1:
            raise ValueError('veRL SFT currently requires one GPU')


class CompletionDataset:
    """veRL's dataset hook, preserving the prompt and completion boundary."""

    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        import torch

        self.samples = []
        rows = json.loads(Path(parquet_files).read_text())
        if max_samples >= 0:
            rows = rows[:max_samples]
        if tokenizer.eos_token_id is None:
            raise ValueError('Tokenizer must define EOS')
        for row in rows:
            prefix = tokenizer.encode(row['prefix'], add_special_tokens=False)
            completion = tokenizer.encode(row['completion'], add_special_tokens=False) + [tokenizer.eos_token_id]
            ids = prefix + completion
            if not prefix or len(ids) > config.max_length:
                raise ValueError('veRL SFT sequence has an empty prompt or exceeds max_length')
            self.samples.append({'input_ids': torch.tensor(ids), 'position_ids': torch.arange(len(ids)),
                                 'loss_mask': torch.tensor([0] * len(prefix) + [1] * len(completion))})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class _VeRLWorker(LoRAEngine):
    def train(self, ref, rows, output, round_number, *, steps=None):
        try:
            observed = version('verl')
        except PackageNotFoundError as error:
            raise RuntimeError('Install the pinned veRL environment on training_python') from error
        if observed != self.config['verl_version']:
            raise ValueError(f'veRL version mismatch: expected {self.config["verl_version"]}, got {observed}')

        config = self.config
        micro_batch = config.get('train_batch_size', 1)
        accumulation = config.get('gradient_accumulation_steps', 1)
        epochs = config.get('epochs', 1)
        if any(type(n) is not int or n < 1 for n in (micro_batch, accumulation, epochs)):
            raise ValueError('veRL SFT batch sizes and epochs must be positive integers')
        batch = micro_batch * accumulation
        if not rows or len(rows) % batch:
            raise ValueError('veRL SFT needs complete batches; reduce train_batch_size or gradient_accumulation_steps')
        available = len(rows) // batch * epochs
        if steps is not None and (type(steps) is not int or not 1 <= steps <= available):
            raise ValueError('veRL SFT step limit must fit the configured data and epochs')
        total_steps = available if steps is None else steps

        from transformers import AutoTokenizer

        base = Path(ref).resolve(strict=True)
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        output = output.resolve(strict=True)
        with tempfile.TemporaryDirectory(prefix='verl-', dir=output) as temporary:
            root = Path(temporary)
            data, metrics, checkpoints = (root / name for name in ('rows.json', 'metrics.jsonl', 'checkpoints'))
            data.write_text(json.dumps([{'prefix': self._prefix(tokenizer, row['prompt']),
                                        'completion': row['completion']} for row in rows]))
            settings = {
                'data.train_batch_size': batch, 'data.micro_batch_size_per_gpu': micro_batch,
                'data.use_dynamic_bsz': False, 'data.max_length': config.get('max_length', 2048),
                'data.num_workers': 0, 'model.lora_rank': 8, 'model.lora_alpha': 16,
                'model.target_modules': ['q_proj', 'v_proj'], 'model.use_remove_padding': False,
                'model.enable_gradient_checkpointing': config.get('train_gradient_checkpointing', False),
                '+model.override_config.attn_implementation': 'sdpa', 'engine.strategy': 'fsdp2',
                'engine.seed': config.get('seed', 73) + round_number,
                'optim.lr': config.get('learning_rate', 2e-4), 'trainer.total_epochs': epochs,
                'trainer.total_training_steps': total_steps, 'trainer.resume_mode': 'disable',
                'trainer.logger': ['file'], 'checkpoint.save_contents': ['model'],
            }
            paths = {'data.train_files': str(data), 'model.path': str(base),
                     'trainer.default_local_dir': str(checkpoints), 'hydra.run.dir': str(root / 'hydra'),
                     'data.custom_cls.path': 'pkg://rsicomposer.backends.training.lora.verl_sft',
                     'data.custom_cls.name': 'CompletionDataset'}
            overrides = [f'{key}={json.dumps(value)}' for key, value in {**settings, **paths}.items()]
            env = {**os.environ, 'VERL_FILE_LOGGER_PATH': str(metrics), 'HF_HUB_OFFLINE': '1'}
            subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                            '--nproc-per-node=1', '-m', 'verl.trainer.sft_trainer', *overrides],
                           env=env, check=True)
            results = [json.loads(line) for line in metrics.read_text().splitlines()]
            losses = [entry['data']['train/loss'] for entry in results]
            if [entry['step'] for entry in results] != list(range(1, total_steps + 1)) or not all(
                    loss is not None and math.isfinite(loss) for loss in losses):
                raise ValueError('veRL did not report finite loss for every scheduled training step')

            exported = root / 'export'
            subprocess.run([sys.executable, '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
                            '--local_dir', str(checkpoints / f'global_step_{total_steps}'),
                            '--target_dir', str(exported)], env=env, check=True)
            adapter = exported / 'lora_adapter'
            files = {name: file_hash(adapter / name)
                     for name in ('adapter_config.json', 'adapter_model.safetensors')}
            shutil.move(adapter, output / 'adapter')

        evidence = {'backend': VeRLSFTBackend.name, 'verl_version': observed, 'steps': total_steps,
                    'loss': losses[-1], 'samples': len(rows), 'adapter_files': files, 'settings': settings,
                    'environment': {name: version(name) for name in ('torch', 'transformers', 'peft', 'tensordict')}}
        (output / 'training.json').write_text(json.dumps(evidence, allow_nan=False))
        return evidence


if __name__ == '__main__':
    run_request(_VeRLWorker)
