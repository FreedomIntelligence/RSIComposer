"""Optional single-device TRL trainer over the LoRA worker contract."""

import json
import math
from importlib.metadata import PackageNotFoundError, version

from rsisys.runtime.storage.files import file_hash

from .engine import LoRAEngine
from .hosted import HostedLoRAEngine, run_request


def validate_trl_version(config):
    pin = config.get('trl_version')
    if not isinstance(pin, str) or not pin.strip():
        raise ValueError('TRL SFT requires an exact trl_version')


class TRLSFTBackend(HostedLoRAEngine):
    name = 'trl-sft-lora/v1'
    train_worker = 'rsisys.backends.training.lora.trl_sft'

    def __init__(self, config):
        validate_trl_version(config)
        super().__init__(config)

    def resource_roles(self):
        return {'trainer': super().resource_roles()['trainer']}

    def bind_execution(self, context):
        super().bind_execution(context)
        requested = self.resources.request('trainer').gpus
        if requested != self.resource_roles()['trainer'].gpus:
            raise ValueError('TRL SFT currently requires one GPU, or zero when device=cpu')


class _TRLWorker(LoRAEngine):
    def train(self, ref, rows, output, round_number, *, steps=None):
        config = self.config
        try:
            observed = version('trl')
        except PackageNotFoundError as error:
            raise RuntimeError('Install the pinned TRL SFT environment on training_python') from error
        if observed != config['trl_version']:
            raise ValueError(f'TRL version mismatch: expected {config["trl_version"]}, got {observed}')
        if not rows or (steps is not None and (type(steps) is not int or steps < 1)):
            raise ValueError('TRL SFT needs nonempty data and a positive step limit')

        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer

        use_cpu = config.get('device') == 'cpu'
        seed = config.get('seed', 73) + round_number
        torch.manual_seed(seed)
        model, tokenizer = self._load(ref, train=True)
        model.config.use_cache = False
        prepared = []
        for row in rows:
            prefix = tokenizer(self._prefix(tokenizer, row['prompt']), add_special_tokens=False)['input_ids']
            completion = tokenizer(row['completion'], add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
            if len(prefix) + len(completion) > config.get('max_length', 2048):
                raise ValueError('Accepted completion exceeds max_length; increase it instead of truncating the answer')
            prepared.append({'input_ids': prefix + completion, 'labels': [-100] * len(prefix) + completion})

        def collate(samples):
            width = max(len(row['input_ids']) for row in samples)
            batch = {'input_ids': [], 'labels': [], 'attention_mask': []}
            for row in samples:
                size = len(row['input_ids'])
                batch['input_ids'].append(row['input_ids'] + [tokenizer.pad_token_id] * (width - size))
                batch['labels'].append(row['labels'] + [-100] * (width - size))
                batch['attention_mask'].append([1] * size + [0] * (width - size))
            return {key: torch.tensor(value) for key, value in batch.items()}

        options = dict(output_dir=str(output / 'adapter'), use_cpu=use_cpu, bf16=not use_cpu,
            seed=seed, data_seed=seed, learning_rate=config.get('learning_rate', 2e-4),
            max_steps=steps if steps is not None else -1, num_train_epochs=config.get('epochs', 1),
            per_device_train_batch_size=config.get('train_batch_size', 1),
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
            gradient_checkpointing=config.get('train_gradient_checkpointing', False),
            report_to='none', save_strategy='no', disable_tqdm=True, remove_unused_columns=False,
            max_length=None, dataset_kwargs={'skip_prepare_dataset': True})
        trainer = SFTTrainer(model=model, args=SFTConfig(**options), processing_class=tokenizer,
            train_dataset=Dataset.from_list(prepared), data_collator=collate,
            peft_config=LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
        result = trainer.train()
        if result.global_step < 1 or not math.isfinite(result.training_loss):
            raise ValueError('TRL did not produce a finite training result')
        trainer.save_model(str(output / 'adapter'))
        adapter = {name: file_hash(output / 'adapter' / name)
                   for name in ('adapter_config.json', 'adapter_model.safetensors')}
        evidence = {'backend': TRLSFTBackend.name, 'trl_version': observed,
                    'environment': {name: version(name) for name in ('torch', 'transformers', 'peft', 'datasets', 'accelerate')},
                    'steps': result.global_step, 'loss': result.training_loss, 'samples': len(prepared),
                    'adapter_files': adapter,
                    'settings': {key: value for key, value in options.items() if key != 'output_dir'}}
        (output / 'training.json').write_text(json.dumps(evidence, allow_nan=False))
        return evidence


if __name__ == '__main__':
    run_request(_TRLWorker)
