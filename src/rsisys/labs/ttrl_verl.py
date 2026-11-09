"""Registered TTRL Lab delegating training to the pinned official verl fork.

External backend (Apache-2.0), installed separately:
https://github.com/PRIME-RL/TTRL/tree/5806e119789132fcb43552beaf2532d7d0447213
"""
import json

from .pinned_upstream import TTRLReward, TTRL_REVISION, checkout
from .ttrl import TTRLLab, validate_config as validate_reference
from .weight_artifacts import file_hash
from .weight_engine import WeightEngine

DEFAULTS = dict(total_epochs=10, train_batch_size=32, samples_per_prompt=32,
    eval_samples=16, evaluation_temperature=0.6, evaluation_top_p=0.95,
    temperature=0.6, top_p=1.0, max_tokens=3072, max_length=4096,
    apply_chat_template=True, worker_timeout=86400)
NATIVE_OPTIONS = {'upstream_root', 'total_epochs', 'train_batch_size', 'samples_per_prompt', 'native_overrides'}


def validate_config(config):
    merged = {**DEFAULTS, **config}
    validate_reference({k: v for k, v in merged.items() if k not in NATIVE_OPTIONS})
    if not isinstance(merged.get('upstream_root'), str) or not merged['upstream_root'].strip():
        raise ValueError('ttrl_verl requires upstream_root pointing to the pinned TTRL checkout')
    for key in ('total_epochs', 'train_batch_size', 'samples_per_prompt'):
        if type(merged[key]) is not int or merged[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if merged['samples_per_prompt'] < 2:
        raise ValueError('TTRL training requires at least two responses')
    overrides = merged.get('native_overrides', {})
    if not isinstance(overrides, dict) or any(not isinstance(k, str) or not k for k in overrides):
        raise ValueError('native_overrides must be a mapping of Hydra config keys to JSON values')
    reserved = ('data.train_files', 'data.val_files', 'data.train_batch_size', 'custom_reward_function', 'ttrl',
                'trainer.default_local_dir', 'trainer.resume', 'trainer.total',
                'trainer.save_freq', 'trainer.test_freq', 'trainer.val_before_train',
                'actor_rollout_ref.model.path', 'actor_rollout_ref.model.lora_rank', 'trainer.n_gpus_per_node', 'trainer.nnodes')
    if any(any(k == r or k.startswith(r + '.') or k.startswith(r + '_') or r.startswith(k + '.')
               for r in reserved) for k in overrides):
        raise ValueError('native_overrides cannot replace owned data, reward, model, schedule or publication paths')


class VerlTTRLEngine(WeightEngine):
    name = 'ttrl-official-verl-' + TTRL_REVISION

    def fit(self, model, rows, directory, checkpoint, *, votes):
        import os
        import sys
        from .math_parallel import run_workers
        root = checkout(self.config['upstream_root'], TTRL_REVISION)
        directory.mkdir(parents=True, exist_ok=False)
        output, request = directory / 'response.json', directory / 'request.json'
        request.write_text(json.dumps(dict(config={**self.config, 'upstream_root': str(root)},
            model=model, items=rows, checkpoint=str(checkpoint), votes=votes, output=str(output))))
        env = dict(self._env(), CUDA_VISIBLE_DEVICES=','.join(map(str, self._gpu_ids())),
            VLLM_USE_V1='1', WANDB_MODE='disabled')
        cuda_home = self.config.get('training_cuda_home')
        if cuda_home:
            env.update(CUDA_HOME=cuda_home, PATH=cuda_home + '/bin' + os.pathsep + env.get('PATH', ''),
                LD_LIBRARY_PATH=cuda_home + '/lib64' + os.pathsep + env.get('LD_LIBRARY_PATH', ''))
        run_workers([[self.config.get('training_python', sys.executable), '-m', 'rsisys.labs.verl_worker', str(request)]],
            [env], directory, self.config['worker_timeout'])
        return json.loads(output.read_text())


class TTRLVerlLab(TTRLLab):
    engine_methods = ('generate', 'fit')
    fields = {'evaluate': {'dataset': 'string'}, 'fit': {'dataset': 'string', 'samples_per_prompt': 'integer'}}
    allow_empty_transfer = True
    validate_config = staticmethod(validate_config)

    def __init__(self, context, config, *, engine=None, verifier=None):
        config = {**DEFAULTS, **config}
        validate_config(config)
        if verifier is None:
            verifier = TTRLReward(config['upstream_root'])
        super().__init__(context, config, engine=engine or VerlTTRLEngine(config), verifier=verifier)

    def _fit(self, args):
        rows = self.read(args['dataset'])
        if not isinstance(rows, list) or not rows or any(not isinstance(r, dict) or set(r) != {'id', 'prompt'}
                or any(not isinstance(r[k], str) or not r[k].strip() for k in ('id', 'prompt')) for r in rows):
            raise ValueError('TTRL learning file must contain only id and prompt; gold labels are forbidden')
        prompts = {r['prompt'].strip() for r in rows}
        if len(prompts) != len(rows) or len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate adaptation questions')
        if self.target_prompts is None or prompts != self.target_prompts or prompts & self.transfer_prompts:
            raise ValueError('Adaptation must exactly match target questions and exclude transfer questions')
        if len(rows) < self.config['train_batch_size']:
            raise ValueError('Native drop-last dataloader would have no training batch')
        if args['samples_per_prompt'] < self.config['samples_per_prompt']:
            raise ValueError('Vote count must be at least the training sample count')
        checkpoint = self.root / f'round-{args["round"]}' / 'model'
        metrics = self.engine.fit(str(self.path(args['model'])), rows,
            self.root / f'round-{args["round"]}' / 'native-workers', checkpoint, votes=args['samples_per_prompt'])
        weights = sorted(checkpoint.glob('*.safetensors'))
        if not weights or not metrics.get('weights_changed') or metrics.get('steps', 0) < 1:
            raise ValueError('Native TTRL did not publish updated weights')
        manifest = dict(parent=str(self.path(args['model'])), initial_model=self.initial_model,
            dataset_sha256=file_hash(self.path(args['dataset'])), engine=self.engine.name,
            algorithm='ttrl-native-grpo', recipe=self.config, metrics=metrics,
            weights={p.name: file_hash(p) for p in weights}, verifier=self.verifier.name)
        (checkpoint / 'provenance.json').write_text(json.dumps(manifest, indent=2))
        self.models[str(checkpoint)] = manifest
        return {'model': str(checkpoint), 'training_performed': not self.simulated, 'training': metrics,
                'provenance_sha256': file_hash(checkpoint / 'provenance.json')}
