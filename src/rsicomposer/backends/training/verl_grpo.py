"""Single-node veRL GRPO under the host's GPU lease."""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from rsicomposer.runtime.execution.executor import WorkerSpec, execution_capabilities
from rsicomposer.runtime.execution.resources import ResourceRequest
from rsicomposer.runtime.execution.workers import worker_environment
from rsicomposer.runtime.storage.files import file_hash


class VeRLGRPOBackend:
    name = 'verl-grpo/v1'

    def __init__(self, config):
        defaults = {'training_strategy': 'fsdp2', 'rollout_engine': 'vllm', 'gpus': 1,
                    'train_batch_size': 1, 'samples_per_prompt': 4, 'steps': 2,
                    'max_prompt_tokens': 256, 'max_new_tokens': 64, 'tensor_parallel_size': 1,
                    'learning_rate': 5e-6, 'temperature': 1.0, 'gpu_memory_utilization': 0.35, 'seed': 73}
        if set(config) - defaults.keys() - {'verl_version', 'rollout_version', 'training_python'}:
            raise ValueError('Unknown veRL GRPO option')
        self.config = {**defaults, **config}
        c = self.config
        for key in ('verl_version', 'rollout_version', 'training_python'):
            if not isinstance(c.get(key), str) or not c[key].strip():
                raise ValueError(f'veRL GRPO requires {key}')
        if c['training_strategy'] not in ('fsdp', 'fsdp2') or c['rollout_engine'] not in ('vllm', 'sglang'):
            raise ValueError('Choose fsdp or fsdp2 and vllm or sglang')
        for key in ('gpus', 'train_batch_size', 'samples_per_prompt', 'steps', 'max_prompt_tokens',
                    'max_new_tokens', 'tensor_parallel_size', 'seed'):
            if type(c[key]) is not int or c[key] < (0 if key == 'seed' else 1):
                raise ValueError(f'Invalid veRL GRPO {key}')
        if (c['samples_per_prompt'] < 2 or c['gpus'] % c['tensor_parallel_size']
                or c['train_batch_size'] * c['samples_per_prompt'] % c['gpus']):
            raise ValueError('GRPO needs multiple samples and batches/TP divisible across the allocated GPUs')
        for key in ('learning_rate', 'temperature', 'gpu_memory_utilization'):
            if type(c[key]) not in (int, float) or not math.isfinite(c[key]) or c[key] <= 0:
                raise ValueError(f'Invalid veRL GRPO {key}')
        if c['gpu_memory_utilization'] >= 1:
            raise ValueError('gpu_memory_utilization must be less than 1')

    def resource_roles(self):
        return {'trainer': ResourceRequest(gpus=self.config['gpus'])}

    def bind_execution(self, context):
        self.resources, self.executor = execution_capabilities(context)
        self.resources.declare(self.resource_roles())
        request = self.resources.request('trainer')
        if request.gpus != self.config['gpus']:
            raise ValueError('veRL GRPO needs its configured GPU count')
        owner = getattr(self.resources, 'owner', self.resources)
        if getattr(owner, 'config', {}).get('executor', 'local') != 'local':
            raise ValueError('veRL GRPO currently needs the local host executor')

    def fit(self, model, rows, dev_rows, output, *, reward_module):
        config = self.config
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with self.resources.acquire('trainer') as lease, tempfile.TemporaryDirectory(
                dir=output.parent, prefix='verl-grpo-') as temporary:
            root = Path(temporary)
            request = root / 'request.json'
            request.write_text(json.dumps({'config': config, 'model': str(model), 'rows': rows, 'dev_rows': dev_rows,
                                           'reward_module': str(reward_module)}, allow_nan=False))
            with (root / 'worker.log').open('w') as log:
                code = self.executor.run(WorkerSpec([config['training_python'], '-m', __name__, str(request)],
                                                    worker_environment(), stdout=log), lease)
            if code:
                raise RuntimeError('veRL GRPO worker failed:\n' + (root / 'worker.log').read_text()[-6500:])
            summary = json.loads((root / 'summary.json').read_text())
            (root / 'effective.json').rename(root / 'candidate' / 'training-config.json')
            (root / 'summary.json').rename(root / 'candidate' / 'training.json')
            if output.exists():
                raise FileExistsError('Refusing to overwrite a published model')
            (root / 'candidate').rename(output)
            return summary


def settings(c, model, root, data, dev, reward_module):
    actor = 'actor_rollout_ref.actor'
    rollout = 'actor_rollout_ref.rollout'
    batch = c['train_batch_size']
    result = {
        'algorithm.adv_estimator': 'grpo', 'algorithm.use_kl_in_reward': False,
        'data.train_files': [str(data)], 'data.val_files': [str(dev)], 'data.shuffle': False, 'data.seed': c['seed'],
        'data.train_batch_size': batch, 'data.max_prompt_length': c['max_prompt_tokens'],
        'data.max_response_length': c['max_new_tokens'], 'data.truncation': 'error',
        'data.filter_overlong_prompts': False, '+data.apply_chat_template_kwargs.enable_thinking': False,
        'actor_rollout_ref.model.path': str(model), 'actor_rollout_ref.model.use_remove_padding': False,
        'actor_rollout_ref.model.enable_gradient_checkpointing': True,
        '+actor_rollout_ref.model.override_config.attn_implementation': 'sdpa',
        f'{actor}.strategy': c['training_strategy'], f'{actor}.fsdp_config.fsdp_size': c['gpus'],
        f'{actor}.fsdp_config.seed': c['seed'], f'{actor}.use_torch_compile': False,
        f'{actor}.ppo_mini_batch_size': batch, f'{actor}.ppo_micro_batch_size_per_gpu': 1,
        f'{actor}.use_kl_loss': False, f'{actor}.optim.lr': c['learning_rate'],
        f'{actor}.checkpoint.save_contents': ['model'],
        f'{rollout}.name': c['rollout_engine'], f'{rollout}.n': c['samples_per_prompt'],
        f'{rollout}.temperature': c['temperature'], f'{rollout}.tensor_model_parallel_size': c['tensor_parallel_size'],
        f'{rollout}.log_prob_micro_batch_size_per_gpu': 1,
        f'{rollout}.calculate_log_probs': True, f'{rollout}.enforce_eager': True,
        f'{rollout}.enable_prefix_caching': False, f'{rollout}.gpu_memory_utilization': c['gpu_memory_utilization'],
        f'{rollout}.max_num_batched_tokens': max(1024, c['max_prompt_tokens'] + c['max_new_tokens']),
        'reward.num_workers': 1, 'reward.custom_reward_function.path': str(reward_module),
        'reward.custom_reward_function.name': 'compute_score',
        'trainer.nnodes': 1, 'trainer.n_gpus_per_node': c['gpus'], 'trainer.device': 'cuda',
        'trainer.total_training_steps': c['steps'], 'trainer.total_epochs': c['steps'],
        'trainer.val_before_train': False, 'trainer.test_freq': -1, 'trainer.save_freq': c['steps'],
        'trainer.resume_mode': 'disable', 'trainer.balance_batch': False, 'trainer.logger': ['console', 'file'],
        'trainer.default_local_dir': str(root / 'checkpoints'), 'trainer.rollout_data_dir': str(root / 'rollouts'),
        '+ray_kwargs.ray_init.include_dashboard': False,
        'hydra.run.dir': str(root / 'hydra'),
    }
    return result


def _versions(c):
    result = {}
    for package, pin in (('verl', c['verl_version']), (c['rollout_engine'], c['rollout_version'])):
        try:
            result[package] = version(package)
        except PackageNotFoundError as exc:
            raise RuntimeError(f'Install {package}=={pin} in training_python') from exc
        if result[package] != pin:
            raise ValueError(f'{package} version mismatch: expected {pin}, got {result[package]}')
    return {**result, **{package: version(package) for package in ('torch', 'transformers', 'ray')}}


def run_worker(request_path):
    path = Path(request_path).resolve()
    request = json.loads(path.read_text())
    c = VeRLGRPOBackend(request['config']).config
    versions = _versions(c)
    rows, dev_rows, model, root = request['rows'], request['dev_rows'], Path(request['model']).resolve(strict=True), path.parent
    if not rows or len(rows) % c['train_batch_size']:
        raise ValueError('GRPO needs complete batches; adjust train_batch_size to the training split')
    from datasets import Dataset

    if not dev_rows:
        raise ValueError('veRL requires a development dataset')
    data, dev = root / 'training.parquet', root / 'dev.parquet'
    for records, target in ((rows, data), (dev_rows, dev)):
        Dataset.from_list([{'prompt': row['messages'], 'data_source': row['data_source'],
            'reward_model': {'style': 'rule', 'ground_truth': row['ground_truth']},
            'extra_info': row['extra_info']} for row in records]).to_parquet(target)
    overrides = settings(c, model, root, data, dev, Path(request['reward_module']).resolve(strict=True))
    effective = root / 'effective.json'
    effective.write_text(json.dumps(overrides, sort_keys=True))
    env = {**os.environ, 'RAY_ADDRESS': 'local', 'HF_HUB_OFFLINE': '1', 'WANDB_MODE': 'disabled',
           'RAY_TMPDIR': tempfile.mkdtemp(prefix='rsisys-rl-ray-'), 'VERL_FILE_LOGGER_PATH': str(root / 'metrics.jsonl')}
    try:
        command = [sys.executable, '-m', 'verl.trainer.main_ppo',
                   *(f'{key}={json.dumps(value)}' for key, value in overrides.items())]
        subprocess.run(command, env=env, cwd=root, check=True)
        checkpoints = root / 'checkpoints'
        step = int((checkpoints / 'latest_checkpointed_iteration.txt').read_text())
        if step != c['steps']:
            raise ValueError(f'veRL published step {step}; expected {c["steps"]}')
        steps = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
        rewards = {entry['step']: entry['data']['critic/score/mean'] for entry in steps
                   if 'critic/score/mean' in entry['data']}
        metrics = {entry['step']: {key: value for key, value in entry['data'].items()
                    if key in {'actor/grad_norm', 'critic/rewards/min', 'critic/rewards/max',
                               'timing_s/update_weights'} or key.startswith(('rollout_corr/', 'training/off_policy/'))}
                   for entry in steps if 'critic/score/mean' in entry['data']}
        if (set(rewards) != set(range(1, step + 1)) or any(not math.isfinite(score) for score in rewards.values())
                or any(not (root / 'rollouts' / f'{i}.jsonl').is_file() for i in rewards)):
            raise ValueError('veRL did not publish finite rewards and rollouts at every step')
        candidate = root / 'candidate'
        subprocess.run([sys.executable, '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
                        '--local_dir', str(checkpoints / f'global_step_{step}' / 'actor'),
                        '--target_dir', str(candidate)], env=env, cwd=root, check=True)
        weights = {file.name: file_hash(file) for file in sorted(candidate.glob('*.safetensors'))}
        if not weights:
            raise ValueError('veRL did not export a loadable checkpoint')
        summary = {'backend': VeRLGRPOBackend.name, 'environment': versions, 'steps': step, 'train_examples': len(rows),
                   'reward_mean_by_step': rewards, 'metrics_by_step': metrics, 'checkpoint_files': weights,
                   'settings_sha256': file_hash(effective), 'training_data_sha256': file_hash(data),
                   'rollout_files': {f'{i}.jsonl': file_hash(root / 'rollouts' / f'{i}.jsonl') for i in rewards}}
        (root / 'summary.json').write_text(json.dumps(summary, allow_nan=False))
    finally:
        shutil.rmtree(env['RAY_TMPDIR'], ignore_errors=True)


if __name__ == '__main__':
    run_worker(sys.argv[1])
