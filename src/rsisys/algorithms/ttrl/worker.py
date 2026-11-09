"""Process bridge to TTRL's native Hydra/Ray/FSDP training schedule.

Training receives only unlabeled prompts. The official original-GT diagnostics
see a placeholder; real labels stay in the parent Lab's evaluation operation.
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
from hashlib import sha256
from pathlib import Path

from .upstream import TTRL_REVISION, checkout

SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.'


def native_settings(request, directory, upstream):
    """Explicit settings from the pinned Qwen2.5/math.sh; no shell interpolation."""
    config = request['config']
    settings = {
        'data.train_files': [str(directory / 'unlabeled.parquet')],
        'data.val_files': [str(directory / 'unlabeled.parquet')],
        'data.max_prompt_length': config['max_length'] - config['max_tokens'],
        'data.max_response_length': config['max_tokens'],
        'data.train_batch_size': config['train_batch_size'],
        'data.seed': 1,  # The pinned upstream dataloader default.
        'data.filter_overlong_prompts': True, 'data.truncation': 'error', 'data.suffix_prompt': SUFFIX,
        'actor_rollout_ref.model.path': request['model'],
        'actor_rollout_ref.model.enable_gradient_checkpointing': True,
        'actor_rollout_ref.model.use_remove_padding': True,
        'actor_rollout_ref.model.lora_rank': 0,
        'actor_rollout_ref.actor.ppo_mini_batch_size': 1,
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': 2,
        'actor_rollout_ref.actor.use_kl_loss': True,
        'actor_rollout_ref.actor.loss_agg_mode': 'token-mean',
        'actor_rollout_ref.actor.optim.lr': config.get('rl_lr', 5e-7),
        'actor_rollout_ref.actor.optim.lr_warmup_steps_ratio': 0.03,
        'actor_rollout_ref.actor.optim.warmup_style': 'cosine',
        'actor_rollout_ref.actor.fsdp_config.param_offload': False,
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload': False,
        'actor_rollout_ref.actor.ppo_max_token_len_per_gpu': config['max_length'],
        'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu': 2,
        'actor_rollout_ref.ref.fsdp_config.param_offload': True,
        'actor_rollout_ref.rollout.name': 'vllm',
        'actor_rollout_ref.rollout.seed': 0,  # The pinned upstream rollout default.
        'actor_rollout_ref.rollout.temperature': config['temperature'],
        'actor_rollout_ref.rollout.top_p': config['top_p'],
        'actor_rollout_ref.rollout.enforce_eager': config.get('enforce_eager', False),
        'actor_rollout_ref.rollout.free_cache_engine': False,
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu': 2,
        'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,
        'actor_rollout_ref.rollout.gpu_memory_utilization': 0.8,
        'actor_rollout_ref.rollout.n': config['samples_per_prompt'],
        'actor_rollout_ref.rollout.val_kwargs.do_sample': True,
        'actor_rollout_ref.rollout.val_kwargs.n': config['eval_samples'],
        'actor_rollout_ref.rollout.val_kwargs.top_p': config['evaluation_top_p'],
        'actor_rollout_ref.rollout.val_kwargs.temperature': config['evaluation_temperature'],
        'actor_rollout_ref.rollout.max_model_len': config['max_length'],
        'actor_rollout_ref.rollout.max_num_batched_tokens': config['max_length'],
        'algorithm.kl_ctrl.kl_coef': 0.0, 'algorithm.adv_estimator': 'grpo',
        'custom_reward_function.path': str(upstream / 'verl/utils/reward_score/ttrl_math/__init__.py'),
        'custom_reward_function.name': 'reward_func',
        'ttrl.enable': True, 'ttrl.n_votes_per_prompt': request['votes'],
        'ttrl.n_samples_per_prompt': config['samples_per_prompt'],
        'trainer.logger': ['console'], 'trainer.project_name': 'rsisys-ttrl',
        'trainer.experiment_name': directory.parent.name,
        'trainer.n_gpus_per_node': len(config.get('gpu_ids', [0])), 'trainer.nnodes': 1,
        'trainer.save_freq': 2000000, 'trainer.test_freq': -1, 'trainer.val_before_train': False,
        'trainer.default_local_dir': str(directory / 'checkpoints'),
        'trainer.total_epochs': config['total_epochs'], 'trainer.resume_mode': 'disable',
    }
    settings.update(config.get('native_overrides', {}))
    return settings


def unlabeled_records(rows):
    return [{'data_source': 'MATH-TTT', 'prompt': [{'role': 'user', 'content': r['prompt']}],
        'ability': 'math', 'reward_model': {'style': 'rule', 'ground_truth': 'None'},
        'extra_info': {'split': 'train', 'index': r['id']}} for r in rows]


def tensor_hashes(path):
    import torch
    from safetensors import safe_open
    result = {}
    for shard in sorted(Path(path).glob('*.safetensors')):
        with safe_open(shard, framework='pt', device='cpu') as stream:
            for key in stream.keys():
                value = stream.get_tensor(key)
                # A dtype-only conversion must not count as an optimizer update.
                # Float64 exactly represents all supported floating weight dtypes.
                digest = sha256(value.to(torch.float64).contiguous().numpy().tobytes()).hexdigest()
                result[key] = (tuple(value.shape), digest)
    if not result:
        raise ValueError('No published tensors found')
    return result


def fit(request):
    from datasets import Dataset
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from rsisys.runtime.storage.files import file_hash

    root = checkout(request['config']['upstream_root'], TTRL_REVISION)
    upstream = root / 'verl'
    directory = Path(request['output']).parent
    data = directory / 'unlabeled.parquet'
    Dataset.from_list(unlabeled_records(request['items'])).to_parquet(data)
    settings = native_settings(request, directory, upstream)
    with initialize_config_dir(config_dir=str(upstream / 'verl/trainer/config'), version_base=None):
        config = compose(config_name='ppo_trainer_ttrl')
    for key, value in settings.items():
        OmegaConf.update(config, key, value, force_add=True)
    # Save the fully resolved configuration that actually reaches upstream.
    target = directory / 'effective.yaml'
    target.write_text(OmegaConf.to_yaml(config, resolve=True))
    env = dict(os.environ, PYTHONPATH=str(upstream) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    command = [sys.executable, '-m', 'verl.trainer.main_ppo', '--config-path', str(directory), '--config-name', 'effective']
    (directory / 'command.json').write_text(json.dumps(command))
    # Own a distinct Ray cluster: never attach to or stop another host job.
    # Ray daemons create their own process groups, so launcher-group cleanup
    # alone is insufficient on cancellation or a failed native worker.
    import ray
    handlers = {}
    def cancelled(signum, frame):
        raise InterruptedError(f'Native TTRL interrupted by signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT):
        handlers[sig] = signal.signal(sig, cancelled)
    try:
        session = ray.init(address='local', include_dashboard=False,
            num_cpus=min(32, os.cpu_count() or 1), num_gpus=len(request['config'].get('gpu_ids', [0])),
            _temp_dir=tempfile.mkdtemp(prefix='rsisys-ray-'))
        env['RAY_ADDRESS'] = session.address_info['address']
        (directory / 'ray-session.json').write_text(json.dumps(session.address_info, default=str))
        (directory / 'execution-environment.json').write_text(json.dumps({k: v for k, v in env.items()
            if k.startswith(('NCCL_', 'VLLM_')) or k == 'CUDA_VISIBLE_DEVICES'}, indent=2))
        subprocess.run(command, cwd=upstream, env=env, check=True)
    finally:
        ray.shutdown()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    native = directory / 'checkpoints'
    step = int((native / 'latest_checkpointed_iteration.txt').read_text().strip())
    expected = len(request['items']) // request['config']['train_batch_size'] * request['config']['total_epochs']
    if step != expected or step < 1:
        raise ValueError(f'Native schedule incomplete: final step {step}, expected {expected}')
    actor = native / f'global_step_{step}' / 'actor'
    checkpoint = Path(request['checkpoint'])
    if checkpoint.exists():
        raise ValueError('Refusing to overwrite checkpoint')
    subprocess.run([sys.executable, '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
        '--local_dir', str(actor), '--target_dir', str(checkpoint)], cwd=upstream, env=env, check=True)
    before, after = tensor_hashes(request['model']), tensor_hashes(checkpoint)
    if before.keys() != after.keys():
        raise ValueError('Native publication changed the model parameter schema')
    changed = sum(before[key] != after[key] for key in before)
    if not changed:
        raise ValueError('Native training did not change published model tensors')
    result = dict(steps=step, samples=len(request['items']), world_size=len(request['config'].get('gpu_ids', [0])),
        algorithm='ttrl-native-grpo', weights_changed=True, published_changed_tensors=changed,
        upstream_revision=TTRL_REVISION, effective_config_sha256=file_hash(target),
        unlabeled_parquet_sha256=file_hash(data), checkpoint_selection='configured-final-step')
    (checkpoint / 'training.json').write_text(json.dumps(result, indent=2))
    Path(request['output']).write_text(json.dumps(result))
    print('RSISYS_WORKER ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    fit(json.loads(Path(sys.argv[1]).read_text()))
