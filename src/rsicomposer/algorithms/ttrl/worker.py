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
from .native_jobs import atomic_json
from .progress_hooks import StopAtCheckpoint

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
        'trainer.n_gpus_per_node': request['world_size'], 'trainer.nnodes': 1,
        'trainer.save_freq': config['save_freq'], 'trainer.test_freq': -1, 'trainer.val_before_train': False,
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


def changed_model_tensors(before, after, before_config, after_config):
    def canonical(weights, config):
        # veRL may serialize the Qwen2 LM head even when it shares the input weights.
        if (config.get('model_type') == 'qwen2' and config.get('tie_word_embeddings') is True
                and 'lm_head.weight' in weights):
            if weights['lm_head.weight'] != weights.get('model.embed_tokens.weight'):
                raise ValueError('Tied weights differ in the native model publication')
            return {key: value for key, value in weights.items() if key != 'lm_head.weight'}
        return weights

    if any(before_config.get(key) != after_config.get(key)
           for key in ('model_type', 'tie_word_embeddings')):
        raise ValueError('Native publication changed the model parameter schema')
    before, after = canonical(before, before_config), canonical(after, after_config)
    if before.keys() != after.keys() or any(before[key][0] != after[key][0] for key in before):
        raise ValueError('Native publication changed the model parameter schema')
    return sum(before[key][1] != after[key][1] for key in before)


def fit(request):
    from datasets import Dataset
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from rsicomposer.api import file_hash

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
    # Run the public official TaskRunner; its trainer body and optimizer schedule
    # are unchanged. The worker hook observes save/log boundaries inside that actor.
    sys.path.insert(0, str(upstream))
    import ray
    from verl.trainer.main_ppo import TaskRunner
    handlers = {}
    def cancelled(signum, frame):
        raise InterruptedError(f'Native TTRL interrupted by signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT):
        handlers[sig] = signal.signal(sig, cancelled)
    stopped = False
    runtime_env = dict(worker_process_setup_hook='rsicomposer.algorithms.ttrl.progress_hooks.install', env_vars={
        'RSISYS_TTRL_JOB': str(directory), 'PYTHONPATH': env['PYTHONPATH'],
        'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': env.get('NCCL_DEBUG', 'WARN'),
        'VLLM_LOGGING_LEVEL': 'WARN', 'VLLM_ALLOW_RUNTIME_LORA_UPDATING': 'true'})
    try:
        session = ray.init(address='local', include_dashboard=False,
            runtime_env=runtime_env,
            num_cpus=min(32, os.cpu_count() or 1), num_gpus=request['world_size'],
            _temp_dir=tempfile.mkdtemp(prefix='rsisys-ray-'))
        env['RAY_ADDRESS'] = session.address_info['address']
        atomic_json(directory / 'ray-session.json', json.loads(json.dumps(session.address_info, default=str)))
        atomic_json(directory / 'execution-environment.json', {k: v for k, v in env.items()
            if k.startswith(('NCCL_', 'VLLM_')) or k == 'CUDA_VISIBLE_DEVICES'})
        profile_steps = OmegaConf.select(config.trainer, 'profile_steps')
        if profile_steps:
            runner = TaskRunner.options(runtime_env={
                'nsight': OmegaConf.to_container(config.trainer.controller_nsight_options)}).remote()
        else:
            runner = TaskRunner.remote()
        try:
            ray.get(runner.run.remote(config))
        except ray.exceptions.RayTaskError as exc:
            if not isinstance(exc.as_instanceof_cause(), StopAtCheckpoint):
                raise
            ack = json.loads((directory / 'stop-ack.json').read_text())
            if ack['call_id'] != request['call_id']:
                raise ValueError('Stop acknowledgment belongs to a different native call') from exc
            stopped = True
    finally:
        ray.shutdown()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    native = directory / 'checkpoints'
    step = int((native / 'latest_checkpointed_iteration.txt').read_text().strip())
    expected = len(request['items']) // request['config']['train_batch_size'] * request['config']['total_epochs']
    if step < 1 or step > expected or (not stopped and step != expected):
        raise ValueError(f'Native schedule incomplete: final step {step}, expected {expected}')
    if stopped and ack['step'] != step:
        raise ValueError('Stopped checkpoint does not match the official publication')
    actor = native / f'global_step_{step}' / 'actor'
    checkpoint = Path(request['checkpoint'])
    if checkpoint.exists():
        raise ValueError('Refusing to overwrite checkpoint')
    subprocess.run([sys.executable, '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
        '--local_dir', str(actor), '--target_dir', str(checkpoint)], cwd=upstream, env=env, check=True)
    base = Path(request['model'])
    changed = changed_model_tensors(tensor_hashes(base), tensor_hashes(checkpoint),
        json.loads((base / 'config.json').read_text()), json.loads((checkpoint / 'config.json').read_text()))
    if not changed:
        raise ValueError('Native training did not change published model tensors')
    result = dict(steps=step, samples=len(request['items']), world_size=request['world_size'],
        algorithm='ttrl-native-grpo', weights_changed=True, published_changed_tensors=changed,
        upstream_revision=TTRL_REVISION, effective_config_sha256=file_hash(target),
        unlabeled_parquet_sha256=file_hash(data), checkpoint_selection='controller-stop' if stopped else 'configured-final-step')
    (checkpoint / 'training.json').write_text(json.dumps(result, indent=2))
    atomic_json(request['output'], result)
    print('RSISYS_WORKER ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    request = json.loads(Path(sys.argv[1]).read_text())
    status_path = Path(request['output']).parent / 'status.json'
    try:
        fit(request)
    except BaseException as exc:
        atomic_json(status_path, dict(status='failed', error=f'{type(exc).__name__}: {exc}'))
        raise
    else:
        atomic_json(status_path, dict(status='succeeded'))
