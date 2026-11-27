"""Emit fully bound AIME experiment configs without loading a model or running it."""
import argparse
from copy import deepcopy
from importlib.resources import files
import json
from pathlib import Path

from .controller import validate_config
from .lab import validate_config as validate_lab


ARMS = ('adaptive', 'frozen', 'reference', 'data', 'prompt', 'model-after-data', 'fixed-order',
        'uniform-admissible', 'diagnostic-router')


def build_config(*, bundle, model, run, arm='adaptive', terms=None, steps=None, seed=23,
                 device='cpu', gpu_index=None, model_kind=None, thinking=False, generation_tokens=None, max_length=2048,
                 generation_batch_size=None, rollout_backend=None, scheduler_only=False, data_policy='rollout',
                 max_completion_chars=None, controller_harness=False):
    if arm not in ARMS or type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError('Choose a known AIME arm and an unsigned 32-bit seed')
    if steps is not None and (type(steps) is not int or not 1 <= steps <= 24):
        raise ValueError('Set an integer proposal horizon between 1 and 24')
    if type(max_length) is not int or not 64 <= max_length <= 65536:
        raise ValueError('Set a local context window between 64 and 65536 tokens')
    if generation_tokens is not None and (type(generation_tokens) is not int or not 1 <= generation_tokens < max_length):
        raise ValueError('Generation cap must leave room for the prompt in the selected context window')
    if generation_batch_size is not None and (type(generation_batch_size) is not int or not 1 <= generation_batch_size <= 16):
        raise ValueError('Set a generation batch size between 1 and 16')
    if gpu_index is not None and (device != 'cuda' or type(gpu_index) is not int or gpu_index < 0):
        raise ValueError('A physical GPU index must be non-negative and requires CUDA')
    preset = files(__package__).joinpath('configs/aime.json')
    config = json.loads(preset.read_text(encoding='utf-8'))
    if device == 'cuda':
        # Child trials inherit the same single-GPU Host placement.
        index = 0 if gpu_index is None else gpu_index
        config['resources'] = {'executor': 'local',
            'pools': {'aime-gpu': {'nodes': 1, 'gpus': [index], 'gpus_per_node': 1}},
            'placement': {role: {'pool': 'aime-gpu', 'gpus': 1} for role in ('math.rollout', 'math.trainer')}}
    method = config['framework']['method']['config']
    policy, lab = method['controller'], method['labs']['math']
    base, data = Path(model).expanduser().resolve(strict=True), Path(bundle).expanduser().resolve(strict=True)
    if not (base / 'config.json').is_file() or not any((base / name).is_file() for name in (
            'model.safetensors', 'model.safetensors.index.json')):
        raise ValueError('Model directory must contain local Transformers config and safetensors')
    if not (data / 'manifest.json').is_file() or not (data / 'meta-manifest.json').is_file():
        raise ValueError('AIME bundle must contain both prepared manifests')
    for options, manifest in ((lab, 'manifest.json'), (policy['trial_lab'], 'meta-manifest.json')):
        options['base_model'] = str(base)
        options['manifest'] = str(data / manifest)
        options['model_options'].update(seed=seed, device=device, max_length=max_length)
        if model_kind is not None:
            options['model_options']['model_kind'] = model_kind
        if thinking:
            options['model_options']['enable_thinking'] = True
        if generation_batch_size is not None:
            options['model_options']['generation_batch_size'] = generation_batch_size
        if rollout_backend is not None:
            options['model_options']['rollout_backend'] = rollout_backend
        options['aime']['data_policy'] = data_policy
        if max_completion_chars is not None:
            options['aime']['max_completion_chars'] = max_completion_chars
        if generation_tokens is not None:
            options['model_options']['max_new_tokens'] = generation_tokens
        if options['model_options']['max_new_tokens'] >= max_length:
            raise ValueError('Generation cap must leave room for the prompt in the selected context window')
    campaign = policy['campaign']
    campaign['policy_seed'] = seed
    if terms is not None:
        campaign['terms'] = terms
    if steps is not None:
        policy.update(steps=steps, warmup=min(policy['warmup'], steps - 1), trial_steps=min(steps, 6))
    if scheduler_only:
        policy['meta_targets'] = [target for target in policy['meta_targets'] if target == 'schedule']
    if controller_harness and arm == 'adaptive':
        policy['meta_targets'].append('controller_harness')
    if arm != 'adaptive':
        campaign['track'] = 'reference' if arm == 'reference' else 'frozen'
        policy['meta_targets'] = []
    if arm == 'reference':
        campaign['terms'] = 1
    elif arm == 'data':
        policy['order'] = ['data']
    elif arm == 'prompt':
        policy['order'] = ['harness']
    elif arm == 'model-after-data':
        policy['order'] = ['data', 'model']
    elif arm == 'fixed-order':
        policy['order'] = ['data', 'model', 'harness']
    elif arm == 'diagnostic-router':
        campaign['routing'] = 'diagnostic'
    elif arm == 'uniform-admissible':
        campaign['routing'] = 'uniform'
    config['data_dir'] = str(Path(run).expanduser().resolve())
    config['objective'] += f' / {arm} / seed {seed}'
    validate_config(policy)
    validate_lab(lab)
    validate_lab(policy['trial_lab'])
    config['max_decision_steps'] = max(config['max_decision_steps'], campaign['terms'] * (4 * policy['steps'] + 44))
    return deepcopy(config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True, help='Prepared AIME data bundle')
    parser.add_argument('--model', type=Path, required=True, help='Pinned local Transformers files')
    parser.add_argument('--run', type=Path, required=True, help='Fresh RSIComposer run directory')
    parser.add_argument('--arm', choices=ARMS, default='adaptive')
    parser.add_argument('--terms', type=int)
    parser.add_argument('--steps', type=int, help='Maximum operator proposals per term')
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--gpu-index', type=int, help='Physical CUDA index for the single-GPU Host pool (default: 0)')
    parser.add_argument('--model-kind', choices=('causal_lm', 'qwen3_5', 'qwen3_5_moe'),
                        help='Explicit Transformers architecture for the staged local checkpoint')
    parser.add_argument('--thinking', action='store_true', help='Enable the model chat template reasoning mode')
    parser.add_argument('--generation-tokens', type=int, help='Fixed cap: choose before comparing any arms')
    parser.add_argument('--max-length', type=int, default=2048, help='Fixed context window including generation')
    parser.add_argument('--generation-batch-size', type=int, help='Fixed worker batch or vLLM concurrency for every arm')
    parser.add_argument('--rollout-backend', choices=('vllm_graph',),
                        help='Use single-GPU vLLM graphs for rollout in a trusted vLLM/PEFT Python environment')
    parser.add_argument('--max-completion-chars', type=int,
                        help='Bound verified SFT trajectory storage (4096 default; at most 131072)')
    parser.add_argument('--data-policy', choices=('rollout', 'affine'), default='rollout',
                        help='Use ordinary verified rollouts or bounded task synthesis with a blind Anchor')
    parser.add_argument('--controller-harness', action='store_true',
                        help='Also trial the Controller Harness evidence projection across terms')
    parser.add_argument('--scheduler-only', action='store_true',
                        help='Keep term 1 operator policies identical to the frozen track')
    args = parser.parse_args(argv)
    try:
        config = build_config(bundle=args.bundle, model=args.model, run=args.run, arm=args.arm, terms=args.terms,
                              steps=args.steps, seed=args.seed, device=args.device, gpu_index=args.gpu_index,
                              model_kind=args.model_kind,
                              thinking=args.thinking, generation_tokens=args.generation_tokens, max_length=args.max_length,
                              generation_batch_size=args.generation_batch_size, rollout_backend=args.rollout_backend,
                              scheduler_only=args.scheduler_only,
                              data_policy=args.data_policy, max_completion_chars=args.max_completion_chars,
                              controller_harness=args.controller_harness)
    except (OSError, ValueError) as error:
        parser.exit(2, f'AIME configuration failed: {error}\n')
    print(json.dumps(config, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
