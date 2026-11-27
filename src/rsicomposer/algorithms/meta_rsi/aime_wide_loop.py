"""Opt-in contracts for the development-only large-pool MetaRSI lab."""
import json
from pathlib import Path

from . import aime
from .engine import file_hash

TRAIN_OPTIONS = {'training_world_size': 4, 'training_global_batch_size': 16,
                 'ordered_training_coverage': True, 'training_allow_uneven_tail': True,
                 'rollout_shards': 4}


def check_loop_config(config, wide):
    options = config.get('model_options')
    required = {'bootstrap', 'bootstrap_sha256', 'checkpoint_run', 'checkpoint_sha256', 'aggregate_sha256'}
    if (not isinstance(wide, dict) or not required <= wide.keys() <= required | {'model_stage'}
            or any(not aime.sha256_text(wide[k]) for k in ('bootstrap_sha256', 'checkpoint_sha256', 'aggregate_sha256'))
            or any(not isinstance(wide[k], str) or not Path(wide[k]).is_absolute()
                   for k in ('bootstrap', 'checkpoint_run'))
            or config.get('engine') != 'transformers' or 'test' in config['split_roles']
            or not isinstance(options, dict) or any(options.get(k) != v for k, v in TRAIN_OPTIONS.items())
            or any(options.get(k) != v for k, v in {'device': 'cuda', 'seed': 23,
                   'model_kind': 'qwen3_5_moe', 'rollout_backend': 'vllm_graph'}.items())
            or options.get('train_batch_size', 1) != 1 or options.get('gradient_accumulation_steps', 1) != 1
            or not isinstance(config.get('aime'), dict) or config['aime'].get('data_policy') != 'affine'
            or aime.data_capacity(config['aime']) < 256):
        raise ValueError('Wide AIME loop requires the pinned four-card development-only affine protocol')
    if 'model_stage' in wide:
        stage = wide['model_stage']
        if (not isinstance(stage, dict) or set(stage) != {'source_run', 'result_sha256'}
                or not isinstance(stage['source_run'], str) or not Path(stage['source_run']).is_absolute()
                or not aime.sha256_text(stage['result_sha256']) or 'initial_snapshot' in config):
            raise ValueError('Invalid isolated Model-stage provenance')
    return TRAIN_OPTIONS.keys()


def check_owned_ref(ref):
    if (not isinstance(ref, dict) or set(ref) != {'schema_version', 'candidate_id', 'base_sha256'}
            or ref['schema_version'] != 3 or not aime.candidate_id(ref['candidate_id'])
            or not aime.sha256_text(ref['base_sha256'])):
        raise ValueError('AIME wide-loop snapshots are reference-only owned v3 snapshots')


def load_bootstrap(config):
    path = Path(config['bootstrap'])
    if path.is_symlink() or file_hash(path) != config['bootstrap_sha256']:
        raise ValueError('Wide AIME bootstrap file does not match its frozen identity')
    payload = json.loads(path.read_bytes())
    if (not isinstance(payload, dict) or set(payload) != {'schema_version', 'rows', 'failed_problem_ids', 'source'}
            or payload['schema_version'] != 1 or not isinstance(payload['rows'], list)
            or not isinstance(payload['failed_problem_ids'], list) or not isinstance(payload['source'], dict)
            or len(payload['rows']) != 228 or len(payload['failed_problem_ids']) != 28
            or not all(isinstance(ident, str) and ident for ident in payload['failed_problem_ids'])):
        raise ValueError('Wide AIME bootstrap requires 228 verified rows and 28 original non-admissions')
    return payload


def build_config(*, bundle, model, checkpoint_run, run, aggregate_sha256, checkpoint_sha256):
    from .controller import validate_config as validate_controller
    from .lab import validate_trial_cohorts
    from .aime_scale_probe import OPTIONS

    bundle, model, checkpoint_run = (Path(p).expanduser().resolve(strict=True)
                                     for p in (bundle, model, checkpoint_run))
    run = Path(run).expanduser().resolve()
    wide = {'bootstrap': str(bundle / 'bootstrap.json'), 'bootstrap_sha256': file_hash(bundle / 'bootstrap.json'),
            'checkpoint_run': str(checkpoint_run), 'aggregate_sha256': aggregate_sha256,
            'checkpoint_sha256': checkpoint_sha256}
    if file_hash(checkpoint_run / 'aggregate.json') != aggregate_sha256:
        raise ValueError('Completed source aggregate did not match its supplied digest')
    load_bootstrap(wide)
    model_options = {**{k: v for k, v in OPTIONS.items()
                        if k not in ('supervised_tokens_per_step', 'report_truncated_attempts')}, **TRAIN_OPTIONS}

    def lab(name, search, validation):
        path = bundle / name
        doc = json.loads(path.read_bytes())
        if (path.is_symlink() or set(doc.get('splits', {})) != {'train', search, validation}
                or set(doc) != {'schema_version', 'splits'}):
            raise ValueError('Wide AIME closed loop cannot mount a sealed or extra split')
        return {'engine': 'transformers', 'base_model': str(model), 'manifest': str(path),
                'split_roles': dict(train='train', search=search, validation=validation),
                'model_options': dict(model_options), 'aime': {'max_verified_rows': 288, 'data_policy': 'affine',
                      'max_completion_chars': aime.MAX_VERIFIED_COMPLETION_CHARS}, 'wide_loop': dict(wide)}

    parent = lab('manifest.json', 'search', 'validation')
    child = lab('meta-manifest.json', 'meta_search', 'meta_validation')
    controller = {'order': ['data', 'model', 'harness'], 'profiles': {'harness': 'compact'},
                  'steps': 3, 'warmup': 1, 'trial_steps': 1,
                  'campaign': {'protocol': 'aime', 'track': 'adaptive', 'terms': 2, 'routing': 'ordered', 'policy_seed': 23},
                  'meta_targets': ['data_policy', 'harness_policy', 'model_policy', 'schedule', 'controller_harness'],
                  'trial_budget': {'seconds': 5400, 'gpu_seconds': 17280, 'model_calls': 1500}, 'trial_lab': child}
    validate_controller(controller)
    validate_trial_cohorts(parent, child)
    resources = {'executor': 'local', 'pools': {'trainer': {'gpus': [0, 1, 2, 3]}},
                 'placement': {'math.trainer': {'pool': 'trainer'}}}
    for gpu in range(4):
        pool = f'rollout-{gpu}'
        resources['pools'][pool] = {'gpus': [gpu]}
        resources['placement'][f'math.rollout_{gpu}'] = {'pool': pool}
    return {'objective': 'Development-only MetaRSI from the verified 256-item MATH pool; no sealed cohort',
            'data_dir': str(run), 'max_decision_steps': 360,
            'budget': {'seconds': 28800, 'model_calls': 24000, 'gpu_seconds': 105000}, 'resources': resources,
            'framework': {'method': {'name': 'meta_rsi', 'config': {'controller': controller,
                          'labs': {'math': parent, 'reflect': {}}}}}}


def model_stage_config(config, source_run):
    """Fit and measure one joint Data+LoRA candidate without automatic promotion."""
    import copy
    source = Path(source_run).expanduser().absolute()
    run = Path(config['data_dir'])
    if source.is_symlink() or not source.is_dir() or source.resolve() == run.resolve() or run.exists():
        raise ValueError('Model stage requires a different new run directory')
    result = source / 'result.json'
    if result.is_symlink() or not result.is_file():
        raise ValueError('Model stage requires an owned terminal campaign result')
    config = copy.deepcopy(config)
    method = config['framework']['method']['config']
    parent = method['labs']['math']
    parent['wide_loop']['model_stage'] = {'source_run': str(source.resolve()), 'result_sha256': file_hash(result)}
    check_loop_config(parent, parent['wide_loop'])
    method['controller'] = {'order': ['model'], 'steps': 1, 'warmup': 0, 'profiles': {'model': 'detailed'},
                            'min_delta': 1., 'campaign': {'protocol': 'aime', 'track': 'adaptive',
                            'terms': 1, 'routing': 'ordered', 'policy_seed': 23}}
    config['objective'] = 'Development-only staged Model continuation; measure, do not promote'
    config['max_decision_steps'] = 35
    config['budget'] = {'seconds': 6400, 'model_calls': 1500, 'gpu_seconds': 23000}
    return config


def model_driven_config(config, *, paper_kernel=False):
    """Construct a new, development-only model-driven campaign; not a legacy resume."""
    from copy import deepcopy
    from .lab import validate_config as validate_lab
    from .model_controller import NAME, validate_model_config
    from .model_policy import seed_scheduler

    config = deepcopy(config)
    if Path(config['data_dir']).exists():
        raise ValueError('Model-driven validation requires a new run directory')
    config['objective'] = 'Validate target-driven MetaRSI mechanisms on historical development only'
    config['max_decision_steps'] = 100
    method = config['framework']['method']
    method['name'] = NAME
    options = method['config']
    old = options['controller']
    options['labs']['math']['model_driven'] = True
    options['controller'] = {'warmup': 0, 'steps': 4, 'campaign': old['campaign'], 'scheduler': seed_scheduler(
        old['order'], operator_budget=4, evidence='ledger',
        diagnostics=('candidate_validation' if paper_kernel else
                     'Probe observed training gaps. Prefer testing an eligible fresh Data→Model chain before spending the last slots.'),
        prefer_axis='balanced', original=paper_kernel)}
    if paper_kernel:
        options['labs']['math']['paper_kernel'] = {}
        options['controller']['paper_kernel'] = True
    validate_lab(options['labs']['math'])
    validate_model_config(options['controller'])
    return config


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'model', 'checkpoint-run', 'run', 'aggregate-sha256', 'checkpoint-sha256'):
        parser.add_argument('--' + key, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--model-stage-from', help='Opt in to a separate, non-promoting Model continuation')
    modes.add_argument('--model-driven', action='store_true', help='Use the target model for all proposal roles')
    modes.add_argument('--paper-kernel', action='store_true', help='Enable full staged evidence and protected paired release')
    args = vars(parser.parse_args())
    stage = args.pop('model_stage_from')
    model_driven = args.pop('model_driven')
    paper_kernel = args.pop('paper_kernel')
    config = build_config(**args)
    if stage is not None:
        config = model_stage_config(config, stage)
    elif model_driven or paper_kernel:
        config = model_driven_config(config, paper_kernel=paper_kernel)
    print(json.dumps(config, indent=2))


if __name__ == '__main__':
    main()
