"""Admit the already-scored wide AIME adapter without a new optimizer run."""
from hashlib import sha256
import json
from pathlib import Path
import re

from rsicomposer.backends.training.lora.engine import LoRAEngine
from rsicomposer.labs.math.verifiers import CompetitionIntegerVerifier

from . import aime, aime_scale_probe as probe
from .aime_scale_receipt import verify_training_receipt
from .engine import digest, inherit_checkpoint, read_checkpoint
from .lab import HARNESSES


_ROWS, _TOKENS, _STEPS = 228, 2_760_753, 15
_WORLD, _BATCH, _QUESTIONS = 4, 16, 256
_ROLLOUT_NAMES = tuple(f'wide-rollout-{i:02d}' for i in range(_QUESTIONS // probe.CHUNK))


def _checked_json(path, expected_sha256):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Expected a regular wide AIME evidence file: {path.name}')
    raw = path.read_bytes()
    if not aime.sha256_text(expected_sha256) or sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f'Wide AIME evidence SHA-256 changed: {path.name}')
    record = json.loads(raw)
    if not isinstance(record, dict):
        raise ValueError(f'Invalid wide AIME evidence record: {path.name}')
    return record


def _checked_plan(source, base_digest, evidence):
    if not isinstance(evidence, dict):
        raise ValueError('Wide AIME checkpoint requires pinned source evidence')
    plan = _checked_json(source / 'plan.json', evidence.get('plan_file_sha256'))
    split = evidence.get('split_sha256')
    if (base_digest != probe.BASE_SHA256 or evidence.get('base_sha256') != base_digest
            or not aime.sha256_text(evidence.get('manifest_sha256'))
            or not isinstance(split, dict) or split.keys() != {'train', 'dev'}
            or not all(map(aime.sha256_text, split.values()))
            or not aime.sha256_text(evidence.get('upstream_rollout_plan_sha256'))
            or not aime.sha256_text(plan.get('code_sha256'))):
        raise ValueError('Wide AIME checkpoint requires the pinned base and train/dev rollout source')
    options = dict(probe.OPTIONS, ordered_training_coverage=True, training_world_size=_WORLD,
                   training_global_batch_size=_BATCH)
    options.pop('supervised_tokens_per_step')
    expected = {'profile': probe.PROFILE, 'dev_only': True, 'mode': 'wide_only', 'seed': 23,
                'base_sha256': base_digest, 'manifest_sha256': evidence['manifest_sha256'], 'split_sha256': split,
                'source_rollout_plan_sha256': evidence['upstream_rollout_plan_sha256'],
                'physical_gpus': [0, 1, 2, 3], 'training_world_size': _WORLD, 'training_global_batch_size': _BATCH,
                'dev_parallel_shards': _WORLD, 'dev_count': 60, 'wide_unique_questions': _QUESTIONS,
                'rollouts_per_arm': _QUESTIONS, 'worker_chunk': probe.CHUNK, 'options': options,
                'training_wave_order': 'stable descending assistant token length within each optimizer batch',
                'per_rank_microbatch': 1, 'accumulation_waves': _BATCH // _WORLD,
                'selection': 'one full verified rollout per distinct question; no synthesis, no length masking'}
    # Compare canonical JSON rather than treating bool and int as interchangeable.
    plan_key = digest(plan)
    if (plan_key != evidence.get('plan_sha256') or 'physical_gpu' in plan
            or digest({key: plan.get(key) for key in expected}) != digest(expected)):
        raise ValueError('Wide AIME source is not the pinned seed-23 four-GPU global-batch-16 plan')
    return plan, plan_key, options


def _checked_rows(source, plan, plan_key, evidence, training_rows):
    receipts, values = evidence.get('receipts'), plan.get('reused_rollout_value_sha256')
    if (not isinstance(receipts, dict) or receipts.keys() != set(_ROLLOUT_NAMES)
            or not isinstance(values, dict) or values.keys() != receipts.keys()):
        raise ValueError('Wide AIME source requires all four frozen rollout receipts')
    root = source / 'owned'
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Wide AIME rollout evidence directory cannot be a symlink')
    original = []
    for name in _ROLLOUT_NAMES:
        pin = receipts[name]
        if (not isinstance(pin, dict) or pin.keys() != {'file_sha256', 'value_sha256'}
                or not aime.sha256_text(values[name]) or pin['value_sha256'] != values[name]):
            raise ValueError('Wide AIME source rollout digest disagrees with the plan')
        receipt = _checked_json(root / f'{name}.json', pin['file_sha256'])
        value = receipt.get('value')
        if (receipt.keys() != {'key', 'value', 'value_sha256'} or receipt['key'] != plan_key
                or receipt['value_sha256'] != values[name] or digest(value) != values[name]
                or not isinstance(value, dict) or value.get('attempts') != probe.CHUNK
                or not isinstance(value.get('rows'), list)):
            raise ValueError('Wide AIME rollout receipt did not match its frozen identity')
        original.extend(value['rows'])
    if not isinstance(training_rows, (list, tuple)) or len(original) != _ROWS or list(training_rows) != original:
        raise ValueError('Wide AIME bootstrap rows do not match the 228 original rollouts in source order')

    system = {'role': 'system', 'content': HARNESSES['seed'] + ' ' + CompetitionIntegerVerifier(999).answer_format}
    problem_ids = set()
    for row in original:
        if not isinstance(row, dict) or row.keys() != {'id', 'problem_id', 'prompt', 'completion'}:
            raise ValueError('Wide AIME requires exact scale probe training rows')
        ident, prompt, completion = row['problem_id'], row['prompt'], row['completion']
        if (not isinstance(ident, str) or not re.fullmatch(r'math-counting-and-probability-train-[0-9]{4}', ident)
                or ident in problem_ids or not isinstance(completion, str) or completion != completion.strip()
                or not 0 < len(completion) <= aime.MAX_VERIFIED_COMPLETION_CHARS
                or row['id'] != digest([ident, completion]) or not isinstance(prompt, list) or len(prompt) != 2
                or prompt[0] != system or not isinstance(prompt[1], dict) or prompt[1].keys() != {'role', 'content'}
                or prompt[1]['role'] != 'user' or not isinstance(prompt[1]['content'], str)
                or not 0 < len(prompt[1]['content']) <= aime.MAX_TRAIN_TEXT):
            raise ValueError('Wide AIME training row does not match the seed verifier format or original identity')
        problem_ids.add(ident)
    return original


def _wide_only(value):
    if not isinstance(value, dict) or value.keys() != {'wide'} or not isinstance(value['wide'], dict):
        raise ValueError('Expected one completed wide AIME arm')
    return value['wide']


def _checked_aggregate(source, plan_key, evidence):
    aggregate = _checked_json(source / 'aggregate.json', evidence.get('aggregate_sha256'))
    if (aggregate.keys() != {'plan', 'plan_sha256', 'rollouts', 'training', 'historical_aime_dev'}
            or aggregate['plan_sha256'] != plan_key or digest(aggregate['plan']) != plan_key):
        raise ValueError('Wide AIME aggregate is not bound to the original probe plan')
    outcomes = _wide_only(aggregate['rollouts'])
    counts = {'attempts': _QUESTIONS, 'distinct_questions': _QUESTIONS, 'distinct_verified_questions': _ROWS,
              'admitted_rollouts_before_dedup': _ROWS, 'eligible_distinct_problems': _ROWS, 'eligible_rollouts': _ROWS}
    score = _wide_only(aggregate['historical_aime_dev'])
    invalid = score.get('invalid_format')
    if (any(type(outcomes.get(key)) is not int or outcomes[key] != count for key, count in counts.items())
            or outcomes.get('outcomes') != {'correct': _ROWS, 'length': 27, 'invalid_format': 1}
            or score.keys() != {'count', 'correct', 'truncated', 'invalid_format', 'pass_at_1'}
            or score['count'] != 60 or score['correct'] != 49 or score['truncated'] != 10
            or type(invalid) is not int or not 0 <= invalid <= 1 or score['pass_at_1'] != 49 / 60):
        raise ValueError('Wide AIME source is not the completed 228-rollout, 49/60 development result')
    return _wide_only(aggregate['training'])


def _verified_source(source, model_path, base_digest, training_rows, source_evidence):
    if source.is_symlink() or source.parent.is_symlink() or not source.is_dir():
        raise ValueError('Wide AIME bootstrap requires a regular local probe run directory')
    plan, plan_key, options = _checked_plan(source, base_digest, source_evidence)
    fit = _checked_aggregate(source, plan_key, source_evidence)
    checkpoint = fit.get('checkpoint')
    if not aime.sha256_text(checkpoint) or not aime.sha256_text(source_evidence.get('checkpoint_sha256')):
        raise ValueError('Wide AIME bootstrap requires a pinned scored adapter identity')
    payload = read_checkpoint(source / 'models', checkpoint, simulated=False)
    if digest(payload) != source_evidence['checkpoint_sha256']:
        raise ValueError('Wide AIME source checkpoint does not match the pinned scored adapter')
    original = _checked_rows(source, plan, plan_key, source_evidence, training_rows)
    counts = LoRAEngine(options).count_supervised_tokens(model_path, original)
    if len(counts) != _ROWS or any(type(count) is not int or count < 1 for count in counts) or sum(counts) != _TOKENS:
        raise ValueError('Wide AIME bootstrap does not reproduce the exact scored run assistant labels')
    ordered = probe.bucket_waves(original, counts, _BATCH)
    recipe = {'profile': probe.PROFILE, 'options': options, 'base_sha256': base_digest, 'steps': _STEPS,
              'round': 0, 'arm': 'wide', 'plan_sha256': plan_key, 'label_scope': 'assistant_only',
              'training_rows_sha256': digest(ordered)}
    if checkpoint != digest(recipe) or payload['recipe'] != recipe or payload['samples'] != _ROWS or payload['steps'] != _STEPS:
        raise ValueError('Wide AIME checkpoint did not use the original length-bucketed training recipe')
    verified_fit = verify_training_receipt(source / 'models' / checkpoint / 'training.json', ordered, checkpoint,
                                           steps=_STEPS, full_token_count=sum(counts), world_size=_WORLD,
                                           global_batch_size=_BATCH)
    if digest(verified_fit) != digest(fit):
        raise ValueError('Wide AIME optimizer receipt disagrees with the scored aggregate')
    return {'checkpoint': checkpoint, 'checkpoint_sha256': source_evidence['checkpoint_sha256']}, payload


def _assert_scored_payload(actual, original):
    if digest(actual) != digest(original):
        raise ValueError('Inherited wide AIME adapter must keep the exact already-scored checkpoint identity')


def verify_bootstrap_recipe(payload, source, model_path, base_digest, training_rows, source_evidence):
    """Verify an inherited manifest against the same scored source; return its model reference.

    Supply the 228 original rows as Lab._training_rows formats them, in bootstrap
    cohort order, excluding any new rows the child added. The caller must use
    read_checkpoint to verify adapter bytes at the inherited location.
    """
    model, original = _verified_source(Path(source), model_path, base_digest, training_rows, source_evidence)
    _assert_scored_payload(payload, original)
    return model


def bootstrap_checkpoint(source, target, model_path, base_digest, training_rows, source_evidence):
    """Copy the already-scored checkpoint to a Lab model root; never fit or score.

    Source is the complete scored scale-probe run directory, not its models
    subdirectory. Evidence is the frozen rollout bundle source plus independent
    aggregate_sha256 and checkpoint_sha256 pins. Only training artifacts and
    aggregate metadata are read; no development rows or per-item scoring records.
    """
    source = Path(source)
    model, original = _verified_source(source, model_path, base_digest, training_rows, source_evidence)
    inherit_checkpoint(source / 'models', Path(target), model, simulated=False,
                       validate=lambda actual: _assert_scored_payload(actual, original))
    return model
