"""A paired dev-only probe of larger original-question pools on the same MetaRSI backend."""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace
import re

from rsicomposer.api import file_hash, json_bytes
from rsicomposer.backends.training.lora.engine import LoRAEngine
from rsicomposer.labs.math.tasks import jsonl_problems
from rsicomposer.labs.math.verifiers import VERIFIERS
from rsicomposer.runtime.execution.budget import HostBudget
from rsicomposer.runtime.execution.executor import Executor
from rsicomposer.runtime.execution.resources import ResourceManager
from rsicomposer.runtime.storage.files import atomic_write

from .aime import MAX_TRAIN_TEXT, MAX_VERIFIED_COMPLETION_CHARS
from .engine import TransformersEngine, digest, fit_checkpoint, model_identity, model_stamp, read_checkpoint
from .lab import HARNESSES
from .prepare_aime import _identity


PROFILE = 'meta-rsi/aime-pool-scale-dev/v1'
MAX_PROBE_SECONDS = 21600
SMALL = 8
TRAIN_STEPS = 48
LABELS = 1024
MAX_COMPLETION_TOKENS = 12288
CHUNK = 64
BASE_SHA256 = 'ff8c14fc92a5a2ccc4f51047aab0912c166af986d9c055315da0eab329f3536b'
OPTIONS = {'device': 'cuda', 'model_kind': 'qwen3_5_moe', 'seed': 23, 'enable_thinking': True,
           'rollout_backend': 'vllm_graph', 'max_length': 65536, 'max_new_tokens': 32768,
           'generation_batch_size': 16, 'temperature': 0.7, 'report_truncated_attempts': True,
           'learning_rate': 2e-4, 'train_batch_size': 1, 'gradient_accumulation_steps': 1,
           'supervised_tokens_per_step': LABELS}
BASE = {'checkpoint': None}


def load_bundle(path):
    """Accept only a dedicated, checksummed train/dev directory, never a sealed manifest."""
    path = Path(path)
    manifest = path.read_bytes()
    data = json.loads(manifest)
    if (not isinstance(data, dict) or set(data) != {'schema_version', 'splits'}
            or data['schema_version'] != 1 or not isinstance(data['splits'], dict)
            or set(data['splits']) != {'train', 'dev'}):
        raise ValueError('Expected exactly train and dev, with no sealed role')
    entries = data['splits']
    problems = {}
    for role, (benchmark, topic, pattern) in {
        'train': ('math-counting-and-probability-train', 'counting_and_probability', r'math-counting-and-probability-train-[0-9]{4}'),
        'dev': ('aime-2022-2024-dev', 'mixed', r'aime-202[234]-i{1,2}-(?:0[1-9]|1[0-5])'),
    }.items():
        entry = entries[role]
        if (not isinstance(entry, dict) or set(entry) != {'file', 'sha256', 'count', 'benchmark', 'verifier', 'topics'}
                or entry['file'] != role + '.jsonl' or entry['benchmark'] != benchmark
                or entry['topics'] != {topic: entry['count']}
                or entry['verifier'] != {'name': 'competition_integer', 'maximum': 999}
                or (path.parent / entry['file']).is_symlink()):
            raise ValueError('Unexpected probe source or verifier')
        rows, _, _ = jsonl_problems(path.parent, {k: v for k, v in entry.items() if k != 'topics'}, VERIFIERS,
                                   budgets=False, max_output_tokens=OPTIONS['max_new_tokens'])
        if any(not re.fullmatch(pattern, p.id) for p in rows):
            raise ValueError('Invalid source identity for nonsealed role')
        problems[role] = rows
    all_rows = (*problems['train'], *problems['dev'])
    if len({p.id for p in all_rows}) != len(all_rows) or len({_identity(p.prompt) for p in all_rows}) != len(all_rows):
        raise ValueError('Train/dev prompts and identities must be distinct')
    n = len(problems['train'])
    if n < 128 or n % SMALL or len(problems['dev']) != 60:
        raise ValueError('Require 60 historical dev questions and at least 128 train questions, a multiple of 8')
    return problems, file_hash(path), {key: entries[key]['sha256'] for key in entries}


def prompts(problems):
    return [{'prompt': [{'role': 'system', 'content': HARNESSES['seed'] + ' ' + p.verifier.answer_format},
                        {'role': 'user', 'content': p.prompt}]} for p in problems]


def generations(engine, problems, *, round_number, sample, model=BASE):
    groups = engine.generate(prompts(problems), 1, model=model, harness=None, round_number=round_number,
                             sample=sample, temperature=OPTIONS['temperature'] if sample else 0.)
    if len(groups) != len(problems) or any(len(group) != 1 for group in groups):
        raise RuntimeError('Incomplete worker response')
    return [group[0] for group in groups]


def grade(problem, reply):
    if isinstance(reply, dict):
        if reply.get('status') != 'truncated' or reply.get('finish_reason') not in ('length', 'prompt_budget'):
            raise RuntimeError('Unexpected generation status')
        return reply['finish_reason']
    guess = problem.verifier.parse(reply)
    if guess is None:
        return 'invalid_format'
    return 'correct' if guess == int(problem.expected) else 'incorrect'


def rollout(engine, problems, round_number):
    replies = generations(engine, problems, round_number=round_number, sample=True)
    stats, accepted = Counter(), []
    for problem, prompt, reply in zip(problems, prompts(problems), replies, strict=True):
        outcome = grade(problem, reply)
        stats[outcome] += 1
        if outcome != 'correct':
            continue
        if len(reply) > MAX_VERIFIED_COMPLETION_CHARS or len(problem.prompt) > MAX_TRAIN_TEXT:
            stats['storage_reject'] += 1
            continue
        completion = reply.strip()
        row_id = digest([problem.id, completion])
        accepted.append({'id': row_id, 'problem_id': problem.id, 'prompt': prompt['prompt'], 'completion': completion})
    if len(replies) != sum(stats[k] for k in ('length', 'prompt_budget', 'invalid_format', 'correct', 'incorrect')):
        raise RuntimeError('Rollout accounting mismatch')
    return {'stats': dict(stats), 'rows': accepted, 'attempts': len(replies)}


def select_training_rows(rows, token_counts, *, minimum=LABELS, maximum=MAX_COMPLETION_TOKENS, steps=TRAIN_STEPS):
    """Sweep one distinct verified rollout per problem before using a second."""
    by_problem = defaultdict(list)
    totals, seen = Counter(), set()
    for row, size in zip(rows, token_counts, strict=True):
        if row['id'] in seen:
            totals['duplicate_rollouts'] += 1
            continue
        seen.add(row['id'])
        if not minimum <= size <= maximum:
            totals['outside_label_length_window'] += 1
            continue
        by_problem[row['problem_id']].append(row)
    selected = []
    for index in range(max(map(len, by_problem.values()), default=0)):
        for group in by_problem.values():
            if index < len(group):
                selected.append(group[index])
                if len(selected) == steps:
                    return selected, {**totals, 'eligible_distinct_problems': len(by_problem),
                                      'eligible_rollouts': sum(map(len, by_problem.values()))}
    raise ValueError('Too few unique verified rollouts to satisfy the fixed training budget')


def stage(cache, name, key, fn):
    file = cache / f'{name}.json'
    if file.exists():
        record = json.loads(file.read_bytes())
        if record['key'] != key or digest(record['value']) != record['value_sha256']:
            raise ValueError('Owned stage receipt changed: ' + name)
        return record['value']
    print('START ' + name, flush=True)
    value = fn()
    atomic_write(file, json_bytes({'key': key, 'value': value, 'value_sha256': digest(value)}))
    print('FINISH ' + name, flush=True)
    return value


def evaluate(engine, problems, adapter=None):
    replies = generations(engine, problems, round_number=0, sample=False, model={'checkpoint': adapter})
    outcomes = [grade(p, r) for p, r in zip(problems, replies, strict=True)]
    return {'count': len(problems), 'correct': outcomes.count('correct'),
            'truncated': outcomes.count('length') + outcomes.count('prompt_budget'),
            'invalid_format': outcomes.count('invalid_format'),
            'correctness': [outcome == 'correct' for outcome in outcomes]}


def score_public(report):
    return {**{k: v for k, v in report.items() if k != 'correctness'}, 'pass_at_1': report['correct'] / report['count']}


def paired(a, b):
    outcomes = list(zip(a['correctness'], b['correctness'], strict=True))
    return {'gains': sum(not x and y for x, y in outcomes), 'losses': sum(x and not y for x, y in outcomes),
            'delta_pp': 100 * (b['correct'] - a['correct']) / len(outcomes)}


def select_all_distinct(rows, token_counts):
    """Return the first verified complete rollout per problem in cohort order."""
    found, selected, tokens = set(), [], 0
    for row, count in zip(rows, token_counts, strict=True):
        if type(count) is not int or count < 1:
            raise ValueError('Expected positive assistant labels in every verified rollout')
        if row['problem_id'] not in found:
            found.add(row['problem_id'])
            selected.append(row)
            tokens += count
    return selected, tokens


def reuse_rollout_receipts(source, plan):
    """Rebind completed original rollouts only; no other stage may migrate."""
    previous = json.loads((source / 'plan.json').read_bytes())
    previous_budget = json.loads((source / 'host_execution' / 'budget.json').read_bytes())
    keys = ('dev_only', 'seed', 'mode', 'manifest_sha256', 'split_sha256', 'base_sha256',
            'wide_unique_questions', 'rollouts_per_arm', 'worker_chunk')
    options = {k: v for k, v in plan['options'].items()
               if k not in ('training_world_size', 'training_global_batch_size')}
    if (previous.get('mode') != 'wide_only' or any(previous.get(k) != plan.get(k) for k in keys)
            or previous['options'] != options or previous_budget['leases']
            or (source / 'aggregate.json').exists()):
        raise ValueError('Cannot migrate an incompatible, active, or already scored source rollout')
    previous_key = digest(previous)
    names = [f'wide-rollout-{i:02d}' for i in range((plan['rollouts_per_arm'] + CHUNK - 1) // CHUNK)]
    receipts = {}
    for name in names:
        record = json.loads((source / 'owned' / f'{name}.json').read_bytes())
        if (set(record) != {'key', 'value', 'value_sha256'} or record['key'] != previous_key
                or record['value_sha256'] != digest(record['value'])):
            raise ValueError('Source rollout receipt did not match its frozen identity')
        receipts[name] = record
    return previous_key, receipts


def bucket_waves(rows, assistant_lengths, global_batch_size):
    """Reduce rank stragglers without moving examples across optimizer batches."""
    examples = list(zip(rows, assistant_lengths, strict=True))
    return [row for i in range(0, len(examples), global_batch_size)
            for row, _ in sorted(examples[i:i + global_batch_size], key=lambda pair: pair[1], reverse=True)]


def run(*, manifest, model, output, gpu, wide_only=False, gpus=None, rollout_source=None, global_batch_size=None):
    from .aime_scale_receipt import check_frozen_code, verify_training_receipt
    devices = tuple(gpus) if gpus else (gpu,)
    world = len(devices)
    if world not in (1, 4) or len(set(devices)) != world or any(type(d) is not int or d < 0 for d in devices):
        raise ValueError('Specify one GPU or four distinct physical GPU indices')
    if world > 1 and (not wide_only or rollout_source is None or devices != (0, 1, 2, 3)):
        raise ValueError('Four-card mode is only for the wide-only run with source rollouts on authorized GPUs 0–3')
    if rollout_source is not None and world == 1:
        raise ValueError('Rollout migration is only for the four-card wide-only run')
    if global_batch_size is not None and (world != 4 or global_batch_size != 16):
        raise ValueError('The explicit global batch size 16 requires the four-card wide-only run')
    global_batch = world if global_batch_size is None else global_batch_size
    options = (dict(OPTIONS, ordered_training_coverage=True) if wide_only else OPTIONS)
    if wide_only:
        options.pop('supervised_tokens_per_step')
    if world > 1:
        options['training_world_size'] = world
    if global_batch_size is not None:
        options['training_global_batch_size'] = global_batch
    model = Path(model).resolve(strict=True)
    output = Path(output).resolve()
    data, manifest_id, splits = load_bundle(manifest)
    train, dev = data['train'], data['dev']
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (output / '.lock').open('a+b') as lock, ExitStack() as stack:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stamp = model_stamp(model)
        plan = {'profile': PROFILE, 'dev_only': True, 'seed': 23, 'manifest_sha256': manifest_id,
                'split_sha256': splits, 'base_sha256': BASE_SHA256, 'code_sha256': check_frozen_code(),
                'small_unique_questions': SMALL, 'wide_unique_questions': len(train), 'rollouts_per_arm': len(train),
                'selection': 'distinct-first, then next distinct rollout per problem; no synthesis',
                'training_steps': TRAIN_STEPS, 'label_window_max': MAX_COMPLETION_TOKENS,
                'worker_chunk': CHUNK, 'options': OPTIONS, 'physical_gpu': gpu, 'dev_count': len(dev),
                'budget': {'seconds': MAX_PROBE_SECONDS, 'gpu_seconds': MAX_PROBE_SECONDS}}
        if wide_only:
            plan.update(mode='wide_only', small_unique_questions=None,
                        selection='one full verified rollout per distinct question; no synthesis, no length masking',
                        training_steps='one per eligible question', minimum_training_steps=TRAIN_STEPS,
                        maximum_training_steps=len(train), label_window_max=None, options=options)
        reused = {}
        if world > 1:
            plan.pop('physical_gpu')
            plan.update(physical_gpus=list(devices), training_world_size=world,
                        training_steps='one global update for each four distinct original questions',
                        minimum_training_steps=(TRAIN_STEPS + world - 1) // world,
                        maximum_training_steps=len(train) // world,
                        dev_parallel_shards=world)
            if global_batch_size is not None:
                plan.update(training_global_batch_size=global_batch,
                            training_steps='one global update per batch of up to 16 distinct original questions',
                            minimum_training_steps=(TRAIN_STEPS + global_batch - 1) // global_batch,
                            maximum_training_steps=(len(train) + global_batch - 1) // global_batch,
                            training_wave_order='stable descending assistant token length within each optimizer batch',
                            per_rank_microbatch=1, accumulation_waves=global_batch // world)
            source_key, reused = reuse_rollout_receipts(Path(rollout_source), plan)
            plan.update(source_rollout_plan_sha256=source_key,
                        reused_rollout_value_sha256={name: item['value_sha256'] for name, item in reused.items()})
        path = output / 'plan.json'
        if path.exists() and json.loads(path.read_bytes()) != plan:
            raise ValueError('Cannot change a frozen probe plan')
        if not path.exists():
            atomic_write(path, json_bytes(plan))
        cache = output / 'owned'
        cache.mkdir(exist_ok=True)
        stamp_key = digest([str(model), stamp])
        evidence = stage(cache, 'base-verified', stamp_key, lambda: model_identity(model))
        if evidence != BASE_SHA256 or model_stamp(model) != stamp:
            raise ValueError('Probe must use the previously pinned model bytes')
        budget_root = output / 'host_execution'
        budget = HostBudget(budget_root, plan['budget'], restoring=(budget_root / 'budget.json').is_file())
        pools = {'aime-scale': {'gpus': list(devices)}}
        placement = {f'math.{role}': {'pool': 'aime-scale'} for role in ('rollout', 'trainer')}
        if world > 1:
            for device in devices:
                name = f'dev-{device}'
                pools[name] = {'gpus': [device]}
                placement[f'{name}.rollout'] = {'pool': name}
        resources = ResourceManager({'executor': 'local', 'pools': pools, 'placement': placement}, budget=budget)
        stack.callback(resources.close)
        executor = Executor(budget_root / 'workers', resources, budget)
        stack.callback(executor.close)
        engine = TransformersEngine(SimpleNamespace(resources=resources.component('math'), executor=executor), model, options)
        dev_engines = []
        if world > 1:
            inference_options = {k: v for k, v in options.items()
                                 if k not in ('training_world_size', 'training_global_batch_size')}
            dev_engines = [TransformersEngine(SimpleNamespace(resources=resources.component(f'dev-{device}'),
                           executor=executor), model, inference_options) for device in devices]
        resources.validate_roles()
        key = digest(plan)
        selected, rollout_evidence, exact_tokens = {}, {}, {}
        arms = [('wide', train)] if wide_only else [('small', train[:SMALL] * (len(train) // SMALL)), ('wide', train)]
        for arm, queries in arms:
            total, rows = Counter(), []
            for i in range(0, len(queries), CHUNK):
                group = queries[i:i + CHUNK]
                name = f'{arm}-rollout-{i // CHUNK:02d}'
                r = stage(cache, name, key, lambda: reused[name]['value'] if world > 1
                          else rollout(engine, group, 1 + i // CHUNK))
                if world > 1 and digest(r) != reused[name]['value_sha256']:
                    raise ValueError('Reused rollout stage changed after migration')
                total.update(r['stats'])
                rows.extend(r['rows'])
                if r['attempts'] != len(group):
                    raise RuntimeError('Unequal rollout accounting')
            counts = LoRAEngine(options).count_supervised_tokens(model, rows)
            if wide_only:
                subset, exact_tokens[arm] = select_all_distinct(rows, counts)
                stats = {'eligible_distinct_problems': len(subset), 'eligible_rollouts': len(rows)}
                if len(subset) < TRAIN_STEPS or len(subset) > len(train) or len(subset) % world:
                    raise RuntimeError('Wide pool did not supply the predetermined minimum verified coverage')
                if global_batch_size is not None:
                    lengths = {row['id']: length for row, length in zip(rows, counts, strict=True)}
                    subset = bucket_waves(subset, [lengths[row['id']] for row in subset], global_batch)
            else:
                subset, stats = select_training_rows(rows, counts)
                distinct = len({r['problem_id'] for r in subset})
                if distinct != (SMALL if arm == 'small' else TRAIN_STEPS):
                    raise RuntimeError(f'{arm} did not satisfy the predetermined distinct-question design')
            selected[arm] = subset
            rollout_evidence[arm] = {'attempts': len(queries), 'distinct_questions': len({p.id for p in queries}),
                                     'distinct_verified_questions': len({r['problem_id'] for r in rows}),
                                     'admitted_rollouts_before_dedup': len(rows), 'outcomes': dict(total), **stats}
            print('COLLECTION ' + arm + ' ' + json.dumps(rollout_evidence[arm]), flush=True)
        if model_stamp(model) != stamp:
            raise ValueError('Base model changed after rollout')
        adapter_root = output / 'models'
        fits, scores = {}, {}
        for arm, rows in selected.items():
            steps = (len(rows) + global_batch - 1) // global_batch if wide_only else TRAIN_STEPS
            recipe = {'profile': PROFILE, 'options': options, 'base_sha256': BASE_SHA256, 'steps': steps,
                      'round': 0, 'arm': arm, 'plan_sha256': key, 'label_scope': 'assistant_only',
                      'training_rows_sha256': digest(rows)}
            checkpoint = fit_checkpoint(adapter_root, engine, recipe, rows)['checkpoint']
            payload = read_checkpoint(adapter_root, checkpoint, simulated=False)
            if payload['recipe'] != recipe or payload['steps'] != steps or payload['samples'] != len(rows):
                raise ValueError('Published checkpoint does not match the probe')
            fits[arm] = verify_training_receipt(adapter_root / checkpoint / 'training.json', rows, checkpoint,
                                                steps=steps, labels=None if wide_only else LABELS,
                                                full_token_count=exact_tokens.get(arm), world_size=world,
                                                global_batch_size=global_batch)
            print('FIT ' + arm + ' ' + json.dumps(fits[arm]), flush=True)
        if model_stamp(model) != stamp:
            raise ValueError('Base model changed before scored evaluations')
        if not wide_only:
            scores['base'] = stage(cache, 'dev-base', key, lambda: evaluate(engine, dev))
        for arm, fit in fits.items():
            ckpt = fit['checkpoint']
            # Bind cached readout to the exact verified adapter manifest and artifact identity.
            adapter_manifest = read_checkpoint(adapter_root, ckpt, simulated=False)
            adapter = adapter_root / ckpt / 'adapter'
            if world > 1:
                from .aime_scale_parallel import evaluate_parallel
                scores[arm] = stage(cache, 'dev-' + arm, digest([key, adapter_manifest]),
                                    lambda: evaluate_parallel(dev_engines, dev, adapter))
            else:
                scores[arm] = stage(cache, 'dev-' + arm, digest([key, adapter_manifest]),
                                    lambda: evaluate(engine, dev, adapter))
        if model_stamp(model) != stamp:
            raise ValueError('Base model changed during scored evaluations')
        result = {'plan': plan, 'plan_sha256': key, 'rollouts': rollout_evidence, 'training': fits,
                  'historical_aime_dev': {k: score_public(v) for k, v in scores.items()}}
        if not wide_only:
            result['paired_dev'] = {f'{b}_vs_{a}': paired(scores[a], scores[b])
                                    for a, b in [('base', 'small'), ('base', 'wide'), ('small', 'wide')]}
        atomic_write(output / 'aggregate.json', json_bytes(result))
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--wide-only', action='store_true', help='fit and score the large pool only, no control arms')
    parser.add_argument('--gpus', type=int, nargs=4, metavar='N', help='physical GPU indices 0 1 2 3 for wide DDP')
    parser.add_argument('--rollouts-from', type=Path, help='verified existing wide-only run directory')
    parser.add_argument('--global-batch-size', type=int, choices=[16],
                        help='use four DDP waves per global update; keep a final smaller batch')
    args = parser.parse_args(argv)
    if args.check_only:
        data, sha, _ = load_bundle(args.manifest)
        print(json.dumps({'dev_only': True, 'manifest_sha256': sha, 'counts': {k: len(v) for k, v in data.items()}}))
        return
    if args.model is None or args.output is None or args.gpu < 0:
        parser.error('--model and --output required and --gpu must be non-negative')
    run(manifest=args.manifest, model=args.model, output=args.output, gpu=args.gpu, wide_only=args.wide_only,
        gpus=args.gpus, rollout_source=args.rollouts_from, global_batch_size=args.global_batch_size)


if __name__ == '__main__':
    main()
