"""Single-node, four-GPU DDP worker for an ordered, full-label wide-only fit.

Supports an opt-in uneven final wave for global batch size 16. Only rank zero
publishes options.output/adapter, options.output/training.json, and request.output.
"""

from contextlib import nullcontext
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
import time

from .engine import LoRAEngine, _QWEN35_MOE_TARGETS


WORLD_SIZE = 4
ACCUMULATED_BATCH_SIZE = 16
LABEL_SELECTION = 'full-causal-assistant-including-eos-v1'


def _global_batch_size(config):
    size = config.get('training_global_batch_size', WORLD_SIZE)
    if type(size) is not int or size not in (WORLD_SIZE, ACCUMULATED_BATCH_SIZE):
        raise ValueError(f'wide_ddp requires training_global_batch_size={WORLD_SIZE} or {ACCUMULATED_BATCH_SIZE}')
    return size


def _validate_request(request):
    if request['mode'] != 'train':
        raise ValueError('wide_ddp only accepts hosted training requests')
    config, options, rows = request['config'], request['options'], request['rows']
    if config.get('device', 'cuda') != 'cuda' or config.get('model_kind') != 'qwen3_5_moe':
        raise ValueError('wide_ddp requires the CUDA Qwen3.5 MoE training recipe')
    if config.get('ordered_training_coverage') is not True or 'supervised_tokens_per_step' in config:
        raise ValueError('wide_ddp requires ordered training coverage without a label quota')
    for key in ('train_batch_size', 'gradient_accumulation_steps'):
        value = config.get(key, 1)
        if type(value) is not int or value != 1:
            raise ValueError(f'wide_ddp requires {key}=1 per rank')
    if not isinstance(rows, list) or not rows:
        raise ValueError('wide_ddp requires a nonempty ordered list of training rows')
    allow_uneven_tail = config.get('training_allow_uneven_tail') is True
    if len(rows) % WORLD_SIZE and not allow_uneven_tail:
        raise ValueError('wide_ddp requires a multiple of four rows; no rows may be dropped or repeated')
    batch_size = _global_batch_size(config)
    if allow_uneven_tail and batch_size != ACCUMULATED_BATCH_SIZE:
        raise ValueError(f'wide_ddp requires training_global_batch_size={ACCUMULATED_BATCH_SIZE} '
                         'with training_allow_uneven_tail=True')
    steps = (len(rows) + batch_size - 1) // batch_size
    requested = options.get('steps')
    if requested is not None and (type(requested) is not int or requested != steps):
        raise ValueError(f'wide_ddp requires steps={steps} for exactly one pass over {len(rows)} rows')
    ids = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f'Training row {index} must be a mapping')
        for key in ('id', 'problem_id', 'completion'):
            value = row.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'Training row {index} requires a nonempty string {key}')
        if row['id'] in ids:
            raise ValueError(f'Training row {index} has a duplicate id')
        ids.add(row['id'])
    return steps


def _rank(environ):
    try:
        rank, local, world, local_world = (int(environ[key]) for key in
                                         ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE'))
    except (KeyError, ValueError) as error:
        raise ValueError('wide_ddp must run under single-node torchrun with four processes') from error
    if world != WORLD_SIZE or local_world != WORLD_SIZE or rank != local or not 0 <= rank < world:
        raise ValueError('wide_ddp must run under single-node torchrun with four processes')
    return rank


def _encode_rank_rows(engine, tokenizer, rows, rank):
    # Rank r gets index 4 * wave + r, with no sampler, padding, cycling, or shuffle.
    for index in range(rank, len(rows), WORLD_SIZE):
        ids, labels = engine._encode_training_row(tokenizer, rows[index])
        count = sum(label != -100 for label in labels[1:])
        if not count:
            raise ValueError(f'Training row {index + 1} has no causal assistant labels')
        yield index, ids, labels, count


def _coverage(rows, rank_reports, input_tokens):
    if len(rank_reports) != WORLD_SIZE or any(
            report['indices'] != list(range(rank, len(rows), WORLD_SIZE))
            for rank, report in enumerate(rank_reports)):
        raise RuntimeError('Distributed training did not consume each ordered row exactly once')
    indices = [index for report in rank_reports for index in report['indices']]
    return len(indices), {
        'distinct_problem_ids_seen': len({rows[index]['problem_id'] for index in indices}),
        'distinct_problem_ids_available': len({row['problem_id'] for row in rows}),
        'distinct_rollout_ids_seen': len({rows[index]['id'] for index in indices}),
        'distinct_rollout_ids_available': len({row['id'] for row in rows}),
        'input_tokens_seen': input_tokens,
    }


def _fit(request, expected_steps, rank, device, torch, dist):
    from peft import LoraConfig, get_peft_model
    from torch.nn.parallel import DistributedDataParallel as DDP

    config, rows, round_number = request['config'], request['rows'], request['round_number']
    batch_size = _global_batch_size(config)
    waves_per_step = batch_size // WORLD_SIZE
    total_waves = (len(rows) + WORLD_SIZE - 1) // WORLD_SIZE
    training_seed = config.get('seed', 73) + round_number
    torch.manual_seed(training_seed)
    engine = LoRAEngine(config)
    # Transformers uses an explicit device map rather than CUDA's current device.
    # Keep device='cuda' for BF16 and pass the rank for unambiguous weight placement.
    model, tokenizer = engine._load(request['ref'], train=True, cuda_index=rank)
    dataset = list(_encode_rank_rows(engine, tokenizer, rows, rank))
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                                           target_modules=_QWEN35_MOE_TARGETS, task_type='CAUSAL_LM'))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    parameters = list(model.parameters())
    if any(parameter.device != device for parameter in parameters):
        raise RuntimeError(f'Model parameters are not all on the local rank device {device}')
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if not trainable:
        raise RuntimeError('wide_ddp requires trainable attention adapters')
    # The frozen checkpoint is identical across ranks. Sync only the adapters to
    # avoid a DDP initialization broadcast of the full 35B-parameter base.
    for parameter in trainable:
        dist.broadcast(parameter.detach(), src=0)
    model = DDP(model, device_ids=[rank], output_device=rank, init_sync=False,
                broadcast_buffers=False, find_unused_parameters=False, gradient_as_bucket_view=True)
    learning_rate = config.get('learning_rate', 2e-4)
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    indices = []
    step = microsteps = examples_seen = input_tokens = completion_tokens = 0
    weighted_loss = 0.0
    started = time.monotonic()
    for step, start in enumerate(range(0, total_waves, waves_per_step), start=1):
        wave_count = min(waves_per_step, total_waves - start)
        group = dataset[start:start + wave_count]
        # Reduce counts for real rows only, before any backward in the step.
        counts = torch.tensor((len(group), sum(len(ids) for _, ids, _, _ in group),
                               sum(count for _, _, _, count in group)), device=device, dtype=torch.int64)
        dist.all_reduce(counts)
        step_examples, step_input, step_completion = counts.tolist()
        if step_examples != min(batch_size, len(rows) - start * WORLD_SIZE) or step_completion <= 0:
            raise RuntimeError(f'Invalid distributed training coverage at step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss_sum = None
        for wave in range(wave_count):
            final_wave = wave == wave_count - 1
            if wave < len(group):
                _, ids, labels, count = group[wave]
            else:
                # A synthetic, two-token sequence with one valid shifted
                # label keeps the model's loss differentiable on this rank.
                # It has zero weight and is absent from coverage and counts.
                ids = [tokenizer.eos_token_id] * 2
                labels, count = [-100, ids[1]], 0
            # DDP requires no_sync to enclose both forward and backward. The
            # actual final wave also synchronizes a short (even one-wave) tail.
            with nullcontext() if final_wave else model.no_sync():
                loss = model(input_ids=torch.tensor([ids], device=device),
                             attention_mask=torch.ones((1, len(ids)), device=device, dtype=torch.long),
                             labels=torch.tensor([labels], device=device), use_cache=False).loss
                detached = loss.detach().to(dtype=torch.float64)
                # A finite dummy loss contributes zero; a negative or
                # nonfinite loss on any rank must still fail collectively.
                wave_loss_sum = detached.masked_fill(detached < 0, float('nan')) * count
                loss_sum = wave_loss_sum if loss_sum is None else loss_sum + wave_loss_sum
                if final_wave:
                    # All ranks reach this check, even if an earlier local
                    # (unsynchronized) wave produced a bad loss. Do not enter
                    # the synchronized backward or step if any rank saw one.
                    dist.all_reduce(loss_sum)
                    step_loss_sum = loss_sum.item()
                    last_loss = step_loss_sum / step_completion
                    if not math.isfinite(last_loss):
                        raise ValueError(f'Invalid distributed training loss at step {step}')
                # HF averages over the row's shifted assistant labels. DDP
                # averages by world size, not by waves or by token count.
                (loss * (WORLD_SIZE * count / step_completion)).backward()
        if any(parameter.grad is None for parameter in trainable):
            raise RuntimeError('A trainable attention adapter did not participate in backward')
        torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        indices.extend(index for index, _, _, _ in group)
        microsteps += wave_count
        examples_seen += step_examples
        input_tokens += step_input
        completion_tokens += step_completion
        weighted_loss += step_loss_sum
        if rank == 0:
            print(json.dumps({'stage': 'train', 'round': round_number, 'epoch': 1,
                              'step': step, 'microsteps': microsteps, 'loss': last_loss,
                              'examples_seen': examples_seen, 'input_tokens_seen': input_tokens,
                              'completion_tokens_seen': completion_tokens,
                              'elapsed_seconds': time.monotonic() - started}, allow_nan=False),
                  file=sys.stderr, flush=True)

    # Gather actual completed row indices and adapter identity, not just the
    # planned coverage. No raw row identifiers or training text enter the receipt.
    digest = sha256()
    for parameter in trainable:
        digest.update(parameter.detach().float().cpu().numpy().tobytes())
    rank_reports = [None] * WORLD_SIZE
    dist.all_gather_object(rank_reports, {'indices': indices, 'digest': digest.hexdigest(),
                                        'peak': torch.cuda.max_memory_allocated(device)})
    examples, coverage = _coverage(rows, rank_reports, input_tokens)
    if (step != expected_steps or microsteps != total_waves or examples != examples_seen
            or len({report['digest'] for report in rank_reports}) != 1):
        raise RuntimeError('Distributed adapters did not finish with identical weights and step counts')
    metrics = {'steps': step, 'microsteps': microsteps, 'microsteps_unit': 'global_waves',
               'last_loss': last_loss, 'token_weighted_loss': weighted_loss / completion_tokens,
               'completion_tokens_seen': completion_tokens, 'samples': len(rows), 'examples_seen': examples,
               'world_size': WORLD_SIZE, 'effective_batch_size': batch_size,
               'final_batch_size': step_examples, 'learning_rate': learning_rate,
               'training_seed': training_seed,
               'loss_normalization': 'global_completion_tokens', 'supervision_label_selection': LABEL_SELECTION,
               'coverage': coverage, 'rank_weights_identical': True, 'trainable_weights_sha256': digest.hexdigest(),
               'peak_allocated_bytes': max(report['peak'] for report in rank_reports),
               'elapsed_seconds': time.monotonic() - started}
    if rank == 0:
        output = Path(request['options']['output'])
        model.module.save_pretrained(output / 'adapter', save_embedding_layers=False)
        (output / 'training.json').write_text(json.dumps(metrics, allow_nan=False), encoding='utf-8')
        # LoRAEngine.train returns None in the hosted protocol.
        Path(request['output']).write_text('null', encoding='utf-8')


def train_request(request):
    steps = _validate_request(request)
    rank = _rank(os.environ)
    output = Path(request['options']['output'])
    if any(path.exists() or path.is_symlink() for path in
           (output / 'adapter', output / 'training.json', Path(request['output']))):
        raise ValueError('wide_ddp requires fresh adapter, receipt, and worker output paths')

    import torch
    import torch.distributed as dist

    if torch.cuda.device_count() != WORLD_SIZE:
        raise ValueError('wide_ddp requires exactly four CUDA devices visible in the trainer lease')
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group('nccl', device_id=device)
    try:
        _fit(request, steps, rank, device, torch, dist)
        dist.barrier(device_ids=[rank])
    finally:
        dist.destroy_process_group()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit('usage: torchrun --standalone --nproc-per-node=4 --module '
                         'rsicomposer.backends.training.lora.wide_ddp REQUEST.json')
    train_request(json.loads(Path(argv[0]).read_text(encoding='utf-8')))


if __name__ == '__main__':
    main()
