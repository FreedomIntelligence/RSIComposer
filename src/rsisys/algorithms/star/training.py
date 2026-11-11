"""Four-rank Qwen3.5 full-text FSDP2 execution for one STaR fit."""
import fcntl
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from . import checkpoint
from .protocol import render_prompt
from .records import ACCUMULATION, GLOBAL_BATCH, WORLD_SIZE, load_records, training_example


def verifier_for(name):
    from rsisys.labs.scenes.math.verifiers import MathVerifyVerifier, StrictNumericVerifier

    if name == 'numeric':
        return StrictNumericVerifier()
    try:
        return MathVerifyVerifier()
    except ImportError as exc:
        raise RuntimeError('math_verify requires Math-Verify in the training environment') from exc


def build_datasets(args, tokenizer, verifier, *, context_limit):
    train = load_records(args.train_jsonl, 'completion', unique_prompts=False)
    dev = load_records(args.validation_jsonl, 'answer')
    if {prompt for prompt, _ in train} & {prompt for prompt, _ in dev}:
        raise ValueError('Training and held-out validation share a prompt')
    if len(train) < GLOBAL_BATCH:
        raise ValueError('Training requires at least eight accepted completions')

    def render(prompt):
        if args.qwen_math_chat:
            return render_prompt(tokenizer, prompt)
        return prompt

    train_examples = []
    for index, (prompt, completion) in enumerate(train, 1):
        try:
            train_examples.append(training_example(tokenizer, render(prompt), completion,
                                                   max_tokens=args.max_length))
        except ValueError as exc:
            raise ValueError(f'Training record {index}: {exc}') from exc
    dev_examples = []
    for index, (prompt, answer) in enumerate(dev, 1):
        if verifier.parse(answer) is None:
            raise ValueError(f'Validation record {index}: answer is invalid for {verifier.name}')
        ids = tokenizer.encode(render(prompt), add_special_tokens=False)
        if not ids or len(ids) + args.validation_max_new_tokens > context_limit:
            raise ValueError(f'Validation record {index}: prompt plus generation exceeds model context')
        dev_examples.append((ids, answer))
    return train_examples, dev_examples


def train_batch(model, optimizer, batch, *, rank, device):
    import torch
    import torch.distributed as dist
    from torch.nn.functional import cross_entropy

    optimizer.zero_grad(set_to_none=True)
    total_tokens = sum(example.supervised_tokens for example in batch)
    local_loss = torch.zeros((), device=device)
    for micro in range(ACCUMULATION):
        example = batch[micro * WORLD_SIZE + rank]
        # Keep the hidden states that predict each completion token, including
        # the last prompt position, and omit the unused final hidden state.
        ids = torch.tensor([example.input_ids[:-1]], dtype=torch.long, device=device)
        targets = torch.tensor(example.labels[-example.supervised_tokens:], dtype=torch.long,
                               device=device)
        logits = model(input_ids=ids, use_cache=False,
                       logits_to_keep=example.supervised_tokens).logits[0]
        loss = cross_entropy(logits.float(), targets, reduction='sum')
        local_loss += loss.detach()
        # FSDP2 averages reduced gradients. This makes the final update a
        # token mean over all eight completions, with sharded accumulation.
        (loss * (WORLD_SIZE / total_tokens)).backward()
        del logits, loss
    dist.all_reduce(local_loss)
    mean_loss = float(local_loss.item()) / total_tokens
    if not math.isfinite(mean_loss):
        raise RuntimeError('Non-finite completion loss')
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return mean_loss


def _publish(output, snapshot, result):
    receipt = output / 'training_result.json'
    partial = snapshot.parent / receipt.name
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(snapshot, output / 'checkpoint')
    os.replace(partial, receipt)


def run(args, *, fit):
    """Initialize the shared four-rank backend and run the selected fit."""
    rank = int(os.environ.get('RANK', '-1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '-1'))
    if (int(os.environ.get('WORLD_SIZE', '0')) != WORLD_SIZE
            or int(os.environ.get('LOCAL_WORLD_SIZE', '0')) != WORLD_SIZE
            or rank != local_rank or not 0 <= rank < WORLD_SIZE):
        raise RuntimeError('Launch with torchrun on one node with exactly four processes')
    # Verify the selected scorer and record format before checking CUDA or
    # materializing checkpoint parameters.
    verifier = verifier_for(args.verifier)
    import torch
    import torch.distributed as dist
    from transformers import AutoConfig, AutoTokenizer

    base = args.base_model.resolve(strict=True)
    output = args.output_dir.resolve()
    shards, weight_map = checkpoint.shard_index(base)
    config = AutoConfig.from_pretrained(base, local_files_only=True)
    if config.model_type != 'qwen3_5' or config.tie_word_embeddings:
        raise ValueError('Expected the untied Qwen3.5 conditional-generation checkpoint')
    if args.max_length > config.text_config.max_position_embeddings:
        raise ValueError('Training max length exceeds the model context')
    config._attn_implementation = 'sdpa'
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    train, dev = build_datasets(args, tokenizer, verifier,
                                context_limit=config.text_config.max_position_embeddings)
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError('The worker requires exactly four visible CUDA GPUs')
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('The worker requires native bfloat16 GPU support')
    device = torch.device('cuda', local_rank)
    torch.manual_seed(args.seed)
    dist.init_process_group(backend='nccl', device_id=device)
    try:
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            with (output / '.training.lock').open('ab') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if any((output / name).exists() for name in ('checkpoint', 'training_result.json')):
                    raise FileExistsError(f'Training output already exists: {output}')
                with TemporaryDirectory(prefix='.staging-', dir=output) as staging:
                    fit(args, base, config, shards, weight_map, tokenizer, verifier, train, dev,
                         rank=rank, device=device, output=output, staging=Path(staging))
        else:
            fit(args, base, config, shards, weight_map, tokenizer, verifier, train, dev,
                 rank=rank, device=device, output=output, staging=None)
    finally:
        dist.destroy_process_group()

