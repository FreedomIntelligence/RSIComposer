"""Private CUDA execution entry point. Policy decisions stay in controllers.

SFT uses completion-only loss. GRPO uses on-policy response-token likelihoods,
group-normalized advantages, PPO clipping and a fixed initial-reference KL.
"""
import json
import os
import random
import sys
import time
from hashlib import sha256
from pathlib import Path


def weights_digest(model):
    digest = sha256()
    for name, value in model.named_parameters():
        if value.requires_grad:
            digest.update(name.encode())
            digest.update(value.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def merged_change_count(model):
    """Check actual published dtype, not just float32 optimizer-side deltas."""
    import torch
    from peft.tuners.lora.layer import LoraLayer

    count = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoraLayer):
                original = module.get_base_layer().weight
                updated = (original.float() + module.get_delta_weight('default').float()).to(original.dtype)
                count += int(torch.count_nonzero(updated != original))
    return count


def load_model(path, config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError('EOS token is required')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(device)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to(device)
    model.config.use_cache = False
    return model, tokenizer


def new_adapter(model, config):
    from peft import LoraConfig, get_peft_model
    rank = config.get('lora_rank', 32)
    modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                                            target_modules=modules, task_type='CAUSAL_LM'))
    if config.get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model


def encode_rows(rows, tokenizer, max_length):
    result = []
    for row in rows:
        prompt = tokenizer.encode(row['prompt'], add_special_tokens=False)
        completion = (list(row['response_ids']) if 'response_ids' in row else
                      tokenizer.encode(row['completion'], add_special_tokens=False) + [tokenizer.eos_token_id])
        if not prompt or not completion or len(prompt) + len(completion) > max_length:
            raise ValueError('Empty or overlong training sequence; truncation is not permitted')
        result.append({**row, 'ids': prompt + completion, 'prompt_len': len(prompt)})
    return result


def tensor_batch(rows, tokenizer, device):
    import torch
    size = max(len(r['ids']) for r in rows)
    ids = torch.tensor([r['ids'] + [tokenizer.pad_token_id] * (size - len(r['ids'])) for r in rows], device=device)
    attention = torch.tensor([[1] * len(r['ids']) + [0] * (size - len(r['ids'])) for r in rows], device=device)
    labels = ids.clone()
    for i, row in enumerate(rows):
        labels[i, :row['prompt_len']] = -100
        labels[i, len(row['ids']):] = -100
    return ids, attention, labels


def generate(request):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    config = request['config']
    tokenizer = AutoTokenizer.from_pretrained(request['model'], local_files_only=True)
    texts = [r['prompt'] for r in request['items']]
    if config.get('apply_chat_template', False):
        texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
            add_generation_prompt=True, tokenize=False) for text in texts]
    tokenized = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    if any(len(p) + request['max_tokens'] > config.get('max_length', 4096) for p in tokenized):
        raise ValueError('Prompt and generation budget exceed max_length')
    llm = LLM(model=request['model'], dtype='bfloat16', tensor_parallel_size=1,
        max_model_len=config.get('max_length', 4096), max_num_seqs=128,
        max_num_batched_tokens=32768, gpu_memory_utilization=0.8,
        enforce_eager=config.get('enforce_eager', False), enable_prefix_caching=True,
        seed=config.get('seed', 73) + request.get('seed_offset', 0) + request['rank'])
    params = SamplingParams(temperature=request.get('temperature', config.get('temperature', 0.7)) if request['sample'] else 0.0,
        top_p=request.get('top_p', config.get('top_p', 0.95)) if request['sample'] else 1.0, max_tokens=request['max_tokens'],
        stop=config.get('stop', None), logprobs=None)
    replies = llm.generate([{'prompt_token_ids': p} for p in tokenized], params, use_tqdm=False)
    result = []
    for row, reply in zip(request['items'], replies, strict=True):
        out = reply.outputs[0]
        result.append({'index': row['index'], 'completion': out.text,
                       'response_ids': list(out.token_ids), 'finish_reason': out.finish_reason,
                       'prompt_tokens': len(reply.prompt_token_ids)})
    return result


def token_logps(model, ids, mask, temperature=1.0):
    import torch
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
    # Chunked log-softmax avoids a full vocabulary-sized float32 copy per token.
    values = []
    for start in range(0, logits.shape[1], 128):
        part = logits[:, start:start + 128] / temperature
        targets = ids[:, 1 + start:1 + start + part.shape[1]]
        values.append(part.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(part, -1))
    return torch.cat(values, dim=1)


def train(request):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    from rsicomposer.backends.training.batching import training_groups
    from rsicomposer.backends.training.objectives import grpo_objective
    config, algorithm = request['config'], request['algorithm']
    rank, local, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    torch.cuda.set_device(local)
    dist.init_process_group('nccl')
    try:
        torch.manual_seed(config.get('seed', 73) + request.get('seed_offset', 0))
        base, tokenizer = load_model(request['model'], config)
        model = new_adapter(base, config)
        before = weights_digest(model)
        model = DDP(model, device_ids=[local])
        rows = encode_rows(request['items'], tokenizer, config.get('max_length', 4096))
        epochs = 1 if algorithm == 'grpo' else config.get('outer_epochs', 2)
        epoch_groups = []
        for epoch in range(epochs):
            shuffled = list(rows)
            random.Random(config.get('seed', 73) + epoch).shuffle(shuffled)
            epoch_groups.append(training_groups(shuffled, config.get('batch_size', 1), 1, world))
        if algorithm == 'grpo':
            # Old-policy and fixed-reference likelihoods are detached; every sampled
            # action is preserved byte-for-byte as the rollout token IDs.
            reference, _ = load_model(request['reference_model'], config)
            reference.eval()
            model.eval()
            needed = {}
            for group in epoch_groups[0]:
                for row in (group[rank:rank + 1] or [group[0]]):
                    needed[id(row)] = row
            with torch.no_grad():
                for row in needed.values():
                    ids, mask, _ = tensor_batch([row], tokenizer, local)
                    temperature = config.get('temperature', 0.6)
                    row['old_logps'] = token_logps(model.module, ids, mask, temperature).cpu()
                    row['reference_logps'] = token_logps(reference, ids, mask, temperature).cpu()
            del reference
            torch.cuda.empty_cache()
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
            lr=config.get('rl_lr', 5e-7) if algorithm == 'grpo' else config.get('outer_lr', 3e-4))
        steps, losses, gradient_norms, kl_means, ratio_deviations = 0, [], [], [], []
        for groups in epoch_groups:
            for group in groups:
                total_tokens = sum(len(r['ids']) - max(r['prompt_len'], 1) for r in group)
                optimizer.zero_grad(set_to_none=True)
                local_rows = group[rank * config.get('batch_size', 1):(rank + 1) * config.get('batch_size', 1)]
                active = bool(local_rows)
                # GRPO microbatch=1 avoids padding likelihood arrays; batch_size=1
                # is required by its Lab validator.
                batch = local_rows or [group[0]]
                ids, mask, labels = tensor_batch(batch, tokenizer, local)
                tokens = int((labels[:, 1:] != -100).sum()) if active else 0
                if algorithm == 'sft':
                    loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
                    loss = loss * (tokens * world / total_tokens)
                elif algorithm == 'grpo':
                    row = batch[0]
                    logps = token_logps(model, ids, mask, config.get('temperature', 0.6))
                    response_mask = (labels[:, 1:] != -100).float()
                    old = row['old_logps'].to(local)
                    ref = row['reference_logps'].to(local)
                    advantage = row['advantage']
                    # Equal sequence weighting as GRPO, not long-answer weighting.
                    loss, ratio, kl = grpo_objective(logps, old, ref, advantage, response_mask, config.get('kl_beta', 0.001))
                    if active:
                        kl_means.append(float((kl * response_mask).sum() / response_mask.sum()))
                        ratio_deviations.append(float(((ratio - 1).abs() * response_mask).sum() / response_mask.sum()))
                    loss = loss * ((world / len(group)) if active else 0)
                else:
                    raise ValueError('Unknown training algorithm')
                finite = torch.tensor(int(torch.isfinite(loss)), device=local)
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise ValueError('Non-finite training objective')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                steps += 1
                stats = torch.tensor(float(loss.detach()), device=local)
                dist.all_reduce(stats)
                losses.append(float(stats) / world)
                gradient_norms.append(float(norm))
                if rank == 0:
                    print('RSISYS_WORKER ' + json.dumps({'stage': algorithm, 'step': steps, 'loss': losses[-1]}), flush=True)
        after = weights_digest(model.module)
        hashes = [None] * world
        dist.all_gather_object(hashes, after)
        if len(set(hashes)) != 1 or before == after:
            raise ValueError('No parameter update or distributed weights diverged')
        result = {'steps': steps, 'samples': len(rows), 'world_size': world,
            'algorithm': algorithm, 'before_sha256': before, 'after_sha256': after,
            'weights_changed': True, 'rank_weights_identical': True,
            'loss_first': losses[0], 'loss_last': losses[-1], 'max_gradient_norm': max(gradient_norms)}
        if algorithm == 'grpo':
            result.update(positive_advantages=sum(r['advantage'] > 0 for r in rows),
                negative_advantages=sum(r['advantage'] < 0 for r in rows),
                reference_model=request['reference_model'], temperature=config.get('temperature', 0.6),
                mean_sample_kl=sum(kl_means) / len(kl_means) if kl_means else 0,
                mean_ratio_deviation=sum(ratio_deviations) / len(ratio_deviations) if ratio_deviations else 0)
        if rank == 0:
            surviving = merged_change_count(model.module)
            if not surviving:
                raise ValueError('Adapter updates vanished when merged into the published weight dtype')
            result['merged_changed_elements'] = surviving
            checkpoint = Path(request['checkpoint'])
            checkpoint.mkdir(parents=True, exist_ok=False)
            model.module.save_pretrained(checkpoint / 'adapter')
            merged = model.module.merge_and_unload()
            merged.save_pretrained(checkpoint, max_shard_size='5GB')
            tokenizer.save_pretrained(checkpoint)
            (checkpoint / 'training.json').write_text(json.dumps(result, indent=2))
            Path(request['output']).write_text(json.dumps(result))
        dist.barrier()
        return None
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    request = json.loads(Path(sys.argv[1]).read_text())
    started = time.monotonic()
    value = {'generate': generate, 'train': train}[request['mode']](request)
    if value is not None:
        Path(request['output']).write_text(json.dumps(value))
    print('RSISYS_WORKER ' + json.dumps({'mode': request['mode'], 'elapsed_seconds': time.monotonic() - started}), flush=True)
