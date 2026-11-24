"""Subprocess entry points for prepared-prompt vLLM inference and LoRA SFT."""
import json
import os
import random
import sys
import time
from pathlib import Path

from rsicomposer.runtime.observation.client import log

from .engine import LoRAEngine


def generate(request, output):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    config = request['config']
    engine = LoRAEngine(config)
    base = request['model']
    adapter = (LoRARequest('adapter', 1, request['adapter'])
               if request.get('adapter') is not None else None)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    prompts = [engine._prefix(tokenizer, item['prompt']) for item in request['requests']]
    tokenized = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
    for tokens in tokenized:
        if len(tokens) + config.get('max_new_tokens', 512) > config.get('max_length', 2048):
            raise ValueError('Prompt and generation budget exceed max_length')
    started = time.monotonic()
    llm = LLM(model=base, tokenizer=base, dtype='bfloat16', tensor_parallel_size=1,
        max_model_len=config.get('max_length', 2048),
        gpu_memory_utilization=config.get('gpu_memory_utilization', 0.85),
        max_num_seqs=config.get('max_num_seqs', 128),
        max_num_batched_tokens=config.get('max_num_batched_tokens', 16384),
        enforce_eager=True, enable_lora=adapter is not None, max_lora_rank=8,
        seed=config.get('seed', 73) + request['round'] + request['rank'],
        enable_prefix_caching=True, enable_chunked_prefill=True)
    load_seconds = time.monotonic() - started
    sample = request['sample']
    params = SamplingParams(temperature=request['temperature'] if sample else 0.0, top_k=40 if sample else -1,
        top_p=1.0, max_tokens=config.get('max_new_tokens', 512),
        stop=config.get('stop_strings'))
    generated = llm.generate([{'prompt_token_ids': tokens} for tokens in tokenized], params, lora_request=adapter, use_tqdm=False)
    if len(generated) != len(prompts):
        raise ValueError('vLLM returned an unexpected request count')
    result = [{**{k: item[k] for k in ('index', 'sample')}, 'completion': reply.outputs[0].text.rstrip()}
              for item, reply in zip(request['requests'], generated, strict=True)]
    Path(output).write_text(json.dumps(result))
    import shutil
    import subprocess
    memory = None
    if shutil.which('nvidia-smi'):
        memory = subprocess.check_output(['nvidia-smi', '--id=' + os.environ['CUDA_VISIBLE_DEVICES'],
            '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip()
    tokens = sum(len(r.outputs[0].token_ids) for r in generated)
    seconds = time.monotonic() - started - load_seconds
    log({'inference/requests': len(result), 'inference/generated_tokens': tokens,
         'inference/tokens_per_second': tokens / max(seconds, 1e-9), 'inference/load_seconds': load_seconds},
        series=f"gpu-{os.environ['CUDA_VISIBLE_DEVICES']}")
    print('RSICOMPOSER_WORKER ' + json.dumps({'stage': 'inference', 'rank': request['rank'],
        'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'requests': len(result),
        'generated_tokens': sum(len(r.outputs[0].token_ids) for r in generated),
        'load_seconds': load_seconds, 'generation_seconds': time.monotonic() - started - load_seconds,
        'device_memory_used_mib': memory}), flush=True)


from rsicomposer.backends.training.batching import training_groups


def train(request):
    from contextlib import nullcontext

    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from torch.nn.parallel import DistributedDataParallel as DDP
    config = request['config']
    requested_steps = request['max_steps']
    if requested_steps is not None and (type(requested_steps) is not int or requested_steps < 1):
        raise ValueError('max_steps must be a positive integer')
    rank, local, world = int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl')
    try:
        torch.manual_seed(config.get('seed', 73) + request['round'])
        engine = LoRAEngine(config)
        model, tokenizer = engine._load(request['model'], train=True)
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
            target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
        model.config.use_cache = False
        if config.get('train_gradient_checkpointing', False):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
        model = DDP(model, device_ids=[local])
        dataset = []
        for row in request['rows']:
            prompt = tokenizer(engine._prefix(tokenizer, row['prompt']), add_special_tokens=False)['input_ids']
            answer = tokenizer(row['completion'], add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
            if len(prompt) + len(answer) > config.get('max_length', 2048):
                raise ValueError('Accepted completion exceeds max_length')
            dataset.append((prompt + answer, [-100] * len(prompt) + answer))
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                     lr=config.get('learning_rate', 2e-4))
        batch_size, accumulation = config.get('train_batch_size', 1), config.get('gradient_accumulation_steps', 1)
        steps = microsteps = tokens_seen = 0
        weighted_loss = 0.0
        started = time.monotonic()
        for epoch in range(requested_steps or config.get('epochs', 1)):
            shuffled = list(dataset)
            random.Random(config.get('seed', 73) + request['round'] + epoch).shuffle(shuffled)
            for group in training_groups(shuffled, batch_size, accumulation, world):
                total_tokens = sum(sum(label != -100 for label in labels[1:]) for _, labels in group)
                optimizer.zero_grad(set_to_none=True)
                # Every rank participates in the same number of collectives. Empty
                # tails use a zero-weight real forward, never duplicated supervision.
                chunks = [group[i:i + batch_size * world] for i in range(0, len(group), batch_size * world)]
                for index, chunk in enumerate(chunks):
                    batch = chunk[rank * batch_size:(rank + 1) * batch_size]
                    active = bool(batch)
                    if not active:
                        batch = [group[0]]
                    size = max(len(ids) for ids, _ in batch)
                    ids = torch.tensor([ids + [tokenizer.pad_token_id] * (size - len(ids)) for ids, _ in batch], device=local)
                    labels = torch.tensor([labels + [-100] * (size - len(labels)) for _, labels in batch], device=local)
                    mask = torch.tensor([[1] * len(ids) + [0] * (size - len(ids)) for ids, _ in batch], device=local)
                    tokens = int((labels[:, 1:] != -100).sum()) if active else 0
                    with model.no_sync() if index + 1 < len(chunks) else nullcontext():
                        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
                        finite = torch.tensor(int(torch.isfinite(loss)), device=local)
                        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                        if not finite.item():
                            raise ValueError('Non-finite distributed training loss')
                        (loss * (tokens * world / total_tokens)).backward()
                    stats = torch.tensor([float(loss.detach()) * tokens, tokens], dtype=torch.float64, device=local)
                    dist.all_reduce(stats)
                    weighted_loss += stats[0].item()
                    tokens_seen += int(stats[1].item())
                    microsteps += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                steps += 1
                if rank == 0:
                    name = f"train/round-{request['round']}"
                    log({f'{name}/loss': weighted_loss / max(tokens_seen, 1), f'{name}/step_loss': float(loss.detach()),
                         f'{name}/completion_tokens': tokens_seen,
                         f'{name}/tokens_per_second': tokens_seen / max(time.monotonic() - started, 1e-9)}, step=steps)
                if requested_steps is not None and steps >= requested_steps:
                    break
            if requested_steps is not None and steps >= requested_steps:
                break
        from hashlib import sha256
        digest = sha256()
        for parameter in model.module.parameters():
            if parameter.requires_grad:
                digest.update(parameter.detach().float().cpu().numpy().tobytes())
        hashes = [None] * world
        dist.all_gather_object(hashes, digest.hexdigest())
        if len(set(hashes)) != 1:
            raise ValueError('Distributed adapter weights diverged')
        dist.barrier()
        if rank == 0:
            output = Path(request['output'])
            model.module.save_pretrained(output / 'adapter')
            metrics = {'steps': steps, 'microsteps': microsteps, 'samples': len(dataset),
                'last_loss': float(loss.detach()), 'token_weighted_loss': weighted_loss / tokens_seen,
                'completion_tokens_seen': tokens_seen, 'world_size': world, 'rank_weights_identical': True,
                'trainable_weights_sha256': hashes[0],
                'effective_batch_size': batch_size * accumulation * world,
                'loss_normalization': 'global_completion_tokens',
                'elapsed_seconds': time.monotonic() - started}
            (output / 'training.json').write_text(json.dumps(metrics))
        dist.barrier()
        print('RSICOMPOSER_WORKER ' + json.dumps({'stage': 'train', 'rank': rank,
            'steps': steps, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'elapsed_seconds': time.monotonic() - started}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    request = json.loads(Path(sys.argv[2]).read_text())
    if sys.argv[1] == 'generate':
        generate(request, sys.argv[3])
    elif sys.argv[1] == 'train':
        train(request)
    else:
        raise ValueError('Unknown worker operation')
