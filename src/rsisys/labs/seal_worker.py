"""Optional SEAL backend using the upstream Trainer and SFTTrainer recipes.

This process has no access to QA labels or candidate-selection logic.
Inner padding labels intentionally reproduce the pinned upstream behavior.
"""
import gc
import json
import os
import sys
from pathlib import Path


def require_versions():
    from importlib.metadata import version
    expected = {'transformers': '4.52.4', 'trl': '0.18.1', 'peft': '0.15.2'}
    actual = {key: version(key) for key in expected}
    if actual != expected:
        raise ValueError(f'SEAL paper backend requires pinned packages {expected}; found {actual}')
    return actual


def inner_rows(sequences, tokenizer, max_length):
    rows = []
    for text in sequences:
        encoded = tokenizer(text, truncation=True, max_length=max_length, padding='max_length')
        rows.append({**encoded, 'labels': list(encoded['input_ids'])})
    return rows


def adapt(request):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

    from .weight_worker import weights_digest
    versions = require_versions()
    config = request['config']
    tokenizer = AutoTokenizer.from_pretrained(request['model'], local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(request['model'], torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='sdpa').cuda()
    directory = Path(request['output']).parent / f"inner-{request['rank']}"
    directory.mkdir(exist_ok=False)
    results, adapters = [], []
    for episode in request['items']:
        seed = episode['seed'] + request.get('seed_offset', 0)
        set_seed(seed)
        model = get_peft_model(base, LoraConfig(r=config['inner_rank'], lora_alpha=2 * config['inner_rank'],
            lora_dropout=0.0, task_type='CAUSAL_LM'))
        before = weights_digest(model)
        target = directory / str(episode['index'])
        arguments = TrainingArguments(output_dir=str(target),
            per_device_train_batch_size=config['batch_size'], gradient_accumulation_steps=1,
            num_train_epochs=config['inner_epochs'], learning_rate=config['inner_lr'],
            logging_steps=1, save_strategy='no', report_to='none', remove_unused_columns=False,
            fp16=False, bf16=True, seed=seed)
        trainer = Trainer(model=model, args=arguments,
            train_dataset=Dataset.from_list(inner_rows(episode['sequences'], tokenizer, config['inner_max_length'])),
            data_collator=DataCollatorWithPadding(tokenizer))
        result = trainer.train()
        after = weights_digest(model)
        adapter = target / 'final_adapter'
        model.save_pretrained(adapter)
        adapters.append(str(adapter))
        losses = [r['loss'] for r in trainer.state.log_history if 'loss' in r]
        results.append(dict(index=episode['index'], seed=seed, steps=result.global_step,
            before_sha256=before, after_sha256=after, weights_changed=before != after,
            loss_first=losses[0], loss_last=losses[-1], versions=versions,
            lr_scheduler=str(arguments.lr_scheduler_type.value), weight_decay=arguments.weight_decay,
            max_length=config['inner_max_length'], padding_labels='upstream-eos-unmasked'))
        base = model.unload()
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        print('RSISYS_WORKER ' + json.dumps({'stage': 'seal_inner', **results[-1]}), flush=True)
    del base
    gc.collect()
    torch.cuda.empty_cache()
    # The upstream server evaluates adapters in vLLM. Do the same in-process,
    # after releasing the Trainer model, without persistent HTTP/ZMQ services.
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=request['model'], dtype='bfloat16', enable_lora=True,
        max_lora_rank=config['inner_rank'], max_loras=1, tensor_parallel_size=1,
        max_model_len=config['max_length'], gpu_memory_utilization=0.8,
        enable_prefix_caching=True, enforce_eager=config.get('enforce_eager', False), seed=config.get('seed', 73))
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=64)
    baselines = {}
    for number, (episode, adapter, result) in enumerate(zip(request['items'], adapters, results, strict=True), 1):
        prompts = [q['prompt'] for q in episode['questions']]
        if tuple(prompts) not in baselines:
            baselines[tuple(prompts)] = [r.outputs[0].text.strip() for r in llm.generate(prompts, params, use_tqdm=False)]
        adapted = llm.generate(prompts, params, lora_request=LoRARequest(f'episode-{number}', number, adapter), use_tqdm=False)
        result['baseline'] = baselines[tuple(prompts)]
        result['predictions'] = [r.outputs[0].text.strip() for r in adapted]
    return results


def train(request):
    import torch
    import torch.distributed as dist
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from .weight_worker import merged_change_count, weights_digest
    versions = require_versions()
    if request['algorithm'] != 'sft':
        raise ValueError('SEAL backend only supports outer SFT')
    config = request['config']
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    tokenizer = AutoTokenizer.from_pretrained(request['model'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(request['model'], torch_dtype='auto', local_files_only=True)
    rows = [{'prompt': r['prompt'], 'completion': r['completion']} for r in request['items']]
    longest = max(len(tokenizer(r['prompt'] + r['completion']).input_ids) for r in rows)
    if longest > config['max_length']:
        raise ValueError('Outer SFT sequence exceeds max_length; refusing silent truncation')
    arguments = SFTConfig(output_dir=str(request['checkpoint']),
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['outer_gradient_accumulation_steps'],
        num_train_epochs=config['outer_epochs'], learning_rate=config['outer_lr'],
        logging_steps=1, max_length=longest, gradient_checkpointing_kwargs={'use_reentrant': False},
        report_to='none', save_strategy='no', seed=config.get('seed', 73) + request.get('seed_offset', 0))
    trainer = SFTTrainer(model=model, args=arguments, train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer, peft_config=LoraConfig(r=config['lora_rank'],
            lora_alpha=2 * config['lora_rank'], lora_dropout=0.0, bias='none', task_type='CAUSAL_LM',
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    before = weights_digest(trainer.model)
    outcome = trainer.train()
    model = trainer.accelerator.unwrap_model(trainer.model)
    after = weights_digest(model)
    hashes = [None] * dist.get_world_size()
    dist.all_gather_object(hashes, after)
    if before == after or len(set(hashes)) != 1:
        raise ValueError('Outer SFT did not update synchronized weights')
    if trainer.is_world_process_zero():
        changed = merged_change_count(model)
        if not changed:
            raise ValueError('Outer update vanished at publication dtype')
        path = Path(request['checkpoint'])
        model.save_pretrained(path / 'adapter')
        model.merge_and_unload().save_pretrained(path, max_shard_size='5GB')
        tokenizer.save_pretrained(path)
        result = dict(steps=outcome.global_step, samples=len(rows), world_size=dist.get_world_size(),
            algorithm='sft', before_sha256=before, after_sha256=after, weights_changed=True,
            rank_weights_identical=True, merged_changed_elements=changed, versions=versions,
            effective_batch_size=config['batch_size'] * dist.get_world_size() * config['outer_gradient_accumulation_steps'],
            lr_scheduler=arguments.lr_scheduler_type.value, weight_decay=arguments.weight_decay)
        (path / 'training.json').write_text(json.dumps(result, indent=2))
        Path(request['output']).write_text(json.dumps(result))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    request = json.loads(Path(sys.argv[1]).read_text())
    if request['mode'] == 'adapt':
        Path(request['output']).write_text(json.dumps(adapt(request)))
    elif request['mode'] == 'train':
        train(request)
    else:
        raise ValueError('Unknown SEAL worker operation')
