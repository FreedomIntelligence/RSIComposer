"""SEAL reference inner adaptation; independent adapters per self-edit."""
import json
import random
import sys
from pathlib import Path

from rsicomposer.api import load_model, tensor_batch, weights_digest


def answer_questions(model, tokenizer, questions, max_length):
    import torch
    model.eval()
    tokenizer.padding_side = 'left'
    prompts = [q['prompt'] for q in questions]
    encoded = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors='pt').to(model.device)
    if encoded.input_ids.shape[1] + 64 > max_length:
        raise ValueError('Answer prompt exceeds budget')
    with torch.no_grad():
        output = model.generate(**encoded, do_sample=False, max_new_tokens=64,
            pad_token_id=tokenizer.pad_token_id, use_cache=True,
            stop_strings=['\nQuestion:', '\n\n'], tokenizer=tokenizer)
    return [tokenizer.decode(r[encoded.input_ids.shape[1]:], skip_special_tokens=True).strip() for r in output]


def adapt(request):
    import torch
    config = request['config']
    torch.manual_seed(config.get('seed', 73) + request.get('seed_offset', 0))
    model, tokenizer = load_model(request['model'], config)
    from peft import LoraConfig, get_peft_model
    rank = config.get('inner_rank', 32)
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
        target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    if config.get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    result = []
    for episode in request['items']:
        # A fresh adapter per independent self-edit; frozen base never changes.
        from peft.tuners.lora.layer import LoraLayer
        for module in model.modules():
            if isinstance(module, LoraLayer):
                module.reset_lora_parameters('default', init_lora_weights=True)
        with model.disable_adapter():
            baseline = answer_questions(model, tokenizer, episode['questions'], config.get('max_length', 4096))
        rows = []
        # Inner TTT trains entire title+synthetic-fact sequences, including context.
        for text in episode['sequences']:
            ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
            if len(ids) > config.get('max_length', 4096) or len(ids) < 2:
                raise ValueError('Invalid inner SFT sequence')
            rows.append({'ids': ids, 'prompt_len': 0})
        before = weights_digest(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.get('inner_lr', 1e-3))
        steps, losses = 0, []
        for epoch in range(config.get('inner_epochs', 10)):
            shuffled = list(rows)
            random.Random(config.get('seed', 73) + epoch).shuffle(shuffled)
            model.train()
            for offset in range(0, len(shuffled), config.get('batch_size', 1)):
                batch = shuffled[offset:offset + config.get('batch_size', 1)]
                ids, mask, labels = tensor_batch(batch, tokenizer, model.device)
                optimizer.zero_grad(set_to_none=True)
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite inner SFT loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
                steps += 1
        after = weights_digest(model)
        predictions = answer_questions(model, tokenizer, episode['questions'], config.get('max_length', 4096))
        result.append({'index': episode['index'], 'baseline': baseline, 'predictions': predictions,
            'steps': steps, 'loss_first': losses[0], 'loss_last': losses[-1],
            'before_sha256': before, 'after_sha256': after, 'weights_changed': before != after})
        print('RSICOMPOSER_WORKER ' + json.dumps({'stage': 'inner_sft', 'index': episode['index'], 'steps': steps}), flush=True)
    return result


if __name__ == '__main__':
    request = json.loads(Path(sys.argv[1]).read_text())
    if request['mode'] != 'adapt':
        raise ValueError('SEAL reference worker only supports inner adaptation')
    Path(request['output']).write_text(json.dumps(adapt(request)))
