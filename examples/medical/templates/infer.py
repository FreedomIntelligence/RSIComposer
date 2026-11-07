"""Optional raw-response baseline; copy/edit this file into a candidate bundle.

Returns the complete generated continuation. The host alone extracts answers.
Print generation diagnostics to stderr; this is candidate telemetry, not host
attestation that a backend consumed an input or used the claimed weights.
"""
import hashlib
import json
from pathlib import Path
import sys


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    config = json.loads(Path('submission.json').read_text())
    base = config['base_model_path']
    checkpoint = config.get('checkpoint_path', base)
    tokenizer_root = config.get('adapter_path', checkpoint)
    if not (Path(tokenizer_root) / 'tokenizer_config.json').is_file():
        tokenizer_root = base
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map='auto',
        torch_dtype=torch.bfloat16, local_files_only=True)
    if config.get('adapter_path'):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, config['adapter_path'], local_files_only=True)
    model.eval()
    payload = json.load(sys.stdin)
    predictions = []
    for item in payload['items']:
        prompt = tokenizer.apply_chat_template(item['messages'], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        encoded = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=64, do_sample=False,
                pad_token_id=tokenizer.eos_token_id)
        tokens = generated[0, encoded['input_ids'].shape[1]:]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        print(json.dumps({'candidate_telemetry': True, 'id': item['id'],
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'input_tokens': encoded['input_ids'].shape[1], 'generated_tokens': len(tokens),
            'generation_limit': 64, 'enable_thinking': False, 'raw_generation': text}), file=sys.stderr, flush=True)
        predictions.append({'benchmark_id': item['benchmark_id'], 'id': item['id'], 'text': text})
    print(json.dumps({'predictions': predictions}), flush=True)


if __name__ == '__main__':
    main()
