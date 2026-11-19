"""Generate one GSM8K shard on its scheduler-assigned GPU."""
import argparse
from pathlib import Path

import csv
import json
import os
import subprocess

from rsisys.api import atomic_write, json_bytes

from .gpu import cuda_pci_devices, pci_address
from .protocol import MAX_NEW_TOKENS, render_prompt


def bind_device():
    expected = os.environ.get('NVIDIA_VISIBLE_DEVICES')
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id',
                                   '--format=csv,noheader'], text=True, timeout=15)
    rows = [(index.strip(), uuid.strip(), pci_address(bus)) for index, uuid, bus in csv.reader(raw.splitlines())]
    matches = [(index, bus) for index, uuid, bus in rows if uuid == expected]
    if len(matches) != 1:
        raise PermissionError('Expected one assigned GPU for STaR sampling')
    index, bus = matches[0]
    os.environ['CUDA_DEVICE_ORDER'], os.environ['CUDA_VISIBLE_DEVICES'] = 'PCI_BUS_ID', index
    if cuda_pci_devices() != [bus]:
        raise PermissionError('STaR sampler GPU differs from the allocation')


def generate(model_path, rows, *, temperature=0.):
    bind_device()
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map='cuda:0',
        attn_implementation='sdpa', local_files_only=True).eval().requires_grad_(False)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    responses = []
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
        prompts = [render_prompt(tokenizer, row['question'] +
                   (f'\nKnown correct answer: {row["hint"]}. Derive it from the given problem.'
                    if 'hint' in row else '')) for row in batch]
        encoded = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors='pt')
        prefix = encoded.input_ids.shape[1]
        if prefix + MAX_NEW_TOKENS > 8192:
            raise ValueError('STaR prompt exceeds the generation context allowance')
        with torch.inference_mode():
            output = model.generate(**encoded.to('cuda:0'), max_new_tokens=MAX_NEW_TOKENS,
                                    do_sample=temperature > 0, pad_token_id=tokenizer.pad_token_id,
                                    **({'temperature': temperature} if temperature > 0 else {}))
        for row, text in zip(batch, tokenizer.batch_decode(output[:, prefix:], skip_special_tokens=True), strict=True):
            responses.append({'id': row['id'], 'text': text})
        print(f'generated {len(responses)}/{len(rows)}', flush=True)
    return responses



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--temperature", type=float, required=True)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text())
    atomic_write(args.output, json_bytes(generate(args.model, rows, temperature=args.temperature)))


if __name__ == "__main__":
    main()
