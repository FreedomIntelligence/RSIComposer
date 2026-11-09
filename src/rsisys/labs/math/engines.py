"""Optional numeric-math engines. ML dependencies are loaded only on execution."""
import gc
import json
import random
import sys
import time
from pathlib import Path
from typing import Protocol


class MathEngine(Protocol):
    name: str
    simulated: bool

    def generate(self, ref, rows, count, round_number, sample): ...
    def train(self, ref, rows, output, round_number): ...


class FixtureMathEngine:
    """Uses supplied answers for protocol tests only; never claims learned weights."""
    name = 'fixture-v1'
    simulated = True

    def generate(self, ref, rows, count, round_number, sample):
        return [[row['answer'] if i % 2 == 0 else 'unparseable fixture response'
                 for i in range(count)] for row in rows]

    def train(self, ref, rows, output, round_number):
        pass


class TransformersMathEngine:
    """Local generation and fresh-base LoRA SFT; no serving daemon."""
    name = 'transformers-lora-v1'
    simulated = False

    def __init__(self, config=None, *, input_root=None):
        self.config = dict(config or {})
        self.input_root = Path(input_root or Path.cwd()).resolve()
        self.demonstrations = []
        if self.config.get('few_shot_data'):
            from rsisys.labs.math.verifiers import numeric_answer, final_boxed
            parse_answer = final_boxed if self.config.get('answer_format') == 'latex' else numeric_answer
            data = self._path(self.config['few_shot_data'])
            self.demonstrations = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
            if not self.demonstrations or any(not isinstance(row, dict) or
                    not isinstance(row.get('prompt'), str) or not row['prompt'].strip() or
                    not isinstance(row.get('answer'), str) or parse_answer(row['answer']) is None
                    for row in self.demonstrations):
                raise ValueError('Few-shot data must contain prompt and an answer in the selected final-answer format')
            if len({r['prompt'].strip() for r in self.demonstrations}) != len(self.demonstrations):
                raise ValueError('Duplicate few-shot questions')

    def _path(self, ref):
        path = Path(ref)
        return (path if path.is_absolute() else self.input_root / path).resolve(strict=True)

    def _load(self, ref, *, train=False):
        import torch
        import transformers
        from peft import PeftModel
        selected = self._path(ref)
        manifest = selected / 'rest-em.json'
        adapter = manifest.is_file()
        if adapter:
            metadata = json.loads(manifest.read_text())
            if metadata['simulated']:
                raise ValueError('Cannot load fixture output as model weights')
            base = self._path(metadata['initial_model'])
        else:
            base = selected
        tokenizer = transformers.AutoTokenizer.from_pretrained(base, local_files_only=True)
        if tokenizer.eos_token_id is None:
            raise ValueError('Tokenizer must define EOS')
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        cls = (transformers.Qwen3_5ForConditionalGeneration if self.config.get('model_kind') == 'qwen3_5'
               else transformers.AutoModelForCausalLM)
        device = self.config.get('device', 'cuda')
        model = cls.from_pretrained(base, local_files_only=True,
            dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
            attn_implementation='sdpa').to(device)
        if adapter:
            if train:
                raise ValueError('M-step must load initial weights, not a prior adapter')
            model = PeftModel.from_pretrained(model, selected / 'adapter', local_files_only=True)
        return model, tokenizer

    def _prefix(self, tokenizer, prompt):
        ending = ('End each answer with the final answer in \\boxed{...}.' if self.config.get('answer_format') == 'latex'
                  else 'End each answer with #### followed by the numeric answer.')
        if self.config.get('prompt_format', 'chat') == 'completion':
            if any(row['prompt'].strip() == prompt.strip() for row in self.demonstrations):
                raise ValueError('Few-shot demonstration overlaps a target question')
            instruction = f'Solve each math problem step by step. {ending}\n\n'
            examples = ''.join(f"Question: {row['prompt']}\nAnswer: {row['answer']}\n\n"
                               for row in self.demonstrations)
            return instruction + examples + f'Question: {prompt}\nAnswer:'
        instruction = ending if self.config.get('answer_format') == 'latex' else 'End with #### followed by the numeric answer.'
        return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt + '\n' + instruction}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def _decode(self, tokenizer, tokens):
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        if self.config.get('prompt_format', 'chat') == 'completion':
            # generate includes the stop string; it belongs to the next problem.
            text = text.split('\nQuestion:', 1)[0].rstrip()
        return text

    def generate(self, ref, rows, count, round_number, sample):
        import torch
        torch.manual_seed(self.config.get('seed', 73) + round_number)
        model, tokenizer = self._load(ref)
        try:
            model.eval()
            tokenizer.padding_side = 'left'
            result = [[] for _ in rows]
            requests = [(index, self._prefix(tokenizer, row['prompt']))
                        for index, row in enumerate(rows) for _ in range(count)]
            batch_size = self.config.get('generation_batch_size', 1)
            started = time.monotonic()
            with torch.no_grad():
                for offset in range(0, len(requests), batch_size):
                    batch = requests[offset:offset + batch_size]
                    encoded = tokenizer([text for _, text in batch], add_special_tokens=False,
                                        padding=True, return_tensors='pt').to(model.device)
                    if encoded.input_ids.shape[1] + self.config.get('max_new_tokens', 512) > self.config.get('max_length', 2048):
                        raise ValueError('Prompt and generation budget exceed max_length')
                    options = {'temperature': 0.7, 'top_k': 40, 'top_p': 1.0} if sample else {}
                    if self.config.get('prompt_format', 'chat') == 'completion':
                        options.update(stop_strings=['\nQuestion:'], tokenizer=tokenizer)
                    tokens = model.generate(**encoded, do_sample=sample,
                        max_new_tokens=self.config.get('max_new_tokens', 512),
                        pad_token_id=tokenizer.pad_token_id, **options)
                    for (index, _), output in zip(batch, tokens, strict=True):
                        result[index].append(self._decode(tokenizer, output[encoded.input_ids.shape[1]:]))
                    print(json.dumps({'stage': 'generate' if sample else 'evaluate', 'round': round_number,
                        'completed': offset + len(batch), 'total': len(requests),
                        'elapsed_seconds': time.monotonic() - started}), file=sys.stderr, flush=True)
            return result
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def train(self, ref, rows, output, round_number):
        import torch
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(self.config.get('seed', 73) + round_number)
        model, tokenizer = self._load(ref, train=True)
        try:
            # A new base and a new adapter for EVERY M-step; optimizer also fresh.
            model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
            model.config.use_cache = False
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            model.train()
            dataset = []
            for row in rows:
                prompt = tokenizer(self._prefix(tokenizer, row['prompt']), add_special_tokens=False)['input_ids']
                answer = tokenizer(row['completion'], add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
                if len(prompt) + len(answer) > self.config.get('max_length', 2048):
                    raise ValueError('Accepted completion exceeds max_length; increase it instead of truncating the answer')
                dataset.append((prompt + answer, [-100] * len(prompt) + answer))
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                         lr=self.config.get('learning_rate', 2e-4))
            steps, microsteps, tokens_seen, weighted_loss = 0, 0, 0, 0.0
            started = time.monotonic()
            batch_size = self.config.get('train_batch_size', 1)
            accumulation = self.config.get('gradient_accumulation_steps', 1)
            for epoch in range(self.config.get('epochs', 1)):
                shuffled = list(dataset)
                random.Random(self.config.get('seed', 73) + round_number + epoch).shuffle(shuffled)
                batches = [shuffled[i:i + batch_size] for i in range(0, len(shuffled), batch_size)]
                optimizer.zero_grad(set_to_none=True)
                for index, batch in enumerate(batches):
                    size = max(len(ids) for ids, _ in batch)
                    ids = torch.tensor([ids + [tokenizer.pad_token_id] * (size - len(ids)) for ids, _ in batch], device=model.device)
                    labels = torch.tensor([labels + [-100] * (size - len(labels)) for _, labels in batch], device=model.device)
                    mask = torch.tensor([[1] * len(row_ids) + [0] * (size - len(row_ids)) for row_ids, _ in batch], device=model.device)
                    loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite training loss')
                    group_start = (index // accumulation) * accumulation
                    group_size = min(accumulation, len(batches) - group_start)
                    (loss / group_size).backward()
                    microsteps += 1
                    token_count = int((labels[:, 1:] != -100).sum())
                    tokens_seen += token_count
                    weighted_loss += float(loss.detach()) * token_count
                    if (index + 1) % accumulation == 0 or index + 1 == len(batches):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        steps += 1
                        print(json.dumps({'stage': 'train', 'round': round_number, 'epoch': epoch + 1,
                            'step': steps, 'microsteps': microsteps, 'loss': float(loss.detach()),
                            'elapsed_seconds': time.monotonic() - started}), file=sys.stderr, flush=True)
            model.save_pretrained(output / 'adapter')
            (output / 'training.json').write_text(json.dumps({'steps': steps, 'microsteps': microsteps,
                'last_loss': float(loss.detach()), 'token_weighted_loss': weighted_loss / tokens_seen,
                'completion_tokens_seen': tokens_seen, 'samples': len(rows),
                'elapsed_seconds': time.monotonic() - started}))
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
