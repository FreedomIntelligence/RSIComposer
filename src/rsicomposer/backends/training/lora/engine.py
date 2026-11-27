"""LoRA execution on prepared text/messages; no dataset, grading or RSI policy."""
import gc
import json
import random
import sys
import time
from pathlib import Path


# Adapt only the text attention linears: Q/V in full-attention layers and
# QKV/output in Gated DeltaNet layers. PEFT must not touch vision, routers or experts.
_QWEN35_MOE_TARGETS = (r'model\.language_model\.layers\.\d+\.'
                      r'(?:self_attn\.(?:q_proj|v_proj)|linear_attn\.(?:in_proj_qkv|out_proj))')


def _mask_supervised_labels(labels, quota, row_number):
    # Position zero has no causal predecessor and does not contribute to the model's loss.
    positions = [i for i in range(1, len(labels)) if labels[i] != -100]
    available = len(positions)
    if available < quota:
        raise ValueError(f'Training row {row_number} has {available} causal assistant labels; '
                         f'supervised_tokens_per_step requires {quota}')
    # Sample evenly across the eligible labels, including both endpoints (and thus
    # the appended EOS). A quota of one uses the middle label, not always EOS.
    ranks = ([(available - 1) // 2] if quota == 1 else
             [i * (available - 1) // (quota - 1) for i in range(quota)])
    masked = [-100] * len(labels)
    for rank in ranks:
        position = positions[rank]
        masked[position] = labels[position]
    return masked


class LoRAEngine:
    name = 'transformers-lora-v1'
    simulated = False

    def __init__(self, config=None):
        self.config = dict(config or {})
        if set(self.config) & {'gpu_ids', 'worker_timeout'}:
            raise ValueError('Device placement and worker timeout belong in host resources')

    def bind_execution(self, context):
        from rsicomposer.runtime.execution.executor import execution_capabilities
        self.resources, self.executor = execution_capabilities(context)
        self.resources.declare(self.resource_roles())

    def resource_roles(self):
        from rsicomposer.runtime.execution.resources import ResourceRequest
        count = 0 if self.config.get('device') == 'cpu' else 1
        return {'rollout': ResourceRequest(gpus=count), 'trainer': ResourceRequest(gpus=count)}

    @staticmethod
    def _load_tokenizer(base):
        import transformers
        tokenizer = transformers.AutoTokenizer.from_pretrained(base, local_files_only=True)
        if tokenizer.eos_token_id is None:
            raise ValueError('Tokenizer must define EOS')
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load(self, ref, *, train=False, adapter=None, cuda_index=None):
        import torch
        import transformers
        from peft import PeftModel
        base = Path(ref).resolve(strict=True)
        if adapter is not None and train:
            raise ValueError('Fresh-adapter training requires base weights')
        tokenizer = self._load_tokenizer(base)
        kind = self.config.get('model_kind')
        device = self.config.get('device', 'cuda')
        if cuda_index is not None and (device != 'cuda' or type(cuda_index) is not int or cuda_index < 0):
            raise ValueError('cuda_index requires CUDA and a nonnegative local device index')
        placement = device if cuda_index is None else f'cuda:{cuda_index}'
        options = dict(local_files_only=True, dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
                       attn_implementation='sdpa')
        if kind == 'qwen3_5_moe':
            cls = getattr(transformers, 'Qwen3_5MoeForConditionalGeneration', None)
            if cls is None:
                raise RuntimeError('qwen3_5_moe requires native Transformers Qwen3.5 MoE support (tested with 5.3+)')
            config = transformers.AutoConfig.from_pretrained(base, local_files_only=True)
            if config.model_type != kind:
                raise ValueError(f'model_kind {kind} requires a {kind} checkpoint, got {config.model_type}')
            # Keep all weights on the worker's leased device. Leave CUDA inference on HF's
            # default expert kernels; avoid grouped-MM backward/fallback for frozen experts.
            options.update(config=config, device_map={'': placement})
            if train or device == 'cpu':
                options['experts_implementation'] = 'eager'
            model = cls.from_pretrained(base, **options)
        else:
            cls = (transformers.Qwen3_5ForConditionalGeneration if kind == 'qwen3_5'
                   else transformers.AutoModelForCausalLM)
            model = cls.from_pretrained(base, **options).to(placement)
        if adapter is not None:
            model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
        return model, tokenizer

    def _prefix(self, tokenizer, prompt):
        if isinstance(prompt, str):
            return prompt
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.config.get('enable_thinking', False))

    def _decode(self, tokenizer, tokens):
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        for stop in self.config.get('stop_strings', ()):
            text = text.split(stop, 1)[0]
        return text.rstrip() if self.config.get('stop_strings') else text

    def generate(self, ref, rows, count, round_number, sample, *, adapter=None, temperature=0.7):
        import torch
        torch.manual_seed(self.config.get('seed', 73) + round_number)
        model, tokenizer = self._load(ref, adapter=adapter)
        try:
            model.eval()
            tokenizer.padding_side = 'left'
            result = [[] for _ in rows]
            requests = [(index, self._prefix(tokenizer, row['prompt']))
                        for index, row in enumerate(rows) for _ in range(count)]
            batch_size = self.config.get('generation_batch_size', 1)
            separate_truncation = self.config.get('report_truncated_attempts', False)
            output_limit = self.config.get('max_new_tokens', 512)
            started = time.monotonic()
            with torch.no_grad():
                for offset in range(0, len(requests), batch_size):
                    batch = requests[offset:offset + batch_size]
                    encoded = tokenizer([text for _, text in batch], add_special_tokens=False,
                                        padding=True, return_tensors='pt').to(model.device)
                    completed = offset + len(batch)
                    if encoded.input_ids.shape[1] + output_limit > self.config.get('max_length', 2048):
                        if not separate_truncation:
                            raise ValueError('Prompt and generation budget exceed max_length')
                        eligible = []
                        for (index, text), length in zip(batch, encoded.attention_mask.sum(dim=1).tolist(), strict=True):
                            if length + output_limit > self.config.get('max_length', 2048):
                                result[index].append({'status': 'truncated', 'finish_reason': 'prompt_budget',
                                                      'completion': None})
                            else:
                                eligible.append((index, text))
                        if not eligible:
                            continue
                        batch = eligible
                        encoded = tokenizer([text for _, text in batch], add_special_tokens=False,
                                            padding=True, return_tensors='pt').to(model.device)
                    options = {'temperature': temperature, 'top_k': 40, 'top_p': 1.0} if sample else {}
                    if self.config.get('stop_strings'):
                        options.update(stop_strings=self.config['stop_strings'], tokenizer=tokenizer)
                    tokens = model.generate(**encoded, do_sample=sample,
                        max_new_tokens=output_limit,
                        pad_token_id=tokenizer.pad_token_id, **options)
                    for (index, _), output in zip(batch, tokens, strict=True):
                        tail = output[encoded.input_ids.shape[1]:]
                        completion = self._decode(tokenizer, tail)
                        if separate_truncation and len(tail) >= output_limit:
                            eos = model.generation_config.eos_token_id
                            eos = [eos if eos is not None else tokenizer.eos_token_id] if not isinstance(eos, list) else eos
                            if not set(eos).intersection(tail.tolist()):
                                completion = {'status': 'truncated', 'finish_reason': 'length', 'completion': completion}
                        result[index].append(completion)
                    print(json.dumps({'stage': 'generate' if sample else 'evaluate', 'round': round_number,
                        'completed': completed, 'total': len(requests),
                        'elapsed_seconds': time.monotonic() - started}), file=sys.stderr, flush=True)
            return result
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _encode_training_row(self, tokenizer, row):
        prompt = tokenizer(self._prefix(tokenizer, row['prompt']), add_special_tokens=False)['input_ids']
        answer = tokenizer(row['completion'], add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
        if len(prompt) + len(answer) > self.config.get('max_length', 2048):
            raise ValueError('Accepted completion exceeds max_length; increase it instead of truncating the answer')
        return prompt + answer, [-100] * len(prompt) + answer

    def count_supervised_tokens(self, ref, rows):
        """Count eligible causal assistant labels (including EOS) in input order, without loading weights.

        Uses the same prompt template, completion tokenization, and max_length check as training.
        Counts are before applying any per-step quota; IDs are not needed for counting.
        """
        tokenizer = self._load_tokenizer(Path(ref).resolve(strict=True))
        return [sum(label != -100 for label in self._encode_training_row(tokenizer, row)[1][1:])
                for row in rows]

    def train(self, ref, rows, output, round_number, *, steps=None):
        """Fit a fresh adapter, optionally cycling preordered rows at an exact label quota.

        Coverage mode takes explicit optimizer steps and one full example per step.
        Quota mode implies coverage and additionally masks labels to the supplied quota.
        Stable rollout/problem IDs are required but never written to metrics.
        """
        requested_steps = steps
        if requested_steps is not None and (type(requested_steps) is not int or requested_steps < 1):
            raise ValueError('steps must be a positive integer')
        quota = None
        ordered_coverage = self.config.get('ordered_training_coverage', False)
        if type(ordered_coverage) is not bool:
            raise ValueError('ordered_training_coverage must be a boolean')
        if 'supervised_tokens_per_step' in self.config:
            quota = self.config['supervised_tokens_per_step']
            if type(quota) is not int or quota < 1:
                raise ValueError('supervised_tokens_per_step must be a positive integer')
        track_coverage = ordered_coverage or quota is not None
        if track_coverage:
            for key in ('train_batch_size', 'gradient_accumulation_steps'):
                value = self.config.get(key, 1)
                if type(value) is not int or value != 1:
                    raise ValueError(f'Ordered training coverage requires {key}=1')
            if requested_steps is None:
                raise ValueError('Ordered training coverage requires explicit steps')
            if not isinstance(rows, (list, tuple)) or not rows:
                raise ValueError('Ordered training coverage requires a nonempty ordered list of training rows')
            rollout_ids, problem_ids = set(), set()
            for index, row in enumerate(rows, start=1):
                if not isinstance(row, dict):
                    raise ValueError(f'Training row {index} must be a mapping')
                for key in ('id', 'problem_id'):
                    value = row.get(key)
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(f'Training row {index} requires a nonempty string {key}')
                if row['id'] in rollout_ids:
                    raise ValueError(f'Training row {index} has a duplicate id')
                rollout_ids.add(row['id'])
                problem_ids.add(row['problem_id'])
            seen_rollout_ids, seen_problem_ids = set(), set()
            input_tokens_seen = 0
        import torch
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(self.config.get('seed', 73) + round_number)
        model, tokenizer = self._load(ref, train=True)
        try:
            moe = self.config.get('model_kind') == 'qwen3_5_moe'
            targets = _QWEN35_MOE_TARGETS if moe else ['q_proj', 'v_proj']
            model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                target_modules=targets, task_type='CAUSAL_LM'))
            model.config.use_cache = False
            # Pass cache control through the conditional wrapper to the nested text model.
            train_options = {'use_cache': False} if moe else {}
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            model.train()
            dataset = []
            for index, row in enumerate(rows, start=1):
                ids, labels = self._encode_training_row(tokenizer, row)
                if quota is not None:
                    labels = _mask_supervised_labels(labels, quota, index)
                dataset.append((ids, labels))
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                         lr=self.config.get('learning_rate', 2e-4))
            steps, microsteps, tokens_seen, weighted_loss = 0, 0, 0, 0.0
            started = time.monotonic()
            batch_size = self.config.get('train_batch_size', 1)
            accumulation = self.config.get('gradient_accumulation_steps', 1)
            for epoch in range(requested_steps or self.config.get('epochs', 1)):
                ordered = list(dataset)
                if not track_coverage:
                    random.Random(self.config.get('seed', 73) + round_number + epoch).shuffle(ordered)
                batches = [ordered[i:i + batch_size] for i in range(0, len(ordered), batch_size)]
                optimizer.zero_grad(set_to_none=True)
                for index, batch in enumerate(batches):
                    size = max(len(ids) for ids, _ in batch)
                    ids = torch.tensor([ids + [tokenizer.pad_token_id] * (size - len(ids)) for ids, _ in batch], device=model.device)
                    labels = torch.tensor([labels + [-100] * (size - len(labels)) for _, labels in batch], device=model.device)
                    mask = torch.tensor([[1] * len(row_ids) + [0] * (size - len(row_ids)) for row_ids, _ in batch], device=model.device)
                    loss = model(input_ids=ids, attention_mask=mask, labels=labels, **train_options).loss
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite training loss')
                    group_start = (index // accumulation) * accumulation
                    group_size = min(accumulation, len(batches) - group_start)
                    (loss / group_size).backward()
                    microsteps += 1
                    token_count = int((labels[:, 1:] != -100).sum())
                    if quota is not None and token_count != quota:
                        raise RuntimeError('Training batch does not match supervised_tokens_per_step')
                    tokens_seen += token_count
                    weighted_loss += float(loss.detach()) * token_count
                    if (index + 1) % accumulation == 0 or index + 1 == len(batches):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        steps += 1
                        if track_coverage:
                            seen_rollout_ids.add(rows[index]['id'])
                            seen_problem_ids.add(rows[index]['problem_id'])
                            input_tokens_seen += size  # Probe batches have one unpadded sequence.
                        print(json.dumps({'stage': 'train', 'round': round_number, 'epoch': epoch + 1,
                            'step': steps, 'microsteps': microsteps, 'loss': float(loss.detach()),
                            'elapsed_seconds': time.monotonic() - started}), file=sys.stderr, flush=True)
                        if requested_steps is not None and steps >= requested_steps:
                            break
                if requested_steps is not None and steps >= requested_steps:
                    break
            model.save_pretrained(output / 'adapter')
            metrics = {'steps': steps, 'microsteps': microsteps,
                'last_loss': float(loss.detach()), 'token_weighted_loss': weighted_loss / tokens_seen,
                'completion_tokens_seen': tokens_seen, 'samples': len(rows),
                'elapsed_seconds': time.monotonic() - started}
            if quota is not None:
                metrics['supervised_tokens_per_step'] = quota
            if track_coverage:
                metrics.update(supervision_label_selection=('full-causal-assistant-including-eos-v1' if quota is None
                                    else 'uniform-causal-assistant-including-eos-v1'),
                    coverage={'distinct_problem_ids_seen': len(seen_problem_ids),
                              'distinct_problem_ids_available': len(problem_ids),
                              'distinct_rollout_ids_seen': len(seen_rollout_ids),
                              'distinct_rollout_ids_available': len(rollout_ids),
                              'input_tokens_seen': input_tokens_seen})
            (output / 'training.json').write_text(json.dumps(metrics))
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
