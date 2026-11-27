"""Single-GPU Qwen3.5 MoE graph rollout, isolated in a Host-owned worker.

Prompts use the hosted Transformers tokenizer protocol; vLLM sampling is a
separate generation protocol, not a claim of exact Transformers parity.
"""
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import sys
import time

from .engine import LoRAEngine


_MAX_CONTEXT = 65536
_GRAPH_SIZES = [1, 2, 4, 8, 16]


def _request_seed(seed, round_number, ordinal):
    # Stable regardless of chunk size or earlier requests skipped for prompt budget.
    key = f'{seed}:{round_number}:{ordinal}'.encode('ascii')
    return int.from_bytes(sha256(key).digest()[:8], 'big') & ((1 << 63) - 1)


def _batch(llm, prompts, params, lora):
    # LLM.generate contracts to return completions in input order. In v0.32 the
    # lower-level enqueue IDs need not match the final RequestOutput IDs.
    replies = llm.generate(prompts, params, lora_request=lora, use_tqdm=False)
    if len(replies) != len(prompts) or len({r.request_id for r in replies}) != len(prompts):
        raise RuntimeError('vLLM returned missing, duplicate or unexpected replies')
    for prompt, reply in zip(prompts, replies, strict=True):
        if reply.prompt_token_ids != prompt['prompt_token_ids']:
            raise RuntimeError('vLLM reply did not match its submitted prompt tokens')
        if not reply.finished or len(reply.outputs) != 1 or reply.outputs[0].index != 0:
            raise RuntimeError('vLLM did not return exactly one finished completion per request')
        answer = reply.outputs[0]
        if answer.finish_reason not in ('stop', 'length'):
            raise RuntimeError('vLLM returned an incomplete completion')
        yield answer


class VLLMGraphRollout(LoRAEngine):
    def generate(self, ref, rows, count, round_number, sample, *, adapter=None, temperature=0.7):
        config = self.config
        if config.get('device', 'cuda') != 'cuda' or config.get('model_kind') != 'qwen3_5_moe':
            raise ValueError('vllm_graph requires CUDA and model_kind=qwen3_5_moe')
        batch_size = config.get('generation_batch_size', 1)
        if type(batch_size) is not int or not 1 <= batch_size <= _GRAPH_SIZES[-1]:
            raise ValueError('vllm_graph generation_batch_size must be in [1, 16]')
        max_length = config.get('max_length', 2048)
        if type(max_length) is not int or not 1 <= max_length <= _MAX_CONTEXT:
            raise ValueError('vllm_graph max_length must be in [1, 65536]')
        max_tokens = config.get('max_new_tokens', 512)
        report_truncated = config.get('report_truncated_attempts', False)
        stops = list(config.get('stop_strings', ()))
        seed = config.get('seed', 73)
        started = time.monotonic()

        from transformers import AutoConfig, AutoTokenizer

        base = Path(ref).resolve(strict=True)
        if AutoConfig.from_pretrained(base, local_files_only=True).model_type != 'qwen3_5_moe':
            raise ValueError('vllm_graph requires a qwen3_5_moe checkpoint')
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        if tokenizer.eos_token_id is None:
            raise ValueError('Tokenizer must define EOS')
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        requests, result = [], [[None] * count for _ in rows]
        skipped = 0
        for index, row in enumerate(rows):
            # vLLM batches unpadded token IDs, never appending a special token or
            # re-tokenizing the templated text across the prompt/answer boundary.
            tokens = tokenizer.encode(self._prefix(tokenizer, row['prompt']), add_special_tokens=False)
            if len(tokens) + max_tokens > max_length:
                if not report_truncated:
                    raise ValueError('Prompt and generation budget exceed max_length')
                result[index] = [dict(status='truncated', finish_reason='prompt_budget', completion=None)
                                 for _ in range(count)]
                skipped += count
            else:
                requests.extend((index, attempt, tokens) for attempt in range(count))

        startup_seconds = decode_seconds = 0.0
        prompt_tokens = output_tokens = length = 0
        if requests:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest

            # An engine exception can otherwise log the raw scheduler inputs.
            logging.getLogger('vllm.logging_utils.dump_input').disabled = True
            lora = (LoRARequest('meta-rsi', 1, str(Path(adapter).resolve(strict=True)))
                    if adapter is not None else None)
            launch = time.monotonic()
            llm = LLM(model=str(base), dtype='bfloat16', tensor_parallel_size=1,
                      max_model_len=_MAX_CONTEXT, enable_lora=True, max_lora_rank=8,
                      gpu_memory_utilization=0.75, disable_log_stats=True, max_num_seqs=_GRAPH_SIZES[-1],
                      limit_mm_per_prompt={'image': 0, 'video': 0},
                      compilation_config={'cudagraph_capture_sizes': _GRAPH_SIZES.copy(),
                                          'max_cudagraph_capture_size': _GRAPH_SIZES[-1]},
                      enable_flashinfer_autotune=False)
            startup_seconds = time.monotonic() - launch
            decode_start = time.monotonic()
            for offset in range(0, len(requests), batch_size):
                batch = requests[offset:offset + batch_size]
                prompts = [{'prompt_token_ids': list(tokens)} for _, _, tokens in batch]
                params = [SamplingParams(max_tokens=max_tokens, stop=stops or None,
                    stop_token_ids=[tokenizer.eos_token_id], temperature=temperature if sample else 0.0,
                    top_k=40 if sample else 0, top_p=1.0,
                    seed=_request_seed(seed, round_number, index * count + attempt) if sample else None)
                    for index, attempt, _ in batch]
                for (index, attempt, tokens), answer in zip(batch, _batch(llm, prompts, params, lora), strict=True):
                    prompt_tokens += len(tokens)
                    output_tokens += len(answer.token_ids)
                    text = answer.text.rstrip() if stops else answer.text
                    if answer.finish_reason == 'length':
                        result[index][attempt] = dict(status='truncated', finish_reason='length', completion=text)
                        length += 1
                    else:
                        result[index][attempt] = text
            decode_seconds = time.monotonic() - decode_start

        print('RSICOMPOSER_WORKER ' + json.dumps(dict(
            backend='vllm_graph', stage='generate' if sample else 'evaluate', round=round_number,
            adapter=adapter is not None, total=len(requests) + skipped, generated=len(requests),
            prompt_budget=skipped, length=length, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
            startup_seconds=startup_seconds, decode_seconds=decode_seconds,
            output_tokens_per_second=output_tokens / decode_seconds if decode_seconds else 0.0,
            elapsed_seconds=time.monotonic() - started)), file=sys.stderr, flush=True)
        return result


if __name__ == '__main__':
    # vLLM and HF inspect their environment at import time. Do not set these
    # in the parent or affect the sibling Transformers/PEFT training worker.
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0', HF_HUB_OFFLINE='1', VLLM_NO_USAGE_STATS='1')
    from .hosted import run_request

    run_request(VLLMGraphRollout)
