"""Transformers chat endpoint with local weights; CPU defaults to loopback binding."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("QWEN_MODEL_PATH"),
                        help="Existing local model directory (or QWEN_MODEL_PATH); never downloaded")
    parser.add_argument("--served-name", default=os.environ.get("QWEN_SERVED_NAME"))
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default=os.environ.get("QWEN_DEVICE", "cpu"))
    parser.add_argument("--host",
                        help="Bind address (default: 127.0.0.1 for CPU, 0.0.0.0 for CUDA)")
    parser.add_argument("--threads", type=positive_int,
                        default=os.environ.get("QWEN_THREADS", "4"))
    parser.add_argument("--port", type=positive_int,
                        default=os.environ.get("QWEN_PORT", "18095"))
    args = parser.parse_args(argv)
    if args.device not in ("cpu", "cuda"):
        parser.error("QWEN_DEVICE must be cpu or cuda")
    if args.host is None:
        args.host = "127.0.0.1" if args.device == "cpu" else "0.0.0.0"
    if args.port > 65535:
        parser.error("port must be at most 65535")
    if not args.model:
        parser.error("provide --model or QWEN_MODEL_PATH; no model is selected by default")
    args.model = str(Path(args.model).expanduser().resolve())
    if not Path(args.model).is_dir():
        parser.error("model must be an existing local directory")
    args.served_name = args.served_name or Path(args.model).name
    return args


def configure_device(device):
    if device == "cpu":
        # Hide accelerators before importing torch/Transformers, including when
        # this script is run directly rather than through serve.sh.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    if device != "cuda":
        raise ValueError("device must be cpu or cuda")
    # This cluster requires physical ordinals: UUID visibility alone lets NVML
    # see a device but causes CUDA cuInit to fail. Never select an unallocated GPU.
    allocated_uuid = os.environ.get("NVIDIA_VISIBLE_DEVICES", "")
    if not allocated_uuid.startswith("GPU-") or "," in allocated_uuid:
        raise RuntimeError("CUDA requires exactly one scheduler-allocated GPU UUID")
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        text=True,
    ).splitlines()
    matches = [row.split(",")[0].strip() for row in rows
               if row.split(",")[1].strip() == allocated_uuid]
    if len(matches) != 1:
        raise RuntimeError("Cannot uniquely map the allocated GPU UUID")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = matches[0]
    print(json.dumps({"allocated_gpu_uuid": allocated_uuid,
                      "cuda_visible_devices": matches[0],
                      "cuda_device_order": "PCI_BUS_ID"}), flush=True)
    return "cuda:0"


def load_runtime(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    device = configure_device(args.device)
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    metadata = {"torch": torch.__version__, "transformers": transformers.__version__,
                "device": args.device, "threads": args.threads}
    if args.device == "cuda":
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one allocated GPU")
        if torch.ones(1, device=device).item() != 1.0:
            raise RuntimeError("CUDA tensor preflight failed")
        metadata.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    print(json.dumps(metadata), flush=True)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    model_class = AutoModelForCausalLM
    if config.model_type == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration
        model_class = Qwen3_5ForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = model_class.from_pretrained(
        args.model, dtype=dtype, device_map=device,
        attn_implementation="sdpa", local_files_only=True,
    ).eval()
    return torch, tokenizer, model, device


def finish_generation(content, generated, max_tokens, eos_token_id, stop=None):
    """Trim user stop markers and distinguish stopping from token exhaustion."""
    eos_ids = ([eos_token_id] if isinstance(eos_token_id, int)
               else list(eos_token_id or []))
    ended_with_eos = len(generated) > 0 and int(generated[-1]) in eos_ids
    markers = [stop] if isinstance(stop, str) else stop or []
    positions = [content.find(marker) for marker in markers if marker and marker in content]
    if positions:
        return content[:min(positions)], 'stop'
    reason = ('stop' if ended_with_eos
              else 'length' if len(generated) >= max_tokens else 'stop')
    return content, reason


def main(argv=None):
    args = parse_args(argv)
    torch, tokenizer, model, device = load_runtime(args)
    model_name = args.served_name

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body):
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/health':
                self.reply(200, {'status': 'ok', 'model': model_name, 'backend': 'transformers',
                                'device': args.device, 'threads': args.threads})
            elif self.path == '/v1/models':
                self.reply(200, {'object': 'list', 'data': [
                    {'id': model_name, 'object': 'model', 'owned_by': 'local', 'created': 0}]})
            else:
                self.reply(404, {'error': {'message': 'Unknown endpoint'}})

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                return self.reply(404, {'error': {'message': 'Unknown endpoint'}})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 1_000_000:
                    raise ValueError('Request body must be 1..1000000 bytes')
                req = json.loads(self.rfile.read(size))
                if req.get('model') != model_name:
                    raise ValueError('Unknown model; use ' + model_name)
                if req.get('stream'):
                    raise ValueError('This temporary endpoint supports stream=false only')
                messages = req['messages']
                if not isinstance(messages, list) or not messages:
                    raise ValueError('messages must be a nonempty list')
                for message in messages:
                    if not isinstance(message.get('content'), str):
                        raise ValueError('Only text message content is supported')
                max_tokens = int(req.get('max_completion_tokens', req.get('max_tokens', 200)))
                if not 1 <= max_tokens <= 200:
                    raise ValueError('max_tokens must be between 1 and 200')
                template_args = req.get('chat_template_kwargs', {})
                if template_args.get('enable_thinking', False):
                    raise ValueError('This endpoint requires enable_thinking=false')
                response_format = req.get('response_format', {}).get('type', 'text')
                if response_format == 'json_object':
                    messages = [{'role': 'system', 'content': 'Return only one valid JSON object, without Markdown fences.'}] + messages
                elif response_format != 'text':
                    raise ValueError('Only text and json_object response formats are supported')
                text = tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                inputs = tokenizer(text, return_tensors='pt').to(device)
                prompt_tokens = inputs.input_ids.shape[-1]
                if prompt_tokens + max_tokens > 8192:
                    raise ValueError('Temporary endpoint context limit is 8192 tokens')
                temperature = float(req.get('temperature', 0.0))
                kwargs = {'do_sample': temperature > 0}
                if temperature > 0:
                    kwargs.update(temperature=temperature, top_p=float(req.get('top_p', 0.8)))
                eos = model.generation_config.eos_token_id
                if eos is None:
                    eos = tokenizer.eos_token_id
                if eos is not None:
                    kwargs['eos_token_id'] = eos
                started = time.monotonic()
                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=max_tokens,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, **kwargs)
                generated = output[0, prompt_tokens:]
                content = tokenizer.decode(generated, skip_special_tokens=True).strip()
                content, finish_reason = finish_generation(
                    content, generated, max_tokens, eos, req.get('stop'))
                if response_format == 'json_object' and not isinstance(json.loads(content), dict):
                    raise ValueError('Model response was not a JSON object')
                count = len(generated)
                self.reply(200, {'id': 'chatcmpl-' + uuid.uuid4().hex, 'object': 'chat.completion',
                    'created': int(time.time()), 'model': model_name,
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content},
                                 'finish_reason': finish_reason}],
                    'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': count,
                              'total_tokens': prompt_tokens + count}})
                print(json.dumps({'event': 'completion', 'prompt_tokens': prompt_tokens,
                                  'completion_tokens': count, 'seconds': round(time.monotonic()-started, 3)}), flush=True)
            except (ValueError, KeyError, TypeError) as exc:
                self.reply(400, {'error': {'message': str(exc), 'type': 'invalid_request_error'}})
            except Exception as exc:
                print('generation_error', type(exc).__name__, str(exc), flush=True)
                self.reply(500, {'error': {'message': 'Generation failed; inspect service log'}})


    server = HTTPServer((args.host, args.port), Handler)
    server.timeout = 30
    print(json.dumps({"event": "ready", "model": model_name, "host": args.host, "port": args.port,
                      "device": args.device, "threads": args.threads,
                      "enable_thinking": False, "max_output_tokens": 200}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
