"""Temporary CPU or single-GPU Transformers/PEFT training and serving worker.

Data schema belongs to this example: JSONL objects with prompt and answer.
No framework-wide dataset conversion is introduced.
"""
import argparse
from contextlib import contextmanager
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import traceback


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--host', help='Bind address; defaults to 127.0.0.1 on CPU, 0.0.0.0 on CUDA')
    parser.add_argument('--served-name', default='Qwen3.5-9B')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--steps', type=int, default=24)
    parser.add_argument('--rounds', type=int, choices=(1, 2), default=1)
    parser.add_argument('--base-revision', default='baseline')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.threads < 1:
        parser.error('steps and threads must be positive integers')
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    if args.host is not None and not args.host.strip():
        parser.error('host must be nonempty')
    device = args.device
    host = args.host if args.host is not None else ('127.0.0.1' if device == 'cpu' else '0.0.0.0')
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ['MKL_NUM_THREADS'] = str(args.threads)
    os.environ['ACCELERATE_USE_CPU'] = 'true' if device == 'cpu' else 'false'
    if device == 'cpu':
        # Hide GPUs before importing ML libraries, including their Trainer helpers.
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        # Resolve the scheduler's UUID after Pixi activation. This cluster's CUDA
        # driver needs a physical ordinal even though NVML accepts the UUID.
        allocated = os.environ.get('NVIDIA_VISIBLE_DEVICES', '')
        if allocated.startswith('GPU-') and ',' not in allocated:
            rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                            '--format=csv,noheader'], text=True).splitlines()
            matches = [row.split(',')[0].strip() for row in rows if row.split(',')[1].strip() == allocated]
            if len(matches) != 1:
                raise RuntimeError('Cannot uniquely map the allocated GPU UUID')
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = matches[0]
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback
    from peft import LoraConfig, get_peft_model
    torch.set_num_threads(args.threads)
    if device == 'cuda':
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES')
        print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,
                          'gpu': torch.cuda.get_device_name(0),
                          'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}), flush=True)
        torch.manual_seed(73)
    else:
        # torch.manual_seed also seeds CUDA; seed only the CPU generator here.
        torch.random.default_generator.manual_seed(73)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer requires an EOS token')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32 if device == 'cpu' else torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True).to(device)
    model.config.use_cache = False
    lora_config = LoraConfig(r=8, lora_alpha=16,
        target_modules=['q_proj', 'v_proj'], lora_dropout=0.0, task_type='CAUSAL_LM')
    model = get_peft_model(model, lora_config)
    model.eval()
    jobs, keys, lock, job_lock = {}, {}, threading.Lock(), threading.Lock()
    cancellations = {}
    revision = 'candidate-1'
    trained = False
    training_attempts = 0
    revisions = {}

    def run_path(name):
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise ValueError('Path escapes run directory')
        return path

    class Cancelled(Exception):
        pass

    def check_cancel(ident):
        if cancellations[ident].is_set():
            raise Cancelled('Cancellation requested')

    class CancelTraining(TrainerCallback):
        def __init__(self, event):
            self.event = event

        def on_step_end(self, args, state, control, **kwargs):
            if self.event.is_set():
                control.should_training_stop = True
                control.should_epoch_stop = True
            return control

        on_train_begin = on_step_end
        on_step_begin = on_step_end
        on_substep_end = on_step_end

    def read_data(name):
        if name not in ('train.jsonl', 'eval.jsonl'):
            raise ValueError('Unknown example dataset')
        rows = [json.loads(line) for line in run_path(name).read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError('Empty dataset')
        if any(not isinstance(row, dict) or any(
                not isinstance(row.get(key), str) or not row[key].strip()
                for key in ('prompt', 'answer')) for row in rows):
            raise ValueError('Dataset rows require nonempty prompt and answer strings')
        return rows

    def encode(row):
        prefix = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
        answer_ids = tokenizer(row['answer'] + tokenizer.eos_token, add_special_tokens=False)['input_ids']
        ids = prompt_ids + answer_ids
        if len(ids) > 256:
            raise ValueError('Toy sample exceeds token budget')
        return {'input_ids': ids, 'labels': [-100] * len(prompt_ids) + answer_ids}

    def collate(rows):
        size = max(len(row['input_ids']) for row in rows)
        return {
            'input_ids': torch.tensor([r['input_ids'] + [tokenizer.pad_token_id] * (size-len(r['input_ids'])) for r in rows]),
            'attention_mask': torch.tensor([[1]*len(r['input_ids'])+[0]*(size-len(r['input_ids'])) for r in rows]),
            'labels': torch.tensor([r['labels']+[-100]*(size-len(r['labels'])) for r in rows]),
        }

    @contextmanager
    def use_revision(selected):
        if selected == args.base_revision:
            with model.disable_adapter():
                yield
        else:
            model.set_adapter(revisions[selected]['adapter_name'])
            yield

    def evaluate(ident, selected):
        metrics = {}
        rows = [encode(row) for row in read_data('eval.jsonl')]
        model.eval()
        for name, selected_revision in [('base', revisions[selected]['base_revision']), ('candidate', selected)]:
            weighted_loss, tokens = 0., 0
            with use_revision(selected_revision), torch.no_grad():
                for row in rows:
                    check_cancel(ident)
                    batch = {k:v.to(device) for k,v in collate([row]).items()}
                    count = int((batch['labels'][:, 1:] != -100).sum())
                    loss = float(model(**batch).loss)
                    if not math.isfinite(loss):
                        raise ValueError('Non-finite evaluation loss')
                    weighted_loss += loss * count
                    tokens += count
            metrics[f'{name}_nll'] = weighted_loss / tokens
        metrics['eval_samples'] = len(rows)
        metrics['delta_nll'] = metrics['base_nll'] - metrics['candidate_nll']
        return metrics

    def run_job(ident, kind, payload):
        nonlocal trained, revision
        try:
            with lock:
                with job_lock:
                    check_cancel(ident)
                    jobs[ident]['status'] = 'running'
                    selected = jobs[ident]['revision']
                    parent = jobs[ident]['base_revision']
                    round_number = jobs[ident]['round']
                output_root = root if round_number == 1 else run_path(f'round-{round_number}')
                try:
                    if kind == 'train':
                        adapter_name = 'default' if round_number == 1 else selected
                        if round_number > 1:
                            if parent == args.base_revision:
                                model.add_adapter(adapter_name, lora_config)
                            else:
                                # Continue from a copy of the parent's learned LoRA weights;
                                # the parent's adapter remains intact for evaluation/serving.
                                model.load_adapter(revisions[parent]['adapter_path'],
                                    adapter_name=adapter_name, is_trainable=True,
                                    local_files_only=True)
                        model.set_adapter(adapter_name)
                        model.config.use_cache = False
                        dataset = [encode(row) for row in read_data('train.jsonl')]
                        options = TrainingArguments(output_dir=str(output_root / 'trainer'), max_steps=args.steps,
                            per_device_train_batch_size=2, gradient_accumulation_steps=2,
                            learning_rate=2e-4, use_cpu=device == 'cpu', bf16=device == 'cuda',
                            logging_steps=2, save_strategy='no',
                            report_to='none', gradient_checkpointing=True,
                            gradient_checkpointing_kwargs={'use_reentrant': False},
                            dataloader_num_workers=0, remove_unused_columns=False, seed=73)
                        trainer = Trainer(model=model, args=options, train_dataset=dataset, data_collator=collate,
                                          callbacks=[CancelTraining(cancellations[ident])])
                        check_cancel(ident)
                        output = trainer.train()
                        check_cancel(ident)
                        if trainer.state.global_step < options.max_steps:
                            raise RuntimeError('Training stopped before all requested steps completed')
                        adapter_root = output_root / 'adapter'
                        if round_number == 1:
                            model.save_pretrained(adapter_root)
                        else:
                            model.save_pretrained(adapter_root, selected_adapters=[adapter_name])
                        adapter_path = adapter_root if adapter_name == 'default' else adapter_root / adapter_name
                        tokenizer.save_pretrained(adapter_path)
                        result = {'adapter_path': str(adapter_path), 'revision': selected,
                                  'base_revision': parent, 'round': round_number,
                                  'metrics': {k:float(v) for k,v in output.metrics.items() if isinstance(v,(int,float))}}
                    elif kind == 'evaluate':
                        if not trained:
                            raise ValueError('Candidate is not trained')
                        result = {'metrics': evaluate(ident, selected), 'revision': selected,
                                  'base_revision': parent, 'round': round_number}
                    else:
                        raise ValueError('Unknown job kind')
                finally:
                    if kind == 'train':
                        model.gradient_checkpointing_disable()
                        model.config.use_cache = True
                    model.eval()
                # Cancellation and successful publication have one ordering point.
                # Never publish a partial/cancelled adapter as a trained candidate.
                with job_lock:
                    check_cancel(ident)
                    result_path = output_root / f'{kind}-result.json'
                    temporary = output_root / f'{kind}-result.json.tmp'
                    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
                    temporary.replace(result_path)
                    if kind == 'train':
                        trained = True
                        revision = selected
                        revisions[selected] = {'adapter_name': adapter_name, 'adapter_path': str(adapter_path),
                                               'base_revision': parent, 'round': round_number}
                    jobs[ident].update(status='succeeded', result=result)
        except Exception as exc:
            if not isinstance(exc, Cancelled):
                traceback.print_exc()
            with job_lock:
                status = 'cancelled' if cancellations[ident].is_set() else 'failed'
                jobs[ident].update(status=status, error=f'{type(exc).__name__}: {exc}')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def reply(self, code, value):
            content = json.dumps(value).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        def do_GET(self):
            if self.path == '/health':
                with job_lock:
                    value = {'ready': True, 'trained': trained, 'revision': revision, 'model': args.served_name,
                             'device': device, 'threads': args.threads,
                             'revisions': list(revisions), 'max_rounds': args.rounds,
                             'base_revision': args.base_revision}
                return self.reply(200, value)
            if self.path.startswith('/jobs/'):
                with job_lock:
                    job = dict(jobs.get(self.path.removeprefix('/jobs/'), {}))
                return self.reply(200 if job else 404, job or {'error':'unknown job'})
            self.reply(404, {'error':'not found'})
        def do_POST(self):
            nonlocal training_attempts
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1024*1024 or self.headers.get('Transfer-Encoding'):
                    raise ValueError('Expected a fixed request body of 1..1048576 bytes')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('Request must be a JSON object')
                if self.path == '/cancel':
                    ident = payload.get('job_id')
                    if not isinstance(ident, str) or not ident:
                        raise ValueError('cancel requires a job_id string')
                    with job_lock:
                        if ident not in jobs:
                            return self.reply(404, {'error': 'unknown job'})
                        if jobs[ident]['status'] in ('queued', 'running'):
                            cancellations[ident].set()
                            jobs[ident]['cancel_requested'] = True
                        value = dict(jobs[ident])
                    return self.reply(200, value)
                if self.path in ('/train', '/evaluate'):
                    kind = self.path[1:]
                    key = payload.get('idempotency_key')
                    if not isinstance(key, str) or not key.strip() or len(key) > 512:
                        raise ValueError('idempotency_key must be a nonempty string of at most 512 characters')
                    signature = json.dumps((kind, payload), sort_keys=True, allow_nan=False)
                    ident = sha256(key.encode()).hexdigest()[:24]
                    with job_lock:
                        if key in keys:
                            if keys[key] != signature:
                                raise ValueError('Key reused with different request')
                        else:
                            if any(j['status'] in ('queued','running') for j in jobs.values()):
                                raise ValueError('Worker is busy')
                            request = payload.get('request', {})
                            if not isinstance(request, dict):
                                raise ValueError('request must be an object')
                            if kind == 'train':
                                if training_attempts >= args.rounds:
                                    raise ValueError('Training attempt budget exhausted; use a fresh worker')
                                if len(revisions) != training_attempts:
                                    raise ValueError('A prior training attempt failed; use a fresh worker')
                                if args.rounds > 1 and 'base' not in request:
                                    raise ValueError('Multi-round training requires an explicit parent model')
                                base = request.get('base', {'revision': args.base_revision,
                                    'model_id': args.served_name, 'backend_id': 'qwen-toy'})
                                if (base.get('model_id') != args.served_name or base.get('backend_id') != 'qwen-toy'
                                        or request.get('method', 'sft') != 'sft'
                                        or request.get('recipe_ref', 'toy-lora') != 'toy-lora'):
                                    raise ValueError('Expected the configured model and toy-lora/sft')
                                parent = base.get('revision')
                                if parent != args.base_revision and parent not in revisions:
                                    raise ValueError('Unknown parent model revision')
                                round_number = training_attempts + 1
                                selected = f'candidate-{round_number}'
                                output_root = root if round_number == 1 else run_path(f'round-{round_number}')
                                output_root.mkdir(parents=True, exist_ok=True)
                                for name in ('trainer', 'adapter', 'train-result.json'):
                                    if (output_root / name).exists():
                                        raise ValueError('Training output already exists; use a fresh run directory')
                                read_data('train.jsonl')
                                with (output_root / 'training-attempt.json').open('x') as claim:
                                    json.dump({'job_id': ident, 'base_revision': parent,
                                               'revision': selected, 'round': round_number}, claim)
                                training_attempts += 1
                            else:
                                selected = request.get('candidate', {}).get('candidate_id', revision)
                                if not trained or selected not in revisions:
                                    raise ValueError('Candidate is not trained')
                                parent = revisions[selected]['base_revision']
                                if request.get('candidate', {}).get('base_version', parent) != parent:
                                    raise ValueError('Evaluation parent revision mismatch')
                                if request.get('protocol_ref', 'heldout-nll') != 'heldout-nll':
                                    raise ValueError('Unknown evaluation protocol')
                                round_number = revisions[selected]['round']
                            keys[key] = signature
                            jobs[ident] = {'job_id':ident, 'status':'queued', 'kind': kind,
                                           'revision': selected, 'base_revision': parent, 'round': round_number}
                            cancellations[ident] = threading.Event()
                            try:
                                threading.Thread(target=run_job, args=(ident,kind,payload), daemon=True).start()
                            except Exception as exc:
                                jobs[ident].update(status='failed', error=f'{type(exc).__name__}: {exc}')
                                raise
                    return self.reply(200, {'job_id':ident})
                if self.path == '/v1/chat/completions':
                    if payload.get('model') != args.served_name:
                        raise ValueError('Unknown model; use ' + args.served_name)
                    if payload.get('stream', False) is not False:
                        raise ValueError('Only stream=false is supported')
                    messages = payload.get('messages')
                    if not isinstance(messages, list) or not messages or any(
                            not isinstance(m, dict) or m.get('role') not in ('system', 'user', 'assistant')
                            or not isinstance(m.get('content'), str) for m in messages):
                        raise ValueError('messages must contain text system/user/assistant messages')
                    max_tokens = payload.get('max_tokens', 128)
                    if type(max_tokens) is not int or not 1 <= max_tokens <= 256:
                        raise ValueError('max_tokens must be an integer between 1 and 256')
                    with lock, torch.no_grad():
                        selected = payload.get('model_revision', revision)
                        if not trained or selected not in revisions:
                            raise ValueError('Candidate revision not ready')
                        model.set_adapter(revisions[selected]['adapter_name'])
                        model.eval()
                        text = tokenizer.apply_chat_template(payload['messages'], tokenize=False,
                            add_generation_prompt=True, enable_thinking=False)
                        inputs = tokenizer(text, return_tensors='pt').to(device)
                        if inputs.input_ids.shape[1] + max_tokens > 8192:
                            raise ValueError('Request exceeds the 8192 token context limit')
                        eos = model.generation_config.eos_token_id
                        if eos is None:
                            eos = tokenizer.eos_token_id
                        eos_ids = [eos] if isinstance(eos, int) else list(eos)
                        outputs = model.generate(**inputs, max_new_tokens=max_tokens,
                            do_sample=False, use_cache=True, eos_token_id=eos_ids,
                            pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=False)
                        generated = outputs[0,inputs.input_ids.shape[1]:]
                        answer = tokenizer.decode(generated, skip_special_tokens=True)
                        finish_reason = ('stop' if len(generated) and int(generated[-1]) in eos_ids
                                         else 'length' if len(generated) >= max_tokens else 'stop')
                    return self.reply(200, {'model':payload['model'], 'model_revision':selected,
                        'choices':[{'finish_reason':finish_reason,'message':{'role':'assistant','content':answer}}],
                        'usage':{'prompt_tokens':inputs.input_ids.shape[1],
                                 'completion_tokens':outputs.shape[1]-inputs.input_ids.shape[1]}})
                self.reply(404, {'error':'not found'})
            except Exception as exc:
                self.reply(400, {'error':f'{type(exc).__name__}: {exc}'})
    with ThreadingHTTPServer((host, args.port), Handler) as server:
        ready = {'host': host, 'port': args.port, 'pid': os.getpid(), 'device': device, 'threads': args.threads}
        run_path('worker-ready.json').write_text(json.dumps(ready))
        print(json.dumps({'event': 'worker_ready', **ready}), flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
