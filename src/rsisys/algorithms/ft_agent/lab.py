"""Frozen-data LLM fine-tuning Lab for the FT-Agent reference policy: a MathLab over checkpoints.

The dataset manifest has train, dev and sealed test splits, each selecting its
answer verifier; the Lab's catalog adds math_verify_gsm8k, which grades GSM8K
`#### n` references with Math-Verify after converting them to a final box.
Development and test scoring reuse MathLab's solve/score/trace.

Data code runs in a filesystem/network-isolated CPU sandbox. The trusted Lab
owns native LLaMA-Factory execution and independent development/test scoring.
Only development metrics are delivered during research.
External training backend (Apache-2.0), installed separately:
https://github.com/hiyouga/LLaMA-Factory/tree/ca75f1edf3cb50343ed1c98605141c3e22075b5f
"""
import json
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

from rsisys.algorithms.ft_agent.upstream import LLAMA_FACTORY_REVISION
from rsisys.api import WorkerLostError
from rsisys.api import (
    MATH_VERIFIERS, AnswerCheck, MathAttempt, MathLab, MathVerifyVerifier, MetricSpec, ModelRef, OperationCall,
    OperationResult,
    OperationSpec,
    WeightEngine,
    atomic_write,
    checkout,
    file_hash,
    validate_engine,
)

TRAIN_FIELDS = {
    'stage', 'do_train', 'finetuning_type', 'template', 'cutoff_len',
    'max_samples', 'val_size', 'packing', 'overwrite_cache', 'preprocessing_num_workers',
    'per_device_train_batch_size', 'per_device_eval_batch_size', 'gradient_accumulation_steps',
    'learning_rate', 'num_train_epochs', 'max_steps', 'lr_scheduler_type', 'warmup_ratio',
    'warmup_steps', 'weight_decay', 'max_grad_norm', 'bf16', 'fp16', 'gradient_checkpointing',
    'logging_strategy', 'logging_steps', 'save_strategy', 'save_steps', 'eval_strategy',
    'eval_steps', 'load_best_model_at_end', 'save_total_limit', 'metric_for_best_model',
    'greater_is_better', 'ddp_find_unused_parameters', 'seed', 'data_seed',
    'lora_rank', 'lora_alpha', 'lora_dropout', 'lora_target', 'flash_attn',
    'optim', 'plot_loss', 'dataloader_num_workers', 'report_to', 'trust_remote_code',
    'model_name_or_path', 'dataset', 'dataset_dir', 'output_dir', 'tokenized_path',
    'overwrite_output_dir', 'ddp_timeout', 'save_safetensors', 'train_on_prompt',
}


def evaluation_answer(answer):
    """Convert GSM8K's final numeric label to the verifier's boxed notation.

    Keep the original reasoning and answer untouched for native SFT; only the
    verifier sees the boxed form.
    """
    from rsisys.api import numeric_answer
    value = numeric_answer(answer)
    return '\\boxed{' + str(value) + '}' if value is not None else answer


class GSM8KMathVerify(MathVerifyVerifier):
    """Math-Verify on the final box, against GSM8K references written as `#### n`."""
    spec = {'name': 'math_verify_gsm8k'}

    def __init__(self):
        super().__init__()
        self.name = self.name + '-gsm8k-reference'

    def parse_reference(self, answer):
        return super().parse_reference(evaluation_answer(answer) if isinstance(answer, str) else answer)

    def check(self, completion, answer):
        return super().check(completion, evaluation_answer(answer) if isinstance(answer, str) else answer)


def validate_config(config):
    allowed = {'initial_model', 'manifest',
        'llama_factory_root', 'training_python', 'inference_python', 'inference_cuda_home',
        'training_cuda_home', 'max_tokens', 'max_length', 'enforce_eager',
        'seed', 'output_subdir', 'max_epochs', 'max_steps', 'max_train_samples',
        'max_batch_size', 'max_cutoff_len', 'data_timeout'}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError('Unknown llm_finetune Lab option')
    for key in ('initial_model', 'manifest', 'llama_factory_root'):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError('Missing llm_finetune ' + key)
    validate_engine(config)
    for key, default in (('max_epochs', 2), ('max_steps', 32), ('max_train_samples', 128),
                         ('max_batch_size', 8), ('max_cutoff_len', 1024), ('data_timeout', 120)):
        if type(config.get(key, default)) is not int or config.get(key, default) < 1:
            raise ValueError(key + ' must be a positive integer')


def bounded_train_config(raw, config, directory, *, world_size=1):
    """Reject unsupported/resource-exceeding proposals; never clamp strategy."""
    if not isinstance(raw, dict) or set(raw) - TRAIN_FIELDS:
        raise ValueError('Unsupported LLaMA-Factory fields: ' + str(set(raw or {}) - TRAIN_FIELDS))
    raw = deepcopy(raw)
    # PyYAML's YAML 1.1 parser represents e.g. 1e-4 as a string. Normalize
    # numerical spelling without changing its value or the proposed schedule.
    numeric = {'num_train_epochs', 'learning_rate', 'cutoff_len', 'per_device_train_batch_size',
        'per_device_eval_batch_size', 'gradient_accumulation_steps', 'lora_rank', 'lora_alpha',
        'val_size', 'lora_dropout', 'warmup_ratio', 'weight_decay', 'max_steps', 'max_samples'}
    import re
    for key in numeric & set(raw):
        if isinstance(raw[key], str) and re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', raw[key]):
            value = float(raw[key])
            raw[key] = int(value) if value.is_integer() else value
    if raw.get('stage', 'sft') != 'sft' or raw.get('finetuning_type', 'lora') != 'lora':
        raise ValueError('This installed reference Lab supports LoRA SFT only')
    for key in ('trust_remote_code', 'fp16', 'train_on_prompt'):
        if raw.get(key, False) is not False:
            raise ValueError(key + ' is not supported by this Lab')
    limits = {'num_train_epochs': (0, config.get('max_epochs', 2)),
        'learning_rate': (0, 0.01), 'cutoff_len': (0, config.get('max_cutoff_len', 1024)),
        'per_device_train_batch_size': (0, config.get('max_batch_size', 8)),
        'per_device_eval_batch_size': (0, config.get('max_batch_size', 8)),
        'gradient_accumulation_steps': (0, 64), 'lora_rank': (0, 128),
        'lora_alpha': (0, 512), 'val_size': (0, 0.3), 'lora_dropout': (0, 0.5),
        'warmup_ratio': (0, 0.3), 'weight_decay': (0, 1)}
    for key, (low, high) in limits.items():
        if key in raw:
            value = raw[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(key + ' exceeds installed resource bounds')
            if key not in ('val_size', 'lora_dropout', 'warmup_ratio', 'weight_decay') and value <= 0:
                raise ValueError(key + ' must be positive')
    if raw.get('max_steps', -1) != -1 and not 1 <= raw['max_steps'] <= config.get('max_steps', 32):
        raise ValueError('max_steps exceeds the declared training budget')
    if raw.get('max_samples', config.get('max_train_samples', 128)) > config.get('max_train_samples', 128):
        raise ValueError('max_samples exceeds the declared training budget')
    if raw.get('flash_attn', 'sdpa') not in ('auto', 'sdpa', 'fa2', 'disabled'):
        raise ValueError('Unsupported attention backend')
    if raw.get('template', 'qwen') != 'qwen':
        raise ValueError('This reference recipe requires the Qwen training template')
    if raw.get('model_name_or_path', '/models/target') not in ('/models/target', str(Path(config['initial_model']).resolve())):
        raise ValueError('Training must start from the configured base model')
    for key, expected in (('dataset', 'processed_data'), ('dataset_dir', './'), ('output_dir', './output'),
                          ('tokenized_path', './tokenized_cache')):
        if key in raw and raw[key] != expected:
            raise ValueError('Proposal changed Lab-owned ' + key)
    if raw.get('report_to') not in (None, 'none', [], ['none']):
        raise ValueError('External reporting is not installed')
    # The directory and base are execution bindings, advertised before proposal.
    result = dict(raw, model_name_or_path=str(Path(config['initial_model']).resolve()),
        dataset='processed_data', dataset_dir=str(directory), output_dir=str(directory / 'output'),
        tokenized_path=str(directory / 'tokenized_cache'), report_to='none',
        trust_remote_code=False, overwrite_output_dir=False, save_safetensors=True)
    result.setdefault('stage', 'sft')
    result.setdefault('do_train', True)
    result.setdefault('finetuning_type', 'lora')
    result.setdefault('template', 'qwen')
    result.setdefault('bf16', True)
    result.setdefault('cutoff_len', config.get('max_cutoff_len', 1024))
    result.setdefault('num_train_epochs', 1)
    result.setdefault('learning_rate', 1e-4)
    result.setdefault('per_device_train_batch_size', 1)
    result.setdefault('per_device_eval_batch_size', 1)
    result.setdefault('gradient_accumulation_steps', 1)
    result.setdefault('ddp_find_unused_parameters', False)
    result.setdefault('dataloader_num_workers', 0)
    result.setdefault('seed', config.get('seed', 7308))
    # Check the effective step count before permitting a full run.
    rows = json.loads((directory / 'data.json').read_text())
    size = min(len(rows), raw.get('max_samples', len(rows)))
    batch = result['per_device_train_batch_size'] * result['gradient_accumulation_steps'] * world_size
    estimated = math.ceil(size / batch) * math.ceil(result['num_train_epochs'])
    effective = result.get('max_steps', -1)
    if (effective if effective > 0 else estimated) > config.get('max_steps', 32):
        raise ValueError('Proposed effective training schedule exceeds max_steps budget')
    return result


class LLMFinetuneLab(MathLab):
    source_budgets = False  # Generation is set by this Lab, not by a benchmark.
    verifiers = {**MATH_VERIFIERS, 'math_verify_gsm8k': GSM8KMathVerify}

    def __init__(self, context, config, *, engine=None):
        validate_config(config)
        from rsisys.api import execution_capabilities
        self.resources, self.executor = execution_capabilities(context)
        self.resources.declare(self.resource_roles())
        self.config = deepcopy(config)
        self.root = Path(context.data_dir).resolve() / config.get('output_subdir', 'llm-finetune')
        # A resumed run rebuilds the Lab over its own existing output directory.
        self.root.mkdir(parents=True, exist_ok=bool(getattr(context, 'restoring', False)))
        self.engine = engine or WeightEngine({**config, 'stop': ['<|im_end|>', '<|endoftext|>']})
        bind = getattr(self.engine, 'bind_execution', None)
        if callable(bind):
            bind(context)
        MathLab.__init__(self, context, {
            'manifest': config['manifest'], 'split_roles': {'train': 'train', 'dev': 'dev', 'test': 'sealed'},
            'max_output_tokens': config.get('max_tokens', 512)})
        if len(self.datasets['train']) > config.get('max_train_samples', 128):
            raise ValueError('Frozen training dataset exceeds max_train_samples')
        self.models = {0: str(Path(config['initial_model']).resolve(strict=True))}
        self.prepared, self.seen = {}, set()
        self.initialized = False
        self.selected = None

    def bind_actor(self, context, config):
        # The measured models are fine-tuned checkpoints the engine loads.
        return ModelRef('local-weights', getattr(self.engine, 'name', 'llm-finetune/v1'), 'checkpoint')

    def resource_roles(self):
        from rsisys.api import ResourceRequest
        return {'trainer': ResourceRequest(gpus=1), 'rollout': ResourceRequest(gpus=1),
                'processor': ResourceRequest()}

    _STATE = ('models', 'prepared', 'seen', 'initialized', 'binding', 'selected')

    def state(self):
        return {'models': dict(self.models), 'seen': sorted(self.seen), 'initialized': self.initialized,
                'prepared': {k: {**v, 'directory': str(v['directory'])} for k, v in self.prepared.items()},
                'binding': self.binding(), 'selected': self.selected}

    def restore(self, state):
        if set(state) != set(self._STATE):
            raise ValueError('Unexpected FT-Agent Lab state')
        if state['binding'] != self.binding():
            raise ValueError('Frozen FT-Agent dataset changed while the run was stopped')
        if state['models'].get(0) != self.models[0]:
            raise ValueError('FT-Agent base model path differs from the checkpoint')
        self.models = dict(state['models'])
        self.seen = {tuple(key) for key in state['seen']}
        self.initialized = state['initialized']
        self.selected = state['selected']
        self.prepared = {k: {**v, 'directory': Path(v['directory'])} for k, v in state['prepared'].items()}
        # The host checkpoints right before an operation starts. When that
        # operation then finished and the host reuses its journaled receipt, its
        # effects are recorded only in this Lab's own receipt files: absorb them.
        for path in sorted(self.root.glob('*-receipt.json')):
            self._absorb(path)
        for iteration, prepared in self.prepared.items():
            if (file_hash(prepared['directory'] / 'process_data.py') != prepared['script']
                    or file_hash(prepared['directory'] / 'data.json') != prepared['data']):
                raise ValueError(f'Prepared FT-Agent data of round {iteration} changed while the run was stopped')
        if self.selected is not None and self.selected not in self.models:
            raise ValueError('FT-Agent selection references an unknown model')
        for iteration, model in self.models.items():
            if iteration and not Path(model, 'config.json').is_file():
                raise ValueError(f'Trained FT-Agent model of round {iteration} is missing: {model}')

    def _absorb(self, path):
        data = json.loads(path.read_text())
        name, request = path.name[:-len('-receipt.json')].rsplit('-', 1)[0], data.get('request', {})
        key = (name, request.get('iteration', 0))
        if path.name != f'{key[0]}-{key[1]}-receipt.json':
            raise ValueError(f'Unexpected FT-Agent receipt file: {path}')
        if key in self.seen:
            return
        self.seen.add(key)
        if not data.get('ok'):
            return
        directory = self.root / f'round-{key[1]}'
        if name == 'baseline':
            self.initialized = True
        elif name == 'prepare':
            self.prepared[key[1]] = {'directory': directory, 'script': file_hash(directory / 'process_data.py'),
                                     'data': data['dataset_sha256']}
        elif name == 'train':
            self.models[key[1]] = str(directory / 'model')
        elif name == 'select':
            self.selected = key[1]

    def profiles(self):
        # Native prepare/train bundles remain intact; this is not weight-training/v1.
        return ()

    def primary_metric(self):
        return MetricSpec('accuracy', 'maximize', self.datasets['dev'][0].verifier.name)

    def final_evaluations(self):
        if self.selected is None:
            return ()
        return tuple(OperationCall('evaluate', {'iteration': iteration, 'split': 'test'})
                     for iteration in dict.fromkeys((0, self.selected)))

    def describe(self):
        specs = []
        for name in ('baseline', 'prepare', 'train', 'select', 'evaluate'):
            fields = {} if name == 'baseline' else {'iteration': {'type': 'integer', 'minimum': 0}}
            if name == 'evaluate':
                fields['split'] = {'type': 'string', 'enum': ['dev', 'test']}
            if name in ('prepare', 'train'):
                fields['files'] = {'type': 'object', 'additionalProperties': {'type': 'string'}}
            specs.append(OperationSpec(name, 'FT-Agent ' + name, {'type': 'object',
                'properties': fields, 'required': list(fields), 'additionalProperties': False}))
        return tuple(specs)

    def execute(self, call):
        if call.name not in {s.name for s in self.describe()}:
            raise ValueError('Unknown FT-Agent Lab operation')
        schema = next(s.input_schema for s in self.describe() if s.name == call.name)
        if set(call.arguments) != set(schema['required']):
            raise ValueError('Use documented FT-Agent operation arguments')
        iteration = call.arguments.get('iteration', 0)
        if type(iteration) is not int or iteration < 0:
            raise ValueError('Invalid FT-Agent iteration')
        split = call.arguments.get('split')
        if call.name == 'evaluate' and split not in ('dev', 'test'):
            raise ValueError('Unknown FT-Agent split')
        key = (call.name + '-' + split if call.name == 'evaluate' else call.name, iteration)
        if key in self.seen:
            raise ValueError('FT-Agent operation replay requires a recovery adapter')
        self.seen.add(key)
        # Domain failures are usable experiment feedback. Transport/host failures
        # remain distinct exceptions from the host operation itself.
        try:
            data = getattr(self, '_' + call.name)(call.arguments)
            data['ok'] = True
        except WorkerLostError:
            raise
        except Exception as exc:
            data = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
        data['request'] = deepcopy(call.arguments)
        # Atomic: on resume these receipts are this Lab's record of completed work.
        atomic_write(self.root / f'{key[0]}-{key[1]}-receipt.json', json.dumps(data, indent=2).encode())
        return OperationResult('FT-Agent Lab ' + call.name, (), False, data=data)

    def _score(self, model, split, directory):
        # The worker directory is where the engine writes samples; it stays out of the stored evidence.
        self._directory = directory
        return self.evaluate(None, {'split': split, 'model': model})

    def problems(self, arguments):
        return self.datasets[arguments['split']]

    def solve(self, problems, arguments):
        prompts = ['<|im_start|>user\n' + problem.prompt +
            '\nPlease reason step by step and put the final answer in \\boxed{}.<|im_end|>\n<|im_start|>assistant\n'
            for problem in problems]
        outputs = self.engine.generate(arguments['model'], prompts, self._directory, sample=False)
        return [MathAttempt(output['completion']) for output in outputs]

    def result_data(self, records, ref, arguments):
        correct = sum(row['correct'] for row in records)
        return {'benchmark': {'accuracy_summary': [{'dataset': 'gsm8k-' + arguments['split'],
                    'accuracy': correct / len(records), 'questions': len(records), 'correct': correct}],
                'error_samples': [{'question': row['question'], 'gold': row['expected'],
                                   'model_output': row['completion']} for row in records if not row['correct']][:5]}}

    def _baseline(self, args):
        if args:
            raise ValueError('baseline accepts no arguments')
        assets = self.root / 'seed-data' / 'seed'
        assets.mkdir(parents=True, exist_ok=False)
        raw = [{'instruction': row.prompt, 'input': '', 'output': row.answer} for row in self.datasets['train']]
        (assets / 'train.json').write_text(json.dumps(raw))
        benchmark = self._score(self.models[0], 'dev', self.root / 'baseline-dev').data['benchmark']
        self.initialized = True
        parameter_info = {'methods': ['lora'], 'supported_models': {'target': 'target'}, 'parameters': {
            'model': {}, 'data': {}, 'training': {}, 'finetuning': {'lora': {}}}}
        for key in sorted(TRAIN_FIELDS):
            bucket = parameter_info['parameters']['finetuning']['lora'] if key.startswith('lora_') else parameter_info['parameters']['training']
            bucket[key] = {'type': 'LLaMA-Factory parameter', 'default': None, 'help': 'Supported parameter: ' + key}
        description = (
            'Frozen GSM8K mathematical reasoning integration task, not an FT-Dojo benchmark reproduction. '
            'The target model is Qwen2.5-7B Base, local alias target. Available training backend: '
            f'LLaMA-Factory v0.9.3 LoRA SFT on {self.resources.request("trainer").gpus} GPUs. '
            'Data processing executes with Python standard library only, no network, LLM API or extra packages. '
            'Read /datasets/seed/train.json (an Alpaca JSON list with instruction/input/output); '
            'write ./data.json in Alpaca format. Existing outputs contain supplied reasoning. '
            f'Use at most {len(raw)} examples, {self.config.get("max_epochs", 2)} epochs, '
            f'{self.config.get("max_steps", 32)} optimizer steps, cutoff_len <= {self.config.get("max_cutoff_len", 1024)}, '
            f'per-device batch <= {self.config.get("max_batch_size", 8)} and LoRA rank <=128. '
            'No DeepSpeed/quantization/PiSSA; installed attention is sdpa or fa2. Qwen template, bf16. '
            'Numerical bounds: learning_rate (0,0.01], gradient accumulation <=64, LoRA alpha <=512, '
            'val_size/warmup_ratio <=0.3, LoRA dropout <=0.5, weight_decay <=1. '
            'Use dataset processed_data, dataset_dir ./, model_name_or_path /models/target, '
            'output_dir ./output, tokenized_path ./tokenized_cache. External logging disabled (report_to none). '
            f'Evaluation uses the fixed Qwen user/assistant chat template, greedy decoding and a {self.config.get("max_tokens", 512)}-token cap. '
            'Return the final numerical answer inside \\boxed{}. Development feedback is visible; final test is withheld. '
            'The configured base is the training start for each experiment. The host rejects unsupported proposals; '
            'it never changes strategy or selects the winner. Your native feedback Decision determines SOTA.')
        preview = {'path': '/datasets/seed/train.json', 'format': 'JSON list', 'rows': len(raw),
            'fields': {'instruction': 'problem text', 'input': 'empty optional context',
                       'output': 'GSM8K reference reasoning with #### numerical final answer'},
            'samples': raw[:3]}
        return {'benchmark': benchmark, 'policy_context': {'benchmark': 'gsm8k-dev', 'scenario': description,
            'dataset_description': json.dumps(preview, ensure_ascii=False, indent=2),
            'dataset_info': {'seed': {'tasks': {'_root': {'files': ['train.json'], 'sample_count': len(raw)}},
                'total_samples': len(raw), 'file_tree': 'train.json', 'readme': json.dumps(preview, ensure_ascii=False)}},
            'parameter_info': parameter_info}}

    def _prepare(self, args):
        if not self.initialized:
            raise ValueError('baseline must precede preparation')
        iteration, files = args['iteration'], args['files']
        if type(iteration) is not int or iteration < 1 or not isinstance(files, dict):
            raise ValueError('Invalid preparation request')
        source = files.get('process_data.py')
        if not isinstance(source, str) or not source.strip() or len(source) > 100000:
            raise ValueError('Missing or oversized generated data script')
        directory = self.root / f'round-{iteration}'
        directory.mkdir(parents=True, exist_ok=False)
        script = directory / 'process_data.py'
        script.write_text(source)
        sandbox = directory / 'sandbox'
        sandbox.mkdir()
        (sandbox / 'process_data.py').write_text(source)
        runner = Path(__file__).with_name('data_sandbox.py')
        command = ['bwrap', '--die-with-parent', '--new-session', '--unshare-all', '--cap-drop', 'ALL',
            '--ro-bind', '/usr', '/usr', '--ro-bind', '/lib', '/lib', '--ro-bind', '/lib64', '/lib64',
            '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp',
            '--ro-bind', str(self.root / 'seed-data'), '/datasets',
            '--bind', str(sandbox), '/work', '--ro-bind', str(runner), '/runner.py',
            '--chdir', '/work', '--clearenv', '--setenv', 'PATH', '/usr/bin',
            '--', '/usr/bin/python3', '-I', '/runner.py']
        with (directory / 'data-processing.log').open('w') as stream:
            with self.resources.acquire('processor') as lease:
                from rsisys.api import WorkerSpec
                code = self.executor.run(WorkerSpec(command, {}, stdout=stream,
                    timeout=self.config.get('data_timeout', 120)), lease)
        if code:
            raise RuntimeError('Data processing failed: ' + (directory / 'data-processing.log').read_text()[-6000:])
        output = sandbox / 'data.json'
        import stat
        if not stat.S_ISREG(output.lstat().st_mode):
            raise ValueError('Data output must be a regular file, not a symlink')
        text = output.read_text()
        rows = json.loads(text)
        if not isinstance(rows, list) or not rows or len(rows) > self.config.get('max_train_samples', 128):
            raise ValueError('Generated dataset is empty or exceeds the declared size limit. Processing log:\n' +
                             (directory / 'data-processing.log').read_text()[-6000:])
        for row in rows:
            if (not isinstance(row, dict) or set(row) != {'instruction', 'input', 'output'}
                    or any(not isinstance(row[key], str) for key in row)
                    or not row['instruction'].strip() or not row['output'].strip()):
                raise ValueError('Invalid Alpaca training record')
        (directory / 'data.json').write_text(text)
        (directory / 'dataset_info.json').write_text(json.dumps({'processed_data': {'file_name': 'data.json'}}))
        self.prepared[iteration] = {'directory': directory, 'script': file_hash(script), 'data': file_hash(directory / 'data.json')}
        stats = {'total_samples': len(rows), 'stdout_summary': f'Estimated full output: {len(rows)}',
            'average_characters': sum(len(row['instruction']) + len(row['output']) for row in rows) / len(rows)}
        (directory / 'data_stats.json').write_text(json.dumps(stats))
        return {'data_stats': stats, 'dataset_sha256': self.prepared[iteration]['data']}

    def _train(self, args):
        import yaml

        from rsisys.api import WorkerSpec
        with self.resources.acquire('trainer') as lease:
            started = time.monotonic()
            iteration, files = args['iteration'], args['files']
            prepared = self.prepared[iteration]
            directory = prepared['directory']
            if files.get('process_data.py') != (directory / 'process_data.py').read_text():
                raise ValueError('Data script changed after preparation')
            if file_hash(directory / 'data.json') != prepared['data']:
                raise ValueError('Prepared data changed')
            raw = yaml.safe_load(files['train.yaml'])
            recipe = bounded_train_config(raw, self.config, directory, world_size=len(lease.devices))
            (directory / 'proposed-train.yaml').write_text(files['train.yaml'])
            (directory / 'test_params.yaml').write_text(files.get('test_params.yaml', ''))
            (directory / 'effective-train.json').write_text(json.dumps(recipe, indent=2))
            upstream = checkout(self.config['llama_factory_root'], LLAMA_FACTORY_REVISION)
            python = self.config.get('training_python', sys.executable)
            env = self.engine._env()
            env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, lease.devices)),
                PYTHONPATH=str(upstream / 'src') + os.pathsep + env.get('PYTHONPATH', ''),
                HF_HUB_OFFLINE='1', WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false')
            if self.config.get('training_cuda_home'):
                cuda = self.config['training_cuda_home']
                env.update(CUDA_HOME=cuda, PATH=cuda + '/bin' + os.pathsep + env.get('PATH', ''),
                           LD_LIBRARY_PATH=cuda + '/lib64' + os.pathsep + env.get('LD_LIBRARY_PATH', ''))
            for mode in ('micro', 'train', 'merge'):
                request = directory / (mode + '-request.json')
                request.write_text(json.dumps({'mode': mode, 'recipe': recipe, 'directory': str(directory),
                    'base_model': self.models[0], 'llama_factory_root': str(upstream),
                    'world_size': len(lease.devices)}))
                logs = directory / (mode + '-workers')
                logs.mkdir()
                command = [python]
                mode_env = dict(env)
                if mode != 'merge':
                    command += ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                                '--nproc-per-node=' + str(len(lease.devices))]
                else:
                    mode_env['CUDA_VISIBLE_DEVICES'] = str(lease.devices[0])
                command += ['-m', 'rsisys.algorithms.ft_agent.train_worker', str(request)]
                with (logs / 'worker-0.log').open('w') as log:
                    code = self.executor.run(WorkerSpec(command, mode_env, stdout=log,
                        devices=lease.devices[:1] if mode == 'merge' else lease.devices), lease)
                if code:
                    raise RuntimeError('FT-Agent worker failed: ' + (logs / 'worker-0.log').read_text()[-6000:])
            training = json.loads((directory / 'training-result.json').read_text())
            if not training.get('weights_changed') or training.get('steps', 0) < 1:
                raise ValueError('No verified trained model was published')
            model = str(directory / 'model')
        benchmark = self._score(model, 'dev', directory / 'dev-evaluation').data['benchmark']
        self.models[iteration] = model
        return {'result': {'benchmark': benchmark, 'training_metrics': training},
                'training_performed': True, 'elapsed_seconds': time.monotonic() - started}

    def _select(self, args):
        iteration = args['iteration']
        if iteration not in self.models:
            raise ValueError('Selected experiment has no verified model')
        self.selected = iteration
        return {'selected_iteration': iteration, 'selected_model': self.models[iteration],
                'selection_recorded': True, 'training_performed': len(self.models) > 1,
                'paper_gain_reproduced': False}

    def _evaluate(self, args):
        iteration, split = args['iteration'], args['split']
        if iteration not in self.models:
            raise ValueError('Experiment has no verified model')
        result = self._score(self.models[iteration], split, self.root / f'evaluate-{split}-{iteration}')
        return {'iteration': iteration, 'split': split, 'benchmark': result.data['benchmark'],
                'quality_metrics': {'accuracy': result.data['quality_metrics']['accuracy']}, 'evidence_refs': result.data['evidence_refs'],
                'training_performed': iteration != 0, 'paper_gain_reproduced': False}
