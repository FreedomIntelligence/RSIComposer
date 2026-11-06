"""Host-owned checkpoint evaluation for the programmable medical Lab.

Only local weights are submitted. No candidate Python, endpoint, tokenizer,
generation configuration or answer parser is executed by this entry point.
This is a trusted executor contract, not isolation from other same-UID programs.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import re
import sys
from uuid import uuid4

from rsisys.adapters.local_lab import LocalProcessLab
from rsisys.loops.contracts.decision import OperationSpec
from rsisys.loops.contracts.execution import LabToolError
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.runtime.files import atomic_write, json_bytes
try:
    from .fixed_evaluation import FixedMedicalLab, PROTOCOL, SYSTEM, fixed_inputs, score_predictions
except ImportError:
    from fixed_evaluation import FixedMedicalLab, PROTOCOL, SYSTEM, fixed_inputs, score_predictions


PROFILE = {'id': 'medical-checkpoint-nonthinking-v1', 'protocol': PROTOCOL,
    'system': SYSTEM, 'enable_thinking': False, 'max_new_tokens': 32,
    'do_sample': False, 'num_beams': 1, 'batch_size': 1, 'generation_defaults': 'fresh_host_config',
    'dtype': 'bfloat16', 'attention': 'eager', 'device_map': 'auto',
    'truncation': False, 'trust_remote_code': False, 'local_files_only': True,
    'weights': 'safetensors', 'answer_parser': 'fullmatch-option-letter-only'}
TOKENIZER_FILES = ('config.json', 'tokenizer_config.json', 'tokenizer.json',
    'vocab.json', 'vocab.txt', 'merges.txt', 'chat_template.jinja', 'added_tokens.json',
    'special_tokens_map.json', 'tokenizer.model', 'spiece.model', 'sentencepiece.bpe.model')


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def file_digest(path):
    before = path.stat()
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('Evaluation input changed while hashing: ' + str(path))
    return {'bytes': after.st_size, 'sha256': value.hexdigest()}


def tokenizer_identity(path):
    path = Path(path)
    if not (path / 'config.json').is_file():
        raise ValueError('Registered base requires config.json')
    names = [name for name in TOKENIZER_FILES if (path / name).is_file()]
    names += [str(p.relative_to(path)) for p in sorted((path / 'chat_templates').glob('*.jinja'))]
    return {name: file_digest(path / name) for name in names}


def weight_identity(path, *, adapter=False):
    path = Path(path)
    if adapter:
        names = ['adapter_config.json', 'adapter_model.safetensors']
        config = json.loads((path / names[0]).read_text())
        if config.get('peft_type') != 'LORA':
            raise ValueError('The fixed checkpoint profile supports built-in LoRA adapters only')
    elif (path / 'model.safetensors').is_file():
        names = ['model.safetensors']
    else:
        index = 'model.safetensors.index.json'
        shards = set(json.loads((path / index).read_text())['weight_map'].values())
        if not shards or any(not isinstance(name, str) or Path(name).name != name
                             or not name.endswith('.safetensors') for name in shards):
            raise ValueError('Invalid local safetensors shard index')
        names = [index, *sorted(shards)]
    files = {name: file_digest(path / name) for name in names}
    return {'files': files, 'sha256': digest(files)}


def source_identity():
    root = Path(__file__).resolve().parent
    return {name: file_digest(root / name)['sha256'] for name in (
        'checkpoint_evaluation.py', 'fixed_evaluation.py', 'candidate_adapter.py')}


def evaluation_inputs(groups):
    payload, gold, datasets = fixed_inputs(groups)
    # Labels come from the option mapping, never from scanning question text.
    labels = {}
    for group in groups:
        for line in Path(group['path']).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                options = list(row['options'])
                if not options or any(re.fullmatch('[A-J]', k) is None for k in options):
                    raise ValueError('Unsupported fixed option label')
                labels[group['benchmark_id'], row['id']] = options
    for item in payload['items']:
        item['option_letters'] = labels[item['benchmark_id'], item['id']]
    return payload, gold, datasets


def answer_letter(text, options):
    match = re.fullmatch(r'([A-J])[.\s]*', text.strip())
    return match.group(1) if match and match.group(1) in options else None


def package_identity():
    result = {}
    for name in ('torch', 'torch-npu', 'transformers', 'accelerate', 'peft', 'safetensors'):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def load_runtime(config):
    import torch
    from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM,
                              AutoModelForImageTextToText, GenerationConfig)
    if config['device_type'] == 'npu':
        import torch_npu  # noqa: F401; registers the installed device backend
        torch.npu.set_compile_mode(jit_compile=False)
    base = config['base_model_path']
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    model_config = AutoConfig.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    loader = (AutoModelForImageTextToText if model_config.model_type in ('qwen3_5', 'qwen3_5_moe')
              else AutoModelForCausalLM)
    model, info = loader.from_pretrained(config['checkpoint_path'], config=model_config,
        local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.bfloat16, attn_implementation=PROFILE['attention'],
        device_map='auto' if config['device_type'] != 'cpu' else None, output_loading_info=True)
    if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('Checkpoint does not exactly fit the registered base architecture: ' + repr(info))
    device_map = {key: str(value) for key, value in getattr(model, 'hf_device_map', {}).items()}
    if config['device_type'] != 'cpu' and (not device_map or any(
            value in ('cpu', 'disk') for value in device_map.values())):
        raise ValueError('Fixed evaluation requires weights on assigned accelerators; no CPU/disk offload')
    model_class = type(model).__name__
    if config.get('adapter_path'):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, config['adapter_path'], is_trainable=False,
            local_files_only=True, use_safetensors=True)
    model.eval()
    # Never load generation defaults from a submitted checkpoint/adapter.
    eos = tokenizer.eos_token_id
    generation = GenerationConfig(max_new_tokens=PROFILE['max_new_tokens'],
        do_sample=False, num_beams=1, use_cache=True,
        eos_token_id=eos, pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos)
    model.generation_config = generation
    if config.get('adapter_path'):
        model.get_base_model().generation_config = generation
    runtime = {'packages': package_identity(), 'loader': loader.__name__, 'model_class': model_class,
        'tokenizer_class': type(tokenizer).__name__,
        'device_map': device_map, 'generation': generation.to_dict(),
        'visibility': {name: os.environ.get(name) for name in (
            'CUDA_VISIBLE_DEVICES', 'ASCEND_RT_VISIBLE_DEVICES')}}
    return torch, tokenizer, model, generation, runtime


def generate_one(torch, tokenizer, model, generation, item):
    prompt = tokenizer.apply_chat_template(item['messages'], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    batch = tokenizer(prompt, return_tensors='pt', add_special_tokens=False, truncation=False)
    batch = batch.to(model.get_input_embeddings().weight.device)
    input_ids = batch['input_ids'][0].tolist()
    with torch.inference_mode():
        generated = model.generate(**batch, generation_config=generation)
    new_ids = generated[0][len(input_ids):].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return {'benchmark_id': item['benchmark_id'], 'id': item['id'], 'raw_text': text,
        'messages_sha256': digest(item['messages']),
        'rendered_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
        'input_ids_sha256': digest(input_ids), 'input_tokens': len(input_ids),
        'generated_token_ids': new_ids, 'output_tokens': len(new_ids),
        'output_limit_hit': len(new_ids) >= PROFILE['max_new_tokens'],
        'answer_letter': answer_letter(text, item['option_letters'])}


def evaluate(config):
    if config['source_sha256'] != source_identity() or config['profile'] != PROFILE:
        raise ValueError('Host evaluation source/profile changed')
    if tokenizer_identity(config['base_model_path']) != config['tokenizer_identity']:
        raise ValueError('Registered tokenizer/config changed')
    payload, gold, datasets = evaluation_inputs(config['groups'])
    weights = weight_identity(config['checkpoint_path'])
    adapter = weight_identity(config['adapter_path'], adapter=True) if config.get('adapter_path') else None
    torch, tokenizer, model, generation, runtime = load_runtime(config)
    # Fingerprint the realized runtime as well as the declared host profile.
    profile_sha = digest({'profile': PROFILE, 'tokenizer': config['tokenizer_identity'],
        'source': config['source_sha256'], 'device_type': config['device_type'], 'runtime': runtime})
    record = {'kind': 'host_checkpoint_development_evaluation', 'protocol': PROTOCOL,
        'candidate_id': config['candidate_id'], 'base_model_path': config['base_model_path'],
        'checkpoint_path': config['checkpoint_path'], 'adapter_path': config.get('adapter_path'),
        'profile': PROFILE, 'profile_sha256': profile_sha, 'runtime': runtime,
        'source_sha256': config['source_sha256'], 'tokenizer_identity': config['tokenizer_identity'],
        'weight_identity': weights, 'adapter_identity': adapter, 'dataset_sha256': datasets,
        'adopted': False, 'traces': []}
    partial = Path(config['output'] + '.partial.json')
    # Completed raw generations survive a later program error/timeout; partial is never a score.
    atomic_write(partial, json_bytes(record))
    for item in payload['items']:
        record['traces'].append(generate_one(torch, tokenizer, model, generation, item))
        atomic_write(partial, json_bytes(record))
    if (weight_identity(config['checkpoint_path']) != weights or
            (weight_identity(config['adapter_path'], adapter=True) if config.get('adapter_path') else None) != adapter or
            tokenizer_identity(config['base_model_path']) != config['tokenizer_identity'] or
            source_identity() != config['source_sha256']):
        raise ValueError('Evaluation inputs changed during inference; no comparable score')
    predictions = [{'benchmark_id': r['benchmark_id'], 'id': r['id'], 'text': r['raw_text']}
                   for r in record['traces']]
    record['metrics'], record['predictions'] = score_predictions(predictions, gold)
    record['diagnostics'] = {'invalid_answers': sum(r['answer_letter'] is None for r in record['traces']),
        'output_limit_hits': sum(r['output_limit_hit'] for r in record['traces'])}
    atomic_write(Path(config['output']), json_bytes(record))
    return record


class CheckpointMedicalLab(FixedMedicalLab):
    """One formal evaluation operation; arbitrary research stays in lab_start."""
    def __init__(self, *args, base_model_paths, device_type, **kwargs):
        super().__init__(*args, **kwargs)
        if device_type not in ('cuda', 'npu', 'cpu'):
            raise ValueError('Unsupported evaluation device type')
        self.checkpoint_runner = Path(__file__).resolve()
        self.formal_configuration = {'profile': deepcopy(PROFILE), 'source_sha256': source_identity(),
            'device_type': device_type, 'bases': {str(Path(p).resolve()): tokenizer_identity(p)
                                                for p in base_model_paths}}
        if not self.formal_configuration['bases']:
            raise ValueError('At least one registered base is required')

    def describe(self):
        text = {'type': 'string', 'minLength': 1}
        return (*LocalProcessLab.describe(self), OperationSpec('medical_evaluate',
            'Submit local weights to the host-owned fixed medical evaluation. Choose a registered '
            'base_model_path; omit checkpoint_path/adapter_path to measure that base, or supply one '
            'workspace checkpoint directory OR LoRA adapter directory. The host owns tokenizer, '
            'chat template, non-thinking greedy generation (32 new tokens), raw output capture, '
            'strict option-letter parsing and scoring. No candidate code or endpoint is used. '
            'Stop other programs using the assigned cards before evaluation. This transient job '
            'uses a program slot and assigned accelerators; collect with lab_poll then lab_result. '
            'Compare only matching profile_sha256 and dataset_sha256; submission is not a score/adoption.',
            {'type': 'object', 'properties': {'key': text, 'candidate_id': text,
                'base_model_path': {'type': 'string', 'enum': list(self.formal_configuration['bases'])},
                'checkpoint_path': text, 'adapter_path': text},
             'required': ['key', 'candidate_id', 'base_model_path'], 'additionalProperties': False}))

    def execute(self, call):
        if call.name != 'medical_evaluate':
            if (call.name == 'lab_start' and isinstance(call.arguments, dict)
                    and call.arguments.get('key') not in self._keys):
                active = [row for row in self._status()['active_jobs'] if row['job_id'] in self._evaluations]
                if active:
                    raise LabToolError('Fixed checkpoint evaluation is using the assigned devices; collect it first',
                        data={'reason': 'evaluation_devices_busy', 'active_jobs': active})
            return LocalProcessLab.execute(self, call)
        args = call.arguments
        if (not isinstance(args, dict) or set(args) - {
                'key', 'candidate_id', 'base_model_path', 'checkpoint_path', 'adapter_path'}
                or not {'key', 'candidate_id', 'base_model_path'} <= set(args)
                or any(not isinstance(v, str) or not v.strip() or '\x00' in v for v in args.values())
                or {'checkpoint_path', 'adapter_path'} <= set(args)):
            raise LabToolError('Submit a registered base and optionally one checkpoint OR adapter; no evaluation overrides')
        if self._closed or not self._accepting:
            raise RuntimeError('Lab is shutting down; new work is closed')
        base = args['base_model_path']
        if base not in self.formal_configuration['bases']:
            raise LabToolError('Unknown registered base_model_path')
        if source_identity() != self.formal_configuration['source_sha256']:
            raise PermissionError('Host checkpoint evaluator source changed')
        ident = self._keys.get(args['key'], (None,))[0]
        if ident is not None:
            previous = self._evaluations.get(ident)
            if previous is None or previous['arguments'] != args:
                raise LabToolError('Submission key already used for different arguments')
            return self._start({'key': args['key'], 'argv': previous['command']})
        active = self._status()['active_jobs']
        if active:
            raise LabToolError('Release other owned programs before fixed checkpoint evaluation',
                               data={'reason': 'evaluation_devices_busy', 'active_jobs': active})
        def candidate_path(name):
            path = (self.workspace / args[name]).resolve()
            if not path.is_dir() or not path.is_relative_to(self.workspace.resolve()):
                raise LabToolError(name + ' must be a directory inside the Lab workspace')
            return str(path)
        config = {'groups': self.evaluation_groups, 'candidate_id': args['candidate_id'],
            'base_model_path': base, 'checkpoint_path': candidate_path('checkpoint_path') if 'checkpoint_path' in args else base,
            'adapter_path': candidate_path('adapter_path') if 'adapter_path' in args else None,
            'tokenizer_identity': self.formal_configuration['bases'][base],
            'source_sha256': self.formal_configuration['source_sha256'], 'profile': PROFILE,
            'device_type': self.formal_configuration['device_type'],
            'output': str(self.evaluation_root / (uuid4().hex + '.json'))}
        command = [self.evaluation_python, str(self.checkpoint_runner), json.dumps(config)]
        result = self._start({'key': args['key'], 'argv': command})
        self._evaluations[result.data['job_id']] = {'arguments': deepcopy(args), 'command': command, 'config': config}
        return result

    def _observe(self, ident):
        result = LocalProcessLab._observe(self, ident)
        binding = self._evaluations.get(ident)
        if (binding is None or result.data['state'] != 'succeeded'
                or 'fixed_evaluation' in result.data or 'fixed_evaluation_error' in result.data):
            return result
        config = binding['config']
        try:
            raw = Path(config['output']).read_bytes()
            record = json.loads(raw)
            payload, gold, datasets = evaluation_inputs(self.evaluation_groups)
            traces = record['traces']
            if len(traces) != len(payload['items']):
                raise ValueError('Missing raw generation evidence')
            for trace, item in zip(traces, payload['items']):
                if (any(trace[k] != item[k] for k in ('benchmark_id', 'id'))
                        or trace['messages_sha256'] != digest(item['messages'])
                        or trace['answer_letter'] != answer_letter(trace['raw_text'], item['option_letters'])
                        or type(trace['input_tokens']) is not int or trace['input_tokens'] < 1
                        or trace['output_tokens'] != len(trace['generated_token_ids'])
                        or not 0 <= trace['output_tokens'] <= PROFILE['max_new_tokens']
                        or trace['output_limit_hit'] != (trace['output_tokens'] >= PROFILE['max_new_tokens'])):
                    raise ValueError('Raw generation identity or parsing differs')
            metrics, scored = score_predictions([{'benchmark_id': r['benchmark_id'], 'id': r['id'],
                'text': r['raw_text']} for r in traces], gold)
            expected_profile = digest({'profile': PROFILE, 'tokenizer': config['tokenizer_identity'],
                'source': config['source_sha256'], 'device_type': config['device_type'], 'runtime': record['runtime']})
            if (record['kind'] != 'host_checkpoint_development_evaluation' or record['protocol'] != PROTOCOL
                    or any(record[k] != config[k] for k in ('candidate_id', 'base_model_path',
                        'checkpoint_path', 'adapter_path', 'source_sha256', 'tokenizer_identity', 'profile'))
                    or record['dataset_sha256'] != datasets or record['predictions'] != scored
                    or record['metrics'] != metrics or record['profile_sha256'] != expected_profile
                    or record['diagnostics'] != {'invalid_answers': sum(r['answer_letter'] is None for r in traces),
                        'output_limit_hits': sum(r['output_limit_hit'] for r in traces)}):
                raise ValueError('Checkpoint evaluation identity or score differs from source evidence')
            for key in ('weight_identity', 'adapter_identity'):
                value = record[key]
                if value is None:
                    if key == 'weight_identity' or config['adapter_path'] is not None:
                        raise ValueError('Missing weight identity')
                elif value['sha256'] != digest(value['files']) or not value['files']:
                    raise ValueError('Invalid weight manifest digest')
            artifact = self.artifacts.put(raw, kind='fixed_medical_checkpoint_evaluation')
            fixed = {key: record[key] for key in ('protocol', 'candidate_id', 'metrics', 'dataset_sha256',
                'profile', 'profile_sha256', 'weight_identity', 'adapter_identity', 'diagnostics')}
            # Detailed manifests/raw generations remain in the artifact, not the decision evidence card.
            for key in ('weight_identity', 'adapter_identity'):
                if fixed[key] is not None:
                    fixed[key] = {'sha256': fixed[key]['sha256']}
            fixed.update(evaluator_sha256=config['source_sha256']['checkpoint_evaluation.py'],
                record_sha256=hashlib.sha256(raw).hexdigest(), adopted=False,
                input_contract={'status': 'host_owned_checkpoint_inference'})
            result = replace(result, artifacts=(*result.artifacts, artifact), data={**result.data, 'fixed_evaluation': fixed})
        except Exception as exc:
            result = replace(result, status=ExecutionStatus.FAILED,
                error=f'Checkpoint evaluation verification failed: {type(exc).__name__}: {exc}',
                data={**result.data, 'failure_origin': 'infrastructure',
                      'fixed_evaluation_error': f'{type(exc).__name__}: {exc}'})
        self._terminal[ident] = result
        return result


if __name__ == '__main__':
    result = evaluate(json.loads(sys.argv[1]))
    print(json.dumps({key: result[key] for key in ('candidate_id', 'profile_sha256', 'metrics', 'diagnostics')}), flush=True)
