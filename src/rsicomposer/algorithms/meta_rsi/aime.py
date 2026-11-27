"""Bounded, declarative state and trusted local inheritance for AIME operators."""
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from .aime_synthesis import KINDS
from .engine import digest


MAX_ROWS, MAX_COMPLETION = 24, 4096
MAX_TRAIN_TEXT = 16384
MAX_VERIFIED_COMPLETION_CHARS = 131072
DEFAULT_AIME = {'max_exemplars': 2, 'replay_tolerance': 0.0, 'data_policy': 'rollout'}


def completion_char_limit(options):
    return (options or {}).get('max_completion_chars', MAX_COMPLETION)


def data_capacity(options):
    return (options or {}).get('max_verified_rows', MAX_ROWS)
DEV_ROLES = ('search', 'validation')


def sha256_text(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def candidate_id(value):
    return isinstance(value, str) and value.startswith('metarsi-') and sha256_text(value[8:])


def check_options(options):
    if not isinstance(options, dict) or options.keys() - (DEFAULT_AIME.keys() | {'max_completion_chars', 'max_verified_rows'}):
        raise ValueError('Invalid AIME option keys')
    limit = completion_char_limit(options)
    if type(limit) is not int or not MAX_COMPLETION <= limit <= MAX_VERIFIED_COMPLETION_CHARS:
        raise ValueError('AIME max_completion_chars must be between 4096 and 131072')
    if type(data_capacity(options)) is not int or not MAX_ROWS <= data_capacity(options) <= 288:
        raise ValueError('AIME verified data capacity must be between 24 and 288')
    options = {**DEFAULT_AIME, **options}
    if options['data_policy'] not in KINDS:
        raise ValueError('AIME data_policy must be rollout or affine')
    cap, tolerance = options['max_exemplars'], options['replay_tolerance']
    if (type(cap) is not int or not 1 <= cap <= 2 or type(tolerance) not in (int, float)
            or not 0 <= tolerance <= 1 or not math.isfinite(tolerance)):
        raise ValueError('Invalid AIME max_exemplars or replay_tolerance')
    return {**options, 'replay_tolerance': float(tolerance)}


def check_components(components, options, instructions, *, allow_model_guidance=False):
    if not isinstance(components, dict) or set(components) != {'harness', 'data', 'model'}:
        raise ValueError('Invalid AIME candidate components')
    harness, data, model = (components[key] for key in ('harness', 'data', 'model'))
    if not isinstance(data, list) or len(data) > data_capacity(options):
        raise ValueError('AIME snapshots require bounded verified train data')
    for row in data:
        if (not isinstance(row, dict) or set(row) != {'id', 'prompt', 'completion'}
                or any(not isinstance(text, str) or not text.strip() for text in row.values())
                or len(row['id']) > MAX_TRAIN_TEXT or len(row['prompt']) > MAX_TRAIN_TEXT
                or len(row['completion']) > completion_char_limit(options)):
            raise ValueError('Malformed AIME verified train trajectory')
    harness_fields = {'instruction', 'few_shot', 'exemplars', 'revision'}
    if (not isinstance(harness, dict) or not set(harness) in (harness_fields, harness_fields | {'guidance'})
            or ('guidance' in harness and not allow_model_guidance)
            or not sha256_text(harness['revision'])
            or harness['instruction'] not in instructions or type(harness['few_shot']) is not bool
            or not isinstance(harness['exemplars'], list) or len(harness['exemplars']) > options['max_exemplars']
            or harness['few_shot'] != bool(harness['exemplars'])
            or any(not sha256_text(item) for item in harness['exemplars'])
            or len(set(harness['exemplars'])) != len(harness['exemplars'])):
        raise ValueError('Invalid AIME prompt-only harness')
    if 'guidance' in harness:
        from .aime_domain_policy import check_guidance
        check_guidance(harness['guidance'])
    # Long SFT rows remain valid training data, not prompt exemplars.
    rows = {digest(row): row['id'] for row in data if len(row['completion']) <= MAX_COMPLETION}
    if (not set(harness['exemplars']) <= rows.keys()
            or len({rows[key] for key in harness['exemplars']}) != len(harness['exemplars'])):
        raise ValueError('AIME exemplars must pin distinct verified train problems')
    if (model != {'checkpoint': None} and (not isinstance(model, dict)
            or set(model) != {'checkpoint', 'checkpoint_sha256'}
            or not all(sha256_text(value) for value in model.values()))):
        raise ValueError('Invalid AIME immutable model reference')


def check_snapshot(value, options, instructions):
    if (not isinstance(value, dict) or set(value) != {
            'schema_version', 'candidate_id', 'base_sha256', 'harness', 'data', 'model'}
            or type(value['schema_version']) is not int or value['schema_version'] != 2
            or not candidate_id(value['candidate_id'])
            or (value['base_sha256'] is not None and not sha256_text(value['base_sha256']))):
        raise ValueError('AIME requires a bounded MetaRSI v2 snapshot')
    check_components({key: value[key] for key in ('harness', 'data', 'model')}, options, instructions)


def owned_parent(data_dir):
    """Use only the runtime-owned NestedRun layout, never a path from a request/snapshot."""
    child = Path(data_dir)
    job = child.parent
    runs = job.parent
    if (child.name != 'child' or runs.name != 'nested_runs' or not re.fullmatch(r'[0-9a-f]{32}', job.name)
            or any(path.is_symlink() for path in (child, job, runs))):
        return None
    bootstrap = job / 'bootstrap.json'
    descriptor = job / 'descriptor.json'
    request = job / 'request.json'
    if any(path.is_symlink() or not path.is_file() for path in (bootstrap, descriptor, request)):
        return None
    boot, desc, req = (json.loads(path.read_bytes()) for path in (bootstrap, descriptor, request))
    if not all(isinstance(doc, dict) for doc in (boot, desc, req)):
        raise ValueError('AIME requires owned nested bootstrap documents')
    parent_call = boot.get('parent_call_id')
    # The trusted host serializes both the submitted request and owned bootstrap.
    if (boot.get('schema') != 1 or boot.get('run_id') != job.name or Path(boot.get('child_dir', '')) != child
            or Path(boot.get('host', {}).get('application', {}).get('data_dir', '')) != runs.parent
            or not isinstance(parent_call, str) or sha256(parent_call.encode()).hexdigest()[:32] != job.name
            or desc.get('schema') != 2 or desc.get('job_id') != job.name or desc.get('request_digest') != digest(req)
            or boot.get('manifest') != req.get('manifest') or boot.get('parent_call_id') != req.get('parent_call_id')):
        raise ValueError('AIME nested run ownership does not match its host bootstrap')
    return runs.parent / 'meta-rsi'


def metrics(candidate, split, protocol, count, correct, invalid_format, truncated, prompt_overlength):
    return {'candidate_id': candidate, 'split': split, 'protocol_id': protocol, 'count': count,
            'correct': correct, 'accuracy': correct / count, 'invalid_format': invalid_format,
            'invalid_format_rate': invalid_format / count, 'truncated': truncated,
            'prompt_overlength': prompt_overlength}


def check_metrics(value, candidate, split, protocol, count):
    if not isinstance(value, dict):
        raise ValueError('Missing AIME aggregate diagnostic')
    counts = [value.get(name) for name in ('correct', 'invalid_format', 'truncated')]
    overlength = value.get('prompt_overlength')
    if (any(type(n) is not int or n < 0 for n in counts) or sum(counts) > count
            or type(overlength) is not int or not 0 <= overlength <= counts[-1]
            or value != metrics(candidate, split, protocol, count, *counts, overlength)):
        raise ValueError('AIME aggregate diagnostic does not match its frozen development binding')
