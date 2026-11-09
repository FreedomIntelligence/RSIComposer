"""File/network-isolated Lab for AIDE's generated LLM inference programs.

The candidate sees public examples and unlabeled current questions. A bounded
Unix broker grants fixed-model inference, not host TCP/network or GPU access.
Validation metrics are host-scored; heldout feedback is delivered only after
upstream AIDE has made its final selection.
"""
from contextlib import contextmanager
import csv
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socketserver
import subprocess
import sys
import tempfile
import threading
import time

from rsisys.labs.harness_runtime import ChatPool, digest
from rsisys.labs.math_verifiers import StrictNumericVerifier
from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.decision import OperationSpec
from rsisys.runtime.isolation import ExecutionPolicy, ProcessIsolation, system_runtime_paths


def validate_config(config):
    allowed = {'input_dir', 'validation_data', 'heldout_data', 'endpoints', 'model',
               'python', 'timeout', 'max_tokens', 'calls_per_question', 'output_subdir'}
    if set(config) - allowed:
        raise ValueError('Unknown aide_program option')
    for key in ('input_dir', 'validation_data', 'heldout_data', 'model'):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError('aide_program requires ' + key)
    endpoints = config.get('endpoints')
    if not isinstance(endpoints, list) or not endpoints or any(
            not isinstance(e, str) or not e.startswith(('http://', 'https://')) for e in endpoints):
        raise ValueError('aide_program requires explicit endpoints')
    for key, default in (('timeout', 300), ('max_tokens', 1536), ('calls_per_question', 4)):
        if type(config.get(key, default)) is not int or config.get(key, default) < 1:
            raise ValueError('Invalid aide_program ' + key)
    subdir = config.get('output_subdir', 'aide-program')
    if not isinstance(subdir, str) or not subdir or Path(subdir).is_absolute() or '..' in Path(subdir).parts:
        raise ValueError('Invalid aide_program output_subdir')


class _Broker(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    block_on_close = False


@contextmanager
def inference_broker(pool, *, budget, max_tokens):
    lock = threading.Lock()
    stats = {'calls': 0, 'completed': 0, 'tokens': 0, 'failures': 0}
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                self.request.settimeout(300)
                request = json.loads(self.rfile.readline(1_000_001))
                if set(request) - {'messages', 'max_tokens', 'temperature', 'seed'}:
                    raise ValueError('Unknown inference argument')
                messages = request.get('messages')
                if not isinstance(messages, list) or not messages or len(messages) > 40 or any(
                        not isinstance(m, dict) or set(m) != {'role', 'content'} or
                        m['role'] not in ('system', 'user', 'assistant') or not isinstance(m['content'], str)
                        for m in messages):
                    raise ValueError('Invalid inference messages')
                tokens = request.get('max_tokens', 1024)
                temperature = request.get('temperature', 0.0)
                seed = request.get('seed', 73)
                if type(tokens) is not int or not 1 <= tokens <= max_tokens:
                    raise ValueError('max_tokens exceeds the installed inference budget')
                if type(temperature) not in (float, int) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
                    raise ValueError('Invalid temperature')
                if type(seed) is not int or not 0 <= seed < 2**32:
                    raise ValueError('Invalid seed')
                with lock:
                    if stats['calls'] >= budget:
                        raise ValueError('Candidate inference-call budget exhausted')
                    stats['calls'] += 1
                response = pool.complete(messages, seed=seed, max_tokens=tokens, temperature=temperature)
                with lock:
                    stats['completed'] += 1
                    stats['tokens'] += response['tokens']
                result = {'content': response['content']}
            except Exception as exc:
                with lock:
                    stats['failures'] += 1
                result = {'error': type(exc).__name__ + ': ' + str(exc)}
            try:
                self.wfile.write(json.dumps(result).encode() + b'\n')
            except (BrokenPipeError, ConnectionResetError):
                pass
    with tempfile.TemporaryDirectory(prefix='aide-broker-') as directory:
        path = str(Path(directory) / 'inference.sock')
        server = _Broker(path, Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield path, stats
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


def _read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or any(not isinstance(r, dict) or not all(isinstance(r.get(k), str) and r[k].strip()
            for k in ('id', 'prompt', 'answer')) for r in rows):
        raise ValueError('Invalid AIDE evaluation JSONL')
    if len({r['id'] for r in rows}) != len(rows) or len({r['prompt'] for r in rows}) != len(rows):
        raise ValueError('Duplicate AIDE evaluation rows')
    return rows


def write_questions(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['id', 'prompt'])
        writer.writeheader()
        writer.writerows({k: row[k] for k in ('id', 'prompt')} for row in rows)


def score_predictions(path, rows):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 8_000_000:
        raise ValueError('Missing or invalid working/submission.csv')
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['id', 'answer']:
            raise ValueError('submission.csv columns must be id,answer')
        predictions = list(reader)
    if len(predictions) != len(rows) or len({r['id'] for r in predictions}) != len(rows):
        raise ValueError('Submission has missing or duplicate question IDs')
    indexed = {r['id']: r['answer'] for r in predictions}
    if set(indexed) != {r['id'] for r in rows}:
        raise ValueError('Submission question IDs mismatch')
    verifier = StrictNumericVerifier()
    correct = 0
    for row in rows:
        answer = indexed[row['id']]
        # Submission accepts a plain final number or the benchmark's #### /
        # boxed notation; free-form reasoning still needs an explicit marker.
        if verifier.parse(answer) is None:
            answer = '#### ' + answer.strip()
        correct += verifier.is_correct(answer, row['answer'])
    return {'accuracy': correct / len(rows), 'correct': correct, 'questions': len(rows)}


class AIDEProgramLab:
    def __init__(self, context, config, *, pool=None, executor=None):
        validate_config(config)
        self.config = dict(config)
        self.root = Path(context.data_dir) / config.get('output_subdir', 'aide-program')
        self.root.mkdir(parents=True, exist_ok=False)
        self.input_dir = Path(config['input_dir']).resolve(strict=True)
        if not self.input_dir.is_dir() or any(not p.is_file() or p.is_symlink() for p in self.input_dir.iterdir()):
            raise ValueError('AIDE input directory must contain ordinary public files')
        if {p.name for p in self.input_dir.iterdir()} != {'train.csv', 'validation.csv'}:
            raise ValueError('AIDE public inputs must be exactly train.csv and unlabeled validation.csv')
        self.input_hashes = {p.name: digest(p) for p in self.input_dir.iterdir()}
        self.datasets = {split: _read_rows(config[key]) for split, key in
                         (('validation', 'validation_data'), ('heldout', 'heldout_data'))}
        for key in ('id', 'prompt'):
            if {r[key] for r in self.datasets['validation']} & {r[key] for r in self.datasets['heldout']}:
                raise ValueError('AIDE validation and heldout data overlap')
        with (self.input_dir / 'train.csv').open(newline='') as stream:
            training = list(csv.DictReader(stream))
        if not training or any(set(r) != {'id', 'prompt', 'answer'} for r in training):
            raise ValueError('AIDE public train.csv requires id,prompt,answer')
        for key in ('id', 'prompt'):
            if {r[key] for r in training} & {r[key] for rows in self.datasets.values() for r in rows}:
                raise ValueError('AIDE public examples overlap evaluation data')
        with (self.input_dir / 'validation.csv').open(newline='') as stream:
            preview = list(csv.DictReader(stream))
        if preview != [{k: r[k] for k in ('id', 'prompt')} for r in self.datasets['validation']]:
            raise ValueError('AIDE preview questions must match the frozen validation split without labels')
        self.data_hashes = {key: digest(config[key]) for key in ('validation_data', 'heldout_data')}
        self.pool = pool or ChatPool(config['endpoints'], config['model'],
            timeout=config.get('timeout', 300), max_tokens=config.get('max_tokens', 1536))
        self.executor = executor or self._execute_isolated
        self.counter = 0

    def describe(self):
        properties = {'candidate': {'type': 'string'}, 'sha256': {'type': 'string'}}
        return tuple(OperationSpec(name, description, {'type': 'object', 'properties': {
            **properties, extra: {'type': kind}}, 'required': ['candidate', 'sha256', extra],
            'additionalProperties': False}) for name, extra, kind, description in (
                ('execute', 'step', 'integer', 'Run native AIDE candidate on validation questions.'),
                ('evaluate', 'label', 'string', 'Evaluate a final, already selected candidate on heldout questions.')))

    def _execute_isolated(self, workspace, broker):
        python = self.config.get('python', sys.executable)
        policy = ExecutionPolicy(read_only=system_runtime_paths(python), sockets=(broker,))
        isolation = ProcessIsolation(policy)
        command, env = isolation.command(workspace, (python, 'candidate.py'), {
            'PATH': str(Path(python).absolute().parent) + ':/usr/bin:/bin',
            'AIDE_LLM_SOCKET': broker, 'PYTHONPATH': str(workspace), 'OMP_NUM_THREADS': '1'})
        stdout = workspace.parent / 'stdout.log'
        started = time.monotonic()
        with stdout.open('w') as output:
            process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                process.wait(timeout=self.config.get('timeout', 300))
                exc_type = None if process.returncode == 0 else 'CandidateProcessError'
            except subprocess.TimeoutExpired:
                exc_type = 'TimeoutError'
            finally:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                # A launcher may exit before its children. Always escalate for
                # the owned group, even when waiting for the parent succeeded.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        text = stdout.read_text(errors='replace')
        # The upstream reviewer receives bounded terminal text. Full logs remain
        # outside the generated program's mutable workspace.
        return {'term_out': [text[-48000:]], 'exec_time': time.monotonic() - started,
                'exc_type': exc_type, 'exc_info': None, 'exc_stack': None}

    def execute(self, call):
        if call.name not in ('execute', 'evaluate'):
            raise ValueError('Unknown AIDE program operation')
        args = dict(call.arguments)
        if call.name == 'evaluate' and args['label'] not in ('initial', 'selected'):
            raise ValueError('Unknown final evaluation label')
        if call.name == 'execute' and (type(args['step']) is not int or args['step'] < 0):
            raise ValueError('Invalid AIDE step')
        source = Path(args['candidate']).resolve(strict=True)
        if digest(source) != args['sha256'] or source.stat().st_size > 500_000:
            raise ValueError('AIDE candidate identity mismatch')
        for key, expected in self.data_hashes.items():
            if digest(self.config[key]) != expected:
                raise ValueError('AIDE evaluation data changed')
        if {p.name: digest(p) for p in self.input_dir.iterdir()} != self.input_hashes:
            raise ValueError('AIDE public input changed')
        split = 'validation' if call.name == 'execute' else 'heldout'
        rows = self.datasets[split]
        directory = self.root / f'{self.counter:03d}-{split}'
        directory.mkdir(exist_ok=False)
        self.counter += 1
        workspace = directory / 'workspace'
        (workspace / 'input').mkdir(parents=True)
        (workspace / 'working').mkdir()
        # Only the public training examples are reused. The current unlabeled
        # question file is reconstructed from the Lab-owned frozen split.
        shutil.copyfile(self.input_dir / 'train.csv', workspace / 'input/train.csv')
        write_questions(workspace / 'input/validation.csv', rows)
        shutil.copyfile(source, workspace / 'candidate.py')
        shutil.copyfile(Path(__file__).with_name('client.py'), workspace / 'llm_client.py')
        candidate_hash = digest(workspace / 'candidate.py')
        if candidate_hash != args['sha256']:
            raise ValueError('AIDE candidate changed while being copied')
        with inference_broker(self.pool, budget=len(rows) * self.config.get('calls_per_question', 4),
                              max_tokens=self.config.get('max_tokens', 1536)) as (broker, stats):
            result = self.executor(workspace, broker)
        measured = None
        if result['exc_type'] is None:
            try:
                measured = score_predictions(workspace / 'working/submission.csv', rows)
                result['term_out'].append('\nHOST_VALIDATION_METRIC ' + json.dumps(measured) + '\n')
            except (ValueError, OSError, KeyError, TypeError) as exc:
                result['exc_type'] = type(exc).__name__
                result['term_out'].append('\nHost submission validation failed: ' + str(exc) + '\n')
        if digest(workspace / 'candidate.py') != candidate_hash:
            measured = None
            result['exc_type'] = 'CandidateMutationError'
            result['term_out'].append('\nCandidate changed its own source during execution.\n')
        data = {'request': args, 'execution_result': result, 'candidate_sha256': candidate_hash,
                'split': split, 'inference': dict(stats), 'training_performed': False,
                'execution_directory': str(directory.resolve()),
                'dataset_sha256': self.data_hashes[split + '_data']}
        if measured is not None:
            data['quality_metrics'] = measured
        (directory / 'receipt.json').write_text(json.dumps(data, indent=2))
        return OperationResult('AIDE candidate execution recorded', (), False, data=data)
