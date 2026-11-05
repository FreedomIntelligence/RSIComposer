"""Host-owned medical evaluation; candidate programs remain freely editable.

The candidate consumes JSON on stdin and returns predictions on stdout. The
evaluator owns data selection, prompts, gold matching and aggregate metrics.
This is a trusted local execution contract, not an OS security boundary.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
from uuid import uuid4

from rsisys.adapters.local_lab import LocalProcessLab
from rsisys.loops.contracts.decision import OperationSpec
from rsisys.loops.contracts.execution import LabToolError
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.runtime.files import atomic_write, json_bytes


PROTOCOL = 'exact-option-letter-v1'
SYSTEM = 'Choose the best answer. Return exactly one option letter, with no explanation.'
MAX_PREDICTION_BYTES = 4 * 1024 * 1024


def fixed_inputs(groups):
    items, gold, datasets = [], {}, {}
    for group in groups:
        content = Path(group['path']).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != group['sha256']:
            raise ValueError('Fixed evaluation dataset identity changed')
        benchmark = group['benchmark_id']
        if benchmark in datasets:
            raise ValueError('Duplicate fixed benchmark identity')
        datasets[benchmark] = digest
        for line in content.decode().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (benchmark, row['id'])
            if key in gold or row['gold'] not in row['options']:
                raise ValueError('Invalid fixed sample identity or gold')
            gold[key] = row['gold']
            prompt = row['question'] + ''.join('\n' + k + '. ' + v for k, v in row['options'].items())
            items.append({'benchmark_id': benchmark, 'id': row['id'], 'messages': [
                {'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]})
    if not gold:
        raise ValueError('Fixed evaluation requires nonempty data')
    return {'protocol': PROTOCOL, 'items': items}, gold, datasets


def score_predictions(predictions, gold):
    if not isinstance(predictions, list) or len(predictions) != len(gold):
        raise ValueError('Candidate must return exactly one prediction per fixed sample')
    seen, scored, counts = set(), [], {}
    for value in predictions:
        if not isinstance(value, dict) or set(value) != {'benchmark_id', 'id', 'text'}:
            raise ValueError('Prediction requires benchmark_id, id and text only')
        key = (value['benchmark_id'], value['id'])
        if key not in gold or key in seen or not isinstance(value['text'], str):
            raise ValueError('Unknown, duplicate or malformed prediction')
        seen.add(key)
        match = re.fullmatch(r'([A-J])[.\s]*', value['text'].strip())
        correct = bool(match and match.group(1) == gold[key])
        scored.append({**value, 'gold': gold[key], 'correct': correct})
        count = counts.setdefault(key[0], [0, 0])
        count[0] += int(correct)
        count[1] += 1
    return {name: {'accuracy': correct / n, 'n': n} for name, (correct, n) in counts.items()}, scored


def run_candidate(argv, payload):
    process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    errors = []
    def feed():
        try:
            process.stdin.write(json_bytes(payload))
        except (BrokenPipeError, OSError) as exc:
            errors.append(exc)
        finally:
            process.stdin.close()
    writer = threading.Thread(target=feed, daemon=True)
    try:
        writer.start()
        output = process.stdout.read(MAX_PREDICTION_BYTES + 1)
        if len(output) > MAX_PREDICTION_BYTES:
            raise ValueError('Candidate prediction output exceeds fixed byte limit')
        code = process.wait()
        writer.join(timeout=5)
        if writer.is_alive() or errors or code != 0:
            raise ValueError('Candidate failed to consume inputs or return predictions successfully')
        result = json.loads(output)
        if not isinstance(result, dict) or set(result) != {'predictions'}:
            raise ValueError('Candidate stdout must be a predictions JSON object')
        return result['predictions']
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()
        if writer.ident is not None:
            writer.join(timeout=5)
        elif not process.stdin.closed:
            process.stdin.close()
        # The outer LocalProcessBackend owns the entire group and cleans up
        # candidate descendants on evaluator exit or timeout.


class FixedMedicalLab(LocalProcessLab):
    """Add one fixed evaluation capability to the otherwise open local Lab."""
    def __init__(self, *args, evaluation_groups, evaluation_root, python, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluation_groups = deepcopy(evaluation_groups)
        self.evaluation_root = Path(evaluation_root).resolve()
        self.evaluation_root.mkdir(parents=True, exist_ok=True)
        self.evaluation_python = str(python)
        self.evaluator_path = Path(__file__).resolve()
        self.evaluator_sha256 = hashlib.sha256(self.evaluator_path.read_bytes()).hexdigest()
        self._evaluations = {}

    def describe(self):
        text = {'type': 'string', 'minLength': 1}
        return (*super().describe(), OperationSpec('medical_evaluate',
            'Run the fixed medical development protocol against your candidate program. '
            'Candidate receives JSON {protocol,items:[{benchmark_id,id,messages}]} on stdin, '
            'with fixed questions/options and no gold in the request. Return only JSON '
            '{predictions:[{benchmark_id,id,text}]} on stdout; send diagnostics to stderr. '
            'The Lab owns datasets, gold, parsing and accuracy. Submission is not a score; '
            'use lab_poll then lab_result. Candidate code, weights and inference remain your choice.',
            {'type': 'object', 'properties': {'key': text, 'candidate_id': text,
                'argv': {'type': 'array', 'minItems': 1, 'items': text}},
             'required': ['key', 'candidate_id', 'argv'], 'additionalProperties': False}))

    def execute(self, call):
        if call.name != 'medical_evaluate':
            return super().execute(call)
        if self._closed or not self._accepting:
            raise RuntimeError('Lab is shutting down; new work is closed')
        args = call.arguments
        if (set(args) != {'key', 'candidate_id', 'argv'}
                or any(not isinstance(args[name], str) or not args[name].strip()
                       for name in ('key', 'candidate_id'))
                or not isinstance(args['argv'], list) or not args['argv']
                or any(not isinstance(item, str) or not item or '\x00' in item for item in args['argv'])):
            raise LabToolError('Invalid fixed evaluation arguments')
        if hashlib.sha256(self.evaluator_path.read_bytes()).hexdigest() != self.evaluator_sha256:
            raise PermissionError('Host evaluator code identity changed')
        # Use deterministic configuration in the submission fingerprint. A replay
        # must not generate a new nonce/output path and conflict with its own key.
        ident = self._keys.get(args['key'], (None,))[0]
        previous = self._evaluations.get(ident)
        if previous is not None and previous['arguments'] == args:
            return self._start({'key': args['key'], 'argv': previous['command']})
        config = {'groups': self.evaluation_groups, 'candidate_id': args['candidate_id'],
            'candidate_argv': args['argv'], 'evaluator_sha256': self.evaluator_sha256,
            'output': str(self.evaluation_root / (uuid4().hex + '.json'))}
        command = [self.evaluation_python, str(self.evaluator_path), json.dumps(config)]
        result = self._start({'key': args['key'], 'argv': command})
        self._evaluations[result.data['job_id']] = {'arguments': deepcopy(args),
            'command': command, 'config': config}
        return result

    def _observe(self, ident):
        result = super()._observe(ident)
        binding = self._evaluations.get(ident)
        if (binding is None or result.data['state'] != 'succeeded'
                or 'fixed_evaluation' in result.data or 'fixed_evaluation_error' in result.data):
            return result
        config = binding['config']
        try:
            raw = Path(config['output']).read_bytes()
            record = json.loads(raw)
            _, gold, datasets = fixed_inputs(self.evaluation_groups)
            predictions = [{key: row[key] for key in ('benchmark_id', 'id', 'text')}
                           for row in record['predictions']]
            metrics, scored = score_predictions(predictions, gold)
            if (record['protocol'] != PROTOCOL or record['candidate_id'] != config['candidate_id']
                    or record['candidate_argv'] != config['candidate_argv']
                    or record['dataset_sha256'] != datasets or record['metrics'] != metrics
                    or record['predictions'] != scored or record['evaluator_sha256'] != self.evaluator_sha256):
                raise ValueError('Fixed evaluation identity or score differs from source evidence')
            artifact = self.artifacts.put(raw, kind='fixed_medical_evaluation')
        except Exception as exc:
            failure = replace(result, status=ExecutionStatus.FAILED,
                error=f'Fixed evaluation verification failed: {type(exc).__name__}: {exc}',
                data={**result.data, 'failure_origin': 'infrastructure',
                      'fixed_evaluation_error': f'{type(exc).__name__}: {exc}'})
            # Public poll/result escalates this via LocalProcessLab.execute;
            # teardown still observes exit and can release already reaped work.
            self._terminal[ident] = failure
            return failure
        result = replace(result, artifacts=(*result.artifacts, artifact), data={**result.data,
            'fixed_evaluation': {'protocol': PROTOCOL, 'candidate_id': config['candidate_id'],
                'metrics': metrics, 'dataset_sha256': datasets, 'evaluator_sha256': self.evaluator_sha256,
                'record_sha256': hashlib.sha256(raw).hexdigest(), 'adopted': False,
                'weight_identity': 'not_verified_by_prediction_protocol'}})
        self._terminal[ident] = result
        return result


def main():
    config = json.loads(sys.argv[1])
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if digest != config['evaluator_sha256']:
        raise ValueError('Fixed evaluator source identity changed')
    payload, gold, datasets = fixed_inputs(config['groups'])
    predictions = run_candidate(config['candidate_argv'], payload)
    metrics, scored = score_predictions(predictions, gold)
    record = {'kind': 'fixed_development_evaluation', 'protocol': PROTOCOL,
        'candidate_id': config['candidate_id'], 'candidate_argv': config['candidate_argv'],
        'dataset_sha256': datasets, 'evaluator_sha256': digest,
        'metrics': metrics, 'predictions': scored, 'adopted': False}
    atomic_write(Path(config['output']), json_bytes(record))
    print(json.dumps({key: value for key, value in record.items() if key != 'predictions'}), flush=True)


if __name__ == '__main__':
    main()
