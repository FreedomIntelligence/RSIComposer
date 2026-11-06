"""Legacy program-driven medical evaluation and shared fixed scoring utilities.

The candidate consumes JSON on stdin and returns predictions on stdout. The
evaluator owns data selection, prompts, gold matching and aggregate metrics.
This is a trusted local execution contract, not an OS security boundary.
New Open Lab runs use protocol_evaluation.ProtocolMedicalLab with frozen bundles,
split-specific feedback and durable campaign admission. This legacy API remains
for historical receipt recovery and explicit exploratory input debugging.
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
import tempfile
from uuid import uuid4

from rsisys.adapters.local_lab import LocalProcessLab
from rsisys.loops.contracts.decision import OperationSpec
from rsisys.loops.contracts.execution import LabToolError
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.runtime.files import atomic_write, json_bytes
try:
    from .candidate_adapter import CONTRACT, CandidateInputError, digest as input_digest, validate_receipts
except ImportError:  # Executed by filename in the installed training environment.
    from candidate_adapter import CONTRACT, CandidateInputError, digest as input_digest, validate_receipts


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


def diagnostic_inputs():
    """Public synthetic inputs; no dataset access, gold or diagnostic-only branch."""
    return {'protocol': PROTOCOL, 'items': [
        {'benchmark_id': 'synthetic-input-contract', 'id': str(index), 'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': f'Synthetic input marker {marker}.\nA. alpha\nB. beta'}]}
        for index, marker in enumerate(('CANARY_ORANGE_41', 'CANARY_PURPLE_73'))]}


def run_candidate(argv, payload, *, require_input_contract=False, return_contract=False,
                  execution=None):
    # Legacy exploratory callers retain their explicit trusted-local behavior.
    # Formal evaluation supplies a host-selected isolation boundary and cwd.
    options = {}
    scratch = None
    try:
        if execution is not None:
            isolation, workspace = execution
            scratch = tempfile.TemporaryDirectory(prefix='rsisys-inference-')
            argv, env = isolation.command(workspace, argv, {'PATH': '/usr/bin:/bin',
                'LANG': 'C.UTF-8', 'PYTHONUNBUFFERED': '1'}, writable=False, scratch=scratch.name)
            options = {'cwd': workspace, 'env': env}
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, **options)
    except BaseException:
        if scratch is not None:
            scratch.cleanup()
        raise
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
        if (not isinstance(result, dict) or 'predictions' not in result
                or set(result) - {'predictions', 'input_receipts'}):
            raise ValueError('Candidate stdout must be a predictions JSON object')
        contract = (validate_receipts(result['input_receipts'], payload) if 'input_receipts' in result else
                    {'contract': CONTRACT, 'status': 'unverified_legacy',
                     'binding': 'candidate_input_consumption_not_verified'})
        if require_input_contract and contract['status'] != 'adapter_verified':
            raise CandidateInputError('Missing standard input adapter receipts; use the supplied '
                'adapter and request.render(...) or request.chat_messages(). Run medical_diagnose first')
        return (result['predictions'], contract) if return_contract else result['predictions']
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()
        if writer.ident is not None:
            writer.join(timeout=5)
        elif not process.stdin.closed:
            process.stdin.close()
        if scratch is not None:
            scratch.cleanup()
        # The outer LocalProcessBackend owns the entire group and cleans up
        # candidate descendants on evaluator exit or timeout.


class FixedMedicalLab(LocalProcessLab):
    """Add one fixed evaluation capability to the otherwise open local Lab."""
    def __init__(self, *args, evaluation_groups, evaluation_root, python,
                 require_input_contract=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluation_groups = deepcopy(evaluation_groups)
        self.evaluation_root = Path(evaluation_root).resolve()
        self.evaluation_root.mkdir(parents=True, exist_ok=True)
        self.evaluation_python = str(python)
        if type(require_input_contract) is not bool:
            raise TypeError('require_input_contract must be a bool')
        self.require_input_contract = require_input_contract
        self.evaluator_path = Path(__file__).resolve()
        self.evaluator_sha256 = hashlib.sha256(self.evaluator_path.read_bytes()).hexdigest()
        self.adapter_path = self.evaluator_path.with_name('candidate_adapter.py')
        self.adapter_sha256 = hashlib.sha256(self.adapter_path.read_bytes()).hexdigest()
        self._evaluations = {}

    def describe(self):
        text = {'type': 'string', 'minLength': 1}
        arguments = {'type': 'object', 'properties': {'key': text, 'candidate_id': text,
                'argv': {'type': 'array', 'minItems': 1, 'items': text}},
             'required': ['key', 'candidate_id', 'argv'], 'additionalProperties': False}
        contract = (' Use the supplied candidate adapter: implement infer(request) or infer_batch(requests). '
            'For local inference tokenize request.render(tokenizer,...); for HTTP put '
            'request.chat_messages() in the actual request. messages[0] is system; the question/options '
            'are in the user message. Input receipts verify the adapter boundary, not backend consumption.')
        return (*super().describe(), OperationSpec('medical_evaluate',
            'Run the fixed medical development protocol against your candidate program. '
            'The supplied adapter consumes JSON {protocol,items:[{benchmark_id,id,messages}]} on stdin, '
            'with fixed questions/options and no gold. It writes JSON '
            '{predictions:[{benchmark_id,id,text}],input_receipts:[...]} on stdout; '
            'send diagnostics to stderr. '
            'The Lab owns datasets, gold, parsing and accuracy. Submission is not a score; '
            'use lab_poll then lab_result. Candidate code, weights and inference remain your choice.' + contract,
            arguments), OperationSpec('medical_diagnose',
            'Run the SAME candidate argv and input adapter on two public synthetic questions. '
            'No development data or gold is read and no accuracy/adoption is reported. '
            'Use lab_poll then lab_result; inspect input_diagnosis and stderr for missing user input. '
            'Do not add a separate candidate diagnosis path.' + contract, deepcopy(arguments)))

    def execute(self, call):
        if call.name not in ('medical_evaluate', 'medical_diagnose'):
            return super().execute(call)
        self.require_admission()
        args = call.arguments
        if (set(args) != {'key', 'candidate_id', 'argv'}
                or any(not isinstance(args[name], str) or not args[name].strip()
                       for name in ('key', 'candidate_id'))
                or not isinstance(args['argv'], list) or not args['argv']
                or any(not isinstance(item, str) or not item or '\x00' in item for item in args['argv'])):
            raise LabToolError('Invalid fixed evaluation arguments')
        if hashlib.sha256(self.evaluator_path.read_bytes()).hexdigest() != self.evaluator_sha256:
            raise PermissionError('Host evaluator code identity changed')
        if hashlib.sha256(self.adapter_path.read_bytes()).hexdigest() != self.adapter_sha256:
            raise PermissionError('Host candidate adapter code identity changed')
        # Use deterministic configuration in the submission fingerprint. A replay
        # must not generate a new nonce/output path and conflict with its own key.
        ident = self._keys.get(args['key'], (None,))[0]
        previous = self._evaluations.get(ident)
        mode = 'diagnose' if call.name == 'medical_diagnose' else 'evaluate'
        if (previous is not None and previous['arguments'] == args
                and previous['config'].get('mode', 'evaluate') == mode):
            return self._start({'key': args['key'], 'argv': previous['command']})
        output_root = self.evaluation_root / 'diagnostics' if mode == 'diagnose' else self.evaluation_root
        output_root.mkdir(exist_ok=True)
        config = {'groups': self.evaluation_groups, 'candidate_id': args['candidate_id'],
            'candidate_argv': args['argv'], 'evaluator_sha256': self.evaluator_sha256,
            'adapter_sha256': self.adapter_sha256, 'mode': mode,
            'require_input_contract': self.require_input_contract or mode == 'diagnose',
            'output': str(output_root / (uuid4().hex + '.json'))}
        command = [self.evaluation_python, str(self.evaluator_path), json.dumps(config)]
        result = self._start({'key': args['key'], 'argv': command})
        self._evaluations[result.data['job_id']] = {'arguments': deepcopy(args),
            'command': command, 'config': config}
        return result

    def _observe(self, ident):
        result = super()._observe(ident)
        binding = self._evaluations.get(ident)
        if (binding is None or result.data['state'] != 'succeeded'
                or 'fixed_evaluation' in result.data or 'fixed_evaluation_error' in result.data
                or 'input_diagnosis' in result.data):
            return result
        config = binding['config']
        try:
            raw = Path(config['output']).read_bytes()
            record = json.loads(raw)
            if config.get('mode') == 'diagnose':
                payload = diagnostic_inputs()
                contract = validate_receipts(record['input_contract']['receipts'], payload)
                if (record['kind'] != 'synthetic_input_diagnosis' or record['candidate_id'] != config['candidate_id']
                        or record['candidate_argv'] != config['candidate_argv']
                        or record['input_payload_sha256'] != input_digest(payload)
                        or record['input_contract'] != contract
                        or record['evaluator_sha256'] != self.evaluator_sha256
                        or record['adapter_sha256'] != self.adapter_sha256):
                    raise ValueError('Input diagnosis identity differs from source evidence')
                artifact = self.artifacts.put(raw, kind='medical_input_diagnosis')
                result = replace(result, artifacts=(*result.artifacts, artifact), data={**result.data,
                    'input_diagnosis': {**record, 'record_sha256': hashlib.sha256(raw).hexdigest()}})
                self._terminal[ident] = result
                return result
            payload, gold, datasets = fixed_inputs(self.evaluation_groups)
            if config.get('require_input_contract'):
                if record['input_contract'] != validate_receipts(record['input_contract']['receipts'], payload):
                    raise ValueError('Candidate input contract differs from fixed source inputs')
            predictions = [{key: row[key] for key in ('benchmark_id', 'id', 'text')}
                           for row in record['predictions']]
            metrics, scored = score_predictions(predictions, gold)
            if (record['protocol'] != PROTOCOL or record['candidate_id'] != config['candidate_id']
                    or record['candidate_argv'] != config['candidate_argv']
                    or record['dataset_sha256'] != datasets or record['metrics'] != metrics
                    or record['predictions'] != scored or record['evaluator_sha256'] != self.evaluator_sha256
                    or record['adapter_sha256'] != self.adapter_sha256):
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
                'input_contract': {key: value for key, value in record.get('input_contract', {}).items()
                                   if key != 'receipts'},
                'weight_identity': 'not_verified_by_prediction_protocol'}})
        self._terminal[ident] = result
        return result


def main():
    config = json.loads(sys.argv[1])
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if digest != config['evaluator_sha256']:
        raise ValueError('Fixed evaluator source identity changed')
    adapter_sha = hashlib.sha256(Path(__file__).with_name('candidate_adapter.py').read_bytes()).hexdigest()
    if config.get('adapter_sha256', adapter_sha) != adapter_sha:
        raise ValueError('Candidate input adapter source identity changed')
    if config.get('mode') == 'diagnose':
        payload = diagnostic_inputs()
        predictions, contract = run_candidate(config['candidate_argv'], payload,
            require_input_contract=True, return_contract=True)
        # Shape/identity only: no gold, accuracy, capability or improvement claim.
        expected = {(i['benchmark_id'], i['id']) for i in payload['items']}
        if (not isinstance(predictions, list) or len(predictions) != len(expected)
                or {(p.get('benchmark_id'), p.get('id')) for p in predictions} != expected):
            raise CandidateInputError('Synthetic prediction IDs differ from the adapter inputs')
        record = {'kind': 'synthetic_input_diagnosis', 'candidate_id': config['candidate_id'],
            'candidate_argv': config['candidate_argv'], 'evaluator_sha256': digest,
            'adapter_sha256': adapter_sha, 'input_payload_sha256': input_digest(payload),
            'input_contract': contract, 'synthetic': True, 'adopted': False}
        atomic_write(Path(config['output']), json_bytes(record))
        print(json.dumps(record), flush=True)
        return
    payload, gold, datasets = fixed_inputs(config['groups'])
    predictions, contract = run_candidate(config['candidate_argv'], payload,
        require_input_contract=config.get('require_input_contract', False), return_contract=True)
    metrics, scored = score_predictions(predictions, gold)
    record = {'kind': 'fixed_development_evaluation', 'protocol': PROTOCOL,
        'candidate_id': config['candidate_id'], 'candidate_argv': config['candidate_argv'],
        'dataset_sha256': datasets, 'evaluator_sha256': digest,
        'metrics': metrics, 'predictions': scored, 'adopted': False,
        'input_contract': contract, 'adapter_sha256': adapter_sha}
    atomic_write(Path(config['output']), json_bytes(record))
    summary = {key: value for key, value in record.items() if key not in ('predictions', 'input_contract')}
    summary['input_contract'] = {key: value for key, value in contract.items() if key != 'receipts'}
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
