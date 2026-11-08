"""Medical scoring protocol accepting frozen candidate weights AND inference code.

Public inference uses a host-isolated frozen bundle worker. Closed evaluation is
delegated to an operator's isolated gateway; no closed data is loaded here.
The gateway must enforce the same campaign quota and freeze/import the bundle
before executing candidate code in a data-, filesystem- and network-isolated
worker. Its private logs/data never enter the Lab backend or artifact store.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

from rsisys.adapters.local_lab import LocalProcessLab
from rsisys.loops.contracts.decision import OperationSpec
from rsisys.loops.contracts.evaluation import EvaluationProtocol
from rsisys.loops.contracts.execution import LabToolError
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.runtime.evaluation import EvaluationLedger, EvaluationAdmissionError
from rsisys.runtime.files import atomic_write, json_bytes
from rsisys.runtime.evaluation_diagnostics import prediction_diagnostics
from rsisys.runtime.isolation import ExecutionPolicy, ProcessIsolation
from rsisys.runtime.runtime_identity import inference_policy, runtime_identity
try:
    from .fixed_evaluation import FixedMedicalLab, fixed_inputs, score_predictions, run_candidate
    from .checkpoint_evaluation import digest, file_digest
    from .official_protocol import scoring_functions
except ImportError:
    from fixed_evaluation import FixedMedicalLab, fixed_inputs, score_predictions, run_candidate
    from checkpoint_evaluation import digest, file_digest
    from official_protocol import scoring_functions


PROTOCOL = EvaluationProtocol()
BUNDLE_SCHEMA = 'candidate-inference-bundle-v1'


def source_identity():
    root = Path(__file__).resolve().parent
    result = {name: file_digest(root / name)['sha256'] for name in (
        'protocol_evaluation.py', 'fixed_evaluation.py', 'candidate_adapter.py', 'checkpoint_evaluation.py',
        'official_protocol.py', 'closed_gateway.py')}
    for name in ('loops/contracts/evaluation.py', 'runtime/evaluation.py', 'runtime/isolation.py',
                 'runtime/runtime_identity.py', 'runtime/evaluation_diagnostics.py'):
        result['rsisys/' + name] = file_digest(root.parents[1] / 'src/rsisys' / name)['sha256']
    return result


def bundle_manifest(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Submission bundles cannot contain symlinks')
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError('Submission bundles contain regular files only')
        result[str(path.relative_to(root))] = file_digest(path)
    return result


def bundle_configuration(root, bases):
    root = Path(root)
    config = json.loads((root / 'submission.json').read_text())
    if (not isinstance(config, dict) or set(config) - {
            'schema', 'entrypoint', 'base_model_path', 'checkpoint_path', 'adapter_path'}
            or not {'schema', 'entrypoint', 'base_model_path'} <= set(config)
            or config['schema'] != BUNDLE_SCHEMA or config['base_model_path'] not in bases
            or {'checkpoint_path', 'adapter_path'} <= set(config)):
        raise ValueError('Invalid submission.json; choose a registered base and a Python entrypoint')
    for name in ('entrypoint', 'checkpoint_path', 'adapter_path'):
        if name not in config:
            continue
        value = config[name]
        if (not isinstance(value, str) or not value or Path(value).is_absolute()
                or '..' in Path(value).parts or '\x00' in value):
            raise ValueError('Bundle paths must be relative and cannot escape the submission')
        path = (root / value).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Bundle path escapes submission')
        if name == 'entrypoint':
            if not path.is_file() or path.suffix != '.py':
                raise ValueError('entrypoint must name a submitted Python file')
        elif not path.is_dir():
            raise ValueError(name + ' must name a submitted directory')
    return config


def registered_base_identity(root):
    root = Path(root)
    if not (root / 'config.json').is_file():
        raise ValueError('Registered base requires config.json')
    # Candidate inference may use generation defaults or other installed base
    # files, so bind the whole registered inventory, not only model shards.
    files = bundle_manifest(root)
    return {'files': files, 'sha256': digest(files)}


def frozen_bundle(source, destination, bases):
    manifest = bundle_manifest(source)
    config = bundle_configuration(source, bases)
    shutil.copytree(source, destination)
    if bundle_manifest(destination) != manifest or bundle_manifest(source) != manifest:
        raise ValueError('Submission changed while freezing; no submission accepted')
    base = registered_base_identity(config['base_model_path'])
    return {'configuration': config, 'files': manifest, 'base_identity': base,
            'sha256': digest({'files': manifest, 'base_identity': base})}


def candidate_axes(submission):
    config, files = submission['configuration'], submission['files']
    roots = [config[name] for name in ('checkpoint_path', 'adapter_path') if name in config]
    def model_file(name):
        file = Path(name).name
        return (file.endswith(('.safetensors', '.bin', '.safetensors.index.json', '.bin.index.json'))
                or file in ('config.json', 'adapter_config.json'))
    weights = {name: value for name, value in files.items() if model_file(name)
               and any(name == root or name.startswith(root.rstrip('/') + '/') for root in roots)}
    inference = {name: value for name, value in files.items()
                 if name != 'submission.json' and name not in weights}
    return {'inference_sha256': digest({'files': inference, 'entrypoint': config['entrypoint']}),
            'weight_sha256': digest({'base': submission['base_identity']['sha256'],
                                    'mode': next((k for k in ('checkpoint_path', 'adapter_path') if k in config),
                                                 'registered_base'), 'files': weights})}


def safe_closed_result(raw, request, gateway):
    """Construct an allowlisted receipt. Never publish arbitrary gateway fields."""
    if (not isinstance(raw, dict) or raw.get('status') not in ('succeeded', 'failed')
            or any(raw.get(k) != request[k] for k in (
                'protocol_sha256', 'campaign_id', 'submission_id', 'candidate_id', 'submission_sha256'))
            or raw.get('service_id') != gateway['service_id']
            or ('runtime_sha256' in request and raw.get('runtime_sha256') != request['runtime_sha256'])
            or raw.get('dataset_sha256') != gateway['dataset_sha256']):
        raise ValueError('Closed gateway receipt identity mismatch')
    result = {k: raw[k] for k in ('status', 'protocol_sha256', 'campaign_id', 'submission_id',
                                  'candidate_id', 'submission_sha256', 'service_id', 'dataset_sha256')}
    if 'runtime_sha256' in request:
        result['runtime_sha256'] = raw['runtime_sha256']
    if raw['status'] == 'succeeded':
        metrics = raw.get('metrics')
        if not isinstance(metrics, dict) or set(metrics) != set(gateway['dataset_sha256']):
            raise ValueError('Missing closed aggregate metrics')
        for values in metrics.values():
            if (not isinstance(values, dict) or set(values) != {'accuracy', 'n'}
                    or type(values['n']) is not int or values['n'] < 1
                    or type(values['accuracy']) not in (int, float)
                    or not math.isfinite(values['accuracy']) or not 0 <= values['accuracy'] <= 1):
                raise ValueError('Invalid closed aggregate metrics')
        result['metrics'] = deepcopy(metrics)
    else:
        # Candidate diagnostics may contain questions. Keep all failure details
        # at the private gateway and expose only this fixed status code.
        result['failure_code'] = 'closed_candidate_failed'
    return result


def evaluate(config):
    if source_identity() != config['source_sha256']:
        raise ValueError('Host evaluation protocol source changed')
    protocol = EvaluationProtocol(**config.get('evaluation_protocol', asdict(PROTOCOL)))
    fixed_inputs, score_predictions = scoring_functions(protocol.score_protocol)
    bundle = Path(config['bundle_path'])
    if bundle_manifest(bundle) != config['submission']['files']:
        raise ValueError('Frozen submission changed')
    if config['split'] == 'closed':
        gateway = config['closed_gateway']
        request = {key: config[key] for key in ('protocol_sha256', 'campaign_id', 'candidate_id', 'submission_id')}
        request.update(submission_sha256=config['submission']['sha256'], bundle_path=str(bundle),
                       submission=config['submission'], protocol=asdict(protocol),
                       runtime_sha256=gateway['runtime_sha256'],
                       execution_budget=config['execution_budget'])
        # The configured command is a transport to an isolated service, not a
        # candidate command. Bounded output stays in memory and is allowlisted.
        raw = run_candidate_gateway(gateway['argv'], request, config['timeout_seconds'])
        record = safe_closed_result(raw, request, gateway)
    else:
        payload, gold, datasets = fixed_inputs(config['groups'])
        smoke = config['split'] == 'smoke'
        if smoke:
            payload = {**payload, 'items': payload['items'][:2]}
            if not payload['items']:
                raise ValueError('Smoke test requires public inputs')
        submission = config['submission']['configuration']
        def verify_base():
            return registered_base_identity(submission['base_model_path']) == config['submission']['base_identity']
        if not verify_base():
            raise ValueError('Registered base changed')
        # Template/tokenizer/decoding are candidate-owned. The host passes full
        # questions/options without gold, then scores strict final letters.
        policy = ExecutionPolicy.from_dict(config['candidate_execution'])
        expected_runtime = config['runtime_identity']
        installed_policy = ExecutionPolicy.from_dict(expected_runtime['policy'])
        if policy != replace(installed_policy,
                read_only=(*installed_policy.read_only, submission['base_model_path'])):
            raise ValueError('Candidate execution exposes undeclared dependencies')
        if runtime_identity(config['python'], installed_policy) != expected_runtime:
            raise ValueError('Installed inference runtime changed')
        isolation = ProcessIsolation(policy)
        predictions = run_candidate([config['python'], submission['entrypoint']], payload,
                                    execution=(isolation, bundle), timeout_seconds=max(1, config['timeout_seconds'] - 5))
        if smoke:
            expected = {(x['benchmark_id'], x['id']) for x in payload['items']}
            if (not isinstance(predictions, list) or len(predictions) != len(expected)
                    or any(not isinstance(x, dict) or set(x) != {'benchmark_id', 'id', 'text'}
                           or not isinstance(x['text'], str) or not x['text'].strip()
                           for x in predictions)
                    or {(x['benchmark_id'], x['id']) for x in predictions} != expected):
                raise ValueError('Smoke test did not return predictions for the supplied public inputs')
        else:
            metrics, scored = score_predictions(predictions, gold)
        if not verify_base():
            raise ValueError('Registered base changed during inference')
        if runtime_identity(config['python'], installed_policy) != expected_runtime:
            raise ValueError('Installed inference runtime changed during inference')
        record = ({'status': 'succeeded', 'inference_completed': True,
                   'prediction_count': len(predictions), 'model_load_verified': False,
                   'runtime_sha256': expected_runtime['sha256']} if smoke else
                  {'status': 'succeeded', 'metrics': metrics, 'predictions': scored,
                  'dataset_sha256': datasets, 'submission': config['submission'],
                  'runtime_sha256': expected_runtime['sha256'],
                  'output_diagnostics': prediction_diagnostics(scored),
                  **candidate_axes(config['submission'])})
    if bundle_manifest(bundle) != config['submission']['files']:
        raise ValueError('Frozen submission changed during inference; no score')
    record.update({key: config[key] for key in ('candidate_id', 'split', 'protocol_sha256', 'source_sha256')})
    record.update(protocol=protocol.score_protocol, evaluation_protocol=asdict(protocol),
                  submission_sha256=config['submission']['sha256'],
                  execution_budget=config['execution_budget'], adopted=False)
    atomic_write(Path(config['output']), json_bytes(record))
    return record


def run_candidate_gateway(argv, request, timeout):
    # Use files, rather than unbounded capture_output, for a potentially buggy
    # transport. Neither its stdout nor stderr becomes a Lab process log.
    import tempfile
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        result = subprocess.run(argv, input=json_bytes(request), stdout=output, stderr=errors,
                                timeout=timeout, check=False)
        output.seek(0)
        data = output.read(1024 * 1024 + 1)
    if result.returncode or len(data) > 1024 * 1024:
        raise ValueError('Closed gateway transport failed')
    return json.loads(data)


class ProtocolMedicalLab(FixedMedicalLab):
    def __init__(self, *args, base_model_paths, campaign_id, ledger_path, closed_gateway=None,
                 require_smoke_test=True,
                 resource_policy=None, candidate_execution=None, score_protocol=PROTOCOL.score_protocol, **kwargs):
        super().__init__(*args, **kwargs)
        self.protocol_runner = Path(__file__).resolve()
        self.closed_gateway = deepcopy(closed_gateway)
        if closed_gateway is not None:
            if (not isinstance(closed_gateway, dict) or set(closed_gateway) != {
                    'argv', 'service_id', 'dataset_sha256', 'runtime_sha256'}
                    or not isinstance(closed_gateway['argv'], list) or not closed_gateway['argv']
                    or any(not isinstance(v, str) or not v or '\x00' in v for v in closed_gateway['argv'])
                    or not isinstance(closed_gateway['service_id'], str) or not closed_gateway['service_id']
                    or not isinstance(closed_gateway['dataset_sha256'], dict) or not closed_gateway['dataset_sha256']):
                raise ValueError('Closed evaluation requires an operator-configured isolated gateway')
            runtime = closed_gateway['runtime_sha256']
            if not isinstance(runtime, str) or len(runtime) != 64:
                raise ValueError('Closed gateway requires a pinned runtime identity')
            int(runtime, 16)
            for name, sha in closed_gateway['dataset_sha256'].items():
                if not isinstance(name, str) or not name or not isinstance(sha, str) or len(sha) != 64:
                    raise ValueError('Invalid closed dataset identity')
                int(sha, 16)
        scoring_functions(score_protocol)
        self.evaluation_protocol = replace(PROTOCOL, score_protocol=score_protocol)
        self.ledger = EvaluationLedger(ledger_path, campaign_id, protocol=self.evaluation_protocol)
        self.candidate_execution = candidate_execution or inference_policy(self.evaluation_python)
        if not isinstance(self.candidate_execution, ExecutionPolicy):
            raise TypeError('candidate_execution must be a host ExecutionPolicy')
        ProcessIsolation(self.candidate_execution).check()
        self.installed_runtime = runtime_identity(self.evaluation_python, self.candidate_execution)
        if type(require_smoke_test) is not bool:
            raise TypeError('require_smoke_test must be bool')
        self.formal_configuration = {'evaluation_protocol': asdict(self.evaluation_protocol),
            'admission': 'same_bundle_runtime_public_smoke/v1',
            'require_smoke_test': require_smoke_test,
            'source_sha256': source_identity(), 'campaign_id': campaign_id,
            'ledger_path': str(Path(ledger_path).resolve()),
            'base_model_paths': list(dict.fromkeys(str(Path(p).resolve()) for p in base_model_paths)),
            'closed_gateway': self.closed_gateway,
            'candidate_execution': self.candidate_execution.configuration(),
            'installed_runtime_sha256': self.installed_runtime['sha256'],
            'execution_budget': {'timeout_seconds': self.timeout_seconds,
                                 'resources': deepcopy(resource_policy)}}
        if not self.formal_configuration['base_model_paths']:
            raise ValueError('At least one registered base is required')
        # Inference code is part of the candidate, so it is deliberately absent
        # from the common comparison protocol identity.
        self.protocol_sha256 = digest({'protocol': asdict(self.evaluation_protocol), 'source': source_identity(),
                                      'runtime_sha256': self.installed_runtime['sha256'],
                                      'closed_runtime_sha256': (closed_gateway or {}).get('runtime_sha256'),
                                      'execution_budget': self.formal_configuration['execution_budget']})

    def describe(self):
        text = {'type': 'string', 'minLength': 1}
        return (*LocalProcessLab.describe(self), OperationSpec('medical_smoke_test',
            'Execute this frozen bundle on up to two public inputs in the formal inference isolation/runtime. '
            'Collect with lab_poll/lab_result. Required before formal evaluation for the exact bundle/runtime. '
            'No scoring, closed data or closed slot; real inference uses assigned Lab devices and time. '
            'A passing run verifies execution, not model loading or medical improvement.',
            {'type': 'object', 'properties': {'key': text, 'submission_path': text},
             'required': ['key', 'submission_path'], 'additionalProperties': False}), OperationSpec('medical_preflight',
            'Validate submission paths, Python syntax and weight layout without loading models, '
            'starting evaluation or consuming the closed submission slot.',
            {'type': 'object', 'properties': {'submission_path': text}, 'required': ['submission_path'],
             'additionalProperties': False}, read_only=True),
            OperationSpec('medical_package',
            'Copy an inference entrypoint and optional checkpoint/LoRA from this workspace into a '
            'fresh self-contained bundle with submission.json. No symlinks or escaping weight paths. '
            'Additional helper modules must be copied into the bundle before medical_preflight.',
            {'type': 'object', 'properties': {k: text for k in ('destination_path', 'entrypoint_path',
                'base_model_path', 'adapter_path', 'checkpoint_path')},
             'required': ['destination_path', 'entrypoint_path', 'base_model_path'],
             'additionalProperties': False}), OperationSpec('medical_evaluate',
            'Submit a workspace bundle containing submission.json, inference Python and checkpoint/LoRA '
            'or a registered base reference. You own inference; the host owns data, scoring and budgets. '
            'public returns detailed development evidence; closed returns aggregate scores/status only '
            'through an isolated gateway, at most once per 1800 seconds across this campaign. '
            'An accepted failure consumes its slot. Reuse the same key only to query the same submission. '
            'First collect a successful medical_smoke_test for these exact files/runtime. '
            'Release active research programs first. Collect with lab_poll then lab_result.',
            {'type': 'object', 'properties': {'key': text, 'candidate_id': text,
                'split': {'enum': ['public', 'closed']}, 'submission_path': text},
             'required': ['key', 'candidate_id', 'split', 'submission_path'], 'additionalProperties': False}))

    def execute(self, call):
        if call.name == 'medical_smoke_test':
            return self._smoke_start(call.arguments)
        if call.name in ('medical_preflight', 'medical_package'):
            from .submission import preflight, package_submission
            from rsisys.runtime.bundles import workspace_path
            from rsisys.loops.contracts.coordination import OperationResult
            if self._status()['active_jobs']:
                raise LabToolError('Release active programs before checking or packaging their mutable files',
                                   data={'reason': 'candidate_artifacts_busy'})
            try:
                bases = self.formal_configuration['base_model_paths']
                args = call.arguments
                if not isinstance(args, dict):
                    raise ValueError('Expected an arguments object')
                if call.name == 'medical_preflight':
                    if set(args) != {'submission_path'}:
                        raise ValueError('Supply submission_path only')
                    result = preflight(workspace_path(self.workspace, args['submission_path']), bases)
                else:
                    self.require_admission()
                    allowed = {'destination_path', 'entrypoint_path', 'base_model_path', 'adapter_path', 'checkpoint_path'}
                    if set(args) - allowed or not {'destination_path', 'entrypoint_path', 'base_model_path'} <= set(args):
                        raise ValueError('Supply destination_path, entrypoint_path and base_model_path; optional weights')
                    result = package_submission(self.workspace, bases, **args)
                return OperationResult('Candidate bundle checked', (), False, data=result)
            except (ValueError, OSError, TypeError, KeyError, SyntaxError) as exc:
                raise LabToolError(str(exc), data={'reason': 'candidate_bundle_invalid',
                    'evaluation_submitted': False, 'next_operation': 'medical_package'}) from exc
        if call.name != 'medical_evaluate':
            if (call.name == 'lab_start' and isinstance(call.arguments, dict)
                    and call.arguments.get('key') not in self._keys
                    and any(row['job_id'] in self._evaluations for row in self._status()['active_jobs'])):
                raise LabToolError('Evaluation is using the assigned devices', data={'reason': 'evaluation_devices_busy'})
            return LocalProcessLab.execute(self, call)
        args = call.arguments
        if (not isinstance(args, dict) or set(args) != {'key', 'candidate_id', 'split', 'submission_path'}
                or any(not isinstance(v, str) or not v.strip() or '\x00' in v for v in args.values())
                or args['split'] not in ('public', 'closed')):
            raise LabToolError('Submit key, candidate_id, split and submission_path only')
        self.require_admission()
        if args['split'] == 'closed' and self.closed_gateway is None:
            raise LabToolError('An isolated closed evaluator has not been configured',
                               data={'reason': 'closed_evaluator_unavailable'})
        if args['split'] == 'closed' and not self.closed_gateway.get('runtime_sha256'):
            raise LabToolError('Closed gateway runtime identity is not configured',
                               data={'reason': 'closed_runtime_identity_unavailable'})
        if self.formal_configuration['source_sha256'] != source_identity():
            raise PermissionError('Host evaluation protocol source changed')
        identity = digest(args)
        previous = self.ledger.lookup(args['key'])
        if previous is not None:
            if previous['identity'] != identity or previous['split'] != args['split']:
                raise LabToolError('Evaluation key already identifies another submission')
            ident = previous['job_id']
            binding = self._evaluations.get(ident)
            if binding is None or args['key'] not in self._keys:
                raise LabToolError('Accepted evaluation requires host recovery; no automatic replay',
                                   data={'reason': 'accepted_submission_requires_recovery'})
            return self._start({'key': args['key'], 'argv': binding['command']})
        if args['key'] in self._keys:
            raise LabToolError('Key is already used by a research program')
        if self._status()['active_jobs']:
            raise LabToolError('Release owned programs before evaluation', data={'reason': 'evaluation_devices_busy'})
        if runtime_identity(self.evaluation_python, self.candidate_execution) != self.installed_runtime:
            raise PermissionError('Installed runtime changed after campaign configuration')
        source = self.workspace / args['submission_path']
        if (Path(args['submission_path']).is_absolute() or '..' in Path(args['submission_path']).parts
                or source.is_symlink() or not source.resolve().is_relative_to(self.workspace.resolve())
                or not source.is_dir()):
            raise LabToolError('submission_path must name a directory inside the workspace')
        ident = uuid4().hex
        destination = self.evaluation_root / (ident + '-bundle')
        try:
            submission = frozen_bundle(source, destination, self.formal_configuration['base_model_paths'])
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise LabToolError('Invalid candidate bundle: ' + str(exc)) from exc
        if self.formal_configuration['require_smoke_test'] and not self._smoke_passed(submission):
            shutil.rmtree(destination)
            raise LabToolError('Run and collect medical_smoke_test for the exact current bundle/runtime first',
                               data={'reason': 'candidate_smoke_required', 'evaluation_submitted': False,
                                     'closed_slot_consumed': False})
        try:
            self.require_admission()
            admission = self.ledger.reserve(args['key'], identity, args['split'])
        except RuntimeError:
            shutil.rmtree(destination)
            raise
        except EvaluationAdmissionError as exc:
            shutil.rmtree(destination)
            raise LabToolError(str(exc), data=exc.data) from exc
        if admission['replayed']:
            shutil.rmtree(destination)
            raise LabToolError('Concurrent submission accepted; reconcile its existing identity',
                               data={'reason': 'accepted_submission_requires_recovery'})
        candidate_policy = replace(self.candidate_execution,
            read_only=(*self.candidate_execution.read_only, submission['configuration']['base_model_path']))
        # The common installed runtime excludes the chosen base, whose complete
        # inventory is separately bound to submission_sha256.
        config = {'candidate_id': args['candidate_id'], 'split': args['split'],
            'protocol_sha256': self.protocol_sha256, 'source_sha256': source_identity(),
            'evaluation_protocol': asdict(self.evaluation_protocol),
            'campaign_id': self.ledger.campaign_id, 'submission_id': args['key'],
            'submission': submission, 'bundle_path': str(destination), 'python': self.evaluation_python,
            'execution_budget': self.formal_configuration['execution_budget'],
            'timeout_seconds': self.timeout_seconds, 'groups': self.evaluation_groups if args['split'] == 'public' else [],
            'closed_gateway': self.closed_gateway if args['split'] == 'closed' else None,
            'candidate_execution': candidate_policy.configuration(), 'runtime_identity': self.installed_runtime,
            'output': str(self.evaluation_root / (ident + '.json'))}
        config_path = self.evaluation_root / (ident + '-request.json')
        atomic_write(config_path, json_bytes(config))
        command = [self.evaluation_python, str(self.protocol_runner), str(config_path)]
        result = self._start({'key': args['key'], 'argv': command},
            executor=getattr(self.backend, 'exec_host', None))
        self._evaluations[result.data['job_id']] = {'arguments': deepcopy(args), 'config': config, 'command': command}
        self.ledger.attach(args['key'], result.data['job_id'])
        return result

    def _observe(self, ident):
        result = LocalProcessLab._observe(self, ident)
        binding = self._evaluations.get(ident)
        if binding is not None and binding.get('kind') == 'smoke':
            return self._smoke_observe(ident, result, binding)
        if (binding is None or result.data['state'] != 'succeeded'
                or 'fixed_evaluation' in result.data or 'fixed_evaluation_error' in result.data):
            return result
        config = binding['config']
        try:
            if bundle_manifest(config['bundle_path']) != config['submission']['files']:
                raise ValueError('Frozen submission changed before collection')
            raw = Path(config['output']).read_bytes()
            record = json.loads(raw)
            if (any(record[k] != config[k] for k in ('candidate_id', 'split', 'protocol_sha256', 'source_sha256'))
                    or record['evaluation_protocol'] != asdict(self.evaluation_protocol)
                    or record['protocol'] != self.evaluation_protocol.score_protocol
                    or record['execution_budget'] != config['execution_budget']
                    or record['submission_sha256'] != config['submission']['sha256']
                    or record['adopted'] is not False):
                raise ValueError('Evaluation receipt identity differs')
            if config['split'] == 'public':
                if record['runtime_sha256'] != config['runtime_identity']['sha256']:
                    raise ValueError('Inference runtime receipt identity differs')
                fixed_inputs, score_predictions = scoring_functions(self.evaluation_protocol.score_protocol)
                _, gold, datasets = fixed_inputs(self.evaluation_groups)
                metrics, scored = score_predictions([{k: v[k] for k in ('benchmark_id', 'id', 'text')}
                                                     for v in record['predictions']], gold)
                if (record['status'] != 'succeeded' or record['dataset_sha256'] != datasets
                        or record['metrics'] != metrics or record['predictions'] != scored):
                    raise ValueError('Public evaluation score differs from evidence')
                if (record.get('output_diagnostics') != prediction_diagnostics(scored)
                        or any(record.get(k) != v for k, v in candidate_axes(config['submission']).items())):
                    raise ValueError('Public output diagnostics or candidate identity differs')
                artifact = self.artifacts.put(raw, kind='public_medical_protocol_evaluation')
                result = replace(result, artifacts=(*result.artifacts, artifact))
            else:
                request = {k: config[k] for k in ('protocol_sha256', 'campaign_id', 'submission_id', 'candidate_id')}
                request['submission_sha256'] = config['submission']['sha256']
                request['runtime_sha256'] = self.closed_gateway['runtime_sha256']
                safe = safe_closed_result(record, request, self.closed_gateway)
                if safe['status'] == 'failed':
                    result = replace(result, status=ExecutionStatus.FAILED, error='Closed candidate failed',
                                     data={**result.data, 'state': 'failed', 'closed_evaluation_status': 'failed'})
                    self._terminal[ident] = result
                    return result
            fixed = {k: record[k] for k in ('protocol', 'evaluation_protocol', 'protocol_sha256', 'candidate_id',
                                            'split', 'metrics', 'dataset_sha256', 'submission_sha256')}
            fixed.update(record_sha256=hashlib.sha256(raw).hexdigest(), adopted=False,
                         evaluator_sha256=config['source_sha256']['protocol_evaluation.py'], verified=True)
            fixed['runtime_sha256'] = record['runtime_sha256']
            if config['split'] == 'public':
                fixed.update(status=record['status'], output_diagnostics=record['output_diagnostics'],
                    execution_budget=record['execution_budget'], **candidate_axes(config['submission']))
            result = replace(result, data={**result.data, 'fixed_evaluation': fixed})
        except Exception:
            # Do not expose gateway payloads, candidate output or private paths
            # in exception strings. Detailed closed diagnostics stay remote.
            result = replace(result, status=ExecutionStatus.FAILED, error='Evaluation receipt verification failed',
                data={**result.data, 'failure_origin': 'infrastructure',
                      'fixed_evaluation_error': 'evaluation_receipt_verification_failed'})
        self._terminal[ident] = result
        return result

    def _smoke_start(self, args):
        from rsisys.runtime.bundles import workspace_path
        if (not isinstance(args, dict) or set(args) != {'key', 'submission_path'}
                or any(not isinstance(v, str) or not v.strip() or '\x00' in v for v in args.values())):
            raise LabToolError('Supply smoke key and submission_path only')
        self.require_admission()
        if args['key'] in self._keys:
            ident, _ = self._keys[args['key']]
            previous = self._evaluations.get(ident)
            if previous is None or previous.get('kind') != 'smoke' or previous['arguments'] != args:
                raise LabToolError('Smoke key already identifies a different submission')
            # Idempotent identity lookup, never rerun an accepted mutation.
            return self._start({'key': args['key'], 'argv': previous['command']})
        if self._status()['active_jobs']:
            raise LabToolError('Release owned programs before smoke inference')
        if runtime_identity(self.evaluation_python, self.candidate_execution) != self.installed_runtime:
            raise PermissionError('Installed runtime changed after campaign configuration')
        if self.formal_configuration['source_sha256'] != source_identity():
            raise PermissionError('Host evaluation protocol source changed')
        try:
            source = workspace_path(self.workspace, args['submission_path'])
            ident = uuid4().hex
            destination = self.evaluation_root / (ident + '-smoke-bundle')
            submission = frozen_bundle(source, destination, self.formal_configuration['base_model_paths'])
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise LabToolError('Invalid candidate smoke bundle: ' + str(exc)) from exc
        try:
            self.require_admission()
        except RuntimeError:
            shutil.rmtree(destination)
            raise
        config = {'candidate_id': 'smoke-only', 'split': 'smoke', 'protocol_sha256': self.protocol_sha256,
            'source_sha256': source_identity(), 'evaluation_protocol': asdict(self.evaluation_protocol),
            'submission': submission, 'bundle_path': str(destination), 'python': self.evaluation_python,
            'execution_budget': self.formal_configuration['execution_budget'],
            'timeout_seconds': min(300, self.timeout_seconds), 'groups': self.evaluation_groups,
            'candidate_execution': replace(self.candidate_execution,
                read_only=(*self.candidate_execution.read_only,
                           submission['configuration']['base_model_path'])).configuration(),
            'runtime_identity': self.installed_runtime,
            'output': str(self.evaluation_root / (ident + '-smoke.json'))}
        config_path = self.evaluation_root / (ident + '-smoke-request.json')
        atomic_write(config_path, json_bytes(config))
        command = [self.evaluation_python, str(self.protocol_runner), str(config_path)]
        result = self._start({'key': args['key'], 'argv': command},
                             executor=getattr(self.backend, 'exec_host', None))
        self._evaluations[result.data['job_id']] = {'kind': 'smoke', 'arguments': deepcopy(args),
                                                   'config': config, 'command': command}
        return result

    def _smoke_observe(self, ident, result, binding):
        if result.data['state'] != 'succeeded' or 'smoke_test' in result.data or 'smoke_error' in result.data:
            return result
        config = binding['config']
        try:
            record = json.loads(Path(config['output']).read_bytes())
            if (record['status'] != 'succeeded' or record['inference_completed'] is not True
                    or type(record['prediction_count']) is not int or not 1 <= record['prediction_count'] <= 2
                    or record['runtime_sha256'] != self.installed_runtime['sha256']
                    or any(record[k] != config[k] for k in
                           ('protocol_sha256', 'source_sha256', 'candidate_id', 'split'))
                    or record['submission_sha256'] != config['submission']['sha256']
                    or bundle_manifest(config['bundle_path']) != config['submission']['files']):
                raise ValueError('Smoke receipt identity mismatch')
            result = replace(result, data={**result.data, 'smoke_test': {
                'verified': True, 'inference_completed': True, 'model_load_verified': False,
                'submission_sha256': record['submission_sha256'],
                'runtime_sha256': record['runtime_sha256'], 'protocol_sha256': record['protocol_sha256'],
                'prediction_count': record['prediction_count'], 'scored': False,
                'closed_slot_consumed': False}})
        except Exception:
            result = replace(result, status=ExecutionStatus.FAILED, error='Smoke receipt verification failed',
                             data={**result.data, 'smoke_error': 'receipt_verification_failed'})
        self._terminal[ident] = result
        return result

    def _smoke_passed(self, submission):
        for ident, binding in self._evaluations.items():
            if (binding.get('kind') != 'smoke'
                    or binding['config']['submission']['sha256'] != submission['sha256']):
                continue
            result = self._observe(ident)
            receipt = (result.data or {}).get('smoke_test', {})
            if (result.status == ExecutionStatus.COMPLETED and receipt.get('verified') is True
                    and receipt['runtime_sha256'] == self.installed_runtime['sha256']
                    and receipt['protocol_sha256'] == self.protocol_sha256):
                return True
        return False


if __name__ == '__main__':
    configuration = json.loads(Path(sys.argv[1]).read_text())
    try:
        value = evaluate(configuration)
        print(json.dumps({'status': value['status'], 'split': value['split']}), flush=True)
    except Exception:
        # Gateway errors can include closed questions. Never print their text.
        if configuration['split'] in ('public', 'smoke'):
            import traceback
            traceback.print_exc()
        else:
            print('evaluation_execution_failed', file=sys.stderr, flush=True)
        sys.exit(1)
