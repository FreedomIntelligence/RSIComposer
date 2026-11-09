"""Operator-side local closed evaluator; never mount its config/data into Lab.

The transport process is trusted host code. It imports a frozen candidate into
an operator-owned directory, then supplies only gold-free questions through a
pipe to a separate filesystem/device/network-isolated inference worker. Raw
outputs and diagnostics stay here; stdout contains only allowlisted aggregates.
"""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
import traceback
from uuid import uuid4

from rsisys.loops.contracts.evaluation import EvaluationProtocol
from rsisys.runtime.evaluation import EvaluationLedger
from rsisys.runtime.files import atomic_write, json_bytes
from rsisys.runtime.isolation import ExecutionPolicy, ProcessIsolation
from rsisys.runtime.runtime_identity import runtime_identity
from .protocol_evaluation import frozen_bundle, bundle_manifest, registered_base_identity, digest, source_identity
from .official_protocol import scoring_functions
from .fixed_evaluation import run_candidate


def serve(config, request):
    root = Path(config['root']).resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    protocol = EvaluationProtocol(**config['protocol'])
    if (request['campaign_id'] != config['campaign_id'] or request['protocol'] != asdict(protocol)
            or request['protocol_sha256'] != config['protocol_sha256']
            or request['execution_budget'] != config['execution_budget']
            or request['runtime_sha256'] != config['runtime_identity']['sha256']
            or source_identity() != config['source_sha256']):
        raise ValueError('Closed evaluator identity differs')
    now = time.time()
    if now >= config['deadline_epoch']:
        raise RuntimeError('Absolute research deadline reached')
    source = Path(request['bundle_path']).resolve(strict=True)
    inbox = Path(config['submission_root']).resolve(strict=True)
    if not source.is_relative_to(inbox) or source == inbox:
        raise ValueError('Submission is outside the host frozen-bundle inbox')
    identity = digest({k: request[k] for k in ('campaign_id', 'candidate_id', 'submission_id',
                                              'submission_sha256', 'protocol_sha256')})
    ledger = EvaluationLedger(root / 'closed-ledger.json', config['campaign_id'], protocol=protocol)
    previous = ledger.lookup(request['submission_id'])
    if previous is not None:
        if previous['identity'] != identity:
            raise ValueError('Closed submission key conflict')
        if previous['job_id'] is None or not (root / previous['job_id'] / 'receipt.json').is_file():
            raise RuntimeError('Unconfirmed closed submission; reconcile without automatic replay')
        return json.loads((root / previous['job_id'] / 'receipt.json').read_text())
    fixed_inputs, score_predictions = scoring_functions(protocol.score_protocol)
    payload, gold, datasets = fixed_inputs(config['groups'])
    if datasets != config['dataset_sha256']:
        raise ValueError('Closed dataset identity changed')
    installed = ExecutionPolicy.from_dict(config['runtime_identity']['policy'])
    if installed.sockets:
        raise ValueError('Closed worker cannot receive model broker or other sockets')
    if runtime_identity(config['python'], installed) != config['runtime_identity']:
        raise ValueError('Closed inference runtime changed')
    ledger.reserve(request['submission_id'], identity, 'closed')
    job_id = uuid4().hex
    job = root / job_id
    job.mkdir(mode=0o700)
    ledger.attach(request['submission_id'], job_id)
    receipt = {k: request[k] for k in ('campaign_id', 'candidate_id', 'submission_id',
                                      'submission_sha256', 'protocol_sha256', 'runtime_sha256')}
    receipt.update(status='failed', service_id=config['service_id'], dataset_sha256=datasets)
    try:
        submission = frozen_bundle(source, job / 'bundle', config['base_model_paths'])
        if submission != request['submission'] or submission['sha256'] != request['submission_sha256']:
            raise ValueError('Imported closed submission identity differs')
        base = submission['configuration']['base_model_path']
        policy = replace(installed, read_only=(*installed.read_only, base))
        # Candidate gets only runtime, registered base and frozen bundle. No
        # source cache, private config, ledger, Lab workspace or host network.
        remaining = min(config['execution_budget']['timeout_seconds'], config['deadline_epoch'] - time.time())
        if remaining <= 5:
            raise RuntimeError('Insufficient time before absolute research deadline')
        with (job / 'private-inference.stderr').open('wb') as errors:
            predictions = run_candidate([config['python'], submission['configuration']['entrypoint']], payload,
                execution=(ProcessIsolation(policy), job / 'bundle'), timeout_seconds=remaining - 5, stderr=errors)
        metrics, scored = score_predictions(predictions, gold)
        if (bundle_manifest(job / 'bundle') != submission['files']
                or registered_base_identity(base) != submission['base_identity']
                or runtime_identity(config['python'], installed) != config['runtime_identity']):
            raise ValueError('Closed submission or runtime changed during inference')
        atomic_write(job / 'private-predictions.json', json_bytes(scored))
        receipt.update(status='succeeded', metrics=metrics)
    except Exception:
        (job / 'private-error.txt').write_text(traceback.format_exc())
    atomic_write(job / 'receipt.json', json_bytes(receipt))
    return receipt


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    request = json.load(sys.stdin)
    result = serve(config, request)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # Detailed operator traces must not cross back into the Lab stream.
        print('closed_evaluator_failed', file=sys.stderr)
        raise SystemExit(1)
