"""Exercise submitted inference code and the redacted closed-gateway boundary.

The closed gateway here is a synthetic transport fixture, not an isolation test
or a medical benchmark. Production deployment must supply the isolated service.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from examples.medical import protocol_evaluation as host
from examples.medical.open_lab_experiment import RetainedAllocatedProcessBackend
from rsisys.adapters.local_lab_state import LocalLabState
from rsisys.loops.contracts.decision import OperationCall
from rsisys.loops.contracts.execution import LabToolError, LabInfrastructureError
from rsisys.runtime.artifacts import InMemoryArtifactStore
from rsisys.runtime.accounting import HostAdmissionStop


INFERENCE = '''import json,sys
from pathlib import Path
payload=json.load(sys.stdin)
assert 'gold' not in json.dumps(payload)
assert 'full question' in json.dumps(payload)
config=json.loads(Path('submission.json').read_text())
assert Path(config['checkpoint_path']).is_dir()
print(json.dumps({'predictions':[{'benchmark_id':r['benchmark_id'],'id':r['id'],'text':'J'} for r in payload['items']]}))
'''


class MedicalProtocolTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.base = self.root / 'base'
        self.base.mkdir()
        (self.base / 'config.json').write_text('{}')
        (self.base / 'model.safetensors').write_bytes(b'synthetic-not-a-model')
        self.data = self.root / 'public.jsonl'
        self.data.write_text(json.dumps({'id': '1', 'question': 'full question',
            'options': {letter: letter for letter in 'ABCDEFGHIJ'}, 'gold': 'J'}) + '\n')
        repo = Path(__file__).resolve().parents[1]
        self.backend = RetainedAllocatedProcessBackend(self.root / 'executor',
            {'CUDA_VISIBLE_DEVICES': '0', 'PYTHONPATH': str(repo / 'src') + ':' + str(repo)})
        self.options = {'python': sys.executable, 'evaluation_root': self.root / 'evaluations',
            'evaluation_groups': [{'benchmark_id': 'public-fixture', 'path': str(self.data),
                'sha256': hashlib.sha256(self.data.read_bytes()).hexdigest()}],
            'base_model_paths': [self.base], 'campaign_id': 'campaign',
            'ledger_path': self.root / 'campaign-ledger.json', 'timeout_seconds': 10}
        self.lab = host.ProtocolMedicalLab(self.backend, InMemoryArtifactStore(), **self.options)
        self.addCleanup(self.lab.close)
        self.bundle = self.lab.workspace / 'candidate'
        self.bundle.mkdir()
        (self.bundle / 'weights').mkdir()
        (self.bundle / 'weights/model.safetensors').write_bytes(b'synthetic-candidate')
        (self.bundle / 'infer.py').write_text(INFERENCE)
        (self.bundle / 'submission.json').write_text(json.dumps({'schema': host.BUNDLE_SCHEMA,
            'entrypoint': 'infer.py', 'base_model_path': str(self.base), 'checkpoint_path': 'weights'}))
        self.args = {'key': 'one', 'candidate_id': 'candidate-1', 'split': 'public', 'submission_path': 'candidate'}

    def submit(self, **changes):
        return self.lab.execute(OperationCall('medical_evaluate', {**self.args, **changes})).data['job_id']

    def complete(self, ident):
        process = self.backend.wait(self.lab._jobs[ident][0])
        self.assertEqual(process.exit_code, 0, process.stderr)
        return self.lab.execute(OperationCall('lab_result', {'job_id': ident}))

    def test_public_executes_frozen_script_with_weights_and_host_recomputes_score(self):
        ident = self.submit()
        (self.bundle / 'infer.py').write_text('raise RuntimeError("later workspace edit")')
        result = self.complete(ident)
        fixed = result.data['fixed_evaluation']
        self.assertEqual(fixed['metrics']['public-fixture'], {'accuracy': 1., 'n': 1})
        self.assertEqual(fixed['split'], 'public')
        self.assertTrue(fixed['verified'])
        self.assertFalse(fixed['adopted'])
        with patch.object(self.backend, 'exec', side_effect=AssertionError('must not reexecute')):
            self.assertEqual(self.submit(), ident)
        self.assertEqual(len(json.loads(self.options['ledger_path'].read_text())['submissions']), 1)

    def test_no_isolated_gateway_means_no_closed_acceptance_or_dataset_access(self):
        with patch.object(host, 'frozen_bundle', side_effect=AssertionError('no freeze')):
            with self.assertRaises(LabToolError) as error:
                self.submit(split='closed')
        self.assertEqual(error.exception.data['reason'], 'closed_evaluator_unavailable')
        self.assertFalse(self.options['ledger_path'].exists())
        self.assertEqual(self.lab._jobs, {})

    def test_absolute_admission_closes_after_freeze_without_consuming_quota_or_starting(self):
        open_gate = [True]
        original = host.frozen_bundle
        def freeze(*args):
            result = original(*args)
            open_gate[0] = False
            return result
        self.lab.admission_open = lambda: open_gate[0]
        with patch.object(host, 'frozen_bundle', freeze):
            with self.assertRaises(HostAdmissionStop):
                self.submit()
        self.assertFalse(self.options['ledger_path'].exists())
        self.assertEqual(self.lab._jobs, {})
        self.assertEqual(list(self.lab.evaluation_root.iterdir()), [])

    def test_formal_candidate_cannot_read_unmanifested_external_dependency(self):
        external = self.root / 'mutable-answer.txt'
        (self.bundle / 'infer.py').write_text('import json,sys\nfrom pathlib import Path\n'
            f'answer=Path({str(external)!r}).read_text()\n'
            'p=json.load(sys.stdin)\nprint(json.dumps({"predictions":[{"benchmark_id":r["benchmark_id"],'
            '"id":r["id"],"text":answer} for r in p["items"]]}))\n')
        identities = []
        for letter in ('A', 'J'):
            external.write_text(letter)
            ident = self.submit(key='external-' + letter)
            identities.append(self.lab._evaluations[ident]['config']['submission']['sha256'])
            state = self.backend.wait(self.lab._jobs[ident][0])
            self.assertNotEqual(state.exit_code, 0)
            result = self.lab.execute(OperationCall('lab_result', {'job_id': ident}))
            self.assertNotIn('fixed_evaluation', result.data)
        self.assertEqual(identities[0], identities[1])

    def test_bundled_dependency_is_frozen_and_runtime_identity_is_in_receipt(self):
        (self.bundle / 'answer.txt').write_text('J')
        (self.bundle / 'infer.py').write_text(INFERENCE.replace("'text':'J'", "'text':Path('answer.txt').read_text()"))
        ident = self.submit()
        (self.bundle / 'answer.txt').write_text('A')
        fixed = self.complete(ident).data['fixed_evaluation']
        self.assertEqual(fixed['metrics']['public-fixture']['accuracy'], 1.)
        self.assertEqual(fixed['runtime_sha256'], self.lab._evaluations[ident]['config']['runtime_identity']['sha256'])

    def test_mutating_declared_installed_dependency_requires_a_new_protocol(self):
        from dataclasses import replace
        library = self.root / 'installed-runtime'
        library.mkdir()
        (library / 'helper.py').write_text('VERSION=1')
        policy = replace(self.lab.candidate_execution, read_only=(*self.lab.candidate_execution.read_only, str(library)))
        other = host.ProtocolMedicalLab(self.backend, self.lab.artifacts, candidate_execution=policy, **self.options)
        self.addCleanup(other.close)
        self.assertNotEqual(other.protocol_sha256, self.lab.protocol_sha256)
        (library / 'helper.py').write_text('VERSION=2')
        with self.assertRaisesRegex(PermissionError, 'runtime changed'):
            other.execute(OperationCall('medical_evaluate', self.args))
        self.assertFalse(self.options['ledger_path'].exists())

    def test_closed_gateway_aggregates_are_allowlisted_and_all_log_routes_stay_redacted(self):
        gateway = self.root / 'gateway-fixture.py'
        gateway.write_text('''import json,sys
r=json.load(sys.stdin)
assert 'groups' not in r
assert 'full question' not in json.dumps(r)
result={k:r[k] for k in ('protocol_sha256','campaign_id','submission_id','candidate_id','submission_sha256')}
result.update(status='succeeded',service_id='synthetic-gateway',dataset_sha256={'closed-fixture':'a'*64},runtime_sha256=r['runtime_sha256'],
 metrics={'closed-fixture':{'accuracy':0.5,'n':2}},predictions=['SECRET_QUESTION'],stderr='SECRET_GOLD')
print('SECRET_QUESTION',file=sys.stderr)
print(json.dumps(result))
''')
        self.lab.closed_gateway = {'argv': [sys.executable, str(gateway)], 'service_id': 'synthetic-gateway',
                                   'dataset_sha256': {'closed-fixture': 'a' * 64}, 'runtime_sha256': 'b' * 64}
        ident = self.submit(split='closed')
        result = self.complete(ident)
        self.assertEqual(result.data['fixed_evaluation']['metrics']['closed-fixture']['accuracy'], .5)
        for stream in ('stdout', 'stderr'):
            log = self.lab.execute(OperationCall('lab_read_log', {'job_id': ident, 'stream': stream}))
            self.assertNotIn('SECRET', json.dumps(log.data))
        self.assertNotIn('SECRET', json.dumps(result.data))
        output = Path(self.lab._evaluations[ident]['config']['output']).read_text()
        self.assertNotIn('SECRET', output)
        with self.assertRaises(LabToolError) as error:
            self.submit(split='closed', key='new-candidate', candidate_id='candidate-2')
        self.assertEqual(error.exception.data['reason'], 'closed_evaluation_rate_limited')

    def test_bad_gateway_errors_and_timeout_messages_never_leak_closed_text(self):
        request = {'protocol_sha256': 'p', 'campaign_id': 'c', 'submission_id': 's',
                   'candidate_id': 'x', 'submission_sha256': 'w'}
        gateway = {'service_id': 'g', 'dataset_sha256': {'test': 'a' * 64}}
        raw = {**request, **gateway, 'status': 'succeeded', 'metrics': {'test': {'accuracy': True, 'n': 1}}}
        with self.assertRaises(ValueError):
            host.safe_closed_result(raw, request, gateway)
        raw.update(status='failed', exception='SECRET', predictions=['SECRET'])
        safe = host.safe_closed_result(raw, request, gateway)
        self.assertNotIn('SECRET', json.dumps(safe))
        self.assertNotIn('metrics', safe)

    def test_failed_start_consumes_slot_and_unknown_effect_is_not_replayed(self):
        self.lab.closed_gateway = {'argv': [sys.executable, '-c', 'pass'], 'service_id': 'g',
                                   'dataset_sha256': {'closed': 'a' * 64}, 'runtime_sha256': 'b' * 64}
        with patch.object(self.backend, 'exec', side_effect=OSError('fixture start failure')):
            with self.assertRaises(OSError):
                self.submit(split='closed')
        with self.assertRaises(LabToolError) as error:
            self.submit(split='closed')
        self.assertEqual(error.exception.data['reason'], 'accepted_submission_requires_recovery')
        with self.assertRaises(LabToolError) as error:
            self.submit(split='closed', key='different')
        self.assertEqual(error.exception.data['reason'], 'closed_evaluation_rate_limited')

    def test_tampered_snapshot_and_forged_score_are_not_published(self):
        ident = self.submit()
        self.backend.wait(self.lab._jobs[ident][0])
        output = Path(self.lab._evaluations[ident]['config']['output'])
        record = json.loads(output.read_text())
        record['metrics']['public-fixture']['accuracy'] = .5
        output.write_text(json.dumps(record))
        with self.assertRaises(LabInfrastructureError):
            self.lab.execute(OperationCall('lab_result', {'job_id': ident}))
        (self.bundle / 'bad-link').symlink_to(self.data)
        with self.assertRaises(LabToolError):
            self.submit(key='symlink')

    def test_uncollected_result_recovers_with_same_identity_and_no_execution(self):
        ident = self.submit()
        self.backend.wait(self.lab._jobs[ident][0])
        def port(lab):
            return LocalLabState(lab, capture_extra=lambda: deepcopy(lab._evaluations),
                restore_extra=lambda state: setattr(lab, '_evaluations', state), configuration=lab.formal_configuration)
        state = port(self.lab).capture()
        self.lab.close()
        backend = RetainedAllocatedProcessBackend(self.root / 'executor', self.backend.environment)
        restored = host.ProtocolMedicalLab(backend, self.lab.artifacts, defer_session=True, **self.options)
        self.addCleanup(restored.close)
        with patch.object(backend, 'exec', side_effect=AssertionError('no replay on recovery')):
            port(restored).restore(state)
            self.assertTrue(restored.execute(OperationCall('medical_evaluate', self.args)).data['replayed'])
            result = restored.execute(OperationCall('lab_result', {'job_id': ident}))
            self.assertEqual(result.data['fixed_evaluation']['metrics']['public-fixture']['accuracy'], 1.)
