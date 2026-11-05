"""Synthetic input faults exercised through the real candidate subprocess path."""
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from examples.medical.candidate_adapter import (
    CandidateInputError, predict_payload, validate_receipts,
)
from examples.medical.fixed_evaluation import FixedMedicalLab, diagnostic_inputs
from examples.medical.open_lab_experiment import AllocatedProcessBackend
from rsisys.loops.contracts.decision import OperationCall
from rsisys.runtime.artifacts import InMemoryArtifactStore


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / 'examples/medical/candidate_adapter.py'


class Tokenizer:
    def apply_chat_template(self, messages, **options):
        return '\n'.join(m['role'] + ':' + m['content'] for m in messages)


class DroppingTokenizer(Tokenizer):
    def apply_chat_template(self, messages, **options):
        return messages[0]['content']


class CandidateAdapterTests(unittest.TestCase):
    def test_system_only_render_and_raw_input_bypass_fail_before_scoring(self):
        payload = diagnostic_inputs()
        with self.assertRaisesRegex(CandidateInputError, 'omits the fixed user'):
            predict_payload(payload, lambda request: request.render(DroppingTokenizer()))
        with self.assertRaisesRegex(CandidateInputError, 'No prepared input'):
            predict_payload(payload, lambda request: request.messages[0]['content'])
        with self.assertRaisesRegex(CandidateInputError, 'Missing user'):
            predict_payload(payload, lambda request: request.render(
                Tokenizer(), messages=[request.messages[0]]))

    def test_batch_keeps_complete_inputs_but_constant_answers_are_not_rejected(self):
        seen = []
        def batch(requests):
            seen.extend(request.render(Tokenizer()) for request in requests)
            return ['A'] * len(requests)
        payload = diagnostic_inputs()
        result = predict_payload(payload, infer_batch=batch)
        self.assertIn('CANARY_ORANGE_41', seen[0])
        self.assertIn('CANARY_PURPLE_73', seen[1])
        self.assertNotEqual(seen[0], seen[1])
        self.assertEqual([p['text'] for p in result['predictions']], ['A', 'A'])
        report = validate_receipts(result['input_receipts'], payload)
        self.assertEqual(report['status'], 'adapter_verified')
        self.assertIn('not_attested', report['binding'])
        changed = json.loads(json.dumps(result['input_receipts']))
        changed[0]['source_messages_sha256'] = '0' * 64
        with self.assertRaisesRegex(CandidateInputError, 'does not match'):
            validate_receipts(changed, payload)

    def test_service_messages_preserve_roles_and_reject_user_omission(self):
        def infer(request):
            messages = request.chat_messages()
            self.assertEqual([m['role'] for m in messages], ['system', 'user'])
            return 'B'
        result = predict_payload(diagnostic_inputs(), infer)
        self.assertTrue(all(r['kind'] == 'chat_messages' for r in result['input_receipts']))
        with self.assertRaisesRegex(CandidateInputError, 'omit or alter'):
            predict_payload(diagnostic_inputs(), lambda request: request.chat_messages(messages=[
                {'role': 'user', 'content': 'wrong input'}]))


class CandidateSubprocessTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'fixture.jsonl'
        self.data.write_text(json.dumps({'id': 'one', 'question': 'Synthetic formal fixture?',
            'options': {'A': 'alpha', 'B': 'beta'}, 'gold': 'A'}) + '\n')
        self.backend = AllocatedProcessBackend(self.root / 'processes', {
            'CUDA_VISIBLE_DEVICES': 'synthetic-no-gpu',
            'PYTHONPATH': str(ROOT / 'src') + ':' + str(ROOT)})
        self.addCleanup(self.backend.close_all)
        self.lab = FixedMedicalLab(self.backend, InMemoryArtifactStore(), timeout_seconds=10,
            python=sys.executable, evaluation_root=self.root / 'evaluations', require_input_contract=True,
            evaluation_groups=[{'benchmark_id': 'fixture', 'path': str(self.data),
                'sha256': hashlib.sha256(self.data.read_bytes()).hexdigest()}])
        self.addCleanup(self.lab.close)
        self.argv = [sys.executable, 'candidate.py']

    def write_candidate(self, *, broken):
        render = "messages[0]['content']" if broken else "'\\n'.join(m['content'] for m in messages)"
        code = f'''import importlib.util
spec=importlib.util.spec_from_file_location('adapter',{str(ADAPTER)!r})
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
class Tokenizer:
    def apply_chat_template(self,messages,**options):return {render}
def infer(request):
    prompt=request.render(Tokenizer())
    return 'B'
adapter.run(infer)
'''
        self.lab.execute(OperationCall('lab_write_file', {'path': 'candidate.py', 'content': code}))

    def execute(self, name, key):
        args = {'key': key, 'candidate_id': 'synthetic-fixture', 'argv': self.argv}
        ident = self.lab.execute(OperationCall(name, args)).data['job_id']
        self.backend.wait(self.lab._jobs[ident][0])
        return self.lab.execute(OperationCall('lab_result', {'job_id': ident}))

    def test_same_argv_fault_repair_diagnosis_and_formal_score(self):
        self.write_candidate(broken=True)
        failed = self.execute('medical_diagnose', 'broken')
        self.assertEqual(failed.data['state'], 'failed')
        self.assertEqual(failed.data['failure_origin'], 'program')
        self.assertIn('omits the fixed user', failed.data['stderr'])
        self.assertNotIn('fixed_evaluation', failed.data)
        self.write_candidate(broken=False)
        # Diagnosis must not touch formal data, even when it is unavailable.
        self.data.rename(self.data.with_suffix('.saved'))
        diagnosis = self.execute('medical_diagnose', 'repaired-diagnosis')
        self.assertEqual(diagnosis.data['state'], 'succeeded', diagnosis.error)
        report = diagnosis.data['input_diagnosis']
        self.assertEqual(report['candidate_argv'], self.argv)
        self.assertTrue(report['synthetic'])
        self.assertNotIn('metrics', report)
        self.assertNotIn('fixed_evaluation', diagnosis.data)
        self.assertEqual(report['input_contract']['samples'], 2)
        self.data.with_suffix('.saved').rename(self.data)
        result = self.execute('medical_evaluate', 'same-path-formal')
        self.assertEqual(result.data['state'], 'succeeded', result.error)
        fixed = result.data['fixed_evaluation']
        self.assertEqual(fixed['input_contract']['status'], 'adapter_verified')
        self.assertEqual(fixed['metrics']['fixture'], {'accuracy': 0.0, 'n': 1})
        self.assertFalse(fixed['adopted'])
        self.assertEqual(len(list((self.root / 'evaluations').glob('*.json'))), 1)

    def test_legacy_stdout_cannot_silently_bypass_required_adapter(self):
        self.lab.execute(OperationCall('lab_write_file', {'path': 'candidate.py', 'content':
            "import json,sys;p=json.load(sys.stdin);print(json.dumps({'predictions':["
            "{'benchmark_id':i['benchmark_id'],'id':i['id'],'text':'A'} for i in p['items']]}))"}))
        result = self.execute('medical_evaluate', 'missing-adapter')
        self.assertEqual(result.data['state'], 'failed')
        self.assertIn('Missing standard input adapter', result.data['stderr'])
        self.assertNotIn('fixed_evaluation', result.data)


if __name__ == '__main__':
    unittest.main()
