from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
import unittest

from examples.medical import closed_gateway as gateway
from examples.medical.official_protocol import SCORE_PROTOCOL, extract_answer, official_inputs, official_score_predictions
from examples.medical.protocol_evaluation import frozen_bundle, source_identity
from examples.medical.fixed_evaluation import run_candidate
from rsisys.loops.contracts.evaluation import EvaluationProtocol
from rsisys.runtime.evaluation import EvaluationAdmissionError
from rsisys.runtime.runtime_identity import inference_policy, runtime_identity


class OfficialMedicalProtocolTests(unittest.TestCase):
    def test_official_first_answer_extraction_and_all_rows_denominator(self):
        for text, expected in [('B', 'B'), ('The answer is (B).', 'B'),
                ('Among A through D, the answer is C. A is implausible.', 'C'),
                ('I understand A through D; D or A.', 'D'), ('no valid label', None), ('b', None)]:
            self.assertEqual(extract_answer(text, 'medqa'), expected)
        predictions = [{'benchmark_id': 'medqa', 'id': str(i), 'text': text}
                       for i, text in enumerate(('B', 'The answer is B.', 'invalid'))]
        metrics, scored = official_score_predictions(predictions, {('medqa', str(i)): 'B' for i in range(3)})
        self.assertEqual(metrics, {'medqa': {'accuracy': 2 / 3, 'n': 3}})
        self.assertIsNone(scored[-1]['extracted_answer'])
        with self.assertRaises(ValueError):
            official_score_predictions(predictions[:-1], {('medqa', str(i)): 'B' for i in range(3)})

    def test_official_prompt_contains_choices_once_and_no_gold(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'test.jsonl'
            question = 'Full question\nAnswer Choices: (A) alpha (B) beta (C) gamma (D) delta'
            path.write_text(json.dumps({'id': 'one', 'question': question, 'options': dict(zip('ABCD', 'abcd')),
                                        'gold': 'B'}) + '\n')
            groups = [{'benchmark_id': 'medqa', 'path': str(path),
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]
            payload, gold, _ = official_inputs(groups)
            self.assertNotIn('gold', json.dumps(payload))
            self.assertEqual(payload['items'][0]['messages'], [
                {'role': 'system', 'content': 'Please answer only.'},
                {'role': 'user', 'content': f'Q: {question}\nA: Among A through D, the answer is'}])
            self.assertEqual(gold[('medqa', 'one')], 'B')

    def test_real_closed_worker_cannot_read_private_files_and_receipts_are_aggregate_only(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            private = root / 'operator-secret.txt'
            private.write_text('private-canary-do-not-release')
            base = root / 'registered-base'
            base.mkdir()
            (base / 'config.json').write_text('{}')
            (base / 'model.safetensors').write_bytes(b'synthetic-not-model-weights')
            source = root / 'candidate'
            source.mkdir()
            (source / 'submission.json').write_text(json.dumps({'schema': 'candidate-inference-bundle-v1',
                'entrypoint': 'infer.py', 'base_model_path': str(base)}))
            (source / 'infer.py').write_text('import json,sys\nfrom pathlib import Path\n'
                f'try:\n secret=Path({str(private)!r}).read_text()\n'
                'except (PermissionError,FileNotFoundError): pass\nelse: raise AssertionError("private read allowed")\n'
                'p=json.load(sys.stdin)\nassert "gold" not in json.dumps(p)\n'
                'print(json.dumps({"predictions":[{"benchmark_id":r["benchmark_id"],"id":r["id"],'
                '"text":"The answer is B."} for r in p["items"]]}))\n')
            inbox = root / 'host-inbox'
            inbox.mkdir()
            frozen = frozen_bundle(source, inbox / 'bundle', [str(base)])
            dataset = root / 'closed.jsonl'
            dataset.write_text(json.dumps({'id': 'one', 'question': 'Synthetic question\nAnswer Choices: '
                '(A) alpha (B) beta (C) gamma (D) delta', 'options': dict(zip('ABCD', 'abcd')), 'gold': 'B'})+'\n')
            groups = [{'benchmark_id': 'medqa', 'path': str(dataset),
                       'sha256': hashlib.sha256(dataset.read_bytes()).hexdigest()}]
            protocol = EvaluationProtocol(score_protocol=SCORE_PROTOCOL)
            runtime = runtime_identity(sys.executable, inference_policy(sys.executable))
            config = {'root': str(root / 'private-service'), 'campaign_id': 'fixture-campaign',
                'protocol': asdict(protocol), 'protocol_sha256': 'f' * 64,
                'execution_budget': {'timeout_seconds': 10, 'resources': {'device_count': 0}},
                'source_sha256': source_identity(), 'runtime_identity': runtime,
                'deadline_epoch': time.time() + 60, 'submission_root': str(inbox),
                'groups': groups, 'dataset_sha256': {'medqa': groups[0]['sha256']},
                'python': sys.executable, 'base_model_paths': [str(base)], 'service_id': 'closed-fixture'}
            request = {k: config[k] for k in ('campaign_id', 'protocol', 'protocol_sha256', 'execution_budget')}
            request.update(candidate_id='fixture', submission_id='one', submission=frozen,
                submission_sha256=frozen['sha256'], runtime_sha256=runtime['sha256'], bundle_path=str(inbox / 'bundle'))
            receipt = gateway.serve(config, request)
            diagnostics = '\n'.join(p.read_text() for p in (root / 'private-service').rglob('private-error.txt'))
            self.assertEqual(receipt['status'], 'succeeded', diagnostics)
            self.assertEqual(receipt['metrics'], {'medqa': {'accuracy': 1.0, 'n': 1}})
            self.assertEqual(gateway.serve(config, request), receipt)
            for forbidden in ('private-canary', str(private), 'question', 'predictions', 'gold'):
                self.assertNotIn(forbidden, json.dumps(receipt))
            with self.assertRaises(EvaluationAdmissionError):
                gateway.serve(config, {**request, 'submission_id': 'two'})
            config['deadline_epoch'] = time.time() - 1
            with self.assertRaises(RuntimeError):
                gateway.serve(config, {**request, 'submission_id': 'three'})

    def test_worker_timeout_stops_exact_owned_process(self):
        with self.assertRaises(TimeoutError):
            run_candidate([sys.executable, '-c', 'import time;time.sleep(60)'],
                          {'items': []}, timeout_seconds=.1)


if __name__ == '__main__':
    unittest.main()
