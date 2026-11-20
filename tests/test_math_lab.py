"""Regression coverage for the shared math scene and its host split boundary."""
from contextlib import redirect_stdout
from hashlib import sha256
from importlib.util import find_spec
from io import BytesIO, StringIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import patch

from rsisys.cli import main

from rsisys.api import DEV, SEALED, GenerationResponse, ModelRef, ModelResponseError, OperationCall, TokenUsage
from rsisys.loops.contracts.execution import LabToolError
from rsisys.loops.contracts.model_service import ModelTransportError
from rsisys.labs.scenes.math.lab import MathLab
from rsisys.runtime.orchestration.profiles import ProfileBinding
from rsisys.runtime.storage.artifact_access import ArtifactAuthority
from rsisys.runtime.storage.artifacts import InMemoryArtifactStore


class MathLabTest(TestCase):
    def setUp(self):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.root = Path(root.name)
        self.actor = ModelRef('test', 'actor', 'v1')
        self.store = InMemoryArtifactStore()
        self.authority = ArtifactAuthority(self.store)
        self.context = SimpleNamespace(roles={'actor': self.actor},
            models=SimpleNamespace(generate=self.generate),
            artifacts=self.authority.component('math'))
        self.responses = []
        self.requests = []
        self.splits = {}
        self.dev = [{'id': f'dev-{i}', 'prompt': f'Development problem {i}', 'answer': 762}
                    for i in range(4)]
        self.splits['development'] = self.write_rows('development.jsonl', self.dev)

    def generate(self, request):
        self.requests.append(request)
        response = self.responses.pop(0) if self.responses else r'\boxed{1}'
        if isinstance(response, Exception):
            raise response
        return GenerationResponse(response, self.actor, True, TokenUsage(12, 4))

    def write_rows(self, filename, rows):
        data = ''.join(json.dumps(row) + '\n' for row in rows).encode()
        (self.root / filename).write_bytes(data)
        return {'file': filename, 'sha256': sha256(data).hexdigest(), 'count': len(rows)}

    def add_official(self):
        datasets = {}
        for name, groups in [('aime2024', [('I', 15), ('II', 15)]),
                             ('aime2025', [('I', 15), ('II', 15)]),
                             ('aimo3_reference', [('reference', 10)])]:
            rows = [{'id': f'{name}:{group}:{i}', 'benchmark': name, 'group': group, 'number': i,
                     'prompt': f'Fixture {name} {group} {i}', 'answer': 1}
                    for group, count in groups for i in range(1, count + 1)]
            datasets[name] = self.write_rows(f'{name}.jsonl', rows)
        data = json.dumps({'schema_version': 1, 'benchmarks': datasets}).encode()
        (self.root / 'benchmarks.json').write_bytes(data)
        self.splits['official'] = {'benchmarks_manifest': 'benchmarks.json',
                                  'sha256': sha256(data).hexdigest()}

    def make_lab(self):
        manifest = self.root / 'scene.json'
        manifest.write_text(json.dumps({'schema_version': 1, 'splits': self.splits}))
        return MathLab(self.context, {'manifest': str(manifest),
            'split_roles': {split: 'sealed' if split == 'official' else 'dev' for split in self.splits},
            'max_output_tokens': 32768})

    def test_development_receipt_and_trace(self):
        self.responses = [r'\boxed{762}', r'\boxed{7}', '762', ModelResponseError(
            'Output limit', finish_reason='length', usage=TokenUsage(12, 32768))]
        lab = self.make_lab()
        call = OperationCall('evaluate', {'split': 'development', 'instruction': 'Check arithmetic.'})
        with self.authority.scope('math', split='development', role=DEV, finishing=False):
            result = lab.execute(call)
        self.assertEqual(result.data['quality_metrics']['accuracy'], 0.25)
        self.assertEqual(result.data['execution_metrics']['truncated'], 1)
        self.assertEqual(result.data['execution_metrics']['invalid_format'], 1)
        self.assertEqual(result.data['execution_metrics']['output_tokens'], 32780)
        self.assertTrue(result.simulated)
        self.assertEqual(result.call_id, call.call_id)
        self.assertTrue(all(r.role == 'actor' and r.max_output_tokens == 32768 for r in self.requests))
        self.assertTrue(all('762' not in r.prompt + r.system for r in self.requests))
        trace_args = {'split': 'development', 'evaluation': result.data['evidence_refs'][0], 'index': 1}
        trace = lab.execute(OperationCall('read_trace', trace_args)).data
        self.assertEqual(trace['sample']['id'], 'dev-1')
        self.assertEqual(trace['completion'], r'\boxed{7}')

    def test_combined_holdout_and_access_controls(self):
        self.add_official()
        lab = self.make_lab()
        self.assertEqual(set(lab.execute(OperationCall('inspect', {})).data['splits']), {'development'})
        profile = ProfileBinding('math', lab, lab.describe())
        bindings = {item['split']: item for item in profile.identity()['data_bindings']}
        self.assertEqual(bindings['development']['row_ids'], [row['id'] for row in self.dev])
        self.assertEqual(set(bindings['official']), {'name', 'split', 'role', 'binding_sha256'})
        call = OperationCall('evaluate', {'split': 'official', 'instruction': 'Solve.'})
        with self.assertRaisesRegex(ValueError, 'only to the host'):
            profile.check(call)
        profile.check(call, finishing=True)
        with self.authority.scope('math', split='official', role=SEALED, finishing=True):
            with self.assertRaises(LabToolError):
                lab.execute(OperationCall('evaluate', {**call.arguments, 'task_ids': ['aime2024:I:1']}))
            result = profile.validate_result(call, lab.execute(call))
            aggregate = profile.aggregate(call, result, self.context.artifacts)
        self.assertEqual(len(self.requests), 70)
        self.assertEqual(aggregate.data['count'], 70)
        scores = aggregate.data['quality_metrics']
        self.assertEqual([scores[f'{name}_count'] for name in
                         ['aime2024', 'aime2025', 'aimo3_reference']], [30, 30, 10])
        self.assertEqual(scores['accuracy'], 1.0)
        self.assertNotIn('records', aggregate.data)
        with self.assertRaises(PermissionError):
            self.context.artifacts.read(result.artifacts[0])
        with self.assertRaises(LabToolError):
            lab.execute(OperationCall('read_trace', {'split': 'official',
                'evaluation': result.data['evidence_refs'][0], 'index': 0}))

    def test_manifest_mismatch_and_cross_split_overlap(self):
        self.add_official()
        self.dev[0]['prompt'] = ' Fixture  aime2024 I 1 '
        self.splits['development'] = self.write_rows('development.jsonl', self.dev)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.make_lab()
        self.dev[0]['prompt'] = 'Another development problem'
        self.splits['development'] = self.write_rows('development.jsonl', self.dev)
        bound = self.make_lab().state()
        self.dev[0]['answer'] = 761
        self.splits['development'] = self.write_rows('development.jsonl', self.dev)
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.make_lab().restore(bound)
        (self.root / 'development.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'hash'):
            self.make_lab()

    def test_transport_failure_is_not_scored(self):
        self.responses = [ModelTransportError('Endpoint unavailable')]
        lab = self.make_lab()
        call = OperationCall('evaluate', {'split': 'development', 'instruction': 'Solve.'})
        with self.assertRaises(ModelTransportError):
            lab.execute(call)

    def test_custom_sealed_set_uses_frozen_rows_and_host_aggregation(self):
        rows = [{'id': 'blind-1', 'prompt': 'Custom sealed problem', 'answer': 1,
                 'benchmark': 'custom'}]
        self.splits['official'] = self.write_rows('sealed.jsonl', rows)
        lab = self.make_lab()
        profile = ProfileBinding('math', lab, lab.describe())
        binding = next(item for item in lab.data_bindings() if item.split == 'official')
        self.assertEqual(binding.role, 'sealed')
        self.assertEqual(binding.row_ids, ('blind-1',))
        call = OperationCall('evaluate', {'split': 'official', 'instruction': 'Solve.'})
        with self.assertRaisesRegex(ValueError, 'only to the host'):
            profile.check(call)
        profile.check(call, finishing=True)
        with self.authority.scope('math', split='official', role=SEALED, finishing=True):
            with self.assertRaisesRegex(LabToolError, 'entire split'):
                lab.execute(OperationCall('evaluate', {**call.arguments, 'task_ids': ['blind-1']}))
            result = profile.validate_result(call, lab.execute(call))
            aggregate = profile.aggregate(call, result, self.context.artifacts)
        self.assertEqual(aggregate.data['count'], 1)
        self.assertEqual(aggregate.data['quality_metrics']['custom_accuracy'], 1.0)
        self.assertNotIn('blind-1', json.dumps(aggregate.data))
        with self.assertRaises(PermissionError):
            self.context.artifacts.read(result.artifacts[0])
        rows[0]['prompt'] = self.dev[0]['prompt']
        self.splits['official'] = self.write_rows('sealed.jsonl', rows)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.make_lab()

    def run_cli(self, respond):
        (self.root / 'scene.json').write_text(json.dumps({'schema_version': 1, 'splits': self.splits}))
        plugin = self.root / 'fixture_method.py'
        plugin.write_text("""
from rsisys.api import Decision, MathLab, MethodDefinition, OperationCall

class FixtureController:
    def __init__(self, context, config):
        self.started = False

    def decide(self, window):
        if not self.started:
            self.started = True
            return Decision.invoke(OperationCall('math.evaluate', {
                'split': 'development', 'instruction': 'Solve.'}))
        if all(feedback.data.get('split') != 'development' for feedback in window.feedbacks):
            raise ValueError('Controller did not receive development feedback')
        return Decision('finish', reason='The fixed candidate was measured')

class FinalFixtureLab(MathLab):
    def final_evaluations(self):
        if 'official' in self.roles:
            return (OperationCall('evaluate', {'split': 'official', 'instruction': 'Solve.'}),)
        return ()

def register(registry):
    registry.register_method('fixture_math', MethodDefinition.from_factories(
        controller=FixtureController, labs={'math': FinalFixtureLab}, version='1'))
""")
        config = self.root / 'run.json'
        config.write_text(json.dumps({
            'objective': 'Measure the fixture actor on development and sealed tasks',
            'max_decision_steps': 8,
            'data_dir': 'run',
            'backends': {'test': {'kind': 'chat_completions',
                                  'endpoint': 'http://127.0.0.1:18098/v1', 'json_mode': False}},
            'roles': {'actor': {'backend_id': 'test', 'model_id': 'actor', 'revision': 'v1'}},
            'framework': {'method': {'name': 'fixture_math', 'config': {'labs': {'math': {
                'manifest': str(self.root / 'scene.json'),
                'split_roles': {split: 'sealed' if split == 'official' else 'dev'
                                for split in self.splits}}}}}},
        }))
        output = StringIO()
        run = self.root / 'run'
        args = ['rsisys', 'run', '--config', str(config), '--plugin', str(plugin), '--data-dir', str(run)]
        opener = SimpleNamespace(open=respond)
        with patch.object(sys, 'argv', args), patch('urllib.request.build_opener', return_value=opener):
            with redirect_stdout(output):
                main()
        return json.loads(output.getvalue()), run

    @skipUnless(find_spec('math_verify'), 'Math-Verify extra is not installed')
    def test_cli_scores_symbolic_answers_from_the_lab_worker(self):
        self.splits['development'] = {**self.write_rows('symbolic.jsonl', [
            {'id': f'symbolic-{i}', 'prompt': str(i), 'answer': r'\frac{1}{2}'} for i in range(3)
        ]), 'verifier': 'math_verify'}
        completions = [r'\boxed{\frac{2}{4}}', r'\boxed{\frac{1}{3}}', 'No boxed answer']

        def respond(request, **kwargs):
            i = int(json.loads(request.data)['messages'][-1]['content'])
            return BytesIO(json.dumps({'model': 'actor', 'model_revision': 'v1', 'choices': [{
                'message': {'content': completions[i]}, 'finish_reason': 'stop',
            }]}).encode())

        result, run = self.run_cli(respond)
        self.assertIn(result['status'], ('finished', 'completed'))
        monitor = json.loads((run / 'monitor.json').read_text())
        self.assertEqual(monitor['evaluations'][0]['metrics']['accuracy'], 1 / 3)
        samples = {row['id']: (row['parsed'], row['correct']) for row in monitor['samples']}
        self.assertEqual(samples, {'symbolic-0': (True, True), 'symbolic-1': (True, False),
                                   'symbolic-2': (False, False)})

    def test_cli_run_with_host_finalization_and_provider_truncation(self):
        self.add_official()
        requests = []

        def respond(request, **kwargs):
            payload = json.loads(request.data)
            requests.append(payload)
            prompt = payload['messages'][-1]['content']
            finish = 'length' if prompt == 'Fixture aime2024 I 1' else 'stop'
            answer = 762 if prompt.startswith('Development') else 1
            return BytesIO(json.dumps({'model': 'actor', 'model_revision': 'v1',
                'choices': [{'message': {'content': rf'\boxed{{{answer}}}'}, 'finish_reason': finish}],
                'usage': {'prompt_tokens': 12, 'completion_tokens': 32768 if finish == 'length' else 4},
            }).encode())

        result, run = self.run_cli(respond)
        self.assertIn(result['status'], ('finished', 'completed'))
        self.assertEqual(len(requests), 74)
        self.assertTrue(all(r['max_tokens'] == 32768 for r in requests))
        monitor = json.loads((run / 'monitor.json').read_text())
        self.assertEqual(len(monitor['evaluations']), 2)
        official = next(row for row in monitor['evaluations'] if row['split'] == 'official')
        self.assertAlmostEqual(official['metrics']['accuracy'], 69 / 70)
        self.assertEqual(official['metrics']['aime2024_correct'], 29)
        self.assertEqual(official['metrics']['aime2025_count'], 30)
        self.assertEqual(official['metrics']['aimo3_reference_count'], 10)
        receipt = next(event['receipt'] for event in result['events'] if event.get('phase') == 'host_finish')
        execution = receipt['data']['execution_metrics']
        self.assertEqual(execution['truncated'], 1)
        self.assertEqual(execution['output_tokens'], 32768 + 69 * 4)
        self.assertEqual(monitor['resources']['latest_execution_metrics']['num_samples'], 4)
        self.assertNotIn('Fixture aime2024', json.dumps(monitor))
        self.assertEqual({sample['id'] for sample in monitor['samples']}, {f'dev-{i}' for i in range(4)})
