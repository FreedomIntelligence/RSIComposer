"""CPU integration fixtures only: no real optimization, GPU, or medical claims."""
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import re
import tempfile
import unittest

from examples.medical.experiment import ExperimentSettings, parser, run
from examples.medical.task import encoded
from rsisys.adapters.training import ProcessTrainingBackend
from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.jobs import JobOutcome, JobProgress, JobRef, JobStatus
from rsisys.loops.contracts.model_service import GenerationResponse
from rsisys.loops.contracts.training import TrainingResult


def row(ident, gold='A'):
    return {'id': ident, 'benchmark_id': 'medqa', 'question': 'Synthetic symbol '+ident,
            'options': dict(A='square', B='circle', C='star', D='triangle'), 'gold': gold}


class Converter:
    conversion_id = 'cpu-fixture-completion-only'
    metadata = {'fixture': True, 'completion_only': True, 'max_length': 2048}

    def filter_samples(self, rows):
        return tuple(rows), ()

    def __call__(self, rows):
        return [{'input_ids': [1, 2], 'labels': [-100, 2]} for _ in rows]


class FixtureProcess(ProcessTrainingBackend):
    """Native bridge fixture preserving request, job and result contracts."""
    def __init__(self, root):
        super().__init__(root)
        self.requests, self.jobs, self.cancelled = {}, {}, []
        self.active_job = None
        self.polls = 0
        self.interrupt = False
        self.lose_submit_response = False
        self.hold_until_cancel = False

    def submit(self, request, *, idempotency_key):
        job = JobRef(sha256(idempotency_key.encode()).hexdigest(), self.backend)
        self.requests[job] = (request, idempotency_key)
        directory = self.root / job.job_id
        directory.mkdir(exist_ok=True)
        (directory/'receipt.json').write_text('{}')
        adapter = directory/'adapter'
        adapter.mkdir(exist_ok=True)
        model = replace(request.base, revision='peft-'+job.job_id)
        candidate = Candidate(job.job_id, request.base.revision,
            {'model': ArtifactRef(job.job_id, 'peft_adapter', adapter.as_uri(), version=model.revision)},
            ('actor',), request.origin.parent_system)
        self.jobs[job] = TrainingResult(request.base, model, candidate, {'loss': .5}, False)
        self.active_job = job
        if self.lose_submit_response:
            raise RuntimeError('fixture lost submit response after external launch')
        return job

    def poll(self, job):
        self.polls += 1
        if self.interrupt:
            self.interrupt = False
            self.hold_until_cancel = True
            raise KeyboardInterrupt('fixture SIGTERM equivalent')
        if job in self.cancelled:
            return JobOutcome(job, JobStatus.CANCELLED)
        if self.hold_until_cancel:
            return JobOutcome(job, JobStatus.RUNNING)
        if self.active_job == job and self.polls == 1:
            return JobOutcome(job, JobStatus.RUNNING, progress=JobProgress(step=1, total_steps=32, metrics={'loss': .7}))
        self.active_job = None
        return JobOutcome(job, JobStatus.SUCCEEDED, self.jobs[job], progress=JobProgress(step=32, total_steps=32, metrics={'loss': .5}))

    def cancel(self, job):
        if job not in self.jobs:
            raise AssertionError('Controller tried to cancel a foreign job')
        self.cancelled.append(job)
        if self.active_job == job:
            self.active_job = None


class FixtureServing:
    def __init__(self, path, *, actor_model, decision_model, artifacts, process, script, **kwargs):
        self.base, self.process, self.script = actor_model, process, iter(script)
        self.contexts, self.prepared, self.activated, self.events = [], [], [], []
        self.weight_metadata = {actor_model: {'fixture': True}}
        self.release_count = 0
        self.model_calls = 0

    def release_memory(self):
        self.release_count += 1
        self.events.append('release')

    def prepare_inference(self, system):
        if self.process.active_job:
            raise AssertionError('Inference preparation while training owns GPU')
        self.prepared.append(system)
        self.events.append('prepare:'+system.version)
        return system

    def activate(self, system, *, idempotency_key):
        self.activated.append(system)
        return self.prepare_inference(system)

    def generate(self, request):
        if self.process.active_job:
            raise AssertionError('Inference while training owns GPU')
        if request.role == 'decision_maker':
            self.model_calls += 1
            context = json.loads(request.prompt)
            self.contexts.append(context)
            value = next(self.script)
            if callable(value):
                value = value(context)
            text = value if isinstance(value, str) else json.dumps(value)
        else:
            if request.model != self.base and not any(s.model == request.model for s in self.prepared):
                raise AssertionError('Candidate generation before explicit preparation')
            self.events.append('generate:'+request.model.revision)
            text = 'A' if request.model == self.base else 'B'
        return GenerationResponse(text, request.model, simulated=False)

    __call__ = generate


def train(context, recipe=None):
    objective = context['objective']
    return {'kind': 'explore', 'mode': 'train', 'strategy': 'Synthetic composition check',
        'input_ids': [re.search(r'Frozen train artifact_id=([^\. ]+)', objective)[1]],
        'arguments': {'recipe_ref': recipe or re.search(r'Default recipe_ref=([^;]+)', objective)[1],
                      'method': 'sft', 'apply_result': False}}


def evaluate(context):
    return {'kind': 'explore', 'mode': 'evaluate', 'strategy': 'Measure synthetic candidate', 'input_ids': [],
        'arguments': {'candidate_id': context['candidates'][-1]['candidate_id'],
                      'protocol_ref': re.search(r'Evaluation protocol_ref=([^\. ]+)', context['objective'])[1]}}


def apply(context):
    return {'kind': 'explore', 'mode': 'apply_candidate', 'strategy': 'Apply after fixed acceptance', 'input_ids': [],
            'arguments': {'candidate_id': context['candidates'][-1]['candidate_id']}}


FINISH = {'kind': 'finish', 'reason': 'Synthetic researcher chose to stop; no scientific conclusion'}


class MedicalExperimentIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root/'train.jsonl').write_bytes(encoded(row('train'))+b'\n')
        (self.root/'dev.jsonl').write_bytes(encoded(row('dev', 'B'))+b'\n')
        self.task = self.root/'task.json'
        self.task.write_text(json.dumps({'train': 'train.jsonl',
            'development': [{'benchmark_id': 'medqa', 'role': 'primary', 'path': 'dev.jsonl'}]}))
        self.process = FixtureProcess(self.root/'native-process')
        self.serving = None

    def execute(self, script, **overrides):
        def serving_factory(path, **kwargs):
            self.serving = FixtureServing(path, process=self.process, script=script, **kwargs)
            return self.serving
        settings = ExperimentSettings(self.task, self.root/'run', self.root/'fixture-model', **overrides)
        return run(settings, serving_factory=serving_factory, converter_factory=lambda *a, **kw: Converter(),
            process_backend=self.process, identity_factory=lambda p: ('fixture-content-pin',
                {'model_id': 'fixture-symbol-model', 'actual_weight_sha256': 'fixture-only'}))

    def read(self, path):
        return json.loads((self.root/'run'/path).read_text())

    def test_early_finish_constructs_real_build_monitor_and_persistence(self):
        result = self.execute([FINISH])
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(result['researcher_finish_reason'], FINISH['reason'])
        self.assertEqual(result['trainings_reserved'], 0)
        context = self.serving.contexts[0]
        self.assertTrue(context['run_control']['researcher_finish_ends_run'])
        names = [spec['name'] for spec in context['operations']]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue({'define_recipe', 'preview_training_data', 'choose_training_data', 'read_artifact'} <= set(names))
        self.assertIn('train', context['directive_modes'])
        self.assertEqual(self.read('monitor.json')['project']['system_id'], 'medical-qwen35-9b')
        self.assertEqual(self.read('monitor.json')['status'], 'completed')
        self.assertTrue(self.read('decision_memory.json')['history'])
        for name in ('experience.json', 'candidates.json', 'evaluations.json', 'exploration_memory.json',
                     'data-manifest.json', 'model_identity.json', 'baseline.json', 'result.json'):
            self.assertTrue((self.root/'run'/name).exists(), name)
        self.assertGreaterEqual(result['baseline_seconds'], 0)
        with self.assertRaises(FileExistsError):
            self.execute([FINISH])

    def test_real_build_sft_prepare_evaluate_apply_and_monitor_progress(self):
        result = self.execute([train, evaluate, apply, FINISH])
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(len(self.process.requests), 1)
        self.assertEqual(result['trainings_reserved'], 1)
        self.assertTrue(result['cleanup']['confirmed'])
        self.assertEqual(len(self.serving.activated), 1)
        self.assertNotEqual(self.serving.activated[0].model, self.serving.base)
        self.assertTrue(any(system.model != self.serving.base for system in self.serving.prepared))
        evaluations = self.read('evaluations.json')
        self.assertEqual(evaluations['initial']['metrics']['score'], 0)
        self.assertEqual(evaluations[self.serving.activated[0].version]['metrics']['score'], 1)
        monitor = self.read('monitor.json')
        self.assertTrue(any(job['kind'] == 'train' and job['step'] == 32 for job in monitor['jobs']))
        self.assertTrue(any(job['kind'] == 'evaluate' and job['step'] == 1 for job in monitor['jobs']))
        self.assertEqual(self.read('systems.json')['current']['version'], self.serving.activated[0].version)
        self.assertTrue(self.read('experience.json'))
        native_request, native_key = next(iter(self.process.requests.values()))
        receipt = next(iter(self.read('training/budget.json').values()))
        self.assertTrue(native_key.startswith(receipt['recipe_ref']+':directive:'))
        self.assertEqual(native_request.base, self.serving.base)
        self.assertGreaterEqual(self.serving.release_count, 2)

    def test_duplicate_training_is_failed_feedback_without_second_process(self):
        result = self.execute([train, train, FINISH])
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(len(self.process.requests), 1)
        feedback = self.serving.contexts[-1]['last_feedback']
        self.assertEqual(feedback['status'], 'failed')
        self.assertIn('Exact duplicate', feedback['error'])

    def test_train_budget_only_removes_train_prompt_and_evaluation_survive(self):
        prompt = {'kind': 'explore', 'mode': 'configure_prompt', 'strategy': 'Test optional prompt research',
                  'arguments': {'instructions': 'Choose the requested synthetic symbol'}, 'input_ids': []}
        result = self.execute([train, prompt, FINISH], max_trainings=1)
        self.assertEqual(result['status'], 'completed', result)
        self.assertNotIn('train', self.serving.contexts[1]['directive_modes'])
        self.assertIn('configure_prompt', self.serving.contexts[1]['directive_modes'])
        self.assertIn('evaluate', self.serving.contexts[1]['directive_modes'])

    def test_operation_business_error_is_feedback_and_schema_retry_is_bounded(self):
        bad = {'kind': 'invoke', 'operation': 'choose_training_data', 'arguments': {'ids': ['unknown']}}
        result = self.execute(['malformed json', bad, FINISH])
        self.assertEqual(result['status'], 'completed', result)
        results = self.serving.contexts[-1]['operation_results']
        self.assertTrue(any(result['status'] == 'failed' for result in results))
        self.assertEqual(self.serving.model_calls, 3)

    def test_interrupt_cancels_only_reserved_process_job(self):
        self.process.interrupt = True
        result = self.execute([train])
        self.assertEqual(result['status'], 'stopped', result)
        self.assertEqual(result['reason'], 'interrupted')
        self.assertEqual(len(self.process.cancelled), 1)
        self.assertTrue(result['cleanup']['confirmed'])
        self.assertIsNone(self.process.active_job)

    def test_lost_external_submit_response_is_not_retried_and_cleanup_reconciles(self):
        self.process.lose_submit_response = True
        result = self.execute([train])
        self.assertEqual(result['status'], 'failed', result)
        self.assertEqual(len(self.process.requests), 1)
        self.assertEqual(len(self.process.cancelled), 1)
        self.assertTrue(result['cleanup']['confirmed'])
        self.assertTrue((self.root/'run'/'exception.txt').exists())

    def test_private_exclusions_block_prepare_data_without_disclosing_rows(self):
        private = self.root/'operator-private.jsonl'
        withheld = row('operator-private')
        private.write_bytes(encoded(withheld)+b'\n')
        proposal = {'kind': 'invoke', 'operation': 'prepare_data',
                    'arguments': {'text': encoded(withheld).decode()}}
        result = self.execute([proposal, FINISH], private_exclusions=(private,))
        self.assertEqual(result['status'], 'completed', result)
        self.assertTrue(any(result['status'] == 'failed' for result in self.serving.contexts[-1]['operation_results']))
        # No private corpus was installed in the task/objective or public monitor.
        self.assertNotIn(withheld['question'], self.serving.contexts[0]['objective'])
        self.assertNotIn(withheld['question'], (self.root/'run'/'monitor.json').read_text())

    def test_cli_defaults_and_validation(self):
        args = parser().parse_args(['--task', str(self.task), '--run-dir', str(self.root/'run'),
                                   '--model-path', 'model', '--private-exclusions', 'a', '--private-exclusions', 'b'])
        self.assertEqual((args.seconds, args.max_trainings, args.max_model_calls, args.max_seq_length,
                          args.max_train_steps), (5400, 3, 96, 2048, 64))
        self.assertEqual(args.private_exclusions, [Path('a'), Path('b')])
        for overrides in ({'max_trainings': 4}, {'max_train_steps': 65}, {'seconds': float('nan')},
                          {'max_trainings': True}):
            with self.assertRaises(ValueError):
                ExperimentSettings(self.task, self.root/'run', self.root/'model', **overrides)


if __name__ == '__main__':
    unittest.main()
