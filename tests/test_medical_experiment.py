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
from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef
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
        self.decision_model = decision_model
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


class FixtureResearcher:
    """Independent remote-style backend; owns no actor or training resources."""
    def __init__(self, model, process, script):
        self.model, self.process, self.script = model, process, iter(script)
        self.requests, self.contexts = [], []
        self.response_model = model

    def generate(self, request):
        if self.process.active_job:
            raise AssertionError('Researcher inference while training is active')
        if request.role != 'decision_maker' or request.model != self.model:
            raise AssertionError('Researcher received an actor request or the wrong identity')
        self.requests.append(request)
        context = json.loads(request.prompt)
        self.contexts.append(context)
        value = next(self.script)
        if callable(value):
            value = value(context)
        return GenerationResponse(value if isinstance(value, str) else json.dumps(value),
            self.response_model, simulated=False, provider_model='researcher-service-alias')


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

    def execute(self, script, *, research_started_callback=None, decision_model=None,
                decision_backend=None, native_model_kwargs=None, **overrides):
        def serving_factory(path, **kwargs):
            self.serving = FixtureServing(path, process=self.process, script=script, **kwargs)
            return self.serving
        settings = ExperimentSettings(self.task, self.root/'run', self.root/'fixture-model', **overrides)
        return run(settings, serving_factory=serving_factory, converter_factory=lambda *a, **kw: Converter(),
            process_backend=self.process, research_started_callback=research_started_callback,
            decision_model=decision_model, decision_backend=decision_backend,
            native_model_kwargs=native_model_kwargs,
            identity_factory=lambda p: ('fixture-content-pin',
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
        self.assertNotIn('evaluate', context['directive_modes'])
        self.assertNotIn('apply_candidate', context['directive_modes'])
        self.assertIn('ALREADY been measured', context['objective'])
        self.assertEqual(self.read('monitor.json')['project'], {
            'project_id': 'run', 'name': '医疗 RSI · run', 'system_id': 'medical-fixture-model'})
        self.assertIsNone(self.read('run.json')['parent_run_id'])
        self.assertEqual(self.read('monitor.json')['status'], 'completed')
        self.assertTrue(self.read('decision_memory.json')['history'])
        for name in ('experience.json', 'candidates.json', 'evaluations.json', 'exploration_memory.json',
                     'data-manifest.json', 'model_identity.json', 'baseline.json', 'result.json'):
            self.assertTrue((self.root/'run'/name).exists(), name)
        self.assertGreaterEqual(result['baseline_seconds'], 0)
        with self.assertRaises(FileExistsError):
            self.execute([FINISH])

    def test_parent_is_only_a_declaration_and_is_kept_in_all_run_outputs(self):
        result = self.execute([FINISH], parent_run_id='run5')
        self.assertEqual(result['status'], 'completed', result)
        for name in ('run.json', 'monitor.json', 'result.json'):
            data = self.read(name)
            self.assertEqual((data['run_id'], data['parent_run_id']), ('run', 'run5'))
        self.assertEqual(result['trainings_reserved'], 0)
        self.assertEqual(self.serving.activated, [])
        args = parser().parse_args(['--task', str(self.task), '--run-dir', 'run6',
                                   '--model-path', 'model', '--parent-run-id', 'run5'])
        self.assertEqual(args.parent_run_id, 'run5')

    def test_independent_researcher_preserves_identity_through_training_and_application(self):
        researcher = ModelRef('researcher-vllm', 'Qwen3.8-27B', 'researcher-revision')
        remote = FixtureResearcher(researcher, self.process, [train, evaluate, apply, FINISH])
        result = self.execute([], decision_model=researcher, decision_backend=remote)
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(result['model_calls'], 4)
        self.assertEqual(self.serving.model_calls, 0)
        # Local actor serving need not accept a different researcher base model.
        self.assertEqual(self.serving.decision_model, self.serving.base)
        self.assertNotEqual(self.serving.activated[0].model, self.serving.base)
        request, _ = next(iter(self.process.requests.values()))
        self.assertEqual(request.base, self.serving.base)
        self.assertTrue(all(r.model == researcher and r.max_output_tokens == 1024 for r in remote.requests))
        self.assertTrue(all(c['decision_model'] == asdict(researcher) for c in remote.contexts))
        self.assertEqual(self.read('decision_model.json'), asdict(researcher))
        self.assertEqual(self.read('model_calls.json')['requested_model'], asdict(researcher))
        responses = [json.loads(line) for line in (self.root/'run/model_responses.jsonl').read_text().splitlines()]
        self.assertEqual(len(responses), 4)
        for response in responses:
            self.assertEqual(response['requested_model'], asdict(researcher))
            self.assertEqual(response['response']['model'], asdict(researcher))
            self.assertEqual(response['response']['provider_model'], 'researcher-service-alias')
            self.assertFalse(response['response']['revision_verified'])
        events = [json.loads(line) for line in (self.root/'run/events.jsonl').read_text().splitlines()]
        requested = [event for event in events if event['kind'] == 'model_decision_requested']
        self.assertEqual(len(requested), 4)
        self.assertTrue(all(e['details']['requested_model'] == asdict(researcher) for e in requested))

    def test_composite_serving_receives_the_independent_researcher_identity(self):
        researcher = ModelRef('composite-researcher', 'Qwen3.8-27B', 'remote-revision')
        result = self.execute([FINISH], decision_model=researcher)
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(self.serving.decision_model, researcher)
        self.assertEqual(self.serving.contexts[0]['decision_model'], asdict(researcher))
        self.assertEqual(self.read('model_calls.json')['requested_model'], asdict(researcher))

    def test_injected_researcher_retains_budget_and_thinking_configuration(self):
        researcher = ModelRef('researcher-vllm', 'Qwen3.8-27B', 'remote-revision')
        remote = FixtureResearcher(researcher, self.process, ['invalid JSON', FINISH])
        result = self.execute([], decision_model=researcher, decision_backend=remote,
                              max_model_calls=1, decision_thinking=True)
        self.assertEqual((result['status'], result['reason']), ('stopped', 'model_call_budget_exhausted'))
        self.assertEqual(len(remote.requests), 1)
        self.assertEqual(remote.requests[0].max_output_tokens, 8192)
        self.assertEqual(result['model_calls'], 1)

    def test_injected_researcher_identity_failure_is_logged_without_retry(self):
        researcher = ModelRef('researcher-vllm', 'Qwen3.8-27B', 'remote-revision')
        remote = FixtureResearcher(researcher, self.process, [FINISH])
        remote.response_model = replace(researcher, revision='unexpected-revision')
        result = self.execute([], decision_model=researcher, decision_backend=remote)
        self.assertEqual(result['status'], 'failed', result)
        self.assertEqual(len(remote.requests), 1)
        response = json.loads((self.root/'run/model_responses.jsonl').read_text())
        self.assertEqual(response['requested_model'], asdict(researcher))
        self.assertEqual(response['response']['model'], asdict(remote.response_model))

    def test_real_build_sft_prepare_evaluate_apply_and_monitor_progress(self):
        result = self.execute([train, evaluate, apply, FINISH])
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(len(self.process.requests), 1)
        self.assertEqual(result['trainings_reserved'], 1)
        self.assertTrue(result['cleanup']['confirmed'])
        self.assertEqual(len(self.serving.activated), 1)
        self.assertIn('evaluate', self.serving.contexts[1]['directive_modes'])
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
        view = self.serving.contexts[-1]
        self.assertIn('research_archive', view['context_summary']['policy_view'])
        self.assertTrue(all('directive' not in row['details'] and 'objective' not in row['details']
                            for row in view['history']))
        self.assertTrue(view['candidate_evaluations'])
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

    def test_training_parallelism_reaches_native_recipe(self):
        result = self.execute([train, FINISH], training_processes=8)
        self.assertEqual(result['status'], 'completed', result)
        request, _ = next(iter(self.process.requests.values()))
        recipe = json.loads(Path(request.recipe_ref).read_text())
        self.assertEqual(recipe['num_processes'], 8)
        self.assertEqual(recipe['device_type'], 'cuda')
        self.assertEqual(recipe['model_kwargs']['attn_implementation'], 'sdpa')
        self.assertEqual(self.read('parallelism.json')['training_processes'], 8)

    def test_npu_device_and_native_model_kwargs_reach_every_recipe(self):
        define = {'kind': 'invoke', 'operation': 'define_recipe', 'arguments': {'steps': 2, 'lr': 2e-4}}
        kwargs = {'attn_implementation': 'eager', 'device_map': 'npu:0'}
        result = self.execute([define, train, FINISH], training_device_type='npu', native_model_kwargs=kwargs)
        self.assertEqual(result['status'], 'completed', result)
        request, _ = next(iter(self.process.requests.values()))
        recipes = [entry['template'] for entry in self.read('training/recipes.json').values()]
        self.assertEqual(len(recipes), 2)
        recipes.append(json.loads(Path(request.recipe_ref).read_text()))
        for recipe in recipes:
            self.assertEqual(recipe['device_type'], 'npu')
            self.assertEqual(recipe['model_kwargs'], {'local_files_only': True, 'dtype': 'bfloat16', **kwargs})
        self.assertEqual(self.read('parallelism.json')['training_device_type'], 'npu')
        self.assertEqual(kwargs, {'attn_implementation': 'eager', 'device_map': 'npu:0'})
        args = parser().parse_args(['--task', str(self.task), '--run-dir', 'npu-run',
                                   '--model-path', 'model', '--training-device-type', 'npu'])
        self.assertEqual(args.training_device_type, 'npu')

    def test_research_clock_is_armed_after_baseline_before_first_decision(self):
        observed = []
        def started(seconds):
            observed.append((seconds, (self.root/'run/baseline.json').exists(), self.serving.model_calls))
        result = self.execute([FINISH], seconds=1800, research_started_callback=started)
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(observed, [(1800, True, 0)])

    def test_tensor_parallelism_cannot_be_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, 'requires the sglang'):
            ExperimentSettings(self.task, self.root/'run', self.root/'model', tensor_parallel_size=8)

    def test_sole_candidate_and_fixed_protocol_defaults_preserve_full_chain(self):
        evaluate_only = {'kind': 'explore', 'mode': 'evaluate', 'strategy': 'Evaluate sole candidate', 'arguments': {}}
        apply_only = {'kind': 'explore', 'mode': 'apply_candidate', 'strategy': 'Apply accepted sole candidate', 'arguments': {}}
        result = self.execute([train, evaluate_only, apply_only, FINISH])
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(len(self.serving.activated), 1)
        self.assertEqual(len(self.read('evaluations.json')), 2)

    def test_unique_registered_action_normalizes_envelope_without_changing_parameters(self):
        recipe = {'kind': 'explore', 'mode': 'define_recipe', 'strategy': 'Researcher recipe',
                  'arguments': {'steps': 32, 'lr': 1e-4, 'seed': 42, 'weight_decay': .01, 'warmup_ratio': .1}}
        def invoke_training(context):
            ref = next(r['data']['recipe_ref'] for r in context['operation_results']
                       if isinstance(r.get('data'), dict) and 'recipe_ref' in r['data'])
            value = train(context, ref)
            value['kind'], value['operation'] = 'invoke', value.pop('mode')
            return value
        result = self.execute([recipe, invoke_training, evaluate, apply, FINISH])
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(len(self.process.requests), 1)
        request, _ = next(iter(self.process.requests.values()))
        native_recipe = json.loads(Path(request.recipe_ref).read_text())
        self.assertEqual(native_recipe['training_args']['seed'], 42)
        self.assertEqual(native_recipe['training_args']['warmup_ratio'], .1)
        events = [json.loads(line) for line in (self.root/'run/events.jsonl').read_text().splitlines()]
        normalized = [event for event in events if event['kind'] == 'decision_envelope_normalized']
        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized[0]['details']['normalized']['arguments'], recipe['arguments'])
        self.assertIn('"mode": "define_recipe"', (self.root/'run/model_responses.jsonl').read_text().replace('\\"', '"'))

    def test_envelope_normalization_does_not_restore_a_disabled_training_action(self):
        denied = {'kind': 'invoke', 'operation': 'train', 'arguments': {}}
        result = self.execute([train, denied, FINISH], max_trainings=1)
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(len(self.process.requests), 1)
        self.assertNotIn('train', self.serving.contexts[1]['directive_modes'])

    def test_named_kind_and_flat_action_use_the_same_registered_contract(self):
        def named_train(context):
            value = train(context)
            value['kind'] = value['operation'] = value.pop('mode')
            return value
        measure = {'action': 'evaluate', 'arguments': {}}
        result = self.execute([named_train, measure, FINISH])
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(len(self.process.requests), 1)
        self.assertEqual(len(self.read('evaluations.json')), 2)

    def test_conflicting_names_do_not_choose_an_action(self):
        conflict = {'kind': 'train', 'operation': 'define_recipe', 'arguments': {}}
        result = self.execute([conflict, FINISH])
        self.assertEqual(result['status'], 'completed', result)
        self.assertFalse(self.process.requests)
        self.assertIn('Conflicting action names', self.serving.contexts[-1]['objective'])

    def test_duplicate_kind_is_rejected_without_repair_for_three_attempts(self):
        duplicate = '{"kind":"research_read","view":"items","kind":"agenda","limit":32}'
        result = self.execute([duplicate, duplicate, duplicate, FINISH])
        self.assertEqual(result['status'], 'failed', result)
        self.assertIn('three bounded schema/validation attempts', result['error'])
        self.assertEqual(result['model_calls'], 3)
        self.assertEqual(result['trainings_reserved'], 0)
        self.assertFalse(self.process.requests)
        events = [json.loads(line) for line in (self.root/'run/events.jsonl').read_text().splitlines()]
        rejected = [event['details'] for event in events if event['kind'] == 'model_decision_rejected']
        self.assertEqual([event['attempt'] for event in rejected], [1, 2, 3])
        self.assertTrue(all("Duplicate JSON key 'kind'" in event['error'] for event in rejected))
        self.assertFalse(any(event['kind'] == 'decision_envelope_normalized' for event in events))

    def test_duplicate_keys_in_nested_arguments_and_array_objects_are_rejected(self):
        nested = ('{"kind":"invoke","operation":"research_read",'
                  '"arguments":{"view":"items","kind":"agenda","kind":"agenda"}}')
        array = ('{"kind":"invoke","operation":"research_patch",'
                 '"arguments":{"upserts":[{"id":"note","text":"first","text":"second"}]}}')
        result = self.execute([nested, array, FINISH])
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(result['model_calls'], 3)
        self.assertIn("Duplicate JSON key 'kind'", self.serving.contexts[1]['objective'])
        self.assertIn("Duplicate JSON key 'text'", self.serving.contexts[2]['objective'])
        self.assertFalse(self.process.requests)
        self.assertEqual(self.serving.contexts[-1]['research_agenda']['total'], 0)

    def test_corrective_hint_explains_generic_envelopes_and_accepts_nested_research_read(self):
        duplicate = '{"kind":"research_read","view":"items","kind":"agenda","limit":32}'
        read = {'kind': 'invoke', 'operation': 'research_read',
                'arguments': {'view': 'items', 'kind': 'agenda', 'limit': 32}}
        result = self.execute([duplicate, read, FINISH])
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(result['model_calls'], 3)
        correction = json.loads(self.serving.contexts[1]['objective'].split(
            '\nCorrect this invalid proposal before execution (error data): ', 1)[1])
        self.assertEqual(correction['previous_output'], duplicate)
        self.assertIn("Duplicate JSON key 'kind'", correction['validation_error'])
        guidance = correction['envelope_guidance']
        for fragment in ('"kind":"invoke"', '"operation":"<name from context.operations>"',
                         '"kind":"explore"', '"mode":"<mode from context.directive_modes>"',
                         'inside arguments', 'top-level kind', 'train requires nonempty top-level input_ids',
                         'resume_mode="restart"', 'omit checkpoint or set it to null',
                         'Choose your own available action and research steps'):
            self.assertIn(fragment, guidance)
        self.assertIn(guidance, self.serving.contexts[0]['objective'])
        receipts = self.serving.contexts[-1]['operation_results']
        self.assertTrue(any(receipt['summary'] == 'Model-owned research notes'
                            and receipt['status'] == 'completed' and receipt['data']['view'] == 'items'
                            for receipt in receipts))
        self.assertFalse(self.process.requests)

    def test_operation_normalization_rejects_nonempty_directive_inputs(self):
        denied = {'kind': 'explore', 'mode': 'define_recipe', 'strategy': 'Invalid input scope',
                  'arguments': {}, 'input_ids': ['unrelated']}
        result = self.execute([denied, FINISH])
        self.assertEqual(result['status'], 'completed', result)
        self.assertFalse(self.process.requests)
        self.assertIn('cannot consume directive inputs', self.serving.contexts[-1]['objective'])

    def test_ambiguous_candidate_requires_model_choice_and_preserves_error_context(self):
        prompt = {'kind': 'explore', 'mode': 'configure_prompt', 'strategy': 'Second candidate',
                  'arguments': {'instructions': 'Return one legal label.'}}
        incomplete = {'kind': 'explore', 'mode': 'evaluate', 'strategy': 'Ambiguous request', 'arguments': {}}
        result = self.execute([train, prompt, incomplete, FINISH])
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(len(self.read('evaluations.json')), 1)  # Only the baseline; no guessed candidate.
        correction = self.serving.contexts[-1]['objective']
        self.assertIn('previous_output', correction)
        self.assertIn('candidate_id is required', correction)
        for candidate in self.serving.contexts[-1]['candidates']:
            self.assertIn(candidate['candidate_id'], correction)

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
        self.assertEqual(args.training_device_type, 'cuda')
        for overrides in ({'max_trainings': 4}, {'max_train_steps': 65}, {'seconds': float('nan')},
                          {'max_trainings': True}, {'training_device_type': 'unknown'}):
            with self.assertRaises(ValueError):
                ExperimentSettings(self.task, self.root/'run', self.root/'model', **overrides)


if __name__ == '__main__':
    unittest.main()
