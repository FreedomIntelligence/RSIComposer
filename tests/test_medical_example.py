"""INTEGRATION VERIFICATION ONLY: synthetic symbols, fixture HTTP, no medical claims.

Scripted HTTP decisions and training receipts verify composition and routing;
normal example entry still constructs ModelDecisionPolicy.
"""
from contextlib import contextmanager
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import sys
import time
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from examples.medical.task import (Benchmark, DataRef, MedicalTask, Sample, encoded,
                                   prepare, samples, validate_training)
from examples.medical.evaluation import MedicalEvaluator, backend_inference
from examples.medical.run import RunSettings, build, main
from rsisys.loops.contracts.candidate import Candidate, SystemVersion
from rsisys.loops.contracts.backend import BackendCapabilities
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.requests import TrainSpec
from rsisys.loops.contracts.jobs import JobOutcome, JobRef, JobStatus
from rsisys.loops.contracts.model_service import BackendBinding, ModelRef, GenerationResponse
from rsisys.loops.contracts.training import TrainingResult, TrainingOrigin, ResumeMode, ModelActivationPending
from rsisys.loops.exploration.harness.evaluation import EvaluationRequest
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.runtime.artifacts import FileArtifactStore
from rsisys.runtime.model import RemoteModelBackend
from rsisys.runtime.systems import ReferenceSystemActivator, SystemRegistry, SystemActivationRejected


MODEL = ModelRef('fixture-http', 'symbol-model', 'initial-weights')
DECISION = ModelRef('fixture-http', 'decision-model', 'fixed-researcher')


def row(ident, question, gold='A', benchmark='medqa'):
    labels = 'ABCDEFGHIJ' if benchmark == 'medxpertqa' else 'ABCD'
    return {'id': ident, 'question': question, 'benchmark_id': benchmark,
            'options': {} if benchmark == 'pubmedqa' else {label: 'symbol ' + label for label in labels},
            'gold': gold}


def write_rows(path, rows):
    path.write_bytes(b'\n'.join(encoded(r) for r in rows) + b'\n')
    return DataRef.freeze(path)


@contextmanager
def fixture_http(mode='prompt'):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = encoded({'data': [{'id': MODEL.model_id}, {'id': 'trained-symbol-model'}]})
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append(request)
            if request['model'] == DECISION.model_id:
                context = json.loads(request['messages'][-1]['content'])
                candidates = context['candidates']
                evaluations = context['candidate_evaluations']
                last = context['last_feedback']
                if not candidates:
                    if mode == 'train':
                        payload = {'kind': 'explore', 'strategy': 'integration verification only', 'mode': 'train',
                            'arguments': {'recipe_ref': 'fixture-native', 'method': 'sft'}}
                    else:
                        payload = {'kind': 'explore', 'strategy': 'integration verification only',
                            'mode': 'configure_prompt', 'arguments': {'instructions': 'choose A'}}
                elif not evaluations:
                    payload = {'kind': 'explore', 'strategy': 'measure fixture outputs', 'mode': 'evaluate',
                        'arguments': {'candidate_id': candidates[0]['candidate_id'],
                                      'protocol_ref': context['objective'].split('Evaluation protocol: ')[1].split('\n')[0]}}
                elif last and (last['summary'].endswith('applied') or last['status'] == 'failed'):
                    payload = {'kind': 'finish', 'reason': 'fixture integration verified'}
                else:
                    payload = {'kind': 'explore', 'strategy': 'apply fixture candidate', 'mode': 'apply_candidate',
                               'arguments': {'candidate_id': candidates[0]['candidate_id']}}
                content = json.dumps(payload)
            else:
                system = request['messages'][0]['content']
                content = 'A' if mode != 'reject' and ('choose A' in system or request['model'] == 'trained-symbol-model') else 'B'
            body = encoded({'model': request['model'], 'choices': [
                {'finish_reason': 'stop', 'message': {'content': content}}]})
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield RemoteModelBackend(BackendBinding(MODEL.backend_id, 'http',
            f'http://127.0.0.1:{server.server_port}/v1', json_mode=False)), calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class FixtureLearner:
    """Scripted receipt only; this fixture performs no optimization."""
    def __init__(self, artifacts):
        self.artifacts, self.requests = artifacts, []

    def submit(self, request, *, idempotency_key):
        self.requests.append(request)
        self.consumed = tuple(samples(self.artifacts.get(r)) for r in request.data_refs)
        self.job = JobRef('integration-verification', 'fixture-learner')
        ref = self.artifacts.put(b'fixture weights', kind='fixture_model')
        self.output = TrainingResult(request.base, replace(MODEL, model_id='trained-symbol-model', revision='trained'),
            Candidate('trained', request.base.revision, {'model': ref}, ('actor',), request.origin.parent_system),
            simulated=True)
        return self.job

    def poll(self, job):
        return JobOutcome(job, JobStatus.SUCCEEDED, self.output, simulated=True)

    def result(self, job):
        return self.output

    def cancel(self, job):
        pass


class MedicalExampleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.train = write_rows(self.root/'train.jsonl', [row('train', 'Select the square symbol')])
        self.dev = write_rows(self.root/'dev.jsonl', [row('dev', 'Select the circle symbol')])
        self.task = MedicalTask(self.train, (Benchmark('medqa', 'primary', self.dev),))
        self.artifacts = FileArtifactStore(self.root/'artifacts')

    def assembly(self, backend, **kwargs):
        return build(self.task, artifacts=self.artifacts, decision_backend=backend, decision_model=DECISION,
                     initial_model=MODEL, inference=kwargs.pop('inference', backend_inference(backend)),
                     system_activator=kwargs.pop('system_activator', ReferenceSystemActivator(MODEL)), **kwargs)

    def test_readonly_prepare_check_and_frozen_bytes(self):
        config = {'train': 'train.jsonl', 'development': [
            {'benchmark_id': 'medqa', 'role': 'primary', 'path': 'dev.jsonl'}]}
        path = self.root/'task.json'
        path.write_text(json.dumps(config))
        before = set(self.root.rglob('*'))
        with patch('sys.stdout', new_callable=io.StringIO):
            main(['prepare', '--config', str(path)])
            main(['check', '--config', str(path)])
        self.assertEqual(before, set(self.root.rglob('*')))
        task = prepare(config, base_directory=self.root)
        self.assertEqual(task.protocol_ref, self.task.protocol_ref)
        with self.assertRaises(Exception):
            task.guardrails = False
        (self.root/'dev.jsonl').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'digest'):
            task.check()
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            prepare({**config, 'final': 'private.jsonl'}, base_directory=self.root)

    def test_legal_labels_trim_only_and_actual_outputs(self):
        for benchmark, gold, wrong in [('medqa', 'A', 'Answer: A'), ('medxpertqa', 'J', 'j'),
                                      ('medmcqa', 'D', '4'), ('pubmedqa', 'yes', 'Yes')]:
            with self.subTest(benchmark=benchmark):
                train = write_rows(self.root/'train.jsonl', [row('t', 'training symbol', gold, benchmark)])
                dev = write_rows(self.root/'dev.jsonl', [row('d', 'development symbol', gold, benchmark)])
                task = MedicalTask(train, (Benchmark(benchmark, 'primary', dev),))
                prompt = self.artifacts.put(b'candidate prefix', kind='medical_prompt')
                systems = SystemRegistry(SystemVersion('initial', {'prompt': prompt}, MODEL), ReferenceSystemActivator(MODEL))
                calls = []
                def infer(request):
                    calls.append(request)
                    return GenerationResponse(' \n' + gold + '\t' if len(calls) == 1 else wrong, request.model, True)
                evaluator = MedicalEvaluator(task, self.artifacts, systems, {}, infer)
                anchor = evaluator.establish_anchor()
                self.assertEqual(anchor.metrics['score'], 1)
                candidate = Candidate('c', MODEL.revision, {}, ('actor',), 'initial')
                systems.register(candidate)
                evaluator.candidates['c'] = candidate
                request = EvaluationRequest(candidate, task.protocol_ref)
                job = evaluator.submit(request, idempotency_key='eval')
                self.assertEqual(job, evaluator.submit(request, idempotency_key='eval'))
                result = evaluator.poll(job).result
                self.assertEqual(result.metrics['score'], 0)
                self.assertEqual(result.metrics[benchmark+'.invalid_format'], 1)
                self.assertNotIn(gold + '\t', calls[0].prompt)
                raw = json.loads(self.artifacts.get(evaluator.evidence[result.evidence_refs[0]]))
                self.assertEqual(raw['predictions'][0]['output'], wrong)
                self.assertTrue(calls[0].system.startswith('candidate prefix'))
                self.assertEqual(evaluator.poll(job).result, result)
                self.assertEqual(len(calls), 2)
                with self.assertRaisesRegex(ValueError, 'frozen'):
                    evaluator.submit(replace(request, protocol_ref='other'), idempotency_key='bad')

    def test_http_model_policy_prompt_eval_apply_and_research_tools(self):
        with fixture_http() as (backend, calls):
            assembly = self.assembly(backend)
            self.assertIsInstance(assembly.runner.decision_making.harness.policy, ModelDecisionPolicy)
            self.assertFalse(calls)
            specs = assembly.runner.decision_making.harness.policy.directives.describe()
            self.assertTrue(all(not spec.accepts_input_refs for spec in specs))
            tools = assembly.runner.decision_making.harness.tools
            harness = assembly.runner.decision_making.harness
            self.assertIs(tools.research_tools.memory, harness.memory)
            self.assertIs(tools.research_tools.agenda, harness.research_agenda)
            self.assertNotIn('search_literature', [s.name for s in assembly.runner.decision_making.action_space.describe()])
            result = assembly.run(RunSettings('Verify the synthetic assembly', 30, 0))
            self.assertEqual(result.status, 'done')
            self.assertTrue(assembly.systems.current.version.startswith('prompt-'))
            self.assertEqual(assembly.systems.current.model, MODEL)
            self.assertTrue(any('choose A' in c['messages'][0]['content'] for c in calls if c['model'] == MODEL.model_id))
            self.assertEqual(assembly.evaluator._anchor.candidate_id, 'initial')
            kinds = {r.kind for r in self.artifacts.refs()}
            self.assertTrue({'medical_raw_evidence', 'medical_interpretation', 'medical_frozen_task'} <= kinds)

    def test_trained_model_origin_inputs_and_routing(self):
        class Activator:
            def activate(self, system, *, idempotency_key):
                return system
        with fixture_http('train') as (backend, calls):
            learner = FixtureLearner(self.artifacts)
            assembly = self.assembly(backend, learner=learner, bind_training_inputs=lambda r, a: r,
                                     system_activator=Activator())
            result = assembly.run(RunSettings('Verify scripted training receipt', 30, 0))
            self.assertEqual(result.status, 'done')
            request = learner.requests[0]
            self.assertEqual(request.origin, TrainingOrigin('initial', MODEL))
            self.assertEqual(request.base, MODEL)
            self.assertEqual(learner.consumed[0], self.train.read())
            self.assertEqual(assembly.systems.current.model.model_id, 'trained-symbol-model')
            self.assertTrue(any(c['model'] == 'trained-symbol-model' for c in calls))
            self.assertEqual(assembly.candidates['trained'].parent_system, request.origin.parent_system)

    def test_overlap_trusted_exclusions_and_unfinished_status(self):
        private = Sample('operator-private', 'medqa', 'Never disclose sentinel', (), 'A')
        validate_training(self.train.read(), (private,))
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            validate_training(samples(encoded(row('x', '  SELECT the CIRCLE symbol  '))), self.dev.read())
        with fixture_http() as (backend, calls):
            assembly = self.assembly(backend, trusted_exclusions=(private,))
            result = assembly.run(RunSettings('Verify pending status', 1, 0))
            self.assertEqual(result.status, 'needs_input')
            self.assertNotIn('Never disclose sentinel', json.dumps(calls))
            self.assertNotIn('operator-private', json.dumps(calls))

    def test_training_schema_parser_and_executor_share_host_initialization(self):
        initializer = replace(MODEL, revision='operator-initializer')
        learner = FixtureLearner(self.artifacts)
        learner.recipe_ref = 'fixed-native-recipe'
        learner.capabilities = lambda: BackendCapabilities(methods=('sft',), resume_modes=tuple(ResumeMode))
        backend = SimpleNamespace(generate=lambda request: self.fail('Contract checks must not infer'))
        assembly = self.assembly(backend, learner=learner, bind_training_inputs=lambda r, a: r,
                                 training_initializer=initializer)
        policy = assembly.runner.decision_making.harness.policy
        catalog = policy.directives
        schema = next(s.input_schema for s in catalog.describe() if s.mode == 'train')
        self.assertNotIn('origin', schema['properties'])
        self.assertFalse(schema['additionalProperties'])
        self.assertEqual(schema['properties']['apply_result']['enum'], [False])
        self.assertEqual(schema['properties']['method']['enum'], ['sft'])
        self.assertEqual(schema['properties']['recipe_ref']['enum'], [learner.recipe_ref])
        generated = policy.decision_schema(SimpleNamespace(operations=(), directive_modes=('train',)))
        self.assertEqual(generated['oneOf'][0]['properties']['arguments'], schema)
        arguments = {'recipe_ref': learner.recipe_ref, 'method': 'sft'}
        spec = catalog.parse('train', arguments)
        self.assertEqual(spec.origin, TrainingOrigin('initial', initializer))
        for fields in ({'origin': None}, {'origin': asdict(spec.origin)},
                       {'origin': {'checkpoint': 'training-data'}}, {'apply_result': True},
                       {'apply_result': 0}, {'apply_result': None}, {'method': 'rl'},
                       {'recipe_ref': 'unconfigured-recipe'}, {'checkpoint': None}):
            with self.subTest(fields=fields), self.assertRaises((ValueError, TypeError)):
                catalog.parse('train', {**arguments, **fields})

        # Parent tracks the incumbent, while the operator's initializer stays fixed.
        candidate = Candidate('prompt-parent', MODEL.revision, {}, ('actor',), 'initial')
        assembly.systems.register(candidate)
        assembly.systems.activate(candidate.candidate_id, idempotency_key='parent')
        spec = catalog.parse('train', arguments)
        self.assertEqual(spec.origin, TrainingOrigin('prompt-parent', initializer))

        # Programmatic directives cannot bypass the model parser's restriction.
        train_ref = assembly.systems.get('initial').components['data']
        for index, origin in enumerate((TrainingOrigin('initial', initializer),
                       TrainingOrigin('prompt-parent', MODEL),
                       TrainingOrigin('prompt-parent', initializer, ResumeMode.WEIGHTS_ONLY))):
            directive = Directive('invalid-origin-'+str(index), 'fixture', 'verify contract', mode='train',
                input_refs=(train_ref,), payload=TrainSpec(**arguments, origin=origin))
            with self.subTest(origin=origin), self.assertRaisesRegex(ValueError, 'host-owned'):
                assembly.runner.exploration.accept(directive)
            assembly.runner.exploration.acknowledge(directive.directive_id)
        self.assertFalse(learner.requests)

    def test_medical_training_requires_restart_capability(self):
        learner = FixtureLearner(self.artifacts)
        learner.capabilities = lambda: BackendCapabilities(methods=('sft',), resume_modes=(ResumeMode.FULL_STATE,))
        with self.assertRaisesRegex(ValueError, 'restart support'):
            self.assembly(SimpleNamespace(generate=lambda request: self.fail('Must not infer')),
                          learner=learner, bind_training_inputs=lambda r, a: r)
        self.assertFalse(learner.requests)

    def test_medical_evaluation_and_application_contract(self):
        assembly = self.assembly(SimpleNamespace(generate=lambda request: self.fail('Must not infer')))
        space = assembly.runner.decision_making.action_space
        catalog = space.directives
        candidate = Candidate('sole', MODEL.revision, {}, ('actor',), 'initial')
        second = replace(candidate, candidate_id='second')
        for mode in ('evaluate', 'apply_candidate'):
            schema = next(s.input_schema for s in catalog.describe() if s.mode == mode)
            self.assertEqual(schema['required'], [])
            self.assertFalse(schema['additionalProperties'])
            if mode == 'evaluate':
                self.assertEqual(schema['properties']['protocol_ref']['enum'], [self.task.protocol_ref])
                self.assertEqual(schema['properties']['protocol_ref']['default'], self.task.protocol_ref)
            payload = catalog.parse(mode, {}, candidates=(candidate,))
            if mode == 'evaluate':
                self.assertEqual(payload.candidate, candidate)
                self.assertEqual(payload.protocol_ref, self.task.protocol_ref)
            else:
                self.assertEqual(payload.candidate_id, candidate.candidate_id)
            for available, arguments in (((), {}), ((candidate, second), {}),
                    ((candidate,), {'candidate_id': 'unknown'}), ((candidate,), {'candidate_id': None}),
                    ((candidate,), {'extra': 1}), ((candidate,), {'protocol_ref': 'changed'})):
                with self.subTest(mode=mode, arguments=arguments, available=available), self.assertRaises((ValueError, TypeError)):
                    catalog.parse(mode, arguments, candidates=available)
            catalog.parse(mode, {'candidate_id': second.candidate_id}, candidates=(candidate, second))
            directive = Directive('no-data-'+mode, 'fixture', 'verify input contract', mode=mode,
                payload=payload, input_refs=(assembly.systems.get('initial').components['data'],))
            with self.assertRaisesRegex(ValueError, 'does not accept input'):
                space.validate_directive(directive)

    def test_evaluation_failure_is_terminal_with_partial_raw_outputs(self):
        dev = write_rows(self.root/'multi.jsonl', [row('d1', 'First symbol'), row('d2', 'Second symbol')])
        task = replace(self.task, development=(Benchmark('medqa', 'primary', dev),))
        prompt = self.artifacts.put(b'', kind='medical_prompt')
        systems = SystemRegistry(SystemVersion('initial', {'prompt': prompt}, MODEL), ReferenceSystemActivator(MODEL))
        calls = []
        def infer(request):
            calls.append(request)
            if len(calls) == 4:
                raise ValueError('Malformed fixture response')
            return GenerationResponse('A', request.model, True)
        evaluator = MedicalEvaluator(task, self.artifacts, systems, {}, infer)
        evaluator.establish_anchor()
        candidate = Candidate('fails', MODEL.revision, {}, ('actor',), 'initial')
        systems.register(candidate)
        evaluator.candidates['fails'] = candidate
        job = evaluator.submit(EvaluationRequest(candidate, task.protocol_ref), idempotency_key='failing-eval')
        self.assertEqual(evaluator.poll(job).status, JobStatus.RUNNING)
        outcome = evaluator.poll(job)
        self.assertEqual(outcome.status, JobStatus.FAILED)
        self.assertEqual(outcome, evaluator.poll(job))
        self.assertEqual(len(calls), 4)
        evidence_id = outcome.error.split('partial evidence: ')[1]
        raw = json.loads(self.artifacts.get(evaluator.evidence[evidence_id]))
        self.assertEqual(raw['error_type'], 'ValueError')
        self.assertEqual(len(raw['completed_predictions']), 1)
        self.assertFalse(raw['complete'])

    def test_native_binding_cannot_replace_validated_training_inputs(self):
        with fixture_http('train') as (backend, calls):
            learner = FixtureLearner(self.artifacts)
            def wrong_binding(request, artifacts):
                return replace(request, data_refs=())
            assembly = self.assembly(backend, learner=learner, bind_training_inputs=wrong_binding)
            with self.assertRaisesRegex(ValueError, 'provenance'):
                assembly.run(RunSettings('Verify native binding rejection', 10, 0))
            self.assertEqual(learner.requests, [])

    def test_explicit_cli_run_and_async_poll_interval(self):
        with fixture_http() as (backend, calls):
            config = self.root/'task.json'
            config.write_text(json.dumps({'train': self.train.path, 'development': [
                {'benchmark_id': 'medqa', 'role': 'primary', 'path': self.dev.path}]}))
            backends = self.root/'backends.json'
            backends.write_text(json.dumps({'decision_binding': asdict(backend.binding),
                'decision_model': asdict(DECISION), 'actor_binding': asdict(backend.binding),
                'initial_model': asdict(MODEL)}))
            settings = self.root/'settings.json'
            settings.write_text(json.dumps({'objective': 'Fixture HTTP integration only',
                'max_runner_steps': 30, 'poll_interval_seconds': 0.25}))
            with patch('examples.medical.run.time.sleep') as sleep, patch('sys.stdout', new_callable=io.StringIO) as output:
                code = main(['run', '--config', str(config), '--backends', str(backends),
                    '--settings', str(settings), '--output-directory', str(self.root/'cli-artifacts')])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.getvalue())['status'], 'done')
            sleep.assert_called_with(0.25)
            self.assertTrue(any(ref.kind == 'medical_run_result'
                for ref in FileArtifactStore(self.root/'cli-artifacts').refs()))

    def test_activation_pending_then_ready_retains_runner_receipt_and_key(self):
        class Activator:
            def __init__(self):
                self.calls = []
            def activate(self, system, *, idempotency_key):
                self.calls.append((system, idempotency_key))
                if len(self.calls) == 1:
                    raise ModelActivationPending('fixture readiness pending')
                return system
        with fixture_http() as (backend, calls):
            activator = Activator()
            assembly = self.assembly(backend, system_activator=activator)
            first = assembly.run(RunSettings('Verify pending deployment', 4, 0))
            self.assertEqual(first.status, 'needs_input')
            self.assertEqual(first.pending.phase, 'applying')
            ident = first.pending.directive_id
            receipt = assembly.runner.exploration.harness.commands.receipts[ident]
            self.assertIsNone(receipt.feedback)
            self.assertTrue(receipt.data['activation_started'])
            self.assertEqual(assembly.systems.current.version, 'initial')
            result = assembly.run(RunSettings('Verify pending deployment', 10, 0))
            self.assertEqual(result.status, 'done')
            self.assertEqual(len(activator.calls), 2)
            self.assertEqual(activator.calls[0], activator.calls[1])
            self.assertEqual(receipt.feedback.status.value, 'completed')
            self.assertIs(receipt, assembly.runner.exploration.harness.commands.receipts[ident])

    def test_cancel_ambiguous_activation_reconciles_instead_of_discarding_receipt(self):
        class Activator:
            def __init__(self):
                self.calls = []
            def activate(self, system, *, idempotency_key):
                self.calls.append((system, idempotency_key))
                if len(self.calls) < 3:
                    raise ModelActivationPending('fixture remote application unconfirmed')
                return system
        with fixture_http() as (backend, calls):
            activator = Activator()
            assembly = self.assembly(backend, system_activator=activator)
            first = assembly.run(RunSettings('Verify cancellation reconciliation', 4, 0))
            ident = first.pending.directive_id
            assembly.runner.cancel(ident)
            second = assembly.run(RunSettings('Verify cancellation reconciliation', 1, 0))
            self.assertEqual(second.status, 'needs_input')
            self.assertEqual(second.pending.phase, 'applying')
            receipt = assembly.runner.exploration.harness.commands.receipts[ident]
            self.assertTrue(receipt.cancel_requested)
            self.assertIsNone(receipt.feedback)
            self.assertEqual(assembly.systems.current.version, 'initial')
            result = assembly.run(RunSettings('Verify cancellation reconciliation', 10, 0))
            self.assertEqual(result.status, 'done')
            self.assertTrue(all(call == activator.calls[0] for call in activator.calls))
            self.assertEqual(len(activator.calls), 3)
            self.assertEqual(receipt.feedback.status.value, 'completed')
            self.assertIn('Cancellation arrived after activation started', receipt.feedback.summary)
            self.assertEqual(assembly.systems.current.version, receipt.request.candidate_id)
            evidence = json.loads(self.artifacts.get(next(r for r in self.artifacts.refs()
                if r.artifact_id == receipt.feedback.evidence_refs[0])))
            self.assertTrue(evidence['cancel_requested'])
            self.assertTrue(evidence['activation_confirmed'])

    def test_interpretation_exception_is_terminal_preserves_measurement_and_blocks_apply(self):
        prompt = self.artifacts.put(b'', kind='medical_prompt')
        systems = SystemRegistry(SystemVersion('initial', {'prompt': prompt}, MODEL), ReferenceSystemActivator(MODEL))
        calls = []
        def infer(request):
            calls.append(request)
            return GenerationResponse('B' if len(calls) == 1 else 'A', request.model, True)
        evaluator = MedicalEvaluator(self.task, self.artifacts, systems, {}, infer)
        evaluator.establish_anchor()
        candidate = Candidate('interpret-fails', MODEL.revision, {}, ('actor',), 'initial')
        systems.register(candidate)
        evaluator.candidates[candidate.candidate_id] = candidate
        job = evaluator.submit(EvaluationRequest(candidate, self.task.protocol_ref), idempotency_key='interpret-fails')
        with patch.object(evaluator, 'interpret', side_effect=ValueError('fixture interpretation error')) as interpret:
            outcome = evaluator.poll(job)
            self.assertEqual(outcome, evaluator.poll(job))
            interpret.assert_called_once()
        self.assertEqual(outcome.status, JobStatus.FAILED)
        self.assertEqual(len(calls), 2)
        failure_id = outcome.error.split('failure evidence: ')[1]
        failure = json.loads(self.artifacts.get(evaluator.evidence[failure_id]))
        self.assertTrue(failure['measurement_complete'])
        self.assertFalse(failure['acceptance_complete'])
        self.assertEqual(failure['source_refs'], list(evaluator._outputs[candidate.candidate_id].evidence_refs))
        raw = json.loads(self.artifacts.get(evaluator.evidence[failure['source_refs'][0]]))
        self.assertEqual(raw['predictions'][0]['output'], 'A')
        with self.assertRaisesRegex(RuntimeError, 'interpretation failed'):
            evaluator.interpret(candidate.candidate_id)

    def test_training_registration_failure_overrides_command_success_but_retains_training_evidence(self):
        with fixture_http('train') as (backend, calls):
            learner = FixtureLearner(self.artifacts)
            assembly = self.assembly(backend, learner=learner, bind_training_inputs=lambda r, a: r)
            with patch.object(assembly.systems, 'register', side_effect=ValueError('fixture system registration rejected')):
                result = assembly.run(RunSettings('Verify registration failure', 2, 0))
            self.assertEqual(result.status, 'needs_input')
            self.assertEqual(assembly.runner.exploration.harness.state.last_feedback.status.value, 'failed')
            self.assertEqual(assembly.runner.decision_making.harness.state.last_feedback.status.value, 'failed')
            self.assertIn('registration rejected', assembly.runner.decision_making.harness.state.last_feedback.error)
            self.assertEqual(assembly.systems.current.version, 'initial')
            self.assertEqual(assembly.candidates, {})
            records = assembly.experience_store.records()
            training = next(r for r in records if r.kind == 'training_run')
            self.assertEqual(training.details['outcome']['status'], JobStatus.SUCCEEDED)
            improvement = next(r for r in records if r.kind == 'improvement')
            self.assertEqual(improvement.details['status'], 'failed')

    def test_rejected_application_is_feedback_then_researcher_finishes_without_activation(self):
        class Activator:
            def activate(self, system, *, idempotency_key):
                raise AssertionError('Rejected candidate must never activate')
        with fixture_http('reject') as (backend, calls):
            assembly = self.assembly(backend, system_activator=Activator())
            result = assembly.run(RunSettings('Verify rejection feedback', 20, 0))
            self.assertEqual(result.status, 'done')
            feedback = assembly.runner.decision_making.harness.state.last_feedback
            self.assertEqual(feedback.status.value, 'failed')
            self.assertIn('at least one accuracy improvement and no accuracy regression', feedback.error)
            self.assertEqual(assembly.systems.current.version, 'initial')
            rejection = json.loads(self.artifacts.get(assembly.evaluator.evidence[feedback.evidence_refs[0]]))
            self.assertEqual(rejection['kind'], 'application_rejection')
            self.assertFalse(rejection['accepted'])
            self.assertTrue(rejection['source_refs'])

    def test_unevaluated_application_is_rejected_as_feedback(self):
        from rsisys.loops.contracts.directive import Directive
        from rsisys.loops.contracts.requests import ApplyCandidateSpec
        class Activator:
            def activate(self, system, *, idempotency_key):
                raise AssertionError('Unevaluated candidate must never activate')
        with fixture_http() as (backend, calls):
            assembly = self.assembly(backend, system_activator=Activator())
            assembly.run(RunSettings('Create unevaluated candidate', 1, 0))
            candidate_id = next(iter(assembly.candidates))
            directive = Directive('apply-unevaluated', 'fixture only', 'verify safe rejection',
                mode='apply_candidate', payload=ApplyCandidateSpec(candidate_id))
            progress = assembly.runner.exploration.accept(directive)
            self.assertFalse(progress.terminal)
            progress = assembly.runner.exploration.advance(directive.directive_id)
            self.assertTrue(progress.terminal)
            self.assertEqual(progress.feedback.status.value, 'failed')
            self.assertIn('no successful completed', progress.feedback.error)
            self.assertEqual(assembly.systems.current.version, 'initial')

    def test_stale_parent_rejected_before_activation_and_researcher_can_finish(self):
        class Activator:
            def __init__(self):
                self.calls = []
            def activate(self, system, *, idempotency_key):
                self.calls.append(system.version)
                return system
        with fixture_http() as (backend, calls):
            activator = Activator()
            assembly = self.assembly(backend, system_activator=activator)
            assembly.run(RunSettings('Verify stale parent', 3, 0))
            candidate = next(iter(assembly.candidates.values()))
            sibling = replace(candidate, candidate_id='operator-sibling')
            assembly.systems.register(sibling)
            assembly.systems.activate(sibling.candidate_id, idempotency_key='operator-fixture')
            result = assembly.run(RunSettings('Verify stale parent', 10, 0))
            self.assertEqual(result.status, 'done')
            self.assertEqual(activator.calls, ['operator-sibling'])
            self.assertEqual(assembly.systems.current.version, 'operator-sibling')
            feedback = assembly.runner.decision_making.harness.state.last_feedback
            self.assertEqual(feedback.status.value, 'failed')
            self.assertIn('parent is stale', feedback.error)

    def test_confirmed_activation_rejection_is_feedback_unknown_failure_retains_intent(self):
        class Rejected:
            def activate(self, system, *, idempotency_key):
                raise SystemActivationRejected('fixture confirms no application occurred')
        with fixture_http() as (backend, calls):
            assembly = self.assembly(backend, system_activator=Rejected())
            result = assembly.run(RunSettings('Verify confirmed deployment rejection', 10, 0))
            self.assertEqual(result.status, 'done')
            self.assertEqual(assembly.systems.current.version, 'initial')
            self.assertIn('no application occurred', assembly.runner.decision_making.harness.state.last_feedback.error)
        class Unknown:
            def activate(self, system, *, idempotency_key):
                raise OSError('fixture uncertain external state')
        with fixture_http() as (backend, calls):
            assembly = self.assembly(backend, system_activator=Unknown())
            with self.assertRaisesRegex(OSError, 'uncertain external'):
                assembly.run(RunSettings('Verify unknown deployment outcome', 10, 0))
            progress = assembly.runner.exploration.pending()
            self.assertFalse(progress.terminal)
            receipt = assembly.runner.exploration.harness.commands.receipts[progress.directive_id]
            self.assertTrue(receipt.data['activation_started'])
            self.assertIsNone(receipt.feedback)

    def test_incremental_poll_serving_pending_and_cancellation_preserve_consumed_rows(self):
        dev = write_rows(self.root/'incremental.jsonl', [row('d1', 'First symbol'), row('d2', 'Second symbol')])
        task = replace(self.task, development=(Benchmark('medqa', 'primary', dev),))
        prompt = self.artifacts.put(b'', kind='medical_prompt')
        systems = SystemRegistry(SystemVersion('initial', {'prompt': prompt}, MODEL), ReferenceSystemActivator(MODEL))
        calls, preparations = [], []
        def infer(request):
            calls.append(request.prompt)
            return GenerationResponse('A', request.model, True)
        def prepare_serving(system):
            preparations.append(system.version)
            if system.version == 'incremental' and preparations.count(system.version) < 3:
                raise ModelActivationPending('fixture candidate route not ready')
        evaluator = MedicalEvaluator(task, self.artifacts, systems, {}, infer, prepare_inference=prepare_serving)
        evaluator.establish_anchor()
        candidate = Candidate('incremental', MODEL.revision, {}, ('actor',), 'initial')
        systems.register(candidate)
        evaluator.candidates[candidate.candidate_id] = candidate
        job = evaluator.submit(EvaluationRequest(candidate, task.protocol_ref), idempotency_key='incremental')
        self.assertEqual(evaluator.poll(job).status, JobStatus.RUNNING)
        self.assertEqual(evaluator.poll(job).status, JobStatus.RUNNING)
        self.assertEqual(len(calls), 2)
        self.assertEqual(evaluator.poll(job).status, JobStatus.RUNNING)
        self.assertEqual(len(calls), 3)
        self.assertEqual(evaluator.poll(job).status, JobStatus.SUCCEEDED)
        self.assertEqual(calls[:2], calls[2:])
        self.assertEqual(preparations.count('incremental'), 3)
        cancelled = replace(candidate, candidate_id='cancel-mid-evaluation')
        systems.register(cancelled)
        evaluator.candidates[cancelled.candidate_id] = cancelled
        job = evaluator.submit(EvaluationRequest(cancelled, task.protocol_ref), idempotency_key='cancel-mid-evaluation')
        self.assertEqual(evaluator.poll(job).status, JobStatus.RUNNING)
        evaluator.cancel(job)
        consumed = len(calls)
        self.assertEqual(evaluator.poll(job).status, JobStatus.CANCELLED)
        self.assertEqual(len(calls), consumed)
        raw = [json.loads(self.artifacts.get(ref)) for ref in evaluator.evidence.values()]
        partial = next(r for r in raw if r['kind'] == 'raw_development_cancelled')
        self.assertEqual(len(partial['completed_predictions']), 1)
        self.assertFalse(partial['complete'])

    def test_evaluator_factory_and_external_component_reads_use_public_contract(self):
        from rsisys.loops.contracts.artifacts import ArtifactRef
        from rsisys.loops.decision_making.action_space import OperationCall
        invocations = []
        def factory(*args, **kwargs):
            invocations.append((args, kwargs))
            return MedicalEvaluator(*args, **kwargs)
        def prepare_serving(system):
            pass
        with fixture_http() as (backend, calls):
            assembly = self.assembly(backend, evaluator_factory=factory, prepare_inference=prepare_serving)
            self.assertEqual(len(invocations), 1)
            self.assertIs(invocations[0][1]['prepare_inference'], prepare_serving)
            external = ArtifactRef('external-checkpoint', 'model', 'file:///operator/weights/directory', 'weights')
            assembly.candidates['external'] = Candidate('external', MODEL.revision, {'model': external}, ('actor',), 'initial')
            tools = assembly.runner.decision_making.harness.tools
            with patch.object(self.artifacts, 'get', side_effect=AssertionError('External file must not be read')):
                result = tools.execute(OperationCall('read_artifact', {'artifact_id': external.artifact_id}))
            self.assertTrue(result.data['requires_external_resolver'])
            self.assertIsNone(result.data['content'])
            self.assertEqual(result.data['reference'], asdict(external))
            self.assertIsNone(assembly.evaluator.measurement('not-measured'))

    def test_medical_serving_native_prepare_hook_routes_trained_model_over_http(self):
        from examples.medical.native import MedicalServing
        from rsisys.adapters.serving import VLLMBackend, VLLMDeploymentBackend
        with fixture_http('train') as (backend, calls):
            candidate_backend = VLLMBackend(backend.binding)
            deployments = VLLMDeploymentBackend(candidate_backend, self.root/'deployments')
            serving = MedicalServing(deployments)
            learner = FixtureLearner(self.artifacts)
            assembly = self.assembly(backend, learner=learner, bind_training_inputs=lambda r, a: r,
                inference=serving, system_activator=serving, prepare_inference=serving.prepare_inference)
            result = assembly.run(RunSettings('Native serving integration verification only', 30, 0))
            self.assertEqual(result.status, 'done')
            trained = assembly.systems.current.model
            self.assertEqual(trained.model_id, 'trained-symbol-model')
            self.assertEqual(candidate_backend.models[trained], 'trained-symbol-model')
            self.assertTrue(any(call['model'] == trained.model_id and call['messages'][-1]['content'] == 'Reply OK.'
                                for call in calls))
            self.assertTrue(any(call['model'] == trained.model_id and 'Select the circle symbol' in call['messages'][-1]['content']
                                for call in calls))
            self.assertTrue(list(deployments.root.glob('medical-activation-*.json')))

    def test_native_medical_conversion_provenance_and_declared_recipe(self):
        from examples.medical.native import MedicalTrainingBackend
        from rsisys.adapters.training import ProcessTrainingBackend, TrainingCommand
        from rsisys.loops.exploration.harness.learner import ActorTrainingRequest
        # Scripted subprocess receipt only: consumes native bytes but does not optimize.
        script = self.root/'native_fixture.py'
        script.write_text("""import json, os, pathlib
request = json.loads(pathlib.Path(os.environ['RSISYS_REQUEST_FILE']).read_text())['request']
root = pathlib.Path(os.environ['RSISYS_JOB_DIR'])
recipe = json.loads(pathlib.Path(request['recipe_ref']).read_text())
rows = pathlib.Path(recipe['train_dataset']).read_bytes()
(root / 'consumed.jsonl').write_bytes(rows)
base = request['base']
result = dict(base=base, model=dict(base, revision='fixture-trained'), simulated=True,
    candidate=dict(candidate_id='native-fixture', base_version=base['revision'],
    changes={}, affected_roles=['actor'], parent_system=request['origin']['parent_system']))
pathlib.Path(os.environ['RSISYS_RESULT_FILE']).write_text(json.dumps(result))
""")
        process = ProcessTrainingBackend(self.root/'native-process', lambda request, directory:
            TrainingCommand((sys.executable, str(script)), str(directory), timeout_seconds=5))
        converted = []
        def tokenize(rows):
            converted.append(rows)
            return [{'input_ids': [1, 2], 'labels': [-100, 2]} for row in rows]
        learner = MedicalTrainingBackend(self.root/'native-jobs', self.artifacts,
            tokenize=tokenize, conversion_id='synthetic-tokenizer-fixture-v1',
            recipe_template={'initial_model': asdict(MODEL)}, process_backend=process)
        recipe_ref = learner.recipe_ref
        source = self.artifacts.put(Path(self.train.path).read_bytes(), kind='training_data')
        request = ActorTrainingRequest(MODEL, (source,), recipe_ref,
            method='sft', origin=TrainingOrigin('initial', MODEL))
        self.assertEqual(learner.bind_inputs(request, self.artifacts), request)
        job = learner.submit(request, idempotency_key='native-conversion-fixture')
        self.assertEqual(learner.submit(request, idempotency_key='native-conversion-fixture'), job)
        deadline = time.monotonic() + 8
        while True:
            outcome = learner.poll(job)
            if outcome.terminal or time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        self.assertEqual(outcome.status, JobStatus.SUCCEEDED, outcome.error)
        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0], self.train.read())
        evidence = json.loads(self.artifacts.get(outcome.result.candidate.changes['medical_training']))
        self.assertEqual(evidence['request']['data_refs'], [asdict(source)])
        self.assertEqual(evidence['sources'][0]['sha256'], source.digest)
        self.assertEqual(evidence['source_ids'], ['train'])
        native_request = evidence['native_request']
        recipe = json.loads(Path(native_request['recipe_ref']).read_text())
        native_data = Path(recipe['train_dataset']).read_bytes()
        consumed = Path(evidence['native_job_directory'])/'consumed.jsonl'
        self.assertEqual(consumed.read_bytes(), native_data)
        self.assertEqual(native_request['origin'], asdict(request.origin))
        self.assertEqual(learner.reattach(job).result, outcome.result)
        with fixture_http() as (backend, calls):
            assembly = self.assembly(backend, learner=learner, bind_training_inputs=learner.bind_inputs)
            spec = next(s for s in assembly.runner.decision_making.harness.policy.directives.describe() if s.mode == 'train')
            self.assertEqual(spec.input_schema['properties']['recipe_ref']['enum'], [learner.recipe_ref])
            with self.assertRaisesRegex(ValueError, 'recipe'):
                learner.bind_inputs(replace(request, recipe_ref='wrong-template'), self.artifacts)

    def test_baseline_pending_resumes_same_assembly_without_failure_or_duplicate_rows(self):
        dev = write_rows(self.root/'baseline-pending.jsonl', [row('d1', 'First symbol'), row('d2', 'Second symbol')])
        self.task = replace(self.task, development=(Benchmark('medqa', 'primary', dev),))
        preparations, second_row_waited = [], []
        def prepare_serving(system):
            preparations.append(system.version)
            if system.version == 'initial' and preparations.count('initial') == 1:
                raise ModelActivationPending('Baseline serving preparation is pending')
        with fixture_http() as (backend, calls):
            def inference(request):
                baseline = 'choose A' not in request.system
                if baseline and request.prompt.startswith('Second symbol') and not second_row_waited:
                    second_row_waited.append(True)
                    raise ModelActivationPending('Baseline route is temporarily pending after one scored row')
                return backend.generate(request)
            assembly = self.assembly(backend, inference=inference, prepare_inference=prepare_serving)
            settings = RunSettings('Verify baseline serving waits', 20, 0)
            first = assembly.run(settings)
            self.assertEqual(first.status, 'needs_input')
            self.assertEqual(calls, [])
            second = assembly.run(settings)
            self.assertEqual(second.status, 'needs_input')
            self.assertIsNone(assembly.evaluator.measurement('initial'))
            self.assertEqual(len(calls), 1)
            third = assembly.run(settings)
            self.assertEqual(third.status, 'done')
            actor_baseline_calls = [call for call in calls if call['model'] == MODEL.model_id
                and 'choose A' not in call['messages'][0]['content']]
            self.assertEqual(len(actor_baseline_calls), 2)
            self.assertTrue(actor_baseline_calls[0]['messages'][-1]['content'].startswith('First symbol'))
            self.assertTrue(actor_baseline_calls[1]['messages'][-1]['content'].startswith('Second symbol'))
            self.assertEqual(preparations.count('initial'), 2)
            self.assertEqual(assembly.evaluator._failures, {})
            evidence = [json.loads(self.artifacts.get(ref)) for ref in assembly.evaluator.evidence.values()]
            self.assertFalse(any(item['kind'] == 'raw_development_failure' for item in evidence))

    def test_pareto_accuracy_comparisons_and_current_incumbent(self):
        from examples.medical.evaluation import pareto_acceptance
        from rsisys.loops.exploration.harness.evaluation import EvaluationResult
        guard = write_rows(self.root/'guard.jsonl', [row('g', 'Triangle', benchmark='medxpertqa')])
        task = replace(self.task, development=self.task.development + (Benchmark('medxpertqa', 'guardrail', guard),))
        def result(a, b):
            return EvaluationResult('fixture', task.protocol_ref,
                {'medqa.accuracy': a, 'medxpertqa.accuracy': b, 'score': a}, (), True)
        self.assertNotEqual(task.protocol_ref, replace(task, selection_rule='primary_guardrail').protocol_ref)
        with self.assertRaises(ValueError):
            replace(task, selection_rule='unknown')
        incumbent = result(0.5, 0.5)
        for a, b, accepted, comparison in [
                (0.6, 0.5, True, 'improvement'), (0.5, 0.6, True, 'improvement'),
                (0.6, 0.6, True, 'improvement'), (0.5, 0.5, False, 'tie'),
                (0.6, 0.4, False, 'tradeoff'), (0.4, 0.6, False, 'tradeoff'),
                (0.4, 0.5, False, 'regression')]:
            with self.subTest(a=a, b=b):
                decision = pareto_acceptance(task, result(a, b), incumbent)
                self.assertEqual(decision['accepted'], accepted)
                self.assertEqual(decision['comparison'], comparison)
        self.assertFalse(pareto_acceptance(task, result(0.6, 0.5), result(0.7, 0.5))['accepted'])
        for invalid in (float('nan'), float('inf'), -0.1, 1.1, True):
            with self.assertRaises(ValueError):
                pareto_acceptance(task, result(invalid, 0.5), incumbent)

    def test_pareto_tradeoff_measurement_is_retained_without_application(self):
        guard = write_rows(self.root/'guard.jsonl', [row('g', 'Triangle', benchmark='medxpertqa')])
        task = replace(self.task, development=self.task.development + (Benchmark('medxpertqa', 'guardrail', guard),),
                       guardrails=False, max_guardrail_drop=1)
        prompt = self.artifacts.put(b'', kind='medical_prompt')
        systems = SystemRegistry(SystemVersion('initial', {'prompt': prompt}, MODEL), ReferenceSystemActivator(MODEL))
        outputs = iter(['B', 'A', 'A', 'B'])
        candidates = {}
        evaluator = MedicalEvaluator(task, self.artifacts, systems, candidates,
            lambda request: GenerationResponse(next(outputs), request.model, True))
        evaluator.establish_anchor()
        candidate = Candidate('tradeoff', MODEL.revision, {}, ('actor',), 'initial')
        systems.register(candidate)
        candidates[candidate.candidate_id] = candidate
        job = evaluator.submit(EvaluationRequest(candidate, task.protocol_ref), idempotency_key='tradeoff')
        while evaluator.poll(job).status == JobStatus.RUNNING:
            pass
        self.assertEqual(evaluator.result(job).candidate_id, 'tradeoff')
        verdict = evaluator.interpret('tradeoff')
        self.assertFalse(verdict['accepted'])
        self.assertEqual(verdict['comparison'], 'tradeoff')
        self.assertIn('tradeoff', candidates)
        self.assertIsNotNone(evaluator.measurement('tradeoff'))
        self.assertEqual(systems.current.version, 'initial')
        replacement = MedicalEvaluator(task, self.artifacts, systems, candidates, lambda r: None,
            acceptance_policy=lambda task, candidate, incumbent: {'accepted': False, 'selection_rule': 'custom'})
        replacement._anchor, replacement._outputs = evaluator._anchor, evaluator._outputs
        self.assertEqual(replacement.interpret('tradeoff')['selection_rule'], 'custom')

    def test_fixed_anchor_guardrail_never_drifts_and_optional_setting(self):
        guard = write_rows(self.root/'guard.jsonl', [row('g', 'Select the triangle symbol', benchmark='medmcqa')])
        task = replace(self.task, development=self.task.development + (Benchmark('medmcqa', 'guardrail', guard),),
                       max_guardrail_drop=0.02, selection_rule="primary_guardrail")
        prompt = self.artifacts.put(b'', kind='medical_prompt')
        systems = SystemRegistry(SystemVersion('initial', {'prompt': prompt}, MODEL), ReferenceSystemActivator(MODEL))
        outputs = iter(['B', 'A', 'A', 'B'])
        evaluator = MedicalEvaluator(task, self.artifacts, systems, {},
            lambda request: GenerationResponse(next(outputs), request.model, True))
        evaluator.establish_anchor()
        candidate = Candidate('c', MODEL.revision, {}, ('actor',), 'initial')
        systems.register(candidate)
        evaluator.candidates['c'] = candidate
        evaluator.measure(systems.get('c'))
        interpretation = evaluator.interpret('c')
        self.assertTrue(interpretation['primary_improved'])
        self.assertFalse(interpretation['accepted'])
        optional = MedicalEvaluator(replace(task, guardrails=False), self.artifacts, systems, {}, lambda r: None)
        optional._anchor, optional._outputs = evaluator._anchor, evaluator._outputs
        self.assertTrue(optional.interpret('c')['accepted'])
        # A weaker incumbent cannot lower the fixed anchor guardrail.
        evaluator._outputs['c'] = replace(evaluator._outputs['c'], metrics={'score': 0.5, 'medmcqa.accuracy': 0})
        systems.activate('c', idempotency_key='operator-fixture')
        evaluator._outputs['next'] = replace(evaluator._outputs['c'], candidate_id='next', metrics={'score': 1, 'medmcqa.accuracy': 0.5})
        self.assertFalse(evaluator.interpret('next')['accepted'])


if __name__ == '__main__':
    unittest.main()
