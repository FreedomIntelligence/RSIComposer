"""CPU-only HTTP contract fixtures; these tests do not claim to run real Qwen."""
from tests import ROOT as REPO_ROOT
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from experiments.qwen_toy.adapters import (
    Deadline, HTTPClient, RevisionBackend, TRAIN_REF, WorkerActivator, WorkerEvaluator, WorkerTrainer,
)
from experiments.qwen_toy.run import BudgetedModel, OBJECTIVE, ResearchBudget, parse_args, run
from rsimonitor import project_snapshot
from rsisys.experience.sqlite import SQLiteExperienceStore
from rsisys.loops.contracts.jobs import JobStatus
from rsisys.loops.contracts.model_service import GenerationRequest, GenerationResponse, ModelRef
from rsisys.loops.exploration.harness.evaluation import EvaluationRequest
from rsisys.loops.exploration.harness.learner import ActorTrainingRequest


class ContextBudgetTests(unittest.TestCase):
    def fixture(self, payload, *, limit=20000):
        model = ModelRef('fixture', 'model', 'baseline')
        request = GenerationRequest('decision_maker', model, json.dumps(payload),
                                    system='Preserve system instructions', max_output_tokens=500)
        response = GenerationResponse('{"kind":"finish"}', model, False)
        backend = Mock(generate=Mock(return_value=response))
        return BudgetedModel(backend, ResearchBudget(max_context_chars=limit)), backend, request, response

    def test_small_context_is_preserved_and_default_budget_is_explicit(self):
        payload = {'objective': 'task', 'history': [], 'experience': [],
                   'operations': [{'name': 'inspect', 'input_schema': {'type': 'object'}}]}
        budgeted, backend, request, response = self.fixture(payload)
        self.assertEqual(budgeted.budget.max_context_chars, 20000)
        self.assertIs(budgeted.generate(request), response)
        sent = backend.generate.call_args.args[0]
        self.assertEqual(json.loads(sent.prompt), {**payload, 'context_omissions': {}})
        self.assertEqual((sent.role, sent.model, sent.system), (request.role, request.model, request.system))
        self.assertEqual(sent.max_output_tokens, budgeted.budget.max_output_tokens)
        self.assertEqual(request.max_output_tokens, 500)
        self.assertEqual(budgeted.calls, 1)

    def test_oldest_evidence_is_removed_with_counts_and_latest_source_preserved(self):
        for field in ('history', 'experience', 'directive_provenance', 'memories', 'operation_results'):
            with self.subTest(field=field):
                latest = {'record_id': 'latest', 'source_refs': ['evidence:last'], 'details': {'loss': 0.7}}
                payload = {'objective': 'task', field: [
                    {'record_id': 'oldest', 'summary': 'old observation ' * 100},
                    {'record_id': 'older', 'summary': 'older observation ' * 100}, latest]}
                original = deepcopy(payload)
                budgeted, backend, request, _ = self.fixture(payload, limit=500)
                original_prompt = request.prompt
                budgeted.generate(request)
                sent = backend.generate.call_args.args[0]
                view = json.loads(sent.prompt)
                self.assertLessEqual(len(sent.prompt), 500)
                self.assertEqual(view[field], [latest])
                self.assertEqual(view['context_omissions'], {field: 2})
                self.assertEqual(payload, original)
                self.assertEqual(request.prompt, original_prompt)
                self.assertEqual(json.loads(request.prompt), original)

    def test_large_last_details_and_tool_data_have_explicit_query_hints(self):
        payload = {'objective': 'task',
                   'history': [{'record_id': 'history:1', 'source_refs': ['feedback:1'],
                                'details': {'trace': ['x' * 4000]}}],
                   'experience': [{'record_id': 'feedback:1', 'source_refs': ['artifact:1'],
                                   'summary': 'Observed loss', 'details': {'trace': ['y' * 4000]}}],
                   'operation_results': [{'summary': 'Queried evidence', 'status': 'completed',
                                          'data': {'items': [{'text': 'z' * 4000}]}}]}
        original = deepcopy(payload)
        budgeted, backend, request, _ = self.fixture(payload, limit=1000)
        budgeted.generate(request)
        sent = backend.generate.call_args.args[0]
        view = json.loads(sent.prompt)
        self.assertLessEqual(len(sent.prompt), 1000)
        for field, contents in (('history', 'details'), ('experience', 'details'), ('operation_results', 'data')):
            row, source = view[field][0], original[field][0]
            self.assertEqual({k: v for k, v in row.items() if k != contents},
                             {k: v for k, v in source.items() if k != contents})
            self.assertEqual(set(row[contents]), {'omitted'})
            self.assertIn('query', row[contents]['omitted'].lower())
        self.assertEqual(payload, original)
        self.assertEqual(json.loads(request.prompt), original)

    def test_fixed_schema_over_budget_stops_before_any_backend_call(self):
        payload = {'objective': 'task', 'operations': [{'name': 'inspect',
            'input_schema': {'type': 'object', 'description': 'Required schema ' * 100}}],
            'history': [{'record_id': 'h1', 'details': {'text': 'x' * 2000}}]}
        original = deepcopy(payload)
        budgeted, backend, request, _ = self.fixture(payload, limit=500)
        with self.assertRaisesRegex(TimeoutError, 'context-character budget exhausted'):
            budgeted.generate(request)
        backend.generate.assert_not_called()
        self.assertEqual(budgeted.calls, 0)
        self.assertEqual(payload, original)
        self.assertEqual(json.loads(request.prompt), original)

    def test_context_budget_rejects_nonpositive_and_noninteger_values(self):
        for limit in (0, -1, True, 1.5, None):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, 'positive integers'):
                ResearchBudget(max_context_chars=limit)


class Services:
    def __init__(self, *, delta=0.5, wrong_revision=False, plan=None,
                 forever=False, failed=False, wrong_activation_json=False,
                 poll_delay=0, cancel_pending=False, invoke_first=False,
                 response_losses=None, lost_response_delay=0, wrong_revision_round=None, max_rounds=2, review_action='continue'):
        self.delta, self.wrong_revision = delta, wrong_revision
        self.plan = list(plan) if plan is not None else None
        self.planned_rounds, self.review_after = 1, None
        self.contexts = []
        self.forever, self.failed = forever, failed
        self.wrong_activation_json = wrong_activation_json
        self.poll_delay, self.cancel_pending = poll_delay, cancel_pending
        self.invoke_first = invoke_first
        self.response_losses = dict(response_losses or {})
        self.lost_response_delay = lost_response_delay
        self.signatures = {}
        self.jobs, self.keys, self.calls, self.decisions = {}, {}, [], []
        self.trained = False
        self.revisions = {}
        self.latest_revision = "candidate-1"
        self.wrong_revision_round, self.max_rounds = wrong_revision_round, max_rounds
        self.review_action, self.reviews = review_action, []

    def reply(self, role, path, payload):
        self.calls.append((role, path, payload))
        if path == '/health':
            return ({'status': 'ok', 'model': 'Qwen3.5-9B'} if role == 'baseline' else
                    {'ready': True, 'trained': self.trained, 'revision': self.latest_revision,
                     'revisions': list(self.revisions), 'max_rounds': self.max_rounds})
        if path in ('/train', '/evaluate'):
            key = payload['idempotency_key']
            signature = (path, json.dumps(payload, sort_keys=True))
            if key in self.signatures and self.signatures[key] != signature:
                raise AssertionError('Idempotency key reused with changed request')
            self.signatures[key] = signature
            if key not in self.keys:
                ident = f'job-{len(self.jobs)}'
                self.keys[key] = ident
                self.jobs[ident] = {'job_id': ident, 'kind': path[1:], 'status': 'queued', 'polls': 0,
                                    'request': payload['request']}
                if path == '/train':
                    number = sum(j['kind'] == 'train' for j in self.jobs.values())
                    self.jobs[ident]['revision'] = f'candidate-{number}'
            return {'job_id': self.keys[key]}
        if path == '/cancel':
            status = 'running' if self.cancel_pending else 'cancelled'
            self.jobs[payload['job_id']]['status'] = status
            return {'job_id': payload['job_id'], 'status': status}
        if path.startswith('/jobs/'):
            job = self.jobs[path.split('/')[-1]]
            if job['status'] == 'cancelled':
                return {'job_id': job['job_id'], 'status': 'cancelled'}
            time.sleep(self.poll_delay)
            job['polls'] += 1
            if self.forever or job['polls'] == 1:
                return {'job_id': job['job_id'], 'status': 'running'}
            if self.failed:
                return {'job_id': job['job_id'], 'status': 'failed', 'error': 'SECRET_REMOTE_ERROR'}
            if job['kind'] == 'train':
                self.trained = True
                revision = job['revision']
                parent = job['request']['base']['revision']
                self.latest_revision = revision
                self.revisions[revision] = parent
                result = {'adapter_path': '/worker/run/' + revision, 'revision': revision,
                          'base_revision': parent, 'metrics': {'train_loss': 0.7, 'train_runtime': 12.5}}
            else:
                revision = job['request']['candidate']['candidate_id']
                delta = self.delta[int(revision.split('-')[-1])-1] if isinstance(self.delta, list) else self.delta
                result = {'revision': revision, 'base_revision': self.revisions[revision], 'metrics': {
                    'base_nll': 2.0, 'candidate_nll': 2.0 - delta,
                    'delta_nll': delta, 'eval_samples': 12}}
            return {'job_id': job['job_id'], 'status': 'succeeded', 'result': result}
        if path == '/v1/chat/completions':
            if payload['max_tokens'] > 200 or payload['chat_template_kwargs']['enable_thinking']:
                raise AssertionError('Service token/thinking contract violated')
            prompt = payload['messages'][-1]['content']
            if payload['messages'][0]['content'].startswith('Review the running execution'):
                context = json.loads(prompt)
                self.reviews.append(context)
                value = {'action': self.review_action, 'next_review_seconds': None,
                         'reason': 'Fixture review of the observed job'}
            elif prompt.startswith('{'):
                context = json.loads(prompt)
                self.contexts.append(context)
                if self.plan is None:
                    # A scripted HTTP fixture, independent of controller prompt wording.
                    # This threshold is a fixture choice, never a production guard.
                    self.plan = []
                    if self.invoke_first:
                        self.plan.append({'kind': 'invoke', 'operation': 'train',
                                          'arguments': {'recipe_ref': 'toy-lora', 'method': 'sft'}})
                    for number in range(self.planned_rounds):
                        delta = self.delta[number] if isinstance(self.delta, list) else self.delta
                        self.plan.extend(['train', 'evaluate'])
                        if delta > 0:
                            self.plan.append('apply_candidate')
                        self.plan.append('execute')
                    self.plan.append('finish')
                if not self.plan:
                    raise AssertionError('Controller requested a decision beyond the fixture plan')
                action = self.plan.pop(0)
                if callable(action):
                    action = action(context)
                if isinstance(action, dict):
                    value = action
                elif action == 'finish':
                    value = {'kind': 'finish', 'reason': 'Fixture plan complete'}
                else:
                    value = {'kind': 'explore', 'strategy': 'Fixture chooses ' + action, 'mode': action}
                    if action == 'train':
                        value['arguments'] = {'recipe_ref': 'toy-lora', 'method': 'sft'}
                        if self.review_after is not None:
                            value['review_after_seconds'] = self.review_after
                    elif action == 'evaluate':
                        value['arguments'] = {'candidate_id': self.latest_revision, 'protocol_ref': 'heldout-nll'}
                    elif action == 'apply_candidate':
                        value['arguments'] = {'candidate_id': self.latest_revision}
                self.decisions.append(value)
            else:
                value = {'parity': 'odd' if '113' in prompt else 'even'}
                if self.wrong_activation_json and role == 'worker':
                    value = {'bad': 'schema'}
            result = {'model': payload['model'],
                      'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}],
                      'usage': {'prompt_tokens': 100, 'completion_tokens': 40}}
            if role == 'worker':
                selected = payload.get('model_revision', self.latest_revision)
                wrong = self.wrong_revision or self.wrong_revision_round == int(selected.split('-')[-1])
                result['model_revision'] = 'wrong' if wrong else selected
            return result
        raise AssertionError('Unexpected endpoint')


@contextmanager
def servers(service):
    active = []
    try:
        for role in ('baseline', 'worker'):
            def handler_for(which):
                class Handler(BaseHTTPRequestHandler):
                    def log_message(self, *args):
                        pass

                    def handle_request(self, payload=None):
                        value = service.reply(which, self.path, payload)
                        kind = self.path.removeprefix('/')
                        if kind in ('train', 'evaluate') and service.response_losses.get(kind, 0):
                            service.response_losses[kind] -= 1
                            time.sleep(service.lost_response_delay)
                            self.close_connection = True
                            try:
                                self.connection.shutdown(socket.SHUT_RDWR)
                            except OSError:
                                pass
                            return  # Remote job exists, but no response reaches the client.
                        data = json.dumps(value).encode()
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.send_header('Content-Length', str(len(data)))
                        self.end_headers()
                        try:
                            self.wfile.write(data)
                        except BrokenPipeError:
                            pass  # Deadline tests intentionally close the original request.

                    def do_GET(self):
                        self.handle_request()

                    def do_POST(self):
                        self.handle_request(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                return Handler
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(role))
            thread = threading.Thread(target=lambda server=server: server.serve_forever(poll_interval=0.01), daemon=True)
            thread.start()
            active.append((server, thread))
        yield tuple(f'http://127.0.0.1:{server.server_port}' for server, _ in active)
    finally:
        for server, thread in active:
            server.shutdown()
            server.server_close()
            thread.join()


class QwenControllerTests(unittest.TestCase):
    def test_declared_parent_is_persisted_without_restoring_actor_or_training(self):
        with tempfile.TemporaryDirectory() as directory, servers(service := Services(plan=['finish'])) as urls:
            root = Path(directory) / 'run6'
            result = self.execute(service, root, urls, extra_args=('--parent-run-id', 'run5'))
            self.assertEqual(result['status'], 'completed')
            for name in ('run.json', 'monitor.json', 'result.json'):
                data = json.loads((root / name).read_text())
                self.assertEqual((data['run_id'], data['parent_run_id']), ('run6', 'run5'))
            self.assertEqual(result['actor_revision'], 'baseline')
            self.assertEqual(service.jobs, {})

    def test_legacy_namespace_without_parent_remains_supported(self):
        with tempfile.TemporaryDirectory() as directory, servers(service := Services(plan=['finish'])) as urls:
            args = parse_args(['--baseline-url', urls[0] + '/v1', '--worker-url', urls[1],
                               '--run-dir', directory, '--poll-interval', '0.001'])
            del args.parent_run_id
            result = run(args)
            self.assertEqual(result['status'], 'completed')
            self.assertIsNone(result['parent_run_id'])

    def execute(self, service, root, urls, *, max_steps=100, timeout=10, rounds=1, review_after=2, require_review=False, extra_args=()):
        service.planned_rounds, service.review_after = rounds, review_after
        args = parse_args(['--baseline-url', urls[0] + '/v1', '--worker-url', urls[1],
                           '--run-dir', str(root), '--poll-interval', '0.001',
                           '--timeout', str(timeout), '--max-steps', str(max_steps), '--rounds', str(rounds),
                           '--review-after-seconds', str(review_after)]
                          + (['--require-training-review'] if require_review else []) + list(extra_args))
        return run(args)

    def test_two_rounds_continue_from_applied_revision_and_keep_baseline_decisions(self):
        with tempfile.TemporaryDirectory() as directory, servers(service := Services()) as urls:
            root = Path(directory)
            result = self.execute(service, root, urls, rounds=2)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['rounds_completed'], 2)
            self.assertEqual(result['actor_revision'], 'candidate-2')
            self.assertEqual(result['decision_revision'], 'baseline')
            self.assertEqual(len(service.jobs), 4)
            trains = [p for _, path, p in service.calls if path == '/train']
            self.assertEqual([p['request']['base']['revision'] for p in trains], ['baseline', 'candidate-1'])
            self.assertNotEqual(trains[0]['idempotency_key'], trains[1]['idempotency_key'])
            self.assertEqual([r['base_revision'] for r in result['round_results']], ['baseline', 'candidate-1'])
            self.assertTrue(all(r['applied'] and r['answer_correct'] for r in result['round_results']))
            modes = [value.get('mode', value['kind']) for value in service.decisions]
            self.assertEqual(modes, ['train', 'evaluate', 'apply_candidate', 'execute'] * 2 + ['finish'])
            for number in (1, 2):
                self.assertEqual(json.loads((root / f'round-{number}.json').read_text()),
                                 result['round_results'][number-1])
            monitor = json.loads((root / 'monitor.json').read_text())
            self.assertEqual([c['base_revision'] for c in monitor['candidates']], ['baseline', 'candidate-1'])
            store = SQLiteExperienceStore(root / 'experience.sqlite3')
            try:
                records = store.records()
            finally:
                store.close()
            self.assertEqual(sum(r.kind == 'training_run' for r in records), 2)
            self.assertEqual(sum(r.kind == 'evaluation' for r in records), 2)

    def test_fixture_nonselection_preserves_current_actor_and_parent_metrics(self):
        for deltas, parents, actors in (([0.5, -0.1], ['baseline', 'candidate-1'], ['candidate-1', 'candidate-1']),
                                       ([-0.1, 0.5], ['baseline', 'baseline'], ['baseline', 'candidate-2'])):
            with self.subTest(deltas=deltas), tempfile.TemporaryDirectory() as directory:
                service = Services(delta=deltas)
                with servers(service) as urls:
                    result = self.execute(service, Path(directory), urls, rounds=2)
                self.assertEqual(result['status'], 'completed')
                self.assertEqual([r['base_revision'] for r in result['round_results']], parents)
                self.assertEqual([r['actor_revision'] for r in result['round_results']], actors)
                monitor = json.loads((Path(directory) / 'monitor.json').read_text())
                self.assertEqual([c['metrics']['delta_nll'] for c in monitor['candidates']], deltas)
                self.assertEqual([c['status'] for c in monitor['candidates']],
                                 ['applied' if delta > 0 else 'proposed' for delta in deltas])

    def test_second_round_activation_failure_retains_first_round_actor(self):
        with tempfile.TemporaryDirectory() as directory, servers(service := Services(wrong_revision_round=2)) as urls:
            result = self.execute(service, Path(directory), urls, rounds=2)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['rounds_completed'], 1)
            self.assertEqual(result['actor_revision'], 'candidate-1')
            self.assertEqual(result['round_results'][0]['candidate_id'], 'candidate-1')

    def test_two_rounds_reject_legacy_worker_before_submitting_any_training(self):
        with tempfile.TemporaryDirectory() as directory, servers(service := Services(max_rounds=1)) as urls:
            result = self.execute(service, Path(directory), urls, rounds=2)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(service.jobs, {})

    def test_each_round_reviews_running_training_via_model_and_persists_evidence(self):
        with tempfile.TemporaryDirectory() as directory, servers(service := Services(poll_delay=0.03)) as urls:
            root = Path(directory)
            result = self.execute(service, root, urls, rounds=2, review_after=0.005, require_review=True)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(len(service.reviews), 2)
            self.assertEqual([r['training_review_count'] for r in result['round_results']], [1, 2])
            self.assertEqual([r['source'] for r in result['training_reviews']], ['llm', 'llm'])
            self.assertEqual([r['actor_revision'] for r in result['training_reviews']], ['baseline', 'candidate-1'])
            for context in service.reviews:
                progress = context['execution']['progress']
                self.assertIsNone(progress['feedback'])
                self.assertEqual(progress['directive']['mode'], 'train')
                self.assertIsNotNone(progress['job'])
            monitor = json.loads((root / 'monitor.json').read_text())
            events = [e for e in monitor['events'] if e['kind'] == 'training_review_completed']
            self.assertEqual(len(events), 2)
            store = SQLiteExperienceStore(root / 'experience.sqlite3')
            try:
                records = [r for r in store.records() if r.details.get('event') == 'review']
            finally:
                store.close()
            self.assertEqual(len(records), 2)
            self.assertTrue(all(r.details['review']['action'] == 'continue' for r in records))

    def test_required_review_cannot_pass_when_training_finishes_before_review_deadline(self):
        with tempfile.TemporaryDirectory() as directory, servers(service := Services()) as urls:
            result = self.execute(service, Path(directory), urls, review_after=100, require_review=True)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['training_reviews'], [])

    def test_model_review_cancel_is_honored_without_starting_another_round(self):
        with tempfile.TemporaryDirectory() as directory, servers(service := Services(
                forever=True, poll_delay=0.03, review_action='cancel', plan=['train', 'finish'])) as urls:
            result = self.execute(service, Path(directory), urls, rounds=2, review_after=0.005)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['experiments'][0]['status'], 'cancelled')
            self.assertEqual(result['training_reviews'][0]['action'], 'cancel')
            self.assertEqual(len(service.jobs), 1)
            self.assertEqual(result['actor_revision'], 'baseline')

    def test_round_alias_is_a_bounded_training_budget(self):
        args = ['--baseline-url', 'http://localhost:1', '--worker-url', 'http://localhost:2']
        self.assertEqual(parse_args(args).rounds, 2)
        self.assertEqual(parse_args(args + ['--max-trainings', '1']).rounds, 1)
        self.assertEqual(parse_args(args + ['--rounds', '2']).rounds, 2)
        for invalid in ('0', '3', '-1'):
            with self.subTest(invalid=invalid), redirect_stdout(io.StringIO()):
                from contextlib import redirect_stderr
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(args + ['--rounds', invalid])

    def test_http_fixture_selection_and_nonselection_have_durable_evidence(self):
        for delta in (0.5, 0, -0.5):
            with self.subTest(delta=delta), tempfile.TemporaryDirectory() as directory:
                service, root = Services(delta=delta), Path(directory) / 'qwen-toy'
                with servers(service) as urls:
                    result = self.execute(service, root, urls)
                self.assertEqual(result['status'], 'completed')
                self.assertFalse(result['simulated'])
                self.assertEqual(result['selection'], 'accepted' if delta > 0 else 'not_selected')
                self.assertEqual(result['actor_revision'], 'candidate-1' if delta > 0 else 'baseline')
                self.assertEqual(result['decision_revision'], 'baseline')
                self.assertEqual(result['applied'], delta > 0)
                self.assertTrue(result['answer_correct'])
                self.assertEqual(result['metrics']['delta_nll'], delta)
                modes = [value.get('mode', value['kind']) for value in service.decisions]
                self.assertEqual(modes, ['train', 'evaluate'] + (['apply_candidate'] if delta > 0 else [])
                                 + ['execute', 'finish'])
                self.assertEqual(len(service.jobs), 2)
                self.assertEqual(json.loads((root / 'result.json').read_text()), result)
                monitor = json.loads((root / 'monitor.json').read_text())
                self.assertEqual(monitor['schema_version'], 1)
                self.assertEqual(monitor['status'], 'completed')
                self.assertEqual(monitor['candidates'][0]['status'], 'applied' if delta > 0 else 'proposed')
                self.assertEqual(monitor['models']['actor']['revision'], result['actor_revision'])
                # The public monitor projection must consume the persisted schema.
                projected = project_snapshot(monitor, 'qwen-toy')
                self.assertEqual(projected['candidates'][0]['metrics']['delta_nll'], delta)
                store = SQLiteExperienceStore(root / 'experience.sqlite3')
                try:
                    records = store.records()
                finally:
                    store.close()
                self.assertTrue(all(not record.simulated for record in records))
                self.assertEqual(len([r for r in records if r.kind == 'training_run']), 1)
                self.assertEqual(len([r for r in records if r.kind == 'evaluation']), 1)
                self.assertTrue((root / 'decision_memory.json').exists())
                self.assertTrue((root / 'exploration_memory.json').exists())
                for name in ('result.json', 'monitor.json'):
                    content = (root / name).read_text()
                    self.assertNotIn(urls[0], content)
                    self.assertNotIn(urls[1], content)

    def test_model_can_finish_immediately_without_training_or_execution(self):
        with tempfile.TemporaryDirectory() as directory, servers(service := Services(plan=['finish'])) as urls:
            result = self.execute(service, Path(directory), urls)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(service.jobs, {})
            self.assertEqual(len(service.decisions), 1)
            self.assertEqual(result['completed_directives'], 0)
            self.assertEqual(result['training_submissions'], 0)
            self.assertEqual(result['round_results'], [])
            self.assertEqual(result['rounds_completed'], 0)
            self.assertEqual(result['selection'], 'not_selected')
            self.assertEqual(result['actor_revision'], 'baseline')

    def test_issued_directives_keep_original_objective_without_research_prompt_accumulation(self):
        with tempfile.TemporaryDirectory() as directory, servers(service := Services(
                plan=['execute', 'execute', 'finish'])) as urls:
            root = Path(directory)
            result = self.execute(service, root, urls)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['rounds_completed'], 2)
            for context in service.contexts:
                self.assertEqual(context['objective'].count('You are the researcher.'), 1)
                for entry in context['history']:
                    directive = entry['details'].get('directive')
                    if directive:
                        self.assertEqual(directive['objective'], OBJECTIVE)
            history = json.loads((root / 'decision_memory.json').read_text())['history']
            directives = [row['details']['directive'] for row in history
                          if row['kind'] == 'decision' and row['details'].get('directive')]
            self.assertEqual([directive['objective'] for directive in directives], [OBJECTIVE, OBJECTIVE])
            self.assertNotIn('You are the researcher.', json.dumps(history))

    def test_unavailable_invoke_train_requires_a_new_model_response(self):
        with tempfile.TemporaryDirectory() as directory, servers(service := Services(invoke_first=True)) as urls:
            result = self.execute(service, Path(directory), urls)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(service.decisions[0]['kind'], 'invoke')
            self.assertEqual(service.decisions[1]['kind'], 'explore')
            self.assertIn('validation error (data):', service.contexts[1]['objective'])
            before_train = service.calls[:next(i for i, (_, path, _) in enumerate(service.calls) if path == '/train')]
            self.assertEqual(sum(path == '/v1/chat/completions' for _, path, _ in before_train), 2)
            self.assertEqual(len(service.jobs), 2)

    def test_model_executes_baseline_then_trains_and_repeats_evaluation_and_execution(self):
        plan = ['execute', 'train', 'evaluate', 'evaluate', 'apply_candidate', 'execute', 'execute', 'finish']
        with tempfile.TemporaryDirectory() as directory, servers(service := Services(plan=plan)) as urls:
            root = Path(directory)
            result = self.execute(service, root, urls, rounds=1)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual([r['mode'] for r in result['experiments']], plan[:-1])
            self.assertEqual(result['training_submissions'], 1)
            self.assertEqual(result['rounds_requested'], 1)
            self.assertEqual(result['rounds_completed'], 3)
            self.assertEqual([r['actor_revision'] for r in result['round_results']],
                             ['baseline', 'candidate-1', 'candidate-1'])
            self.assertEqual([r['selection'] for r in result['round_results']],
                             ['not_selected', 'accepted', 'accepted'])
            self.assertIsNone(result['round_results'][0]['candidate_id'])
            self.assertTrue(all(r['answer_correct'] for r in result['round_results']))
            evaluations = [payload for _, path, payload in service.calls if path == '/evaluate']
            self.assertEqual(len(evaluations), 2)
            self.assertNotEqual(evaluations[0]['idempotency_key'], evaluations[1]['idempotency_key'])
            self.assertEqual(evaluations[0]['request'], evaluations[1]['request'])
            for number, observation in enumerate(result['round_results'], 1):
                self.assertEqual(json.loads((root / f'round-{number}.json').read_text()), observation)

    def test_model_can_apply_a_version_valid_candidate_despite_negative_delta(self):
        with tempfile.TemporaryDirectory() as directory, servers(service := Services(
                delta=-0.5, plan=['train', 'evaluate', 'apply_candidate', 'execute', 'finish'])) as urls:
            root = Path(directory)
            result = self.execute(service, root, urls)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['metrics']['delta_nll'], -0.5)
            self.assertEqual(result['selection'], 'accepted')
            self.assertTrue(result['applied'])
            self.assertEqual(result['actor_revision'], 'candidate-1')
            self.assertEqual(service.contexts[2]['last_feedback']['quality_metrics']['delta_nll'], -0.5)
            self.assertEqual(json.loads((root / 'monitor.json').read_text())['candidates'][0]['status'], 'applied')

    def test_evaluated_candidates_remain_proposed_without_an_apply_decision(self):
        for delta in (-0.5, 0.5):
            with self.subTest(delta=delta), tempfile.TemporaryDirectory() as directory:
                service = Services(delta=delta, plan=['train', 'evaluate', 'finish'])
                with servers(service) as urls:
                    result = self.execute(service, Path(directory), urls)
                self.assertEqual(result['status'], 'completed')
                self.assertEqual(result['selection'], 'not_selected')
                self.assertFalse(result['applied'])
                self.assertEqual(result['actor_revision'], 'baseline')
                self.assertEqual(result['rounds_completed'], 0)
                monitor = json.loads((Path(directory) / 'monitor.json').read_text())
                self.assertEqual(monitor['candidates'][0]['status'], 'proposed')
                self.assertEqual(monitor['candidates'][0]['metrics']['delta_nll'], delta)

    def test_model_invokes_an_advertised_tool_and_receives_its_observation(self):
        operation = {'kind': 'invoke', 'operation': 'query_exploration_model', 'arguments': {}}
        with tempfile.TemporaryDirectory() as directory, servers(service := Services(plan=[operation, 'finish'])) as urls:
            root = Path(directory)
            result = self.execute(service, root, urls)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(service.jobs, {})
            self.assertEqual(result['completed_directives'], 0)
            observed = service.contexts[1]['operation_results'][0]
            self.assertEqual(observed['status'], 'completed')
            self.assertEqual(observed['data']['data']['items'][0]['model']['revision'], 'baseline')
            monitor = json.loads((root / 'monitor.json').read_text())
            self.assertEqual(monitor['decisions'][0]['operation'], 'query_exploration_model')
            self.assertEqual(monitor['decisions'][0]['status'], 'completed')

    def test_training_and_directive_budgets_reject_extra_model_actions(self):
        for mode, extra in (('train', ()), ('execute', ('--max-directives', '1'))):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                service = Services(plan=[mode] * 4)
                with servers(service) as urls:
                    result = self.execute(service, Path(directory), urls, rounds=1, extra_args=extra)
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['completed_directives'], 1)
                self.assertEqual(len(service.decisions), 4)
                self.assertEqual(len(service.jobs), 1 if mode == 'train' else 0)
                self.assertEqual(len(result['experiments']), 1)
                self.assertNotIn(mode, service.contexts[1]['directive_modes'])
                self.assertIn('validation error (data):', service.contexts[2]['objective'])

    def test_model_call_budget_counts_validation_retries(self):
        invalid = {'kind': 'invoke', 'operation': 'not_installed', 'arguments': {}}
        with tempfile.TemporaryDirectory() as directory, servers(service := Services(plan=[invalid] * 3)) as urls:
            result = self.execute(service, Path(directory), urls, extra_args=('--max-model-calls', '2'))
            self.assertEqual(result['status'], 'timed_out')
            self.assertEqual(result['model_calls'], 2)
            self.assertEqual(len(service.decisions), 2)
            self.assertEqual(service.jobs, {})
            self.assertEqual(result['completed_directives'], 0)

    def test_permissions_reject_directives_and_tools_without_side_effects(self):
        operation = {'kind': 'invoke', 'operation': 'query_exploration_model', 'arguments': {}}
        for action, permissions in (('train', {'directives': []}), (operation, {'operations': []})):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = root / 'permissions.json'
                config.write_text(json.dumps(permissions))
                service = Services(plan=[action] * 3)
                with servers(service) as urls:
                    result = self.execute(service, root, urls, extra_args=('--permissions', str(config)))
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(service.jobs, {})
                self.assertEqual(result['completed_directives'], 0)
                self.assertEqual(len(service.decisions), 3)
                self.assertTrue(all(not context['operation_results'] for context in service.contexts))
                self.assertIn('validation error (data):', service.contexts[1]['objective'])

    def test_model_may_omit_review_schedule_despite_controller_hint(self):
        train = {'kind': 'explore', 'mode': 'train', 'strategy': 'No scheduled review',
                 'arguments': {'recipe_ref': 'toy-lora', 'method': 'sft'}}
        with tempfile.TemporaryDirectory() as directory, servers(service := Services(
                plan=[train, 'finish'], poll_delay=0.03)) as urls:
            result = self.execute(service, Path(directory), urls, review_after=0.005)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['training_submissions'], 1)
            self.assertEqual(result['training_reviews'], [])
            self.assertEqual(service.reviews, [])

    def test_wrong_revision_or_bad_smoke_inference_never_binds(self):
        for kwargs in ({'wrong_revision': True}, {'wrong_activation_json': True}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                service = Services(**kwargs)
                with servers(service) as urls:
                    result = self.execute(service, Path(directory), urls)
                self.assertEqual(result['status'], 'failed')
                self.assertFalse(result['applied'])
                self.assertEqual(result['actor_revision'], 'baseline')

    def test_worker_failure_is_persisted_without_remote_error_secrets(self):
        with tempfile.TemporaryDirectory() as directory, servers(service := Services(failed=True, plan=['train', 'execute', 'finish'])) as urls:
            root = Path(directory)
            result = self.execute(service, root, urls)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['experiments'][0]['status'], 'failed')
            self.assertEqual(service.contexts[1]['last_feedback']['status'], 'failed')
            self.assertTrue(result['answer_correct'])
            self.assertFalse(result['applied'])
            for path in root.rglob('*.json'):
                self.assertNotIn('SECRET_REMOTE_ERROR', path.read_text())

    def test_step_budget_requests_remote_cancellation_and_does_not_claim_success(self):
        with tempfile.TemporaryDirectory() as directory, servers(service := Services(forever=True)) as urls:
            result = self.execute(service, Path(directory), urls, max_steps=2)
            self.assertEqual(result['status'], 'timed_out')
            self.assertFalse(result['remote_job_may_still_run'])
            self.assertTrue(result['cancel_requested'])
            self.assertTrue(result['cancellation_confirmed'])
            cancels = [payload for _, path, payload in service.calls if path == '/cancel']
            self.assertEqual(cancels, [{'job_id': 'job-0'}])

    def test_expired_deadline_still_uses_independent_cancellation_client(self):
        with tempfile.TemporaryDirectory() as directory, servers(service := Services(
                forever=True, poll_delay=0.25)) as urls:
            result = self.execute(service, Path(directory), urls, timeout=0.15)
            self.assertEqual(result['status'], 'timed_out')
            self.assertTrue(result['cancel_attempted'])
            self.assertTrue(result['cancel_requested'])
            self.assertTrue(result['cancellation_confirmed'])
            self.assertFalse(result['remote_job_may_still_run'])

    def test_lost_train_and_evaluate_responses_retry_same_keys_without_new_jobs(self):
        with tempfile.TemporaryDirectory() as directory, servers(service := Services(
                response_losses={'train': 1, 'evaluate': 1})) as urls:
            result = self.execute(service, Path(directory), urls)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(len(service.jobs), 2)
            self.assertEqual(len(service.decisions), 5)
            for kind in ('train', 'evaluate'):
                posts = [payload for _, path, payload in service.calls if path == '/' + kind]
                self.assertEqual(len(posts), 2)
                self.assertEqual(posts[0], posts[1])
            self.assertFalse(result['cancel_attempted'])

    def test_cleanup_recovers_uncertain_training_or_evaluation_then_cancels_original_job(self):
        for kind in ('train', 'evaluate'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                service = Services(response_losses={kind: 2})
                with servers(service) as urls:
                    result = self.execute(service, Path(directory), urls)
                self.assertEqual(result['status'], 'failed')
                self.assertTrue(result['cancel_requested'])
                self.assertTrue(result['cancellation_confirmed'])
                self.assertFalse(result['remote_job_may_still_run'])
                self.assertEqual(len(service.jobs), 1 if kind == 'train' else 2)
                posts = [payload for _, path, payload in service.calls if path == '/' + kind]
                self.assertEqual(len(posts), 3)
                self.assertTrue(all(payload == posts[0] for payload in posts))
                ident = service.keys[posts[0]['idempotency_key']]
                cancels = [payload for _, path, payload in service.calls if path == '/cancel']
                self.assertEqual(cancels, [{'job_id': ident}])

    def test_expired_submit_deadline_recovers_with_cleanup_client_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as directory, servers(service := Services(
                response_losses={'train': 1}, lost_response_delay=0.25)) as urls:
            started = time.monotonic()
            result = self.execute(service, Path(directory), urls, timeout=0.15)
            self.assertLess(time.monotonic() - started, 5.5)
            self.assertEqual(result['status'], 'timed_out')
            self.assertTrue(result['cancel_requested'])
            self.assertTrue(result['cancellation_confirmed'])
            self.assertFalse(result['remote_job_may_still_run'])
            self.assertEqual(len(service.jobs), 1)
            posts = [payload for _, path, payload in service.calls if path == '/train']
            self.assertEqual(len(posts), 2)
            self.assertEqual(posts[0], posts[1])

    def test_failed_cleanup_recovery_is_bounded_and_reports_remote_uncertainty(self):
        with tempfile.TemporaryDirectory() as directory, servers(service := Services(
                response_losses={'train': 10})) as urls:
            result = self.execute(service, Path(directory), urls)
            self.assertEqual(result['status'], 'failed')
            self.assertFalse(result['cancel_attempted'])
            self.assertFalse(result['cancel_requested'])
            self.assertFalse(result['cancellation_confirmed'])
            self.assertTrue(result['remote_job_may_still_run'])
            self.assertEqual(len(service.jobs), 1)
            posts = [payload for _, path, payload in service.calls if path == '/train']
            self.assertEqual(len(posts), 4)
            self.assertTrue(all(payload == posts[0] for payload in posts))
            self.assertEqual(json.loads((Path(directory) / 'result.json').read_text()), result)

    def test_cancel_ack_is_not_reported_as_termination(self):
        with tempfile.TemporaryDirectory() as directory, servers(service := Services(
                forever=True, cancel_pending=True)) as urls:
            result = self.execute(service, Path(directory), urls, max_steps=2)
            self.assertEqual(result['status'], 'timed_out')
            self.assertTrue(result['cancel_requested'])
            self.assertFalse(result['cancellation_confirmed'])
            self.assertTrue(result['remote_job_may_still_run'])

    def test_fixture_skips_nonfinite_measurement_and_executes_baseline_without_nan_json(self):
        with tempfile.TemporaryDirectory() as directory, servers(service := Services(delta=float('nan'))) as urls:
            result = self.execute(service, Path(directory), urls)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['selection'], 'not_selected')
            self.assertEqual(result['actor_revision'], 'baseline')
            self.assertNotIn('delta_nll', result['metrics'])
            self.assertEqual(result['metrics']['measurement_valid'], 0)
            self.assertTrue(result['answer_correct'])
            json.dumps(result, allow_nan=False)



class QwenMultiRoundWorkerTests(unittest.TestCase):
    def fixture(self):
        spec = importlib.util.spec_from_file_location('qwen_worker_fixture', ROOT / 'tests/experiments/qwen_toy/test_qwen_worker.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.WorkerFixture()

    def test_failed_or_cancelled_second_training_keeps_parent_and_blocks_more_attempts(self):
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled):
                fixture = self.fixture()
                def scenario():
                    parent = {'backend_id': 'qwen-toy', 'model_id': 'Qwen3.5-9B', 'revision': 'baseline'}
                    payload = {'idempotency_key': 'train-1', 'request': {'base': parent}}
                    code, first = fixture.post('/train', payload)
                    self.assertEqual(code, 200)
                    fixture.drain()
                    self.assertEqual(fixture.job(first['job_id'])['status'], 'succeeded')
                    self.assertEqual(fixture.post('/train', {'idempotency_key': 'missing-parent'})[0], 400)
                    unknown = {'idempotency_key': 'unknown-parent', 'request': {'base': dict(parent, revision='unknown')}}
                    self.assertEqual(fixture.post('/train', unknown)[0], 400)
                    second_payload = {'idempotency_key': 'train-2',
                                      'request': {'base': dict(parent, revision='candidate-1')}}
                    code, second = fixture.post('/train', second_payload)
                    self.assertEqual(code, 200)
                    if cancelled:
                        fixture.post('/cancel', {'job_id': second['job_id']})
                    else:
                        fixture.trainer.train.side_effect = RuntimeError('fixture training failure')
                    fixture.drain()
                    self.assertEqual(fixture.job(second['job_id'])['status'], 'cancelled' if cancelled else 'failed')
                    self.assertEqual(fixture.get('/health')['revisions'], ['candidate-1'])
                    self.assertEqual(fixture.get('/health')['revision'], 'candidate-1')
                    self.assertEqual(fixture.post('/train', payload), (200, first))
                    self.assertEqual(fixture.post('/train', dict(second_payload, idempotency_key='third'))[0], 400)
                    code, response = fixture.post('/v1/chat/completions', {'model': 'Qwen3.5-9B',
                        'model_revision': 'candidate-1', 'messages': [{'role': 'user', 'content': 'p'}]})
                    self.assertEqual(code, 200)
                    self.assertEqual(response['model_revision'], 'candidate-1')
                    self.assertFalse((fixture.root / 'round-2/train-result.json').exists())
                fixture.run(scenario, '--rounds', '2')

    def test_worker_copies_parent_weights_preserves_old_revision_and_compares_parent(self):
        from contextlib import contextmanager
        from types import SimpleNamespace
        fixture = self.fixture()
        weights, active = {'default': 0}, ['default']
        def select(name):
            active[0] = name
        def train():
            weights[active[0]] += 1
            return SimpleNamespace(metrics={'train_loss': 0.5})
        def save(path, selected_adapters=None):
            name = (selected_adapters or ['default'])[0]
            target = Path(path) if name == 'default' else Path(path) / name
            target.mkdir(parents=True, exist_ok=True)
            (target / 'fixture-weight.json').write_text(json.dumps(weights[name]))
        def load(path, *, adapter_name, is_trainable, local_files_only):
            self.assertTrue(is_trainable and local_files_only)
            weights[adapter_name] = json.loads((Path(path) / 'fixture-weight.json').read_text())
        @contextmanager
        def disable():
            previous = active[0]
            active[0] = None
            try:
                yield
            finally:
                active[0] = previous
        fixture.model.set_adapter.side_effect = select
        fixture.model.save_pretrained.side_effect = save
        fixture.model.load_adapter.side_effect = load
        fixture.model.disable_adapter.side_effect = disable
        fixture.model.side_effect = lambda **kw: SimpleNamespace(loss=2.0 - 0.5 * weights.get(active[0], 0))
        fixture.trainer.train.side_effect = train
        def scenario():
            first_path = None
            for number, parent in ((1, 'baseline'), (2, 'candidate-1')):
                base = {'backend_id': 'qwen-toy', 'model_id': 'Qwen3.5-9B', 'revision': parent}
                payload = {'idempotency_key': f'train-{number}', 'request': {'base': base, 'method': 'sft'}}
                code, receipt = fixture.post('/train', payload)
                self.assertEqual(code, 200)
                self.assertEqual(fixture.post('/train', payload), (code, receipt))
                fixture.drain()
                outcome = fixture.job(receipt['job_id'])
                self.assertEqual(outcome['status'], 'succeeded')
                trained = outcome['result']
                self.assertEqual(trained['base_revision'], parent)
                if number == 1:
                    first_path = trained['adapter_path']
                else:
                    self.assertEqual(json.loads((Path(first_path) / 'fixture-weight.json').read_text()), 1)
                    self.assertEqual(weights, {'default': 1, 'candidate-2': 2})
                candidate = {'candidate_id': f'candidate-{number}', 'base_version': parent}
                code, receipt = fixture.post('/evaluate', {'idempotency_key': f'eval-{number}',
                    'request': {'candidate': candidate, 'protocol_ref': 'heldout-nll'}})
                self.assertEqual(code, 200)
                fixture.drain()
                evaluated = fixture.job(receipt['job_id'])['result']
                self.assertEqual(evaluated['metrics']['base_nll'], 2.0 if number == 1 else 1.5)
                self.assertEqual(evaluated['metrics']['candidate_nll'], 1.5 if number == 1 else 1.0)
            for revision in ('candidate-1', 'candidate-2'):
                code, response = fixture.post('/v1/chat/completions', {'model': 'Qwen3.5-9B',
                    'model_revision': revision, 'messages': [{'role': 'user', 'content': 'p'}]})
                self.assertEqual(code, 200)
                self.assertEqual(response['model_revision'], revision)
                self.assertEqual(active[0], 'default' if revision == 'candidate-1' else 'candidate-2')
            self.assertEqual(fixture.get('/health')['revisions'], ['candidate-1', 'candidate-2'])
            self.assertEqual(fixture.post('/train', {'idempotency_key': 'third'})[0], 400)
            self.assertTrue((fixture.root / 'train-result.json').exists())
            self.assertTrue((fixture.root / 'round-2/train-result.json').exists())
            self.assertTrue((fixture.root / 'round-2/evaluate-result.json').exists())
        fixture.run(scenario, '--rounds', '2')


class QwenAdapterTests(unittest.TestCase):
    def test_multi_round_adapter_retries_keep_keys_and_old_revisions_routable(self):
        with servers(service := Services()) as urls:
            baseline, worker = (HTTPClient(url, Deadline(10)) for url in urls)
            base = ModelRef('qwen-toy', 'Qwen3.5-9B', 'baseline')
            models = RevisionBackend(baseline, worker, base)
            trainer, activator = WorkerTrainer(worker, models), WorkerActivator(models)
            request = ActorTrainingRequest(base, (), 'toy-lora', method='sft')
            first_job = trainer.submit(request, idempotency_key='train-1')
            trainer.poll(first_job)
            first = trainer.poll(first_job).result
            activator.activate(first, idempotency_key='apply-1')
            self.assertEqual(trainer.submit(request, idempotency_key='train-1'), first_job)
            with self.assertRaises(ValueError):
                trainer.submit(request, idempotency_key='stale-base')
            second_job = trainer.submit(replace(request, base=first.model), idempotency_key='train-2')
            trainer.poll(second_job)
            second = trainer.poll(second_job).result
            activator.activate(second, idempotency_key='apply-2')
            self.assertEqual(second.base, first.model)
            self.assertEqual(models.active, second.model)
            old = models.generate(GenerationRequest('actor', first.model, 'parity 113'))
            self.assertTrue(old.revision_verified)
            self.assertEqual(old.model, first.model)
            activator.activate(first, idempotency_key='apply-1')
            self.assertEqual(models.active, second.model)

    def test_pending_submission_freezes_request_and_rejects_changed_request_or_key(self):
        with servers(service := Services(response_losses={'train': 2})) as urls:
            baseline, worker = (HTTPClient(url, Deadline(10)) for url in urls)
            base = ModelRef('qwen-toy', 'Qwen3.5-9B', 'baseline')
            trainer = WorkerTrainer(worker, RevisionBackend(baseline, worker, base))
            request = ActorTrainingRequest(base, (), 'toy-lora', method='sft')
            with self.assertRaises(OSError):
                trainer.submit(request, idempotency_key='original')
            pending = trainer.pending_submission
            self.assertTrue(trainer.uncertain_submission)
            self.assertEqual(pending.request, request)
            self.assertIsNot(pending.request, request)
            for changed, key in ((request, 'new-key'), (replace(request, data_refs=(TRAIN_REF,)), 'original')):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'pending submission'):
                    trainer.submit(changed, idempotency_key=key)
            self.assertEqual(len([c for c in service.calls if c[1] == '/train']), 2)
            trainer.worker.deadline.end = 0
            cleanup = HTTPClient(urls[1], Deadline(5), timeout=5)
            job = trainer.recover_submission(cleanup)
            self.assertEqual(job.job_id, service.keys['original'])
            self.assertFalse(trainer.uncertain_submission)
            self.assertIsNone(trainer.pending_submission)
            self.assertEqual(trainer.submit(request, idempotency_key='original'), job)
            self.assertEqual(trainer.requests[job], request)
            self.assertEqual(len(service.jobs), 1)

    def test_async_results_revision_routing_idempotency_and_cancel_contract(self):
        with servers(service := Services()) as urls:
            deadline = Deadline(10)
            baseline, worker = (HTTPClient(url, deadline) for url in urls)
            base = ModelRef('qwen-toy', 'Qwen3.5-9B', 'baseline')
            models = RevisionBackend(baseline, worker, base)
            trainer = WorkerTrainer(worker, models)
            request = ActorTrainingRequest(base, (), 'toy-lora', method='sft')
            job = trainer.submit(request, idempotency_key='same')
            self.assertEqual(trainer.submit(request, idempotency_key='same'), job)
            with self.assertRaises(ValueError):
                trainer.submit(replace(request, recipe_ref='other'), idempotency_key='same')
            with self.assertRaises(RuntimeError):
                trainer.result(job)
            self.assertEqual(trainer.poll(job).status, JobStatus.RUNNING)
            trained = trainer.poll(job).result
            self.assertEqual(trainer.result(job), trained)
            self.assertFalse(trained.simulated)
            self.assertEqual(trained.model.backend_id, base.backend_id)
            self.assertEqual(trainer.poll(job).result, trained)
            evaluator = WorkerEvaluator(worker, models)
            evaluation = evaluator.submit(EvaluationRequest(trained.candidate, 'heldout-nll'), idempotency_key='eval')
            evaluator.cancel(evaluation)
            self.assertEqual(evaluator.poll(evaluation).status, JobStatus.CANCELLED)
            baseline_result = models.generate(GenerationRequest('actor', base, 'parity 113'))
            self.assertFalse(baseline_result.revision_verified)
            activator = WorkerActivator(models)
            self.assertEqual(activator.activate(trained, idempotency_key='apply'), trained.model)
            count = len(service.calls)
            self.assertEqual(activator.activate(trained, idempotency_key='apply'), trained.model)
            self.assertEqual(len(service.calls), count)
            with self.assertRaises(ValueError):
                models.generate(GenerationRequest('actor', replace(base, revision='unknown'), 'test'))

    def test_urls_with_embedded_secrets_and_expired_deadlines_are_rejected(self):
        for url in ('http://user:password@localhost', 'http://localhost?api_key=secret',
                    'http://localhost#secret', 'file:///private', 'http://localhost/unexpected'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                HTTPClient(url, Deadline(1))
        deadline = Deadline(1)
        deadline.end = 0
        with self.assertRaises(TimeoutError):
            deadline.remaining()


if __name__ == '__main__':
    unittest.main()
