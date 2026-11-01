"""CPU-only HTTP contract fixtures; these tests do not claim to run real Qwen."""
from contextlib import contextmanager, redirect_stdout
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.qwen_toy.adapters import (
    Deadline, HTTPClient, RevisionBackend, TRAIN_REF, WorkerActivator, WorkerEvaluator, WorkerTrainer,
)
from examples.qwen_toy.run import eligible, parse_args, run
from rsisys.experience.sqlite import SQLiteExperienceStore
from rsisys.loops.contracts.jobs import JobStatus
from rsisys.loops.contracts.model_service import GenerationRequest, ModelRef
from rsisys.loops.exploration.harness.evaluation import EvaluationRequest
from rsisys.loops.exploration.harness.learner import ActorTrainingRequest


class Services:
    def __init__(self, *, delta=0.5, wrong_revision=False, bad_decisions=False,
                 forever=False, failed=False, wrong_activation_json=False,
                 poll_delay=0, cancel_pending=False, invoke_first=False,
                 response_losses=None, lost_response_delay=0, wrong_revision_round=None, max_rounds=2, review_action='continue'):
        self.delta, self.wrong_revision, self.bad_decisions = delta, wrong_revision, bad_decisions
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
                delta = self.delta[int(self.latest_revision.split('-')[-1])-1] if isinstance(self.delta, list) else self.delta
                objective = context['objective']
                phase = objective.split('Current workflow phase: ')[1].split('.')[0]
                if self.bad_decisions:
                    value = {'kind': 'finish', 'reason': 'skip all work'}
                elif self.invoke_first and not self.decisions:
                    value = {'kind': 'invoke', 'operation': 'train',
                             'arguments': {'recipe_ref': 'toy-lora', 'method': 'sft', 'apply_result': False}}
                elif phase == 'train':
                    value = {'kind': 'explore', 'strategy': 'train once', 'mode': 'train',
                             'arguments': {'recipe_ref': 'toy-lora', 'method': 'sft'}}
                elif phase == 'evaluate':
                    value = {'kind': 'explore', 'strategy': 'heldout validation', 'mode': 'evaluate',
                             'arguments': {'candidate_id': self.latest_revision,
                                           'protocol_ref': 'heldout-nll'}}
                elif phase == 'select' and delta > 0:
                    assert context['last_feedback']['candidates'] == []
                    value = {'kind': 'explore', 'strategy': 'positive heldout delta', 'mode': 'apply_candidate',
                             'arguments': {'candidate_id': self.latest_revision}}
                elif phase in ('select', 'execute'):
                    value = {'kind': 'explore', 'strategy': 'execute chosen model', 'mode': 'execute'}
                else:
                    value = {'kind': 'finish', 'reason': 'one cycle complete'}
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
    def execute(self, service, root, urls, *, max_steps=100, timeout=10, rounds=1, review_after=2, require_review=False):
        args = parse_args(['--baseline-url', urls[0] + '/v1', '--worker-url', urls[1],
                           '--run-dir', str(root), '--poll-interval', '0.001',
                           '--timeout', str(timeout), '--max-steps', str(max_steps), '--rounds', str(rounds),
                           '--review-after-seconds', str(review_after)]
                          + (['--require-training-review'] if require_review else []))
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

    def test_round_rejection_preserves_current_actor_and_parent_metrics(self):
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
                                 ['applied' if delta > 0 else 'rejected' for delta in deltas])

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
            self.assertEqual([r['training_review_count'] for r in result['round_results']], [1, 1])
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
                forever=True, poll_delay=0.03, review_action='cancel')) as urls:
            result = self.execute(service, Path(directory), urls, rounds=2, review_after=0.005)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['training_reviews'][0]['action'], 'cancel')
            self.assertEqual(len(service.jobs), 1)
            self.assertEqual(result['actor_revision'], 'baseline')

    def test_round_argument_defaults_to_one_and_is_bounded(self):
        args = ['--baseline-url', 'http://localhost:1', '--worker-url', 'http://localhost:2']
        self.assertEqual(parse_args(args).rounds, 1)
        self.assertEqual(parse_args(args + ['--rounds', '2']).rounds, 2)
        for invalid in ('0', '3', '-1'):
            with self.subTest(invalid=invalid), redirect_stdout(io.StringIO()):
                from contextlib import redirect_stderr
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(args + ['--rounds', invalid])

    def test_real_protocol_adapters_accept_and_reject_paths_with_durable_evidence(self):
        for delta in (0.5, 0, -0.5):
            with self.subTest(delta=delta), tempfile.TemporaryDirectory() as directory:
                service, root = Services(delta=delta), Path(directory) / 'qwen-toy'
                with servers(service) as urls:
                    result = self.execute(service, root, urls)
                self.assertEqual(result['status'], 'completed')
                self.assertFalse(result['simulated'])
                self.assertEqual(result['selection'], 'accepted' if delta > 0 else 'rejected')
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
                self.assertEqual(monitor['candidates'][0]['status'], 'applied' if delta > 0 else 'rejected')
                self.assertEqual(monitor['models']['actor']['revision'], result['actor_revision'])
                # The exact web projection must consume the persisted schema.
                spec = importlib.util.spec_from_file_location('qwen_monitor_server', ROOT / 'web/server.py')
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                projected = module.project_snapshot(monitor, 'qwen-toy')
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

    def test_guard_cannot_replace_bad_model_decisions_with_scripted_training(self):
        with tempfile.TemporaryDirectory() as directory, servers(services := Services(bad_decisions=True)) as urls:
            output = io.StringIO()
            with redirect_stdout(output):
                result = self.execute(services, Path(directory), urls)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(len(services.decisions), 3)
            self.assertEqual(services.jobs, {})
            self.assertEqual(result['completed_directives'], 0)
            diagnostics = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual([entry['attempt'] for entry in diagnostics], [1, 2, 3])
            for entry in diagnostics:
                self.assertEqual(entry['event'], 'decision_validation_error')
                self.assertEqual(entry['phase'], 'train')
                self.assertEqual(entry['error_type'], 'ValueError')
                self.assertEqual(entry['error_summary'],
                                 'Expected kind="explore" with mode, strategy and arguments; received kind="finish"')
            monitor = (Path(directory) / 'monitor.json').read_text()
            self.assertNotIn('received kind=', monitor)
            self.assertNotIn('error_summary', monitor)

    def test_invoke_train_requires_a_new_model_decision_with_specific_retry_feedback(self):
        with tempfile.TemporaryDirectory() as directory, servers(service := Services(invoke_first=True)) as urls:
            output = io.StringIO()
            with redirect_stdout(output):
                result = self.execute(service, Path(directory), urls)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(service.decisions[0]['kind'], 'invoke')
            self.assertEqual(service.decisions[1]['kind'], 'explore')
            diagnostics = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(len(diagnostics), 1)
            chats = [payload for role, path, payload in service.calls
                     if role == 'baseline' and path == '/v1/chat/completions']
            second = json.loads(chats[1]['messages'][-1]['content'])['objective']
            self.assertIn('Validation feedback (data):', second)
            self.assertIn(json.dumps(diagnostics[0]['error_summary']), second)
            before_train = service.calls[:next(i for i, (_, path, _) in enumerate(service.calls) if path == '/train')]
            self.assertEqual(sum(path == '/v1/chat/completions' for _, path, _ in before_train), 2)
            self.assertEqual(len(service.jobs), 2)

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
        with tempfile.TemporaryDirectory() as directory, servers(service := Services(failed=True)) as urls:
            root = Path(directory)
            result = self.execute(service, root, urls)
            self.assertEqual(result['status'], 'failed')
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

    def test_nonfinite_measurements_reject_then_execute_baseline_without_nan_json(self):
        with tempfile.TemporaryDirectory() as directory, servers(service := Services(delta=float('nan'))) as urls:
            result = self.execute(service, Path(directory), urls)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['selection'], 'rejected')
            self.assertEqual(result['actor_revision'], 'baseline')
            self.assertNotIn('delta_nll', result['metrics'])
            self.assertEqual(result['metrics']['measurement_valid'], 0)
            self.assertTrue(result['answer_correct'])
            json.dumps(result, allow_nan=False)

    def test_guard_fails_closed_for_nonfinite_missing_and_nonpositive_values(self):
        for value in (None, True, '1', float('nan'), float('inf'), -float('inf'), 0, -1):
            with self.subTest(value=value):
                self.assertFalse(eligible({'delta_nll': value}))
        self.assertFalse(eligible({}))
        self.assertTrue(eligible({'delta_nll': 0.01}))


class QwenMultiRoundWorkerTests(unittest.TestCase):
    def fixture(self):
        spec = importlib.util.spec_from_file_location('qwen_worker_fixture', ROOT / 'tests/test_qwen_worker.py')
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
