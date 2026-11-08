from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from rsisys.application import Application, ApplicationConfig
from rsisys.loops._memory import Memory
from rsisys.loops.decision_making.harness.memory import DecisionMemory
from rsisys.loops.exploration.harness.memory import ExplorationMemory
from rsisys.experience.records import ExperienceRecord
from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.sqlite import SQLiteExperienceStore
from rsisys.loops.contracts.jobs import JobStatus
from rsisys.loops.contracts.model_service import BackendBinding, ModelRef, GenerationRequest
from rsisys.loops.exploration.harness.learner import ActorTrainingRequest
from rsisys.runtime.artifacts import FileArtifactStore
from rsisys.runtime.local_process import LocalProcessBackend
from rsisys.runtime.model import RemoteModelBackend, ModelResponseError, ModelTransportError
from rsisys.runtime.sandbox import SandboxSpec, ExecRequest
from rsisys.snapshot.local import LocalSnapshotCapture, restore_snapshot, load_manifest
from rsisys.snapshot.codec import decode


def record(ident='record', scope='actor', summary='experience'):
    return ExperienceRecord(ident, 'runtime_feedback', scope, 'directive', summary, False)


@contextmanager
def model_server():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            self.server.requests.append((self.path, payload, self.headers.get('Authorization')))
            if self.server.delay:
                time.sleep(self.server.delay)
            context = json.loads(payload['messages'][-1]['content'])
            if 'operations' in context:
                content = ({'kind': 'invoke', 'operation': 'prepare_data', 'arguments': {'text': context['objective']}}
                           if not context['operation_results'] else {'kind': 'explore', 'strategy': 'execute prepared task'})
            else:
                content = {'name': 'echo', 'arguments': {'text': context.get('observation', 'hello')}}
            data = {'model': payload['model'], 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(content)}}],
                    'usage': {'prompt_tokens': 7, 'completion_tokens': 3}}
            data.update(self.server.overrides)
            self.send_response(self.server.status)
            if self.server.status == 302:
                self.send_header('Location', '/redirected')
            self.end_headers()
            try:
                self.wfile.write(self.server.raw if self.server.raw is not None else json.dumps(data).encode())
            except OSError:
                pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.requests, server.overrides = [], {}
    server.status, server.raw, server.delay = 200, None, 0
    worker = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
    worker.start()
    try:
        yield server, f'http://127.0.0.1:{server.server_port}/v1'
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


class FoundationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def config(self, name='run', execution='mock', learning='none', endpoint=None):
        backend = BackendBinding('model', 'chat_completions', endpoint) if endpoint else BackendBinding('model', 'mock')
        model = ModelRef('model', 'served-model', 'deployment-label')
        return ApplicationConfig('Inspect a prepared text artifact', self.root/name,
                                 {'model': backend}, {'decision_maker': model, 'actor': model},
                                 execution=execution, learning=learning)

    def test_artifacts_survive_reopen_and_relocation(self):
        store = FileArtifactStore(self.root/'artifacts')
        ref = store.put(b'persisted content', kind='input')
        self.assertEqual(FileArtifactStore(store.root).get(ref), b'persisted content')
        shutil.copytree(store.root, self.root/'relocated')
        self.assertEqual(FileArtifactStore(self.root/'relocated').get(ref), b'persisted content')
        source = self.root/'large.bin'
        source.write_bytes(b'x' * (2*1024*1024 + 5))
        file_ref = store.put_file(source, kind='checkpoint')
        self.assertEqual(store.resolve(file_ref).stat().st_size, source.stat().st_size)

    def test_artifacts_detect_corruption_and_path_escape(self):
        store = FileArtifactStore(self.root/'artifacts')
        ref = store.put(b'original', kind='input')
        with self.assertRaises(ValueError):
            store.get(replace(ref, artifact_id='../outside'))
        with self.assertRaises(ValueError):
            store.get(replace(ref, kind='wrong'))
        store.resolve(ref).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'digest'):
            store.get(ref)

    def test_sqlite_reopen_queries_and_idempotent_records(self):
        path = self.root/'experience.sqlite3'
        store = SQLiteExperienceStore(path)
        original = record()
        store.append(original)
        store.append(original)
        with self.assertRaises(ValueError):
            store.append(replace(original, summary='different'))
        store.append(record('shared', 'shared', 'shared result'))
        store.append(record('decision', 'decision_maker', 'private decision'))
        self.assertEqual([r.record_id for r in store.query('actor', limit=1)], ['shared'])
        self.assertEqual([r.record_id for r in store.query('actor', text='experience')], ['record'])
        store.close()
        reopened = SQLiteExperienceStore(path)
        try:
            self.assertEqual(len(reopened.records()), 3)
        finally:
            reopened.close()

    def test_memory_is_owned_by_each_harness_and_persisted_separately(self):
        path = self.root/'decision_memory.json'
        memory = DecisionMemory(path)
        note = Memory('memory', 'procedural', 'Inspect outputs before accepting', ('source',), 'execution')
        memory.remember(note)
        memory.remember(replace(note, status='supported'))
        self.assertEqual(DecisionMemory(path).recall(text='outputs')[0].status, 'supported')
        self.assertEqual(ExplorationMemory().recall(), ())
        with self.assertRaises(ValueError):
            ExplorationMemory(path)

    def test_local_command_exports_actual_output_and_filters_environment(self):
        backend = LocalProcessBackend(self.root/'work')
        session = backend.create(SandboxSpec('local'))
        with patch.dict(os.environ, {'RSISYS_TEST_SECRET': 'not-for-child'}):
            process = backend.exec(session, ExecRequest((sys.executable, '-c',
                'import os; print("actual-output"); print(os.getenv("RSISYS_TEST_SECRET"))')))
        result = backend.wait(process)
        self.assertEqual(result.status, JobStatus.SUCCEEDED)
        self.assertEqual(result.stdout, 'actual-output\nNone\n')
        self.assertFalse(result.simulated)
        directory = backend.workspace(session)
        backend.close(session)
        self.assertFalse(directory.exists())
        self.assertEqual(backend.poll(process), result)

    def test_local_timeout_and_cancel_are_terminal(self):
        backend = LocalProcessBackend(self.root/'work')
        session = backend.create(SandboxSpec('local'))
        try:
            process = backend.exec(session, ExecRequest((sys.executable, '-c', 'import time; time.sleep(10)'), 0.05))
            result = backend.wait(process)
            self.assertEqual(result.status, JobStatus.FAILED)
            self.assertIn('timed out', result.stderr)
            other = backend.exec(session, ExecRequest((sys.executable, '-c', 'import time; time.sleep(10)')))
            backend.cancel(other)
            self.assertEqual(backend.wait(other).status, JobStatus.CANCELLED)
        finally:
            backend.close(session)

    def test_local_output_is_bounded_and_close_stops_children(self):
        backend = LocalProcessBackend(self.root/'work', max_output_bytes=1024)
        session = backend.create(SandboxSpec('local'))
        output = backend.exec(session, ExecRequest((sys.executable, '-c', 'print("x" * 100000)')))
        result = backend.wait(output)
        self.assertEqual(len(result.stdout), 1024)
        self.assertIn('truncated', result.stderr)
        pending = backend.exec(session, ExecRequest((sys.executable, '-c', 'import time; time.sleep(10)')))
        backend.close(session)
        self.assertEqual(backend.poll(pending).status, JobStatus.CANCELLED)
        self.assertEqual(backend.active_count, 0)

    def test_command_environment_exports_files_before_cleanup(self):
        from rsisys.loops.contracts.directive import Directive
        from rsisys.loops.exploration.action_space import ActionProposal
        from rsisys.loops.exploration.action_space.local import CommandActionSpace
        from rsisys.loops.exploration.harness.environment.local import LocalProcessEnvironment
        artifacts = FileArtifactStore(self.root/'artifacts')
        backend = LocalProcessBackend(self.root/'work')
        environment = LocalProcessEnvironment(backend, artifacts)
        environment.reset(Directive('file-task', 'write a file', 'execute', allowed_actions=('execute',)))
        action = CommandActionSpace().validate(ActionProposal('execute', {
            'argv': [sys.executable, '-c', 'from pathlib import Path; Path("out.txt").write_text("result")'],
            'output_paths': ['out.txt'],
        }, False), ('execute',))
        transition = environment.step(action)
        environment.close()
        self.assertIsNone(transition.error)
        self.assertEqual(artifacts.get(transition.artifacts[1]), b'result')
        self.assertEqual(backend.active_count, 0)

    def test_snapshot_refuses_unrecorded_component_reconfiguration(self):
        with Application(self.config()) as app:
            app.runner.decision_making.harness.policy.allowed_actions = ('different',)
            with self.assertRaisesRegex(TypeError, 'bindings'):
                LocalSnapshotCapture(app).capture('unsupported')

    def test_snapshot_rejects_custom_execution_components_before_writing(self):
        with Application(self.config()) as app:
            actor = app.runner.exploration.harness
            for name in ('episodes', 'learning', 'evaluations', 'local_rl'):
                with self.subTest(component=name):
                    original = getattr(actor, name)
                    try:
                        setattr(actor, name, object())
                        with self.assertRaisesRegex(TypeError, 'execution components'):
                            LocalSnapshotCapture(app).capture('custom-execution')
                        self.assertFalse((app.config.data_dir / 'snapshots').exists())
                    finally:
                        setattr(actor, name, original)

    def test_snapshot_rejects_divergent_execution_state(self):
        from rsisys.loops.exploration.harness.state import ExplorationState
        with Application(self.config()) as app:
            actor = app.runner.exploration.harness
            for component in (actor.learning, actor.evaluations, actor.local_rl):
                with self.subTest(component=type(component).__name__), patch.object(component, 'state', ExplorationState()):
                    with self.assertRaisesRegex(TypeError, 'execution component bindings'):
                        LocalSnapshotCapture(app).capture('divergent-execution')
                    self.assertFalse((app.config.data_dir / 'snapshots').exists())

    def test_snapshot_restores_query_result_and_rebinds_provider(self):
        from rsisys.loops.decision_making.action_space import OperationCall
        from rsisys.loops.decision_making.harness.decision import Decision
        with Application(self.config()) as app:
            dm = app.runner.decision_making.harness
            call = OperationCall('query_exploration_model', {})
            with patch.object(dm.policy, 'decide', return_value=Decision('invoke', operation=call)):
                app.runner.step(app.config.objective)
            observed = dm.state.operation_results[-1].data
            manifest = LocalSnapshotCapture(app).capture('query-result')
            directory = app.config.data_dir / 'snapshots' / manifest.snapshot_id
            dm.tools._bindings[call.name] = lambda args: None
            with self.assertRaisesRegex(TypeError, 'bindings'):
                LocalSnapshotCapture(app).capture('custom-query')
        with restore_snapshot(directory, self.root / 'query-restored') as restored:
            dm = restored.runner.decision_making.harness
            self.assertEqual(dm.state.operation_results[-1].data, observed)
            self.assertIs(dm.tools.query_tools.provider, restored.runner.exploration)
            self.assertEqual(dm.tools.execute(call).data['data']['items'], observed['data']['items'])

    def test_local_paths_reject_traversal_and_symlink_escape(self):
        backend = LocalProcessBackend(self.root/'work')
        session = backend.create(SandboxSpec('local'))
        try:
            with self.assertRaises(ValueError):
                backend.file_path(session, '../outside')
            (backend.workspace(session)/'escape').symlink_to(self.root, target_is_directory=True)
            with self.assertRaises(ValueError):
                backend.file_path(session, 'escape/file')
        finally:
            backend.close(session)

    def test_http_model_messages_auth_usage_and_revision_uncertainty(self):
        with model_server() as (server, endpoint), patch.dict(os.environ, {'MODEL_TEST_KEY': 'fixture-key'}):
            backend = RemoteModelBackend(BackendBinding('model', 'http', endpoint, 'MODEL_TEST_KEY'))
            request = GenerationRequest('actor', ModelRef('model', 'served-model', 'deployment-label'), '{}', system='Return JSON')
            response = backend.generate(request)
            self.assertFalse(response.simulated)
            self.assertFalse(response.revision_verified)
            self.assertEqual(response.usage.input_tokens, 7)
            path, payload, auth = server.requests[0]
            self.assertEqual(path, '/v1/chat/completions')
            self.assertEqual(auth, 'Bearer fixture-key')
            self.assertEqual(payload['messages'][0]['content'], 'Return JSON')
            self.assertEqual(payload['response_format'], {'type': 'json_object'})

    def test_http_rejects_wrong_identity_truncation_and_bad_json(self):
        with model_server() as (server, endpoint):
            backend = RemoteModelBackend(BackendBinding('model', 'http', endpoint))
            request = GenerationRequest('actor', ModelRef('model', 'served-model', 'v1'), '{}')
            server.overrides = {'model': 'other'}
            with self.assertRaises(ModelResponseError):
                backend.generate(request)
            server.overrides = {'model_revision': 'different'}
            with self.assertRaises(ModelResponseError):
                backend.generate(request)
            server.overrides = {'choices': [{'finish_reason': 'length', 'message': {'content': '{}'}}]}
            with self.assertRaises(ModelResponseError):
                backend.generate(request)
            server.overrides = {}
            server.raw = b'not json'
            with self.assertRaises(ModelResponseError):
                backend.generate(request)

    def test_http_errors_do_not_expose_response_body_or_follow_redirects(self):
        with model_server() as (server, endpoint):
            backend = RemoteModelBackend(BackendBinding('model', 'http', endpoint))
            request = GenerationRequest('actor', ModelRef('model', 'served-model', 'v1'), '{}')
            server.status, server.raw = 401, b'sensitive server detail'
            with self.assertRaises(ModelTransportError) as caught:
                backend.generate(request)
            self.assertNotIn('sensitive', str(caught.exception))
            server.status = 302
            with self.assertRaises(ModelTransportError):
                backend.generate(request)
            self.assertEqual(len(server.requests), 2)

    def test_http_missing_usage_is_unknown_and_timeouts_are_explicit(self):
        with model_server() as (server, endpoint):
            request = GenerationRequest('actor', ModelRef('model', 'served-model', 'v1'), '{}')
            backend = RemoteModelBackend(BackendBinding('model', 'http', endpoint, timeout_seconds=0.03))
            server.overrides = {'usage': None}
            self.assertIsNone(backend.generate(request).usage)
            server.delay = 0.1
            with self.assertRaises(ModelTransportError):
                backend.generate(request)

    def test_http_size_limit_and_missing_credential_fail_explicitly(self):
        with model_server() as (server, endpoint):
            request = GenerationRequest('actor', ModelRef('model', 'served-model', 'v1'), '{}')
            backend = RemoteModelBackend(BackendBinding('model', 'http', endpoint, max_response_bytes=8))
            with self.assertRaises(ModelResponseError):
                backend.generate(request)
            backend = RemoteModelBackend(BackendBinding('model', 'http', endpoint, 'UNSET_RSISYS_TEST_KEY'))
            with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
                backend.generate(request)
            self.assertEqual(len(server.requests), 1)

    def test_http_and_real_execution_work_together(self):
        with model_server() as (_, endpoint), Application(self.config(execution='local_process', endpoint=endpoint)) as app:
            result = app.run()
            self.assertEqual(result.status, 'completed')
            self.assertFalse(any(r.simulated for r in app.store.records()))
            self.assertEqual(app.runner.decision_making.harness.state.measured_input_tokens, 14)
            self.assertEqual(app.runner.exploration.harness.state.measured_output_tokens, 3)
            self.assertEqual(app.runner.exploration.harness.state.calls_with_missing_usage, 0)
            self.assertTrue(all(r.run_id == app.config.run_id for r in app.store.records()))

    def test_snapshot_restore_forks_exact_experience_and_memory(self):
        with Application(self.config(execution='local_process')) as app:
            app.runner.decision_making.harness.review_tool_limit = 2
            app.run()
            evidence = app.store.records()[0].record_id
            app.runner.exploration.harness.memory.remember(Memory('m', 'semantic', 'Persist evidence', (evidence,), 'task'))
            app.runner.decision_making.harness.memory.remember(Memory('m', 'procedural', 'Compare training efficiency'))
            manifest = LocalSnapshotCapture(app).capture('first')
            directory = app.config.data_dir/'snapshots'/manifest.snapshot_id
            saved_count = len(app.store.records())
            saved_refs = app.artifacts.refs()
            app.run()
            self.assertGreater(len(app.store.records()), saved_count)
        with restore_snapshot(directory, self.root/'restored') as restored:
            self.assertEqual(restored.runner.decision_making.harness.review_tool_limit, 2)
            self.assertEqual(len(restored.store.records()), saved_count)
            self.assertEqual(restored.runner.exploration.harness.memory.recall()[0].content, 'Persist evidence')
            self.assertEqual(restored.runner.decision_making.harness.memory.recall()[0].content, 'Compare training efficiency')
            self.assertTrue(restored.runner.exploration.harness.memory.history())
            self.assertTrue(restored.runner.decision_making.harness.memory.history())
            for ref in saved_refs:
                self.assertTrue(restored.artifacts.get(ref))
            self.assertEqual(restored.run().status, 'completed')
            self.assertEqual(restored.runner.decision_making.harness.state.completed, 2)

    def test_snapshot_preserves_preparation_without_reexecuting_it(self):
        with Application(self.config()) as app:
            self.assertEqual(app.run(max_decision_steps=1).status, 'step_limit')
            refs = app.artifacts.refs()
            manifest = LocalSnapshotCapture(app).capture('prepared')
            directory = app.config.data_dir/'snapshots'/manifest.snapshot_id
        with restore_snapshot(directory, self.root/'resumed') as restored:
            self.assertEqual(restored.run().decision_steps, 1)
            self.assertEqual(len([r for r in restored.artifacts.refs() if r.kind == 'input_data']), 1)
            self.assertEqual(restored.runner.exploration.harness.state.last_feedback.directive_id,
                             restored.runner.decision_making.harness.state.last_feedback.directive_id)
            self.assertEqual(restored.artifacts.get(refs[0]), b'Inspect a prepared text artifact')

    def test_snapshot_preserves_applied_model_binding_but_archives_old_jobs(self):
        with Application(self.config(learning='mock')) as app:
            actor = app.runner.exploration
            data = app.artifacts.put(b'data', kind='training_data')
            job = actor.harness.submit_learning(ActorTrainingRequest(actor.harness.policy.model, (data,), 'recipe'), idempotency_key='train')
            for _ in range(3):
                result = actor.harness.poll_learning(job)
                if result.terminal:
                    break
            model = actor.harness.apply_learning(job)
            manifest = LocalSnapshotCapture(app).capture('trained')
            directory = app.config.data_dir/'snapshots'/manifest.snapshot_id
        with restore_snapshot(directory, self.root/'restored') as restored:
            self.assertEqual(restored.runner.exploration.harness.policy.model, model)
            with self.assertRaises(ValueError):
                restored.runner.exploration.harness.poll_learning(job)
            self.assertEqual(restored.run().status, 'completed')

    def test_snapshot_rejects_pending_work(self):
        with Application(self.config()) as app:
            app.runner.decision_making.run(app.config.objective)
            with self.assertRaisesRegex(RuntimeError, 'pending'):
                LocalSnapshotCapture(app).capture('invalid')
        with Application(self.config('training', learning='mock')) as app:
            actor = app.runner.exploration
            data = app.artifacts.put(b'data', kind='training_data')
            actor.harness.submit_learning(ActorTrainingRequest(actor.harness.policy.model, (data,), 'recipe'), idempotency_key='train')
            with self.assertRaisesRegex(RuntimeError, 'idle'):
                LocalSnapshotCapture(app).capture('invalid')

    def test_snapshot_corruption_is_rejected_before_creating_target(self):
        with Application(self.config()) as app:
            app.run()
            manifest = LocalSnapshotCapture(app).capture('valid')
            directory = app.config.data_dir/'snapshots'/manifest.snapshot_id
        (directory/'state.json').write_text('{}')
        target = self.root/'restored'
        with self.assertRaisesRegex(ValueError, 'digest'):
            restore_snapshot(directory, target)
        self.assertFalse(target.exists())
        with self.assertRaises(ValueError):
            decode({'$type': 'os.system', 'fields': {'cmd': 'bad'}})

    def test_workspace_lock_and_restore_never_overwrite_existing_directory(self):
        config = self.config()
        with Application(config) as app:
            with self.assertRaisesRegex(RuntimeError, 'already open'):
                Application(config)
            manifest = LocalSnapshotCapture(app).capture('idle')
            directory = app.config.data_dir/'snapshots'/manifest.snapshot_id
            with self.assertRaises(FileExistsError):
                restore_snapshot(directory, config.data_dir)
        with self.assertRaisesRegex(ValueError, 'not empty'):
            Application(config)

    def test_local_literature_operations_and_memory_reach_policy_context(self):
        with Application(self.config()) as app:
            ref = app.artifacts.put(b'Example paper: reusable training trajectories', kind='literature')
            from rsisys.loops.decision_making.action_space import OperationCall
            result = app.runner.decision_making.harness.tools.execute(OperationCall('search_literature', {'query': 'trajectories'}))
            matches = json.loads(app.artifacts.get(result.artifacts[0]))
            self.assertEqual(matches[0]['artifact_id'], ref.artifact_id)
            app.experience.record(record('evidence'))
            app.runner.decision_making.harness.memory.remember(Memory('m', 'procedural', 'Check results', ('evidence',), 'task'))
            policy = app.runner.decision_making.harness.policy
            with patch.object(policy, 'decide', wraps=policy.decide) as decide:
                app.runner.decision_making.step(app.config.objective)
                self.assertEqual(decide.call_args.args[0].memories[0].memory_id, 'm')

    def test_cli_new_run_inspect_and_resume_in_separate_processes(self):
        config = self.config(execution='local_process')
        path = self.root/'config.json'
        path.write_text(json.dumps(config.to_dict()))
        def cli(*args):
            result = subprocess.run([sys.executable, '-m', 'rsisys', *args], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        first = cli('run', '--config', str(path), '--max-decision-steps', '1')
        self.assertEqual(first['status'], 'step_limit')
        self.assertTrue(first['snapshot'])
        inspect = cli('inspect', '--snapshot', first['snapshot'])
        self.assertEqual(inspect['kind'], 'local_quiescent')
        resumed = cli('resume', '--snapshot', first['snapshot'], '--data-dir', str(self.root/'cli-resumed'))
        self.assertEqual(resumed['status'], 'completed')
        self.assertEqual(resumed['run_id'], first['run_id'])
        self.assertEqual(resumed['decision_steps'], 1)


if __name__ == '__main__':
    unittest.main()
