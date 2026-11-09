"""Native search handshake, private scoring and bounded inference integration."""
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rsisys.controllers.aide import AIDEController, validate_config
from rsisys.controllers.research_bridge import ResearchBridge
from rsisys.labs.aide_program import (AIDEProgramLab, inference_broker,
                                     score_predictions, write_questions)
from rsisys.labs.harness_runtime import digest
from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.decision import DecisionInput, OperationCall


class AIDEAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.public = self.root / 'public'
        self.public.mkdir()
        self.validation = [{'id': 'v', 'prompt': 'validation one', 'answer': '#### 1'}]
        self.heldout = [{'id': 'h', 'prompt': 'heldout one', 'answer': '#### 1'}]
        write_questions(self.public / 'validation.csv', self.validation)
        with (self.public / 'train.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'prompt', 'answer'])
            writer.writeheader()
            writer.writerow({'id': 't', 'prompt': 'public example two', 'answer': '#### 2'})
        for name in ('validation', 'heldout'):
            (self.root / (name + '.jsonl')).write_text(json.dumps(getattr(self, name)[0]) + '\n')
        self.config = {'input_dir': str(self.public), 'validation_data': str(self.root / 'validation.jsonl'),
                       'heldout_data': str(self.root / 'heldout.jsonl'),
                       'endpoints': ['http://127.0.0.1:1/v1'], 'model': 'fixed'}

    def _candidate(self, source='pass\n'):
        p = self.root / 'candidate.py'
        p.write_text(source)
        return {'candidate': str(p), 'sha256': digest(p)}

    def test_strict_submission_ids_and_plain_numbers(self):
        path = self.root / 'predictions.csv'
        path.write_text('id,answer\nv,1\n')
        self.assertEqual(score_predictions(path, self.validation)['accuracy'], 1)
        path.write_text('id,answer\nv,1\nv,1\n')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            score_predictions(path, self.validation)
        path.write_text('id,answer\nother,1\n')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            score_predictions(path, self.validation)

    def test_lab_hides_labels_and_uses_frozen_heldout_only_for_final_call(self):
        seen = []
        def execute(workspace, broker):
            seen.append((workspace / 'input/validation.csv').read_text())
            self.assertNotIn('answer', seen[-1])
            self.assertEqual(sorted(p.name for p in (workspace / 'input').iterdir()),
                             ['train.csv', 'validation.csv'])
            with (workspace / 'input/validation.csv').open() as f:
                rows = list(csv.DictReader(f))
            (workspace / 'working/submission.csv').write_text('id,answer\n' + rows[0]['id'] + ',1\n')
            return {'term_out': ['untrusted accuracy=0.1'], 'exec_time': 0.01, 'exc_type': None,
                    'exc_info': None, 'exc_stack': None}
        lab = AIDEProgramLab(SimpleNamespace(data_dir=self.root / 'run'), self.config,
                             pool=object(), executor=execute)
        candidate = self._candidate()
        first = lab.execute(OperationCall('execute', {**candidate, 'step': 0}))
        last = lab.execute(OperationCall('evaluate', {**candidate, 'label': 'selected'}))
        self.assertEqual(first.data['quality_metrics']['accuracy'], 1)
        self.assertEqual(last.data['quality_metrics']['accuracy'], 1)
        self.assertIn('HOST_VALIDATION_METRIC', first.data['execution_result']['term_out'][-1])
        self.assertIn('validation one', seen[0])
        self.assertNotIn('heldout one', seen[0])
        self.assertIn('heldout one', seen[1])

    def test_mutated_candidate_cannot_publish_a_quality_metric(self):
        def execute(workspace, broker):
            (workspace / 'candidate.py').write_text('changed\n')
            (workspace / 'working/submission.csv').write_text('id,answer\nv,1\n')
            return {'term_out': [], 'exec_time': .1, 'exc_type': None}
        lab = AIDEProgramLab(SimpleNamespace(data_dir=self.root / 'run'), self.config, executor=execute)
        result = lab.execute(OperationCall('execute', {**self._candidate(), 'step': 0}))
        self.assertEqual(result.data['execution_result']['exc_type'], 'CandidateMutationError')
        self.assertNotIn('quality_metrics', result.data)

    def test_candidate_failure_remains_feedback_for_native_debugging(self):
        def execute(workspace, broker):
            return {'term_out': ['Traceback'], 'exec_time': 0.1, 'exc_type': 'RuntimeError',
                    'exc_info': None, 'exc_stack': None}
        lab = AIDEProgramLab(SimpleNamespace(data_dir=self.root / 'run'), self.config,
                             pool=object(), executor=execute)
        result = lab.execute(OperationCall('execute', {**self._candidate(), 'step': 0}))
        self.assertEqual(result.data['execution_result']['exc_type'], 'RuntimeError')
        self.assertNotIn('quality_metrics', result.data)

    def test_overlap_labels_in_preview_and_changed_data_are_rejected(self):
        (self.public / 'validation.csv').write_text('id,prompt,answer\nv,validation one,1\n')
        with self.assertRaisesRegex(ValueError, 'without labels'):
            AIDEProgramLab(SimpleNamespace(data_dir=self.root / 'bad'), self.config, pool=object())
        write_questions(self.public / 'validation.csv', self.validation)
        lab = AIDEProgramLab(SimpleNamespace(data_dir=self.root / 'run'), self.config, pool=object())
        (self.root / 'heldout.jsonl').write_text(json.dumps(self.validation[0]) + '\n')
        with self.assertRaisesRegex(ValueError, 'data changed'):
            lab.execute(OperationCall('execute', {**self._candidate(), 'step': 0}))

    def test_wrong_candidate_hash_rejected_before_execution(self):
        lab = AIDEProgramLab(SimpleNamespace(data_dir=self.root / 'run'), self.config, pool=object())
        candidate = self._candidate()
        Path(candidate['candidate']).write_text('changed\n')
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            lab.execute(OperationCall('execute', {**candidate, 'step': 0}))

    def test_broker_binds_model_and_enforces_call_and_token_budgets(self):
        calls = []
        class Pool:
            def complete(self, messages, **kwargs):
                calls.append((messages, kwargs))
                return {'content': 'fixed model reply', 'tokens': 13}
        with inference_broker(Pool(), budget=1, max_tokens=10) as (path, stats):
            def request(value):
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.connect(path)
                    client.sendall(json.dumps(value).encode() + b'\n')
                    return json.loads(client.makefile('rb').readline())
            messages = [{'role': 'user', 'content': 'q'}]
            self.assertIn('error', request({'messages': messages, 'max_tokens': 11}))
            self.assertIn('error', request({'messages': messages, 'model': 'another'}))
            self.assertEqual(request({'messages': messages, 'max_tokens': 10})['content'], 'fixed model reply')
            self.assertIn('error', request({'messages': messages, 'max_tokens': 10}))
        self.assertEqual(len(calls), 1)
        self.assertEqual(stats['completed'], 1)
        self.assertEqual(stats['tokens'], 13)

    def test_controller_uses_worker_selection_and_matching_receipts(self):
        config = {'upstream_root': str(self.root), 'input_dir': str(self.public),
                  'task_file': str(self.root / 'task.md'), 'endpoint': 'http://local/v1', 'model': 'fixed'}
        events, sent = [], []
        class Bridge:
            def __init__(inner, command, **kwargs):
                root = Path(json.loads(Path(command[-1]).read_text())['root'])
                path = root / 'candidate-0.py'
                path.write_text('pass\n')
                args = {'candidate': str(path), 'sha256': digest(path)}
                events.extend([{'kind': 'execute', **args, 'step': 0},
                    {'kind': 'finished', 'report': {'selected_node': 'native-choice'},
                     'evaluations': [{**args, 'label': 'selected'}]}])
            def receive(inner, timeout): return events.pop(0)
            def send(inner, message): sent.append(message)
            def close(inner): pass
        with patch('rsisys.controllers.aide.checkout'):
            controller = AIDEController(SimpleNamespace(data_dir=self.root / 'run'), config,
                                        bridge_factory=Bridge)
        first = controller.decide(DecisionInput('task'))
        with self.assertRaisesRegex(RuntimeError, 'matching'):
            controller.decide(DecisionInput('task'))
        feedback = OperationResult('done', (), False, data={'request': first.operation.arguments,
             'execution_result': {'term_out': ['failure'], 'exec_time': .1, 'exc_type': 'ValueError'}})
        final = controller.decide(DecisionInput('task', last_operation=first.operation, feedback=feedback))
        self.assertEqual(sent[0]['execution_result']['exc_type'], 'ValueError')
        self.assertEqual(final.operation.arguments['label'], 'selected')
        feedback = OperationResult('scored', (), False, data={'request': final.operation.arguments,
                                                            'quality_metrics': {'accuracy': .5}})
        self.assertEqual(controller.decide(DecisionInput('task', last_operation=final.operation,
                                                        feedback=feedback)).kind, 'finish')
        self.assertEqual(controller.report()['native']['selected_node'], 'native-choice')

    def test_config_rejects_unknown_options_and_invalid_steps(self):
        config = {'upstream_root': 'u', 'task_file': 't', 'input_dir': 'i',
                  'endpoint': 'http://local/v1', 'model': 'm'}
        for options in ({'iterations': 0}, {'iterations': True}, {'lab': 'bad.alias'},
                        {'output_subdir': '../escape'}, {'native_config': {'search_policy': 'replace'}}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_config({**config, **options})

    def test_final_execution_failure_is_not_reported_as_completed(self):
        config = {'upstream_root': str(self.root), 'input_dir': str(self.public),
                  'task_file': str(self.root / 'task.md'), 'endpoint': 'http://local/v1', 'model': 'fixed'}
        class Bridge:
            def __init__(inner, command, **kwargs):
                root = Path(json.loads(Path(command[-1]).read_text())['root'])
                path = root / 'candidate-0.py'
                path.write_text('pass\n')
                inner.event = {'kind': 'finished', 'report': {'selected_node': 'upstream'},
                              'evaluations': [{'candidate': str(path), 'sha256': digest(path), 'label': 'selected'}]}
            def receive(inner, timeout): return inner.event
            def close(inner): pass
        with patch('rsisys.controllers.aide.checkout'):
            controller = AIDEController(SimpleNamespace(data_dir=self.root / 'run'), config,
                                        bridge_factory=Bridge)
        selected = controller.decide(DecisionInput('task'))
        feedback = OperationResult('candidate failed', (), False, data={'request': selected.operation.arguments,
            'execution_result': {'exc_type': 'CandidateProcessError'}})
        self.assertEqual(controller.decide(DecisionInput('task', last_operation=selected.operation,
                                                        feedback=feedback)).kind, 'finish')
        self.assertEqual(controller.report()['status'], 'failed')

    def test_simulated_receipt_is_rejected(self):
        config = {'upstream_root': str(self.root), 'input_dir': str(self.public),
                  'task_file': str(self.root / 'task.md'), 'endpoint': 'http://local/v1', 'model': 'fixed'}
        with patch('rsisys.controllers.aide.checkout'):
            controller = AIDEController(SimpleNamespace(data_dir=self.root / 'run'), config)
        controller.bridge = SimpleNamespace()
        controller.pending = OperationCall('aide.execute', {'step': 0})
        feedback = OperationResult('simulated', (), True, data={'request': {'step': 0},
            'execution_result': {'exc_type': None}})
        decision = controller.decide(DecisionInput('task', last_operation=controller.pending, feedback=feedback))
        self.assertEqual(decision.kind, 'finish')
        self.assertEqual(controller.report()['status'], 'failed')

    def test_lab_cleanup_kills_descendant_after_launcher_exit(self):
        import signal
        child = ('import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                 'print("ready",flush=True); time.sleep(30)')
        workspace = self.root / 'sandbox'
        workspace.mkdir()
        (workspace / 'candidate.py').write_text('import subprocess,sys\n'
            f'p=subprocess.Popen([sys.executable,"-c",{child!r}],stdout=subprocess.PIPE,text=True)\n'
            'p.stdout.readline()\nprint(p.pid,flush=True)\n')
        lab = AIDEProgramLab(SimpleNamespace(data_dir=self.root / 'run'), self.config)
        broker = self.root / 'unused.sock'
        endpoint = socket.socket(socket.AF_UNIX)
        self.addCleanup(endpoint.close)
        endpoint.bind(str(broker))
        def command(inner, path, argv, env):
            return [sys.executable, str(path / 'candidate.py')], dict(os.environ)
        with patch('rsisys.labs.aide_program.ProcessIsolation.command', command):
            result = lab._execute_isolated(workspace, str(broker))
        self.assertIsNone(result['exc_type'])
        pid = int((self.root / 'stdout.log').read_text().strip())
        self.addCleanup(lambda: _kill_if_alive(pid, signal.SIGKILL))
        until = time.monotonic() + 3
        while time.monotonic() < until:
            if _process_is_dead_or_zombie(pid):
                break
            time.sleep(.05)
        else:
            self.fail('Candidate descendant survived launcher cleanup')


def _kill_if_alive(pid, sig):
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def _process_is_dead_or_zombie(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    if sys.platform == 'linux':
        try:
            return Path(f'/proc/{pid}/stat').read_text().split()[2] == 'Z'
        except FileNotFoundError:
            # init may reap the orphan between kill(0) and reading /proc.
            return True
    return False


class ResearchBridgeTests(unittest.TestCase):
    def test_upstream_stdout_is_separate_from_events_and_group_is_reaped(self):
        with tempfile.TemporaryDirectory() as root:
            bridge = ResearchBridge([sys.executable, '-u', '-c',
                'import sys,json; print("upstream chatter"); '
                'print("RSISYS_RESEARCH " + json.dumps({"kind":"ready"})); '
                'x=json.loads(sys.stdin.readline()); '
                'print("RSISYS_RESEARCH " + json.dumps({"kind":"reply", "x":x}))'],
                log=Path(root) / 'worker.log')
            try:
                self.assertEqual(bridge.receive(3), {'kind': 'ready'})
                bridge.send({'exact': [1, 2]})
                self.assertEqual(bridge.receive(3)['x'], {'exact': [1, 2]})
            finally:
                bridge.close()
            self.assertIsNotNone(bridge.process.poll())
            self.assertIn('upstream chatter', (Path(root) / 'worker.log').read_text())

    def test_timeout_terminates_owned_worker(self):
        with tempfile.TemporaryDirectory() as root:
            bridge = ResearchBridge([sys.executable, '-c', 'import time; time.sleep(30)'],
                                    log=Path(root) / 'worker.log')
            with self.assertRaises(TimeoutError):
                bridge.receive(.05)
            self.assertIsNotNone(bridge.process.poll())

    def test_close_kills_descendant_after_parent_exits_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as root:
            child = ('import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                     'print("child-ready",flush=True); time.sleep(30)')
            worker = ('import subprocess,sys,json; '
                      f'p=subprocess.Popen([sys.executable,"-c",{child!r}],stdout=subprocess.PIPE,text=True); '
                      'p.stdout.readline(); '
                      'print("RSISYS_RESEARCH "+json.dumps({"kind":"ready","child":p.pid}),flush=True)')
            bridge = ResearchBridge([sys.executable, '-c', worker], log=Path(root) / 'worker.log')
            pid = bridge.receive(3)['child']
            bridge.process.wait(timeout=3)
            os.kill(pid, 0)
            bridge.close()
            bridge.close()
            # On Linux an orphan can briefly remain as a zombie until init
            # reaps it; it is no longer a running process or resource owner.
            until = time.monotonic() + 3
            while time.monotonic() < until:
                if _process_is_dead_or_zombie(pid):
                    break
                time.sleep(.05)
            else:
                self.fail('TERM-ignoring descendant survived owned-group cleanup')


@unittest.skipUnless(os.environ.get('AIDE_UPSTREAM_ROOT') and os.environ.get('AIDE_WORKER_PYTHON'),
                     'requires the pinned native AIDE runtime in an independent environment')
class AIDENativeProtocolTests(unittest.TestCase):
    setUp = AIDEAdapterTests.setUp
    def test_native_agent_draft_improve_debug_and_final_selection_through_registry(self):
        # This scripted endpoint verifies plumbing and native branch semantics,
        # not LLM quality. Candidate execution itself uses real Linux isolation.
        from rsisys.api import Application, ApplicationConfig, Registry
        from rsisys.controllers.aide import build_aide
        requests = []
        code_index = [0]
        outside = str(self.root / 'heldout.jsonl')
        class Handler(BaseHTTPRequestHandler):
            def do_POST(inner):
                body = json.loads(inner.rfile.read(int(inner.headers['Content-Length'])))
                requests.append(body)
                if body.get('tools'):
                    text = '\n'.join(m['content'] for m in body['messages'])
                    buggy = 'HOST_VALIDATION_METRIC' not in text
                    metric = None
                    if not buggy:
                        metric_line = text.split('HOST_VALIDATION_METRIC ')[-1].splitlines()[0]
                        metric = float(json.loads(metric_line)['accuracy'])
                    result = {'is_bug': buggy, 'summary': 'fixture review', 'metric': metric,
                              'lower_is_better': False}
                    message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                        'id': 'call_fixture', 'type': 'function', 'function': {
                            'name': 'submit_review', 'arguments': json.dumps(result)}}]}
                else:
                    index = code_index[0]
                    code_index[0] += 1
                    if index == 1:
                        code = 'raise RuntimeError("fixture candidate failure")\n'
                    else:
                        code = ('import csv, socket\nfrom pathlib import Path\n'
                            f'assert not Path({outside!r}).exists(), "gold exposed"\n'
                            'blocked = False\n'
                            'try:\n'
                            f' socket.create_connection(("127.0.0.1", {server.server_port}),timeout=.2)\n'
                            'except OSError:\n blocked = True\n'
                            'assert blocked, "host TCP exposed"\n'
                            'rows=list(csv.DictReader(open("input/validation.csv")))\n'
                            'with open("working/submission.csv","w",newline="") as f:\n'
                            ' w=csv.DictWriter(f,fieldnames=["id","answer"]);w.writeheader()\n'
                            f' w.writerows({{"id":r["id"],"answer":"{0 if index == 0 else 1}"}} for r in rows)\n')
                    message = {'role': 'assistant', 'content': 'Fixture plan.\n```python\n' + code + '```'}
                output = {'id': 'fixture', 'object': 'chat.completion', 'created': 1,
                          'model': 'native-fixture', 'choices': [{'index': 0, 'message': message,
                           'finish_reason': 'tool_calls' if body.get('tools') else 'stop'}],
                          'usage': {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 20}}
                data = json.dumps(output).encode()
                inner.send_response(200)
                inner.send_header('Content-Type', 'application/json')
                inner.send_header('Content-Length', str(len(data)))
                inner.end_headers()
                inner.wfile.write(data)
            def log_message(inner, *args): pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        task = self.root / 'task.md'
        task.write_text('Predict numbers. The host metric is validation accuracy, maximize it.')
        controller = {'upstream_root': os.environ['AIDE_UPSTREAM_ROOT'],
                      'worker_python': os.environ['AIDE_WORKER_PYTHON'],
                      'input_dir': str(self.public), 'task_file': str(task),
                      'endpoint': f'http://127.0.0.1:{server.server_port}/v1',
                      'model': 'native-fixture', 'iterations': 3, 'timeout': 30,
                      'native_config': {'num_drafts': 1, 'debug_prob': 1.0}}
        lab = {**self.config, 'python': os.environ['AIDE_WORKER_PYTHON']}
        registry = Registry(builtins=False)
        registry.register_controller('aide', build_aide, version='1', validate_config=validate_config)
        registry.register_lab('program', AIDEProgramLab, version='1')
        modules = registry.build({'controller': {'name': 'aide', 'config': controller},
                                 'labs': {'aide': {'name': 'program', 'config': lab}}})
        appconfig = ApplicationConfig('native fixture', self.root / 'native-run', {}, {}, max_decision_steps=12)
        with Application(appconfig, modules=modules) as app:
            self.assertEqual(app.run().status, 'finished')
            report = app.controller_report()
            self.assertEqual(report['native']['stages'], ['draft', 'improve', 'debug'])
            self.assertEqual(report['native']['selected_step'], 2)
            self.assertEqual(report['evaluations']['initial']['quality_metrics']['accuracy'], 0)
            self.assertEqual(report['evaluations']['selected']['quality_metrics']['accuracy'], 1)
            self.assertEqual(len(app.runner.decision_making.harness.state.operation_results), 5)
        self.assertEqual(code_index[0], 3)
        self.assertEqual(len(requests), 6)


if __name__ == '__main__':
    unittest.main()
