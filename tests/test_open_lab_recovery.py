"""Real local-process stop/restart boundaries, with no medical data or model API."""
from copy import deepcopy
import hashlib
import importlib.util
import io
from contextlib import redirect_stdout, redirect_stderr
import json
import os
from pathlib import Path
import signal
import sys
import time
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from examples.medical import open_lab_experiment as launcher
from examples.medical.fixed_evaluation import FixedMedicalLab
from examples.medical.open_lab_recovery import controller_lock, inspect_recovery
from rsisys.adapters.local_lab import LocalProcessLab
from rsisys.adapters.local_lab_state import RetainedLocalProcessBackend, LocalLabState
from rsisys.loops.contracts.decision import OperationCall
from rsisys.loops.contracts.model_service import GenerationResponse
from rsisys.runtime.artifacts import InMemoryArtifactStore
from rsisys.runtime.isolation import system_runtime_paths
from rsisys.runtime.recovery import DurableRecovery, RecoveryRequired


class Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return list(range(100))


class OpenLabRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.artifacts = InMemoryArtifactStore()

    def test_real_process_receipt_key_code_and_logs_survive_without_resubmission(self):
        backend = RetainedLocalProcessBackend(self.root / 'executor')
        lab = LocalProcessLab(backend, self.artifacts)
        workspace = lab.workspace
        lab.execute(OperationCall('lab_write_file', {'path': 'owned.py', 'content': 'print("real result")'}))
        request = {'key': 'one', 'argv': [sys.executable, 'owned.py']}
        ident = lab.execute(OperationCall('lab_start', request)).data['job_id']
        backend.wait(lab._jobs[ident][0])
        expected = lab.execute(OperationCall('lab_result', {'job_id': ident}))
        state = LocalLabState(lab).capture()
        lab.close()
        self.assertTrue(workspace.is_dir())
        restored_backend = RetainedLocalProcessBackend(self.root / 'executor')
        restored = LocalProcessLab(restored_backend, self.artifacts, defer_session=True)
        self.addCleanup(restored.close)
        with patch.object(restored_backend, 'exec', side_effect=AssertionError('resubmitted')):
            LocalLabState(restored).restore(state)
            self.assertEqual(restored.workspace, workspace)
            self.assertEqual(restored.execute(OperationCall('lab_result', {'job_id': ident})), expected)
            waited = restored.execute(OperationCall('lab_poll', {'job_id': ident, 'wait_seconds': 1}))
            self.assertEqual(waited.data['poll_wait']['reason'], 'terminal')
            self.assertEqual(restored.execute(OperationCall('lab_result', {'job_id': ident})), expected)
            self.assertTrue(restored.execute(OperationCall('lab_start', request)).data['replayed'])
            log = restored.execute(OperationCall('lab_read_log', {'job_id': ident, 'stream': 'stdout'}))
            self.assertEqual(log.data['content'], 'real result\n')

    def test_fixed_evaluation_binding_survives_and_is_not_reexecuted(self):
        data = self.root / 'dev.jsonl'
        data.write_text(json.dumps({'id': 'fixture', 'question': 'fixture',
            'options': {'A': 'a', 'B': 'b'}, 'gold': 'A'}) + '\n')
        options = {'evaluation_root': self.root / 'evaluations', 'python': sys.executable,
            'evaluation_groups': [{'benchmark_id': 'fixture', 'path': str(data),
                                  'sha256': hashlib.sha256(data.read_bytes()).hexdigest()}]}
        environment = {'CUDA_VISIBLE_DEVICES': '0',
            'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
        backend = launcher.RetainedAllocatedProcessBackend(self.root / 'executor', environment)
        lab = FixedMedicalLab(backend, self.artifacts, **options)
        lab.execute(OperationCall('lab_write_file', {'path': 'candidate.py', 'content':
            'import json,sys\np=json.load(sys.stdin)\nprint(json.dumps({"predictions":['
            '{"benchmark_id":r["benchmark_id"],"id":r["id"],"text":"A"} for r in p["items"]]}))\n'}))
        args = {'key': 'fixed', 'candidate_id': 'fixture', 'argv': [sys.executable, 'candidate.py']}
        ident = lab.execute(OperationCall('medical_evaluate', args)).data['job_id']
        completed = backend.wait(lab._jobs[ident][0])
        self.assertEqual(completed.status.value, 'succeeded', completed.stderr)
        def port(value):
            return LocalLabState(value, capture_extra=lambda: deepcopy(value._evaluations),
                restore_extra=lambda state: setattr(value, '_evaluations', state))
        state = port(lab).capture()  # completed process, not yet collected or scored by host
        lab.close()
        restored_backend = launcher.RetainedAllocatedProcessBackend(self.root / 'executor', environment)
        restored = FixedMedicalLab(restored_backend, self.artifacts, defer_session=True, **options)
        self.addCleanup(restored.close)
        with patch.object(restored_backend, 'exec', side_effect=AssertionError('reexecuted evaluation')):
            port(restored).restore(state)
            receipt = restored.execute(OperationCall('lab_result', {'job_id': ident}))
            self.assertEqual(receipt.data['fixed_evaluation']['metrics']['fixture'], {'accuracy': 1.0, 'n': 1})
            self.assertTrue(restored.execute(OperationCall('medical_evaluate', args)).data['replayed'])

    def test_missing_terminal_receipt_or_extra_identity_fails_closed(self):
        backend = RetainedLocalProcessBackend(self.root / 'executor')
        lab = LocalProcessLab(backend, self.artifacts)
        ident = lab.execute(OperationCall('lab_start', {'key': 'one',
            'argv': [sys.executable, '-c', 'print("done")']})).data['job_id']
        backend.wait(lab._jobs[ident][0])
        state = LocalLabState(lab).capture()
        lab.close()
        path = backend._receipt_root / (ident + '.json')
        saved = path.read_bytes()
        path.unlink()
        other = RetainedLocalProcessBackend(self.root / 'executor')
        with self.assertRaisesRegex(RecoveryRequired, 'terminal receipt'):
            other.restore_state(state['backend'])
        self.assertEqual(other.active_count, 0)
        path.write_bytes(saved)
        altered = deepcopy(state['backend'])
        altered['processes'].clear()
        with self.assertRaisesRegex(RecoveryRequired, 'identities changed'):
            other.restore_state(altered)

    def test_corrupt_retained_log_is_rejected_before_attaching(self):
        backend = RetainedLocalProcessBackend(self.root / 'executor')
        lab = LocalProcessLab(backend, self.artifacts)
        ident = lab.execute(OperationCall('lab_start', {'key': 'one',
            'argv': [sys.executable, '-c', 'print("done")']})).data['job_id']
        ref = lab._jobs[ident][0]
        backend.wait(ref)
        state = LocalLabState(lab).capture()
        log = Path(backend.log_info(ref)['stdout']['path'])
        lab.close()
        log.write_text('corrupt')
        other = RetainedLocalProcessBackend(self.root / 'executor')
        with self.assertRaisesRegex(ValueError, 'log identity/digest'):
            other.restore_state(state['backend'])
        self.assertEqual(other.active_count, 0)

    def test_controller_lock_and_legacy_run_are_not_bypassed(self):
        with controller_lock(self.root, resume=False):
            with self.assertRaisesRegex(RuntimeError, 'already running'):
                with controller_lock(self.root, resume=True):
                    self.fail('duplicate controller admitted')
        with self.assertRaisesRegex(ValueError, 'no durable launch state'):
            launcher.main(['--root', str(self.root), '--resume'])

    def test_unsubmitted_feedback_survives_pause_and_repeated_cleanup(self):
        from rsisys.loops.decision_making.feedback.learning import FeedbackLearning, SerializedLearningBackend
        from tests.test_feedback_learning_integration import Backend, ImmediateExecutor
        from tests.test_feedback_required_outputs import MODEL, observation
        backend = Backend()
        path = self.root / 'feedback.json'
        def build():
            return FeedbackLearning(SerializedLearningBackend(backend), path=path, executor=ImmediateExecutor())
        learning = build()
        ident = learning.capture_receipt(observation('queued-before-pause'), MODEL)
        learning.close(preserve_queued=True)
        learning.close()
        self.assertEqual(learning.state['jobs'][ident]['status'], 'queued')
        self.assertEqual(backend.requests, [])
        resumed = build()
        try:
            self.assertEqual(resumed.capture_receipt(observation('queued-before-pause'), MODEL), ident)
            resumed.poll()
            resumed.poll(dispatch=False)
            self.assertEqual(len(backend.requests), 1)
            self.assertEqual(resumed.state['jobs'][ident]['status'], 'completed')
        finally:
            resumed.close()

    def test_late_provider_response_cannot_start_work_after_absolute_deadline(self):
        inputs, model, run = self.root / 'inputs', self.root / 'model', self.root / 'late-run'
        inputs.mkdir()
        model.mkdir()
        (model / 'config.json').write_text('{}')
        row = {'id': 'fixture', 'question': 'fixture', 'options': {'A': 'a', 'B': 'b'}, 'gold': 'A'}
        for name in ('train.jsonl', 'dev.jsonl'):
            (inputs / name).write_text(json.dumps(row) + '\n')
        (inputs / 'task.json').write_text(json.dumps({'train': 'train.jsonl',
            'development': [{'path': 'dev.jsonl', 'benchmark_id': 'fixture'}]}))
        clock = SimpleNamespace(time=time.time, monotonic=time.monotonic, sleep=time.sleep)
        class Transport:
            def __init__(self, *args, **kwargs):
                pass
            def _generation_payload(self, request):
                return {'model': request.model.model_id, 'messages': request.chat_messages()}
            def generate_with_response_observer(self, request, observer):
                saved = json.loads((run / 'recovery-launch.json').read_text())
                clock.time = lambda: saved['arguments']['deadline_epoch'] + 1
                observer({'body': 'synthetic late response'})
                return GenerationResponse(json.dumps({'kind': 'invoke', 'operation': 'lab_start',
                    'arguments': {'key': 'too-late', 'argv': [sys.executable, '-c',
                        'from pathlib import Path; Path("must-not-run").touch()']}}), request.model, False)
        transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: Tokenizer()))
        with patch.dict(sys.modules, {'transformers': transformers}), \
             patch.dict(os.environ, {'ASCEND_RT_VISIBLE_DEVICES': '8,13'}), \
             patch.object(launcher, 'VLLMBackend', Transport), patch.object(launcher, 'time', clock), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = launcher.main(['--root', str(run), '--task', str(inputs / 'task.json'),
                '--model-path', str(model), '--python', sys.executable, '--endpoint', 'http://127.0.0.1:1/v1',
                '--run-id', 'late-response-fixture', '--seconds', '120', '--max-calls', '5', '--device-type', 'npu',
                '--execution-policy', json.dumps({'read_only': list(system_runtime_paths())}),
                '--resource-allocation', json.dumps({'physical_devices': [8,13], 'logical_devices': [0,1]})])
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['stop']['reason'], 'absolute_host_deadline_or_signal')
        self.assertEqual(result['model_calls'], 1)
        recovery = json.loads((run / 'application/allocated-workspaces/process-recovery.json').read_text())
        self.assertEqual(recovery['payload']['processes'], {})
        self.assertFalse(any((run / 'application/allocated-workspaces').rglob('must-not-run')))

    def test_launcher_signal_pause_resume_preserves_state_memory_budget_and_deadline(self):
        inputs, model, run = self.root / 'inputs', self.root / 'model', self.root / 'run'
        inputs.mkdir()
        model.mkdir()
        (model / 'config.json').write_text('{}')
        row = {'id': 'fixture', 'question': 'fixture', 'options': {'A': 'a', 'B': 'b'}, 'gold': 'A'}
        for name in ('train.jsonl', 'dev.jsonl'):
            (inputs / name).write_text(json.dumps(row) + '\n')
        (inputs / 'task.json').write_text(json.dumps({'train': 'train.jsonl',
            'development': [{'path': 'dev.jsonl', 'benchmark_id': 'fixture'}]}))
        transports = []
        compact_state = []
        class Transport:
            def __init__(self, *args, **kwargs):
                self.calls = 0
                transports.append(self)
            def _generation_payload(self, request):
                return {'model': request.model.model_id, 'messages': request.chat_messages()}
            def generate_with_response_observer(self, request, observer):
                self.calls += 1
                count = json.loads((run / 'model_calls.json').read_text())['attempted']
                if count == 2:
                    self.restored_context = json.loads(request.prompt)
                operation = ({'name': 'lab_write_file', 'arguments': {'path': 'resume-marker.txt',
                    'content': 'preserve me'}} if count == 1 else
                    {'name': 'lab_start', 'arguments': {'key': 'owned-during-pause',
                        'argv': [sys.executable, '-c', 'import time; time.sleep(20)']}})
                # Exercise the actual installed signal handler during a decision.
                os.kill(os.getpid(), signal.SIGTERM)
                observer({'body': 'fixture response'})
                return GenerationResponse(json.dumps({'kind': 'invoke', 'operation': operation['name'],
                    'arguments': operation['arguments']}), request.model, False)
        transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: Tokenizer()))
        patches = [patch.dict(sys.modules, {'transformers': transformers}),
            patch.dict(os.environ, {'ASCEND_RT_VISIBLE_DEVICES': '8,13'}),
            patch.object(launcher, 'VLLMBackend', Transport)]
        from examples.medical.open_lab_monitor import OpenLabMonitor
        original_update = OpenLabMonitor.update
        def update(monitor, harness, backend, compactor, step):
            if backend.calls == 1 and not compact_state:
                state = compactor.export_state()
                state['summary'] = {'research_notes': ['fixture compressed observation']}
                state['compactions'] = 3
                compactor.restore_state(state)
                compact_state.append(state)
                memory = harness.agent_memory
                memory.execute(OperationCall('research_task', {'expected_revision': 0,
                    'task_id': 'resume-debug', 'action': 'open', 'title': 'Retain raw marker feedback'}))
                memory.execute(OperationCall('memory_write', {'expected_revision': 1,
                    'memory_id': 'retained-memory', 'content': 'fixture reusable observation',
                    'source_refs': [], 'status': 'unverified'}))
                # Exercise the real medical optimizer's reflect ports, then
                # resume with the same task/memory/model/resource accounts.
                lab = harness.tools.lab
                manifest = {'candidate_id': 'reflect-fixture', 'parent_system': 'initial',
                    'base_version': harness.current_model().revision, 'affected_roles': ['decision_maker'],
                    'components': {'meta.prompts': 'reflect-prompts.json', 'meta.harness': 'reflect-harness.json'}}
                config = {key: getattr(harness, key) for key in
                    ('experience_limit', 'memory_limit', 'review_tool_limit', 'max_outstanding')}
                config['memory_limit'] = 7
                for name, value in [('candidate.json', manifest), ('reflect-prompts.json',
                        {'research': 'CPU synthetic reflected research instructions'}), ('reflect-harness.json', config)]:
                    lab.execute(OperationCall('lab_write_file', {'path': name, 'content': json.dumps(value)}))
                lab.execute(OperationCall('system_register', {'manifest_path': 'candidate.json'}))
                lab.execute(OperationCall('system_activate', {'candidate_ids': ['reflect-fixture'],
                    'version': 'reflected-v1', 'idempotency_key': 'reflect-apply'}))
                self.assertEqual(harness.memory_limit, 7)
            if backend.calls == 2:
                self.assertEqual(harness.memory_limit, 7)
                self.assertEqual(harness.policy.prompts.render('research'), 'CPU synthetic reflected research instructions')
            original_update(monitor, harness, backend, compactor, step)
        with patches[0], patches[1], patches[2], patch.object(OpenLabMonitor, 'update', update), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            first = launcher.main(['--root', str(run), '--task', str(inputs / 'task.json'),
                '--model-path', str(model), '--python', sys.executable,
                '--endpoint', 'http://127.0.0.1:1/v1', '--run-id', 'recovery-fixture',
                '--seconds', '120', '--max-calls', '5', '--device-type', 'npu',
                '--execution-policy', json.dumps({'read_only': list(system_runtime_paths())}),
                '--resource-allocation', json.dumps({'physical_devices': [8,13], 'logical_devices': [0,1]})])
            self.assertEqual(first['status'], 'paused')
            saved = json.loads((run / 'recovery-launch.json').read_text())
            checkpoint = DurableRecovery.read(run / 'application')
            decisions = checkpoint['components']['decision']['state']['state'].decision_records
            self.assertEqual(len(decisions), 1)
            path = next((run / 'application/allocated-workspaces').glob('session-*'))
            self.assertEqual((path / 'resume-marker.txt').read_text(), 'preserve me')
            second = launcher.main(['--root', str(run), '--resume'])
            self.assertEqual(second['status'], 'paused')
            self.assertEqual(second['model_calls'], 2)
            self.assertEqual(DurableRecovery.read(run / 'application')['components']['evolution']['state']
                ['registry']['current'].version, 'reflected-v1')
            self.assertEqual(second['compactions'], 3)
            self.assertIn('fixture reusable observation', json.dumps(transports[1].restored_context))
            active = transports[1].restored_context['research_agenda']['active_feedback']
            self.assertEqual(active['tasks'][0]['task_id'], 'resume-debug')
            self.assertIn('resume-marker.txt', json.dumps(active['observations']))
            self.assertEqual(DurableRecovery.read(run / 'application')['components']['agent_memory']['state']['revision'], 2)
            resources = transports[1].restored_context['project_context']['resources']
            self.assertEqual(resources['logical_to_physical'], {'0': '8', '1': '13'})
            self.assertEqual(resources, json.loads((run / 'resource-allocation.json').read_text()))
            self.assertEqual([t.calls for t in transports], [1, 1])
            self.assertEqual(json.loads((run / 'recovery-launch.json').read_text()), saved)
            restored = DurableRecovery.read(run / 'application')
            records = restored['components']['decision']['state']['state'].decision_records
            self.assertEqual(len(records), 2)
            self.assertEqual(restored['components']['decision']['state']['compactor'], compact_state[0])
            self.assertTrue(set(decisions) < set(records))
            self.assertEqual(len(list((run / 'application/allocated-workspaces').glob('session-*'))), 1)
            self.assertEqual(len(json.loads((run / 'application/feedback_learning.json').read_text())['jobs']), 2)
            terminal = restored['components']['lab_execution']['state']['terminal']
            self.assertEqual(len(terminal), 1)
            self.assertEqual(next(iter(terminal.values())).data['state'], 'cancelled')
            before = (run / 'model_calls.json').read_bytes()
            report = inspect_recovery(run)
            self.assertIsNone(report['intervention'])
            self.assertFalse(report['uncommitted_mutation'])
            with self.assertRaises(SystemExit):
                launcher.main(['--root', str(run), '--resume', '--max-calls', '999'])
            self.assertEqual((run / 'model_calls.json').read_bytes(), before)
            with patch.dict(os.environ, {'ASCEND_RT_VISIBLE_DEVICES': '8,14'}), \
                 self.assertRaisesRegex(ValueError, 'deployment identity changed|device mapping'):
                launcher.main(['--root', str(run), '--resume'])
            self.assertEqual((run / 'model_calls.json').read_bytes(), before)
            # A write-ahead step without a committed checkpoint is uncertain.
            # Resume may inspect it but must make no model call or clear it.
            journal = run / 'application/recovery/journal.json'
            original = journal.read_bytes()
            checkpoint_bytes = (run / 'application/recovery/checkpoint.json').read_bytes()
            journal.write_text(json.dumps({'generation': restored['generation'] + 1,
                'status': 'started', 'operation': 'step', 'started_at': 1,
                'arguments': None}))
            with self.assertRaisesRegex(RecoveryRequired, 'Interrupted mutation'):
                launcher.main(['--root', str(run), '--resume'])
            self.assertEqual(transports[-1].calls, 0)
            self.assertEqual((run / 'model_calls.json').read_bytes(), before)
            self.assertEqual((run / 'application/recovery/checkpoint.json').read_bytes(), checkpoint_bytes)
            self.assertTrue(inspect_recovery(run)['uncommitted_mutation'])
            journal.write_bytes(original)  # isolated fixture only; production has no clear/retry switch
            with patch.object(launcher.time, 'time', return_value=saved['arguments']['deadline_epoch'] + 1):
                last = launcher.main(['--root', str(run), '--resume'])
            self.assertEqual(last['status'], 'completed')
            self.assertEqual(last['model_calls'], 2)
            self.assertEqual(transports[-1].calls, 0)
            with self.assertRaisesRegex(ValueError, 'terminal'):
                launcher.main(['--root', str(run), '--resume'])

    def test_new_attempt_cli_imports_notes_before_checkpoint_and_resume_does_not_reapply(self):
        from examples.medical.continuation import export_attempt
        inputs, model = self.root / 'inputs', self.root / 'model'
        inputs.mkdir()
        model.mkdir()
        (model / 'config.json').write_text('{}')
        row = {'id': 'fixture', 'question': 'fixture', 'options': {'A': 'a', 'B': 'b'}, 'gold': 'A'}
        for name in ('train.jsonl', 'dev.jsonl'):
            (inputs / name).write_text(json.dumps(row) + '\n')
        (inputs / 'task.json').write_text(json.dumps({'train': 'train.jsonl',
            'development': [{'path': 'dev.jsonl', 'benchmark_id': 'fixture'}]}))
        contexts = []
        class Transport:
            def __init__(self, *args, **kwargs):
                pass
            def _generation_payload(self, request):
                return {'model': request.model.model_id, 'messages': request.chat_messages()}
            def generate_with_response_observer(self, request, observer):
                contexts.append(json.loads(request.prompt))
                operation = ('memory_search' if len(contexts) > 1 else 'memory_write')
                arguments = {} if len(contexts) > 1 else {'expected_revision': 0,
                    'memory_id': 'prior-note', 'content': 'Prior CPU fixture hypothesis',
                    'source_refs': [], 'status': 'unverified'}
                os.kill(os.getpid(), signal.SIGTERM)
                observer({'body': 'synthetic fixture response'})
                return GenerationResponse(json.dumps({'kind': 'invoke', 'operation': operation,
                    'arguments': arguments}), request.model, False)
        transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: Tokenizer()))
        common = ['--task', str(inputs / 'task.json'), '--model-path', str(model), '--python', sys.executable,
            '--endpoint', 'http://127.0.0.1:1/v1', '--seconds', '120', '--max-calls', '5', '--device-type', 'npu',
            '--evaluation-campaign-id', 'continuation-fixture', '--evaluation-ledger', str(self.root / 'campaign.json'),
            '--execution-policy', json.dumps({'read_only': list(system_runtime_paths())}),
            '--resource-allocation', json.dumps({'physical_devices': [8,13], 'logical_devices': [0,1]})]
        old, new, handoff = self.root / 'old', self.root / 'new', self.root / 'handoff'
        with patch.dict(sys.modules, {'transformers': transformers}), \
             patch.dict(os.environ, {'ASCEND_RT_VISIBLE_DEVICES': '8,13'}), \
             patch.object(launcher, 'VLLMBackend', Transport), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(launcher.main(['--root', str(old), '--run-id', 'old', *common])['status'], 'paused')
            exported = export_attempt(old, handoff)
            deadline = json.loads((old / 'recovery-launch.json').read_text())['arguments']['deadline_epoch']
            self.assertEqual(launcher.main(['--root', str(new), '--run-id', 'new', *common,
                '--continuation', str(handoff), '--deadline-epoch', str(deadline)])['status'], 'paused')
            self.assertIn('Prior CPU fixture hypothesis', contexts[1]['memory_context'])
            self.assertEqual(contexts[1]['research_agenda']['continuation']['handoff_sha256'], exported['sha256'])
            new_launch = json.loads((new / 'recovery-launch.json').read_text())['arguments']
            self.assertEqual(new_launch['max_calls'], 4)
            self.assertEqual(new_launch['deadline_epoch'], deadline)
            first = DurableRecovery.read(new / 'application')
            self.assertIn('continuation', first['components'])
            self.assertEqual(first['components']['agent_memory']['state']['revision'], 1)
            self.assertEqual(launcher.main(['--root', str(new), '--resume'])['status'], 'paused')
            self.assertEqual(DurableRecovery.read(new / 'application')['components']['agent_memory']['state']['revision'], 1)
            renewal = self.root / 'renewal.json'
            renewed = self.root / 'renewed'
            authorization = {'schema': 'rsisys-host-continuation-renewal/v1',
                'handoff_sha256': exported['sha256'], 'campaign_id': 'continuation-fixture',
                'target_attempt_id': 'renewed', 'previous_deadline_epoch': deadline,
                'authorized_at': deadline + 1, 'deadline_epoch': deadline + 3600,
                'reason': 'Explicit new user-authorized run window'}
            renewal.write_text(json.dumps(authorization))
            with patch.object(launcher.time, 'time', return_value=deadline + 2):
                self.assertEqual(launcher.main(['--root', str(renewed), '--run-id', 'renewed', *common,
                    '--continuation', str(handoff), '--continuation-authorization', str(renewal),
                    '--deadline-epoch', str(deadline + 3600)])['status'], 'paused')
                saved = json.loads((renewed / 'recovery-launch.json').read_text())['arguments']
                self.assertEqual(saved['deadline_epoch'], deadline + 122)
                self.assertEqual(saved['max_calls'], 4)
                self.assertEqual(json.loads((renewed / 'continuation-authorization.json').read_text()), authorization)
                renewal.write_text('{}')  # external operator file is no longer authoritative
                self.assertEqual(launcher.main(['--root', str(renewed), '--resume'])['status'], 'paused')
                (renewed / 'continuation-authorization.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'authorization changed'):
                    launcher.main(['--root', str(renewed), '--resume'])


if __name__ == '__main__':
    unittest.main()
