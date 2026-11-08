"""Execution audit regressions: CPU processes and synthetic receipts only."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments.medical.evaluation import MedicalEvaluator
from experiments.medical.run import _ApplyMedical
from experiments.medical.task import Benchmark, MedicalTask
from rsisys.adapters.training import TrainingCommand, _result, _supervise
from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.sqlite import SQLiteExperienceStore
from rsisys.loops.contracts.candidate import Candidate, SystemVersion
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.jobs import JobOutcome, JobRef, JobStatus
from rsisys.loops.contracts.model_service import GenerationRequest, GenerationResponse, ModelRef
from rsisys.loops.contracts.requests import ApplyCandidateSpec
from rsisys.loops.contracts.training import TrainingResult, TrainingOrigin
from rsisys.loops.decision_making.action_space import ModularDecisionActionSpace, OperationCall
from rsisys.loops.decision_making.action_space.access import ActionPermissions, authorized_space
from rsisys.loops.decision_making.harness.tools import BoundDecisionTools, execute_operation
from rsisys.loops.decision_making.harness.tools.workspace import WorkspaceTools
from rsisys.loops.exploration.harness.commands import CommandState
from rsisys.loops.exploration.harness.learner import ActorTrainingRequest, LearningExecution
from rsisys.loops.exploration.harness.memory import ExplorationMemory
from rsisys.loops.exploration.harness.state import ExplorationState
from rsisys.runtime.artifacts import FileArtifactStore
from rsisys.runtime.jobs import JobBook, InvalidJobResult
from rsisys.runtime.local_process import LocalProcessBackend
from rsisys.runtime.sandbox import ExecRequest, SandboxSpec
from rsisys.runtime.systems import ReferenceSystemActivator, SystemRegistry
from tests.experiments.medical.test_medical_example import row, write_rows


BASE = ModelRef('audit', 'fixture', 'base')


class TrainingMetricsAudit(unittest.TestCase):
    def test_invalid_metrics_do_not_hide_base_role_or_parent_identity_errors(self):
        origin = TrainingOrigin('expected-parent', BASE)
        for mismatch in ('base', 'role', 'parent'):
            with self.subTest(mismatch=mismatch):
                result_base = replace(BASE, revision='wrong') if mismatch == 'base' else BASE
                candidate = Candidate('candidate', result_base.revision, {},
                    ('decision_maker',) if mismatch == 'role' else ('actor',),
                    'wrong-parent' if mismatch == 'parent' else origin.parent_system)
                result = TrainingResult(result_base, replace(BASE, revision='trained'),
                    candidate, {'loss': float('nan')})
                job = JobRef('job', 'fixture')
                backend = SimpleNamespace(submit=Mock(return_value=job), cancel=Mock(),
                    poll=Mock(return_value=JobOutcome(job, JobStatus.SUCCEEDED, result)))
                experience, state = Mock(), ExplorationState()
                jobs = JobBook(backend)
                execution = LearningExecution(SimpleNamespace(model=BASE), ExplorationMemory(), experience, state, jobs)
                execution.submit(ActorTrainingRequest(BASE, (), 'fixture', origin=origin), idempotency_key='one')
                with self.assertRaises(InvalidJobResult):
                    execution.poll(job)
                self.assertEqual(jobs._handles[job].outcome.status, JobStatus.SUCCEEDED)
                self.assertFalse(state.candidate_results)
                experience.record.assert_not_called()

    def test_native_worker_does_not_publish_success_after_nonfinite_loss(self):
        from tests.runtime.test_framework_adapters import NativeDistributedWorkerTests
        from rsisys.adapters.training import _transformers_peft
        fixture = NativeDistributedWorkerTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        modules, _, _ = fixture.fake_frameworks(0, 1)
        train = modules['transformers'].Trainer.train
        def nonfinite(trainer):
            trained = train(trainer)
            trained.metrics['loss'] = float('nan')
            return trained
        fixture.write_recipe()
        with patch.object(modules['transformers'].Trainer, 'train', nonfinite), \
                patch.dict(sys.modules, modules), patch.dict(os.environ, {
                    'RSISYS_JOB_DIR': str(fixture.root), 'RSISYS_REQUEST_FILE': str(fixture.request_file),
                    'RSISYS_RESULT_FILE': str(fixture.root / 'result.json')}, clear=True):
            with self.assertRaisesRegex(ValueError, 'finite'):
                _transformers_peft(fixture.recipe)
        self.assertFalse((fixture.root / 'result.json').exists())

    def test_invalid_provider_metrics_never_enter_success_cache_or_candidate_store(self):
        for metrics in ({'loss': float('nan')}, {'loss': float('inf')}, {'loss': True},
                        {'loss': None}, {'loss': '0'}, {'loss': 10**1000}, {'': 0}, {1: 0}, []):
            for cancel_first in (False, True):
                with self.subTest(metrics=repr(metrics)[:50], cancel=cancel_first), TemporaryDirectory() as tmp:
                    store = SQLiteExperienceStore(Path(tmp) / 'experience.sqlite')
                    self.addCleanup(store.close)
                    result = TrainingResult(BASE, replace(BASE, revision='trained'),
                        Candidate('candidate', BASE.revision, {}, ('actor',)), metrics, True)
                    job = JobRef('job', 'fixture')
                    backend = SimpleNamespace(submit=Mock(return_value=job), cancel=Mock(),
                        poll=Mock(return_value=JobOutcome(job, JobStatus.SUCCEEDED, result, simulated=True)))
                    state = ExplorationState()
                    jobs = JobBook(backend)
                    execution = LearningExecution(SimpleNamespace(model=BASE), ExplorationMemory(),
                        BasicExperienceSystem(store), state, jobs)
                    request = ActorTrainingRequest(BASE, (), 'fixture')
                    execution.submit(request, idempotency_key='one')
                    first = (execution.cancel if cancel_first else execution.poll)(job)
                    self.assertEqual(first.status, JobStatus.FAILED)
                    self.assertIn('metrics', first.error)
                    self.assertTrue(first.simulated)
                    self.assertIsNone(first.result)
                    self.assertEqual(jobs._handles[job].outcome, first)
                    self.assertFalse(state.candidate_results)
                    self.assertEqual(execution.poll(job), first)
                    self.assertEqual(execution.cancel(job), first)
                    self.assertEqual(len(store.records()), 1)
                    backend.poll.assert_called_once()
                    store.close()

    def test_native_result_reader_rejects_invalid_metric_names(self):
        result = TrainingResult(BASE, replace(BASE, revision='trained'),
            Candidate('candidate', BASE.revision, {}, ('actor',)), {'': 1})
        with self.assertRaises(ValueError):
            _result(asdict(result))

    def test_negative_zero_and_missing_training_metrics_are_preserved(self):
        for metrics in ({'reward': -3, 'loss': 0.0}, {}):
            with self.subTest(metrics=metrics):
                result = TrainingResult(BASE, replace(BASE, revision='trained'),
                    Candidate('candidate', BASE.revision, {}, ('actor',)), metrics)
                self.assertEqual(_result(asdict(result)).metrics, metrics)


class MedicalBindingAudit(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        train = write_rows(self.root / 'train.jsonl', [row('train', 'training symbol')])
        dev = write_rows(self.root / 'dev.jsonl', [row('dev', 'development symbol')])
        self.task = MedicalTask(train, (Benchmark('medqa', 'primary', dev),))
        self.artifacts = FileArtifactStore(self.root / 'artifacts')
        self.prompt = self.artifacts.put(b'initial prompt', kind='medical_prompt')
        self.initial = SystemVersion('initial', {'prompt': self.prompt}, BASE)
        self.activator = Mock(wraps=ReferenceSystemActivator(BASE))
        self.systems = SystemRegistry(self.initial, self.activator)
        self.candidates = {}
        self.inference = Mock(side_effect=lambda request: GenerationResponse('A', request.model, True))
        self.evaluator = MedicalEvaluator(self.task, self.artifacts, self.systems, self.candidates, self.inference)

    def test_cached_measurement_refuses_identity_reuse_without_poisoning_original(self):
        original = self.evaluator.measure(self.initial)
        other_prompt = self.artifacts.put(b'other prompt', kind='medical_prompt')
        for changed in (replace(self.initial, model=replace(BASE, revision='other')),
                        replace(self.initial, components={'prompt': other_prompt})):
            with self.subTest(system=changed), self.assertRaisesRegex(ValueError, 'identity'):
                self.evaluator.measure(changed)
            self.assertEqual(self.evaluator.measure(self.initial), original)
        self.inference.assert_called_once()

    def test_failed_measurement_identity_is_also_pinned(self):
        self.inference.side_effect = RuntimeError('fixture failure')
        with self.assertRaises(RuntimeError):
            self.evaluator.measure(self.initial)
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.evaluator.measure(replace(self.initial, model=replace(BASE, revision='other')))
        self.assertEqual(len(self.evaluator.evidence), 1)

    def test_acceptance_extension_cannot_replace_canonical_identity_or_facts(self):
        self.evaluator.establish_anchor()
        candidate = Candidate('candidate', BASE.revision, {'prompt': self.prompt}, ('actor',), 'initial')
        self.candidates[candidate.candidate_id] = candidate
        self.evaluator.measure(self.systems.register(candidate))
        self.evaluator.acceptance_policy = lambda *args: {'accepted': True,
            'candidate_id': 'fabricated', 'incumbent': 'fabricated', 'fixed_anchor': 'fabricated',
            'kind': 'fabricated', 'protocol_ref': 'fabricated', 'primary_improved': True,
            'source_refs': ['fabricated']}
        result = self.evaluator.interpret('candidate')
        self.assertEqual(result['candidate_id'], 'candidate')
        self.assertEqual(result['incumbent'], 'initial')
        self.assertEqual(result['fixed_anchor'], 'initial')
        self.assertEqual(result['protocol_ref'], self.task.protocol_ref)
        self.assertEqual(result['kind'], 'acceptance_interpretation')
        self.assertFalse(result['primary_improved'])
        self.assertTrue(result['accepted'])  # The injected policy still owns its verdict.
        self.assertNotIn('fabricated', result['source_refs'])

    def test_prompt_read_failure_precedes_activation_and_role_binding(self):
        candidate = Candidate('candidate', BASE.revision, {'prompt': self.prompt}, ('actor',), 'initial')
        self.systems.register(candidate)
        self.candidates['candidate'] = candidate
        evaluator = SimpleNamespace(completed_evaluation=lambda ident: True,
            interpret=lambda ident: {'accepted': True}, measurement=lambda ident: SimpleNamespace(simulated=True))
        handler = _ApplyMedical(self.systems, evaluator, self.candidates, self.artifacts)
        state = CommandState(Directive('apply', 'objective', 'apply', mode='apply_candidate',
            payload=ApplyCandidateSpec('candidate')))
        harness = SimpleNamespace(policy=SimpleNamespace(update_model=Mock()), state=SimpleNamespace(instructions='old'))
        handler.prepare(harness, state)
        get = self.artifacts.get
        with patch.object(self.artifacts, 'get', side_effect=OSError('fixture unavailable')):
            with self.assertRaises(OSError):
                handler.advance(harness, state, None)
        self.assertEqual(self.systems.current, self.initial)
        self.activator.activate.assert_not_called()
        harness.policy.update_model.assert_not_called()
        self.assertNotIn('activation_started', state.data)
        result = handler.advance(harness, state, None)
        self.assertEqual(result.adopted_candidate_ids, ('candidate',))
        self.assertEqual(harness.state.instructions, get(self.prompt).decode())
        self.activator.activate.assert_called_once()


class LocalExecutionAudit(unittest.TestCase):
    def test_process_bridge_does_not_leave_children_after_native_command_exit(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = ('import pathlib,subprocess,sys; '
                    'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); '
                    'pathlib.Path("child.pid").write_text(str(p.pid))')
            command = TrainingCommand((sys.executable, '-c', code), str(root))
            (root / 'receipt.json').write_text(json.dumps({'command': asdict(command)}))
            _supervise(root)
            pid = int((root / 'child.pid').read_text())
            try:
                state = subprocess.run(['ps', '-p', str(pid), '-o', 'stat='],
                    capture_output=True, text=True).stdout.strip()
                self.assertTrue(not state or state.startswith('Z'), 'Native command left a live child')
                self.assertEqual(json.loads((root / 'state.json').read_text())['status'], 'succeeded')
            finally:
                if state and not state.startswith('Z'):
                    os.kill(pid, 9)

    def test_reaped_process_never_receives_a_group_signal(self):
        from rsisys.runtime.local_process import _Process
        entry = _Process(SimpleNamespace(pid=12345, returncode=0), 1)
        with patch('rsisys.runtime.local_process.os.killpg') as kill:
            LocalProcessBackend._kill(entry)
        kill.assert_not_called()

    def test_normal_exit_kills_owned_background_child_before_reaping_leader(self):
        with TemporaryDirectory() as tmp:
            backend = LocalProcessBackend(tmp)
            session = backend.create(SandboxSpec('local'))
            code = ('import subprocess,sys; '
                    'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); '
                    'print(p.pid, flush=True)')
            killpg = os.killpg
            reaped = []
            def signal_owned(pid, sig):
                entry = next(e for e in backend._processes.values() if e.process.pid == pid)
                reaped.append(entry.process.returncode is not None)
                return killpg(pid, sig)
            with patch('rsisys.runtime.local_process.os.killpg', side_effect=signal_owned):
                job = backend.exec(session, ExecRequest((sys.executable, '-c', code), 3))
                result = backend.wait(job)
            self.assertEqual(result.status, JobStatus.SUCCEEDED)
            self.assertTrue(result.stdout.strip().isdigit())
            self.assertTrue(reaped)
            self.assertFalse(any(reaped), 'A reaped leader no longer reserves the group identity')
            backend.close_all()

    def test_invalid_exec_and_timeout_specs_fail_before_process_creation(self):
        for argv in ('echo', ('',), ('echo', 'bad\0argument')):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                ExecRequest(argv)
        for timeout in (True, float('nan'), float('inf'), 0, -1, '1'):
            for spec in (SandboxSpec, lambda timeout_seconds: ExecRequest(('echo',), timeout_seconds)):
                with self.subTest(timeout=timeout, spec=spec), self.assertRaises(ValueError):
                    spec(timeout_seconds=timeout)

    def test_permission_revocation_after_catalog_read_has_no_workspace_effect(self):
        with TemporaryDirectory() as tmp:
            workspace = WorkspaceTools({'audit': tmp})
            space = ModularDecisionActionSpace(workspace.definitions)
            view = authorized_space(space)
            self.assertTrue(view.describe())
            space.permissions = ActionPermissions(operations=())
            tools = Mock(wraps=BoundDecisionTools(workspace.bindings))
            receipt = execute_operation(tools, view, OperationCall('workspace_list', {'root': 'audit'}))
            self.assertIsInstance(receipt.error, PermissionError)
            tools.execute.assert_not_called()

    def test_workspace_escape_and_escaped_unicode_obey_output_bound(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'inside').mkdir()
            (root / 'outside').write_text('secret')
            (root / 'inside' / 'link').symlink_to(root / 'outside')
            (root / 'inside' / 'text').write_text(('中😀"\\\t' * 2000))
            tools = WorkspaceTools({'audit': root / 'inside'}, max_output_bytes=4096)
            result = tools.bindings['workspace_read']({'root': 'audit', 'path': 'link'})
            self.assertFalse(result.data['available'])
            self.assertNotIn('secret', json.dumps(asdict(result)))
            result = tools.bindings['workspace_read']({'root': 'audit', 'path': 'text'})
            self.assertLessEqual(len(json.dumps(result.data).encode()), 4096)
            self.assertTrue(result.data['truncated'])
            self.assertGreater(result.data['next_offset'], 0)


class LocalInferenceAudit(unittest.TestCase):
    def test_text_chat_history_reaches_both_local_generation_adapters(self):
        from experiments.medical.transformers_runtime import LocalTransformerServing
        from experiments.medical.sglang_runtime import SGLangServing
        from tests.experiments.medical.test_medical_transformers import Tokenizer, Model, Peft, BASE as LOCAL_BASE, DECISION
        from tests.experiments.medical.test_medical_sglang import Engine
        from contextlib import nullcontext
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / 'config.json').write_text('{"max_position_embeddings":10000}')
            (base / 'model.safetensors').write_bytes(b'synthetic fixture')
            for cls in (LocalTransformerServing, SGLangServing):
                tokenizer = Tokenizer()
                rendered = []
                render = tokenizer.apply_chat_template
                tokenizer.apply_chat_template = lambda messages, **kw: (rendered.append(deepcopy(messages)) or render(messages, **kw))
                if cls is LocalTransformerServing:
                    runtime = (SimpleNamespace(inference_mode=nullcontext), tokenizer, Model([]), Peft)
                else:
                    runtime = (tokenizer, Engine())
                serving = cls(base, actor_model=LOCAL_BASE, decision_model=DECISION,
                    artifacts=None, native_model_class='causal_lm', device='cuda:0',
                    runtime_loader=lambda s: runtime)
                messages = ({'role': 'user', 'content': 'negative result: -3'},
                            {'role': 'assistant', 'content': 'Observed'},
                            {'role': 'user', 'content': 'Continue'})
                with self.subTest(adapter=cls.__name__):
                    serving.generate(GenerationRequest('actor', LOCAL_BASE, '', messages=messages, max_output_tokens=1))
                    self.assertEqual(rendered[-1], list(messages))

    def test_unsupported_native_constraints_fail_before_loading_local_weights(self):
        from experiments.medical.transformers_runtime import LocalTransformerServing
        from experiments.medical.sglang_runtime import SGLangServing
        for cls in (LocalTransformerServing, SGLangServing):
            loader = Mock(side_effect=AssertionError('Unsupported request must not acquire weights'))
            serving = cls('.', actor_model=BASE, decision_model=BASE,
                artifacts=None, native_model_class='causal_lm', runtime_loader=loader)
            for extra in ({'response_schema': {'type': 'object'}},
                          {'tools': ({'type': 'function', 'function': {'name': 'inspect', 'parameters': {}}},)}):
                with self.subTest(adapter=cls.__name__, extra=extra), self.assertRaises(NotImplementedError):
                    serving.generate(GenerationRequest('actor', BASE, 'prompt', **extra))
            loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
