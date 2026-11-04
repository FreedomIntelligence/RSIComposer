"""Open Lab calls run through actual policy/receipts and owned local processes."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from rsisys.adapters.execution import LabExecutionTools
from rsisys.adapters.local_lab import LocalProcessLab
from rsisys.application import Application, ApplicationConfig
from rsisys.config import DemoConfig, build_demo
from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.decision import Decision, OperationCall, OperationSpec
from rsisys.loops.contracts.jobs import JobRef
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import GenerationResponse
from rsisys.loops.decision_making.action_space import ModularDecisionActionSpace
from rsisys.loops.decision_making.action_space.access import ActionPermissions, authorized_space
from rsisys.loops.decision_making.harness.tools import execute_operation
from rsisys.runtime.artifacts import InMemoryArtifactStore
from rsisys.runtime.local_process import LocalProcessBackend
from rsisys.runtime.run_policy import TimeRunPolicy
from rsisys.runtime.sandbox import SandboxSpec, ExecRequest
from rsisys.research import ResearchAssembly
from rsisys.snapshot.local import validate_local_application


ROOT = Path(__file__).resolve().parents[1]


class NativeLab:
    """No Actor, private harness state, internal definitions or training enums."""
    def __init__(self):
        self.calls, self.descriptions, self.closed = [], 0, 0
        self.spec = OperationSpec('invent_an_algorithm', 'Execute a Lab-specific research program', {
            'type': 'object', 'properties': {'source': {'type': 'string'}},
            'required': ['source'], 'additionalProperties': False,
        })

    def describe(self):
        self.descriptions += 1
        return (self.spec,)

    def execute(self, call):
        if set(call.arguments) != {'source'} or not isinstance(call.arguments['source'], str):
            raise ValueError('Lab source must be text')
        self.calls.append(deepcopy(call))
        return OperationResult('native result', (), False, data={
            'quality_metrics': {'score': 0.0}, 'evidence_refs': ['native:result'],
            'native': {'algorithm': call.arguments['source'], 'nested': [False, None]},
        })

    def close(self):
        self.closed += 1


class OpenLabAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.demo = DemoConfig.load(ROOT / 'configs/demo.json')
        self.config = replace(ApplicationConfig.load(ROOT / 'configs/local.json'),
                              data_dir=self.root / 'run', execution='mock')

    @contextmanager
    def owner(self, kind, lab, **options):
        if kind == 'application':
            config = replace(self.config, data_dir=self.root / ('run-' + str(time.monotonic_ns())),
                             roles={'decision_maker': self.config.roles['decision_maker']})
            with Application(config, lab_factory=lambda context: lab, **options) as app:
                yield app.runner, app
        else:
            config = replace(self.demo, roles={'decision_maker': self.demo.roles['decision_maker']})
            runner, _ = build_demo(config, lab_factory=lambda context: lab, **options)
            with runner:
                yield runner, runner

    def wire_call(self, runner, name='invent_an_algorithm', arguments=None):
        policy = runner.decision_making.harness.policy
        requests = []
        def generate(request):
            requests.append(request)
            return GenerationResponse(json.dumps({'kind': 'invoke', 'operation': name,
                'arguments': arguments if arguments is not None else {'source': 'my entirely new algorithm'}}),
                policy.model, simulated=False)
        policy.backend.generate = generate
        return requests

    def test_both_roots_expose_native_catalog_without_actor_or_domain_introspection(self):
        for kind in ('application', 'demo'):
            lab = NativeLab()
            with self.subTest(kind=kind), patch('rsisys.assembly._decision_directives',
                    side_effect=AssertionError('No domain introspection')):
                with self.owner(kind, lab) as (runner, _):
                    self.assertEqual(lab.calls, [])  # construction cannot submit work
                    space = runner.decision_making.action_space
                    self.assertEqual(space.directive_modes(), ())
                    self.assertIn('invent_an_algorithm', [s.name for s in space.describe()])
                    self.assertNotIn('prepare_data', [s.name for s in space.describe()])
                    requests = self.wire_call(runner)
                    receipts = []
                    runner.decision_making.harness.add_receipt_listener(receipts.append)
                    step = runner.step('Improve the fixed Lab objective')
                    self.assertEqual(step.kind, 'progressed')
                    raw = step.event['operation_result']['data']
                    self.assertEqual(raw['quality_metrics']['score'], 0.0)
                    self.assertEqual(raw['evidence_refs'], ['native:result'])
                    self.assertEqual(receipts[0].result['data'], raw)
                    self.assertIsNotNone(receipts[0].decision_id)
                    self.assertIsNotNone(receipts[0].issuing_model)
                    context = json.loads(requests[0].prompt)
                    self.assertEqual(context['directive_modes'], [])
                    spec = next(s for s in context['operations'] if s['name'] == lab.spec.name)
                    self.assertEqual(spec['input_schema'], lab.spec.input_schema)
                    self.assertEqual(spec['description'], lab.spec.description)
                    self.assertNotIn('no new experiment can be submitted', requests[0].system)
                    self.assertIn('your own data construction, SFT, RL, evaluation', requests[0].system)
                    self.assertEqual(lab.descriptions, 1)
                self.assertEqual(lab.closed, 1)

    def test_permissions_hide_and_deny_effects_at_dispatch(self):
        lab = NativeLab()
        with self.owner('demo', lab, action_permissions=ActionPermissions(operations=frozenset())) as (runner, _):
            requests = self.wire_call(runner)
            with self.assertRaises(PermissionError):
                runner.step('objective')
            self.assertEqual(json.loads(requests[0].prompt)['operations'], [])
            self.assertEqual(lab.calls, [])

    def test_parameter_admission_happens_before_lab_effects(self):
        lab = NativeLab()
        permissions = ActionPermissions(check=lambda r: r.action is None or
            r.kind != 'operation' or r.action.arguments.get('source') != 'denied')
        with self.owner('demo', lab, action_permissions=permissions) as (runner, _):
            self.wire_call(runner, arguments={'source': 'denied'})
            with self.assertRaises(PermissionError):
                runner.step('objective')
            self.assertEqual(lab.calls, [])

    def test_invalid_provider_and_later_factory_failure_close_once(self):
        for kind in ('application', 'demo'):
            lab = NativeLab()
            lab.spec = 'invalid'
            with self.subTest(kind=kind), self.assertRaisesRegex(TypeError, 'OperationSpec'):
                with self.owner(kind, lab):
                    pass
            self.assertEqual(lab.closed, 1)
            lab = NativeLab()
            def failing(context, exploration):
                raise RuntimeError('decision factory failed')
            with self.assertRaisesRegex(RuntimeError, 'decision factory failed'):
                with self.owner(kind, lab, decision_factory=failing):
                    pass
            self.assertEqual(lab.closed, 1)

    def test_no_implicit_snapshot_or_resume_of_lab_work(self):
        lab = NativeLab()
        with self.owner('application', lab) as (_, app):
            directory = app.config.data_dir
            self.assertTrue((directory / 'lab_execution.json').exists())
            with self.assertRaisesRegex(TypeError, 'Lab execution.*snapshot/recovery'):
                validate_local_application(app)
        factory = Mock(return_value=lab)
        with self.assertRaisesRegex(TypeError, 'Lab execution.*restore'):
            Application.resume(directory)
        with self.assertRaisesRegex(TypeError, 'durable execution is unsupported'):
            Application.resume(directory, lab_factory=factory)
        factory.assert_not_called()
        with self.assertRaisesRegex(TypeError, 'durable execution is unsupported'):
            Application(replace(self.config, data_dir=self.root / 'durable'), durable=True, lab_factory=factory)
        factory.assert_not_called()

    def test_lab_only_json_config_load_and_legacy_actor_requirement(self):
        raw = self.config.to_dict()
        del raw['roles']['actor']
        config = ApplicationConfig.from_dict(raw, base_dir=self.root)
        with Application(config, lab_factory=lambda c: NativeLab()):
            pass
        config = replace(config, data_dir=self.root / 'legacy')
        with self.assertRaisesRegex(ValueError, 'Legacy exploration requires an actor'):
            Application(config)

    def test_minimal_synchronous_provider_needs_only_describe_and_execute(self):
        source = NativeLab()
        provider = SimpleNamespace(describe=source.describe, execute=source.execute)
        with self.owner('demo', provider) as (runner, _):
            self.wire_call(runner)
            self.assertEqual(runner.step('objective').kind, 'progressed')
            self.assertIsNone(runner._execution_shutdown)
        self.assertEqual(len(source.calls), 1)

    def test_program_failure_and_repair_flow_through_decision_feedback_memory_and_archive(self):
        # Deterministic model fixture, real CPU programs, actual default prompts
        # and research pipeline. This is not an autonomous real-model trial.
        config = replace(self.config, execution='local_process',
                         roles={'decision_maker': self.config.roles['decision_maker']})
        labs = []
        def factory(context):
            labs.append(LocalProcessLab(context.sandbox, context.artifacts, timeout_seconds=10))
            return labs[-1]
        with Application(config, lab_factory=factory, research=ResearchAssembly()) as app:
            lab = labs[0]
            harness = app.runner.decision_making.harness
            requests, receipts = [], []
            harness.add_receipt_listener(receipts.append)
            actions = iter(('write_bad', 'start_bad', 'poll', 'result',
                            'write_fixed', 'start_fixed', 'poll', 'result', 'finish'))
            def generate(request):
                requests.append(request)
                context = json.loads(request.prompt)
                action = next(actions)
                if action == 'finish':
                    payload = {'kind': 'finish'}
                else:
                    if action.startswith('write'):
                        name, args = 'lab_write_file', {'path': 'experiment.py', 'content':
                            'raise ValueError("fix this observed failure")\n' if action == 'write_bad'
                            else 'print("fixed result")\n'}
                    elif action.startswith('start'):
                        name, args = 'lab_start', {'key': action, 'argv': [sys.executable, 'experiment.py']}
                    else:
                        ident = next(row['data']['job_id'] for row in reversed(context['operation_results'])
                                     if 'job_id' in row['data'])
                        if action == 'poll':
                            # Synchronize this short fixture, not a host execution rule.
                            lab.backend.wait(lab._jobs[ident][0])
                        name, args = 'lab_' + action, {'job_id': ident}
                    payload = {'kind': 'invoke', 'operation': name, 'arguments': args}
                return GenerationResponse(json.dumps(payload), harness.current_model(), simulated=False)
            harness.policy.backend.generate = generate
            for _ in range(9):
                last = app.runner.step('fixed research objective')
            self.assertEqual(last.kind, 'finished')
            self.assertTrue(last.event['run_stop']['shutdown_complete'])
            self.assertEqual(len(receipts), 8)
            self.assertEqual(receipts[3].request['name'], 'lab_result')
            self.assertEqual(receipts[3].result['status'], ExecutionStatus.FAILED)
            self.assertEqual(receipts[7].result['status'], ExecutionStatus.COMPLETED)
            self.assertIn('fix this observed failure', requests[4].prompt)
            self.assertIn('fixed result', requests[-1].prompt)
            service = app.runner.decision_making.feedback_service
            until = time.monotonic() + 5
            while time.monotonic() < until:
                service.refresh()
                if all(row['status'] in ('completed', 'archived') for row in service.learning.state['jobs'].values()):
                    break
                time.sleep(0.01)
            rows = list(service.learning.state['jobs'].values())
            collected = [row for row in rows if row['raw_input']['receipt']['operation']['name'] == 'lab_result']
            self.assertEqual(len(collected), 2)
            self.assertTrue(all(row['status'] == 'completed' for row in collected))
            self.assertTrue(service.working_memory())
            self.assertGreaterEqual(len(service.trajectories()), 8)


class LabPortAdapterTests(unittest.TestCase):
    def test_provider_catalog_is_frozen_and_input_mutations_are_isolated(self):
        lab = NativeLab()
        tools = LabExecutionTools(lab)
        lab.spec.input_schema['properties'].clear()
        space = ModularDecisionActionSpace(tools.definitions, directive_modes=())
        self.assertIn('source', space.describe()[0].input_schema['properties'])
        call = OperationCall('invent_an_algorithm', {'source': 'custom'})
        def execute(value):
            value.arguments['source'] = 'mutated by Lab'
            return OperationResult('done', (), False)
        lab.execute = execute
        result = execute_operation(tools, space, call)
        self.assertIsNone(result.error)
        self.assertEqual(call.arguments['source'], 'custom')
        self.assertEqual(result.call.arguments['source'], 'custom')

    def test_bad_catalogs_and_collisions_fail_before_execution(self):
        for specs, error in ((['bad'], TypeError),
                             ([OperationSpec('duplicate', '', {}), OperationSpec('duplicate', '', {})], ValueError),
                             ([OperationSpec(' ', '', {})], ValueError),
                             ([OperationSpec('bad', '', {'value': float('nan')})], ValueError)):
            execute = Mock()
            with self.subTest(specs=specs), self.assertRaises(error):
                LabExecutionTools(SimpleNamespace(describe=lambda: specs, execute=execute))
            execute.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'unique'):
            LabExecutionTools(NativeLab(), extra_bindings={'invent_an_algorithm': Mock()})

    def test_malformed_envelope_and_native_argument_errors_are_recordable_failures(self):
        lab = NativeLab()
        tools = LabExecutionTools(lab)
        space = ModularDecisionActionSpace(tools.definitions, directive_modes=())
        for arguments in ({1: 'bad'}, {'source': float('nan')}, {'wrong': 'shape'}):
            with self.subTest(arguments=arguments):
                result = execute_operation(tools, space, OperationCall(lab.spec.name, arguments))
                self.assertIsNotNone(result.error)
                self.assertEqual(result.outcome.status, ExecutionStatus.FAILED)
        self.assertEqual(lab.calls, [])

    def test_optional_jobref_remains_submission_with_external_resume_semantics(self):
        lab = NativeLab()
        lab.execute = lambda call: JobRef('job', 'remote')
        tools = LabExecutionTools(lab)
        space = ModularDecisionActionSpace(tools.definitions, directive_modes=())
        execution = execute_operation(tools, space, OperationCall(lab.spec.name, {'source': 'code'}))
        self.assertEqual(execution.observation().data['status'], 'submitted')
        self.assertIsInstance(execution.outcome, JobRef)


class LocalProgrammableLabTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.backend = LocalProcessBackend(self.root / 'workspaces', max_output_bytes=8192)
        self.artifacts = InMemoryArtifactStore()
        self.lab = LocalProcessLab(self.backend, self.artifacts, timeout_seconds=10)
        self.addCleanup(self.backend.close_all)
        self.addCleanup(self.lab.close)

    def call(self, name, **arguments):
        return self.lab.execute(OperationCall('lab_' + name, arguments))

    def wait(self, job_id):
        until = time.monotonic() + 5
        while time.monotonic() < until:
            result = self.call('poll', job_id=job_id)
            if result.data['state'] != 'running':
                return result
            time.sleep(0.01)
        self.fail('Local program did not terminate')

    def start(self, key, *argv, **options):
        return self.call('start', key=key, argv=list(argv), **options).data['job_id']

    def test_editable_training_and_evaluation_are_programs_not_registered_actions(self):
        # These are CPU arithmetic fixtures, not real SFT/RL or validation.
        first = 'import json\nprint(json.dumps({"implementation":"custom-sft", "score":0}))\n'
        second = 'import json\nprint(json.dumps({"implementation":"custom-rl-and-evaluate", "score":7}))\n'
        self.call('write_file', path='research/train.py', content=first)
        one = self.wait(self.start('first', sys.executable, 'research/train.py'))
        self.call('write_file', path='research/train.py', content=second)
        two = self.wait(self.start('second', sys.executable, 'research/train.py'))
        self.assertEqual(json.loads(one.data['stdout'])['score'], 0)
        self.assertEqual(json.loads(two.data['stdout'])['score'], 7)
        self.assertNotIn('quality_metrics', two.data)  # raw stdout is not authoritative evaluation
        self.assertFalse(two.simulated)
        exported = self.call('export', path='research/train.py')
        self.assertEqual(self.artifacts.get(exported.artifacts[0]), second.encode())
        saved = self.artifacts.get(two.artifacts[0])
        self.lab.close()
        self.assertIn(b'custom-rl-and-evaluate', saved)
        self.assertEqual(self.artifacts.get(exported.artifacts[0]), second.encode())

    def test_real_code_failure_is_feedback_then_repair_succeeds(self):
        self.call('write_file', path='experiment.py', content='raise RuntimeError("observable failure")\n')
        failure = self.wait(self.start('broken', sys.executable, 'experiment.py'))
        self.assertEqual(failure.status, ExecutionStatus.FAILED)
        self.assertIn('observable failure', failure.error)
        self.assertNotEqual(failure.data['exit_code'], 0)
        self.call('write_file', path='experiment.py', content='print("repaired")\n')
        fixed = self.wait(self.start('fixed', sys.executable, 'experiment.py'))
        self.assertEqual(fixed.status, ExecutionStatus.COMPLETED)
        self.assertEqual(fixed.data['stdout'], 'repaired\n')
        archived_failure = json.loads(self.artifacts.get(failure.artifacts[0]))
        self.assertIn('observable failure', archived_failure['stderr'])

    def test_submission_key_never_reexecutes_and_conflicting_reuse_is_rejected(self):
        arguments = {'key': 'once', 'argv': [sys.executable, '-c', 'open("counter", "a").write("x")']}
        first = self.lab.execute(OperationCall('lab_start', arguments))
        self.wait(first.data['job_id'])
        second = self.lab.execute(OperationCall('lab_start', arguments))
        self.assertEqual(second.data['job_id'], first.data['job_id'])
        self.assertEqual(self.call('read_file', path='counter').data['content'], 'x')
        with self.assertRaisesRegex(ValueError, 'different arguments'):
            self.call('start', key='once', argv=[sys.executable, '-c', 'print(2)'])
        self.assertEqual(first.data['state'], 'submitted')
        self.assertIn('completion is not confirmed', first.summary)

    def test_host_timeout_and_unknown_job_cannot_be_overridden(self):
        with self.assertRaisesRegex(ValueError, 'host limit'):
            self.start('excess', sys.executable, '-c', 'print(1)', timeout_seconds=11)
        for name in ('poll', 'cancel'):
            with self.assertRaisesRegex(ValueError, 'owned Lab job'):
                self.call(name, job_id='another-labs-job')
        result = self.wait(self.start('timeout', sys.executable, '-c', 'import time; time.sleep(30)', timeout_seconds=0.05))
        self.assertEqual(result.status, ExecutionStatus.FAILED)
        self.assertIn('timed out', result.error)

    def test_read_windows_paths_and_file_exports_have_truthful_bounds(self):
        self.call('write_file', path='utf8.txt', content='abcdef')
        first = self.call('read_file', path='utf8.txt', max_bytes=3)
        self.assertEqual(first.data['content'], 'abc')
        self.assertEqual(first.data['next_offset'], 3)
        self.assertEqual(self.call('read_file', path='utf8.txt', offset=3).data['content'], 'def')
        for path in ('../escape', str(self.root / 'absolute'), '.', ''):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.call('write_file', path=path, content='no')
        (self.lab.workspace / 'outside').symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(OSError):
            self.call('write_file', path='outside/escape', content='no')
        self.assertFalse((self.root / 'escape').exists())

    def test_cancel_close_only_owned_work_and_reject_new_admission(self):
        other = self.backend.create(SandboxSpec('local', 10))
        other_job = self.backend.exec(other, ExecRequest((sys.executable, '-c', 'import time; time.sleep(30)'), 10))
        ident = self.start('owned', sys.executable, '-c', 'import time; time.sleep(30)')
        ack = self.call('cancel', job_id=ident)
        self.assertTrue(ack.data['cancellation_requested'])
        cancelled = self.wait(ident)
        self.assertEqual(cancelled.status, ExecutionStatus.CANCELLED)
        self.lab.close()
        self.assertEqual(self.backend.poll(other_job).status.value, 'running')
        self.assertEqual(self.backend.active_count, 1)
        self.assertTrue(self.lab.shutdown('cancel'))
        self.lab.close()

    def test_evidence_failure_still_cleans_owned_processes(self):
        ended = self.start('done', sys.executable, '-c', 'print(1)')
        ref = self.lab._jobs[ended][0]
        self.backend.wait(ref)
        running = self.start('running', sys.executable, '-c', 'import time; time.sleep(30)')
        self.artifacts.put = Mock(side_effect=OSError('store unavailable'))
        with self.assertRaisesRegex(OSError, 'store unavailable'):
            self.lab.close()
        self.assertEqual(self.backend.active_count, 0)
        self.assertEqual(self.backend.poll(self.lab._jobs[running][0]).status.value, 'cancelled')
        self.artifacts.put = InMemoryArtifactStore().put

    def test_host_job_limit_rejects_new_code_but_allows_observation(self):
        self.lab.max_jobs = 1
        job = self.start('one', sys.executable, '-c', 'print(1)')
        self.wait(job)
        with self.assertRaisesRegex(PermissionError, 'admission limit'):
            self.start('two', sys.executable, '-c', 'print(2)')
        self.assertEqual(self.call('poll', job_id=job).data['state'], 'succeeded')

    def test_stdout_bounds_are_explicit_in_live_and_archived_evidence(self):
        job = self.start('long-output', sys.executable, '-c', 'print("x" * 100000)')
        result = self.wait(job)
        self.assertLessEqual(len(result.data['stdout'].encode()), 8192)
        self.assertIn('Output was truncated', result.data['stderr'])
        archived = json.loads(self.artifacts.get(result.artifacts[0]))
        self.assertEqual(archived['stdout'], result.data['stdout'])
        self.assertIn('Output was truncated', archived['stderr'])

    def test_drain_blocks_admission_but_does_not_cancel_running_work(self):
        ident = self.start('drain', sys.executable, '-c', 'import time; time.sleep(.1); print("done")')
        self.assertFalse(self.lab.shutdown('drain'))
        with self.assertRaisesRegex(RuntimeError, 'shutting down'):
            self.call('write_file', path='late.py', content='pass')
        result = self.wait(ident)
        self.assertEqual(result.data['state'], 'succeeded')
        self.assertTrue(self.lab.shutdown('drain'))


class OpenLabHostLifecycleTests(unittest.TestCase):
    setUp = OpenLabAssemblyTests.setUp
    owner = OpenLabAssemblyTests.owner
    wire_call = OpenLabAssemblyTests.wire_call

    def test_deadline_cancels_program_then_confirms_cleanup_on_later_ticks(self):
        backend = LocalProcessBackend(self.root / 'jobs', max_output_bytes=4096)
        self.addCleanup(backend.close_all)
        lab = LocalProcessLab(backend, InMemoryArtifactStore(), timeout_seconds=30)
        with self.owner('demo', lab, run_policy=TimeRunPolicy(1)) as (runner, _):
            now = [0.0]
            runner._clock = lambda: now[0]
            self.wire_call(runner, 'lab_start', {'key': 'slow',
                'argv': [sys.executable, '-c', 'import time; time.sleep(30)']})
            self.assertEqual(runner.step('objective').kind, 'progressed')
            now[0] = 2
            result = runner.step('objective')
            self.assertFalse(result.event['run_stop']['shutdown_complete'])
            until = time.monotonic() + 5
            while result.kind != 'finished' and time.monotonic() < until:
                time.sleep(0.01)
                result = runner.step('objective')
            self.assertEqual(result.kind, 'finished')
            self.assertTrue(result.event['run_stop']['shutdown_complete'])
            self.assertFalse(result.event['run_stop']['unresolved_execution'])
            self.assertTrue(all(item.data['state'] == 'cancelled' for item in lab._terminal.values()))
            with self.assertRaisesRegex(RuntimeError, 'shutting down'):
                lab.execute(OperationCall('lab_start', {'key': 'new', 'argv': ['echo', 'no']}))


if __name__ == '__main__':
    unittest.main()
