"""Local Lab receipt identities survive storage and restart independently of filenames."""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from rsicomposer.application import Application, ApplicationConfig
from rsicomposer.harnesses.decision.tools import execute_operation
from rsicomposer.harnesses.decision.tools.lab import LabExecutionTools
from rsicomposer.loops.contracts.coordination import OperationResult
from rsicomposer.loops.contracts.decision import Decision, OperationArgumentError, OperationCall, OperationSpec
from rsicomposer.loops.contracts.execution import LabToolError
from rsicomposer.loops.contracts.jobs import JobOutcome, JobProgress, JobRef, JobStatus, OperationProgress
from rsicomposer.loops.contracts.lifecycle import ExecutionStatus
from rsicomposer.loops.contracts.profiles import Invariant
from rsicomposer.loops.decision_making.loop import _ControllerToolRouter
from rsicomposer.methods import MethodDefinition
from rsicomposer.runtime.execution.errors import WorkerLostError
from rsicomposer.runtime.orchestration.lab_group import LabGroup
from rsicomposer.snapshot.codec import decode, encode


class _Lab:
    def __init__(self, job=None):
        self.calls = []
        self.job = job

    def describe(self):
        return (OperationSpec('run', 'Test operation', {'type': 'object'}),)

    def execute(self, call):
        self.calls.append(call.call_id)
        return self.job or OperationResult('Completed', (), False)

    def poll_job(self, job):
        return JobOutcome(job, JobStatus.SUCCEEDED, OperationResult('Job completed', (), False))


class _SubmitOne:
    def __init__(self, call):
        self.call = call
        self.submitted = False

    def decide(self, window):
        if self.submitted:
            return Decision('finish')
        self.submitted = True
        return Decision.invoke(self.call)

    def state(self):
        return {'submitted': self.submitted}

    def restore(self, state):
        self.submitted = state['submitted']


def _assemblies(call, factory, capacity=1):
    return MethodDefinition.from_factories(controller=lambda *_: _SubmitOne(call),
        labs={'test': factory}, version='1').factory({'labs': {'test': {'max_concurrent': capacity}}})


class ReceiptStorageTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / 'lab_receipts'

    def group(self, *, job=None):
        lab = _Lab(job)
        group = LabGroup(receipts=self.root)
        group.add('test', lab)
        return group, lab

    def legacy_receipt(self, group, call, outcome):
        path = self.root / (call.call_id + '.json')
        path.write_text(json.dumps(encode({
            'schema': 24, 'name': call.name, 'arguments': call.arguments, 'outcome': outcome,
            'lab_state': None, 'profile_state': group.profile_state()['bindings']['test'],
            'sequence': 1, 'elapsed_seconds': 0.5,
        })), encoding='utf-8')
        return path

    def test_domain_rejection_is_a_journaled_receipt_even_after_reopen(self):
        group, lab = self.group()
        call = OperationCall('test.run', {'value': -1}, call_id='domain-error')
        error = LabToolError('Choose a positive value', data={'field': 'value'})
        with patch.object(lab, 'execute', side_effect=error) as invoke:
            result = group.execute(call)
            self.assertEqual((result.status, result.call_id, result.data),
                             (ExecutionStatus.FAILED, call.call_id, error.data))
            self.assertIn('Choose a positive value', result.error)
            self.assertEqual(group.execute(call), result)
            invoke.assert_called_once()
            self.assertEqual(invoke.call_args.args[0].call_id, call.call_id)
        reopened, reader = self.group()
        self.assertEqual(reopened.stored_receipt(call), result)
        self.assertEqual(reopened.in_flight_resolution(call), 'receipt')
        self.assertEqual(reopened.execute(call), result)
        self.assertEqual(reader.calls, [])

    def test_direct_lab_routes_bind_domain_errors_without_treating_them_as_evaluations(self):
        lab = _Lab()
        call = OperationCall('run', {}, call_id='original-id')
        data = {'field': 'value', 'quality_metrics': {'partial': 0.1}}
        with (patch.object(lab, 'execute', side_effect=LabToolError('invalid value', data=data)),
              patch.object(lab, 'primary_metric', create=True, side_effect=AssertionError('not an evaluation'))):
            for route in (LabExecutionTools(lab), _ControllerToolRouter(None, lab)):
                result = route.execute(call)
                self.assertEqual((result.status, result.call_id, result.data), (ExecutionStatus.FAILED, call.call_id, data))

    def test_ordinary_execution_faults_propagate_and_only_validators_reject_arguments(self):
        group, lab = self.group()
        call = OperationCall('run', {}, call_id='original-id')
        mounted = OperationCall('test.run', {}, call_id=call.call_id)
        tools = LabExecutionTools(lab)
        for error in (OperationArgumentError('not a validator'), ValueError('bug'), PermissionError('denied')):
            with self.subTest(error=type(error)), patch.object(lab, 'execute', side_effect=error):
                for route, issued in ((tools, call), (_ControllerToolRouter(None, lab), call), (group, mounted)):
                    with self.assertRaises(type(error)) as raised:
                        route.execute(issued)
                    self.assertIs(raised.exception, error)
                self.assertIsNone(group.stored_receipt(mounted))

        class Validator:
            def validate(self, call):
                raise OperationArgumentError('invalid envelope')

        with patch.object(lab, 'execute', side_effect=AssertionError('a validator must reject before execution')):
            attempt = execute_operation(tools, Validator(), call)
        self.assertIsNone(attempt.error)
        self.assertEqual((attempt.outcome.status, attempt.outcome.call_id), (ExecutionStatus.FAILED, call.call_id))

    def test_opaque_ids_have_stable_distinct_filenames_and_replay(self):
        ids = ('模型 @ v1!', 'Café', 'Cafe\u0301', 'Example-1', 'example-1', '界' * 200)
        calls = [OperationCall('test.run', {}, call_id=ident) for ident in ids]
        group, writer = self.group()
        reopened, reader = self.group()
        with patch.object(Path, 'iterdir', side_effect=AssertionError('Unexpected receipt scan')):
            outcomes = [group.execute(call) for call in calls]
            self.assertEqual([reopened.execute(call) for call in calls], outcomes)
        self.assertEqual(writer.calls, list(ids))
        self.assertEqual(reader.calls, [])
        self.assertEqual(set(self.root.iterdir()), {
            self.root / f'sha256-{sha256(ident.encode("utf-8")).hexdigest()}.receipt' for ident in ids
        })
        self.assertEqual([outcome.call_id for outcome in outcomes], list(ids))
        self.assertEqual([reopened.receipt_sequence(call) for call in calls], list(range(1, len(ids) + 1)))
        reused = OperationCall(calls[0].name, {'different': True}, call_id=ids[0])
        with self.assertRaisesRegex(ValueError, 'already used for a different operation'):
            reopened.execute(reused)
        self.assertEqual(reader.calls, [])

    def test_safe_legacy_terminal_replays_without_rewrite(self):
        group, lab = self.group()
        call = OperationCall('test.run', {}, call_id='模型 + v1 @ Lab')
        outcome = OperationResult('Legacy terminal', (), False, call_id=call.call_id)
        path = self.legacy_receipt(group, call, outcome)
        self.assertEqual(group.execute(call), outcome)
        self.assertEqual(group.receipt_elapsed(call), 0.5)
        self.assertEqual(lab.calls, [])
        self.assertEqual(set(self.root.iterdir()), {path})

    def test_legacy_job_terminal_update_prefers_canonical_receipt(self):
        job = JobRef('job-1', 'fixture')
        group, lab = self.group(job=job)
        call = OperationCall('test.run', {}, call_id='Legacy-Job.1')
        legacy_path = self.legacy_receipt(group, call, job)
        legacy = legacy_path.read_bytes()
        self.assertEqual(group.execute(call), job)
        result, _ = group.poll_job(call, job, decision_id='decision-1', after=0)
        reopened, reader = self.group()
        self.assertEqual(reopened.execute(call), result)
        self.assertEqual(reopened.receipt_sequence(call), 2)
        self.assertEqual(reopened.receipt_elapsed(call), 0.5)
        self.assertEqual(legacy_path.read_bytes(), legacy)
        self.assertEqual(lab.calls, [])
        self.assertEqual(reader.calls, [])

    def test_legacy_job_requires_exact_filename_spelling(self):
        job = JobRef('job-1', 'fixture')
        group, _ = self.group(job=job)
        call = OperationCall('test.run', {}, call_id='Legacy-Case')
        self.legacy_receipt(group, call, job)
        other = OperationCall(call.name, call.arguments, call_id='legacy-case')
        if (self.root / 'legacy-case.json').exists():  # Case-insensitive filesystem
            with self.assertRaisesRegex(ValueError, 'filename does not match'):
                group.stored_receipt(other)
        else:
            self.assertIsNone(group.stored_receipt(other))
        self.assertEqual(group.stored_receipt(call), job)

    def test_invalid_canonical_job_identity_never_falls_back(self):
        job = JobRef('job-1', 'fixture')
        group, writer = self.group(job=job)
        call = OperationCall('test.run', {}, call_id='Check Job!')
        self.assertEqual(group.execute(call), job)
        [path] = self.root.glob('*.receipt')
        row = decode(json.loads(path.read_text(encoding='utf-8')))
        self.assertEqual(row['call_id'], call.call_id)
        self.legacy_receipt(group, call, job)
        path.write_text(json.dumps(encode({**row, 'call_id': 'Other Job!'})), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'call_id does not match'):
            group.execute(call)
        self.assertEqual(group._recovery_receipt_evidence(call), {
            'receipt': 'unreadable_or_invalid', 'receipt_error_type': 'ValueError',
        })
        self.assertEqual(writer.calls, [call.call_id])

    def test_final_progress_delivery_retries_from_receipt_after_host_error(self):
        call = OperationCall('test.run', {}, call_id='one-shot')
        job = JobRef('worker', 'fixture')
        events = tuple(OperationProgress(call.call_id, job, i, 'progress', JobProgress(step=i)) for i in (1, 2))
        lab = _Lab(job)
        terminal = JobOutcome(job, JobStatus.SUCCEEDED, OperationResult('done', (), False, data={'count': 7}))
        cfg = ApplicationConfig('one-shot job', self.root.parent / 'run', {}, {})
        with (patch.object(lab, 'poll_job', side_effect=(terminal, WorkerLostError('worker released'))) as poll,
              patch.object(lab, 'job_events', create=True, return_value=events) as read_events,
              Application(cfg, modules=_assemblies(call, lambda *_: lab)) as app):
            group, host = app.assembly.lab_execution, app.assembly.host_execution
            observe = host.observe

            def partial_delivery(event):
                if event.output['sequence'] == 2:
                    raise OSError('interrupted final progress')
                observe(event)

            with patch.object(host, 'observe', side_effect=partial_delivery):
                with self.assertRaisesRegex(OSError, 'interrupted final progress'):
                    app.run(wait_for_jobs=True)
            saved = group.stored_receipt(call)
            sequence = group.receipt_sequence(call)
            self.assertEqual((saved.status, saved.data), (ExecutionStatus.COMPLETED, {'count': 7}))
            self.assertEqual(host.state.operation_progress, list(events[:1]))
            self.assertEqual(app.run(wait_for_jobs=True).status, 'finished')
            self.assertEqual(host.state.operation_progress, list(events))
            self.assertEqual(group.stored_receipt(call), saved)
            self.assertEqual(group.receipt_sequence(call), sequence)
            self.assertEqual(group.poll_job(call, job, decision_id='any', after=2), (saved, ()))
            poll.assert_called_once_with(job)
            read_events.assert_called_once_with(job, after=0)

    def test_live_and_terminal_progress_replay_across_reopen_without_provider(self):
        call = OperationCall('test.run', {}, call_id='progress')
        job = JobRef('job', 'fixture')
        one, two = (OperationProgress(call.call_id, job, i, 'progress', JobProgress(step=i)) for i in (1, 2))
        group, lab = self.group(job=job)
        self.assertEqual(group.execute(call), job)
        done = JobOutcome(job, JobStatus.SUCCEEDED, OperationResult('final', (), False))
        with (patch.object(lab, 'poll_job', side_effect=(JobOutcome(job, JobStatus.RUNNING), done)) as poll,
              patch.object(lab, 'job_events', create=True, side_effect=((one,), (two,)))):
            self.assertEqual(group.poll_job(call, job, decision_id='d', after=0), (None, (one,)))
            self.assertEqual(group.poll_job(call, job, decision_id='d', after=0), (None, (one,)))
            result, events = group.poll_job(call, job, decision_id='d', after=1)
            self.assertEqual(events, (two,))
            self.assertEqual(poll.call_count, 2)
        reader, provider = self.group(job=job)
        with (patch.object(provider, 'poll_job', side_effect=AssertionError('worker gone')),
              patch.object(provider, 'job_events', create=True, side_effect=AssertionError('worker gone'))):
            self.assertEqual(reader.poll_job(call, job, decision_id='d', after=0), (result, (one, two)))
            self.assertEqual(reader.poll_job(call, job, decision_id='d', after=1), (result, (two,)))
            wrong = JobRef(job.job_id, 'other-backend')
            self.assertIsNone(reader.in_flight_resolution(call, job=wrong))
            with self.assertRaisesRegex(ValueError, 'journaled Lab submission'):
                reader.poll_job(call, wrong, decision_id='d', after=0)

    def test_settling_one_reopened_job_preserves_the_other_jobs_once_key(self):
        effects = []

        class Lab(_Lab):
            def invariants(self):
                return (Invariant.once_per('round'),)

            def execute(self, call):
                effects.append(call.call_id)
                return JobRef(call.call_id, 'fixture')

            def poll_job(self, job):
                return JobOutcome(job, JobStatus.CANCELLED)

            def reattach_job(self, job):
                raise WorkerLostError('fixture worker lost')

        def open_group():
            group = LabGroup(receipts=self.root)
            group.add('test', Lab())
            return group

        def call(key, ident):
            return OperationCall('test.run', {'round': key}, call_id=ident)

        group = open_group()
        first, second = call(1, 'first'), call(2, 'second')
        first_job = group.execute(first)
        second_job = group.execute(second)
        reader = open_group()
        cancel, _ = reader.poll_job(second, second_job, decision_id='d', after=0)
        self.assertEqual((cancel.status, reader.receipt_sequence(second)), (ExecutionStatus.CANCELLED, 3))
        self.assertIn('in-flight once_per', reader.execute(call(1, 'blocked')).error)
        self.assertEqual(reader.execute(call(2, 'replacement')), JobRef('replacement', 'fixture'))
        failed = reader.reattach_job(first, first_job)
        self.assertEqual((failed.status, failed.data['effects_uncertain']), (ExecutionStatus.FAILED, True))
        self.assertEqual(reader.execute(call(1, 'recovered')), JobRef('recovered', 'fixture'))
        self.assertIn('in-flight once_per', reader.execute(call(2, 'still-blocked')).error)
        self.assertEqual(effects, ['first', 'second', 'replacement', 'recovered'])

    def test_once_per_survives_live_job_resume_and_failed_terminal_releases_key(self):
        first = OperationCall('test.run', {'round': 1}, call_id='first')
        second = OperationCall('test.run', {'round': 1}, call_id='second')
        third = OperationCall('test.run', {'round': 1}, call_id='third')
        effects, reattached = [], []

        class Jobs(_Lab):
            concurrent_state_safe = True
            status = JobStatus.RUNNING

            def invariants(self):
                return (Invariant.once_per('round'),)

            def execute(self, call):
                effects.append(call.call_id)
                return JobRef(call.call_id, 'fixture')

            def reattach_job(self, job):
                reattached.append(job)

            def poll_job(self, job):
                if self.status == JobStatus.SUCCEEDED:
                    return super().poll_job(job)
                return JobOutcome(job, self.status, error='fixture failure' if self.status == JobStatus.FAILED else None)

            def state(self):
                return {}

            def restore(self, state):
                pass

        cfg = ApplicationConfig('once per round', self.root.parent / 'durable', {}, {})
        modules = _assemblies(first, lambda *_: Jobs(), capacity=2)
        job = JobRef(first.call_id, 'fixture')
        with Application(cfg, modules=modules, durable=True) as app:
            from verification.test_run_control import ImmediateExecutor
            with patch('rsicomposer.host.execution.ThreadPoolExecutor', ImmediateExecutor):
                self.assertEqual(app.runner.step(cfg.objective).job, job)
            self.assertEqual(effects, [first.call_id])
        with Application.resume(cfg.data_dir, modules=modules) as app:
            self.assertEqual(reattached, [job])
            group = app.assembly.lab_execution
            rejected = group.execute(second)
            self.assertEqual((rejected.status, group.stored_receipt(second)), (ExecutionStatus.FAILED, rejected))
            self.assertIn('in-flight once_per', rejected.error)
            self.assertEqual(effects, [first.call_id])
            group.labs['test'].status = JobStatus.FAILED
            with patch.object(group, '_write_receipt', side_effect=OSError('fixture disk error')):
                with self.assertRaisesRegex(OSError, 'disk error'):
                    group.poll_job(first, job, decision_id='d', after=0)
            blocked = group.execute(OperationCall(first.name, first.arguments, call_id='after-write-failure'))
            self.assertIn('in-flight once_per', blocked.error)
            self.assertEqual(group.poll_job(first, job, decision_id='d', after=0)[0].status, ExecutionStatus.FAILED)
            self.assertEqual(group.execute(second), rejected)  # A new attempt needs a new ID.
            next_job = group.execute(third)
            group.labs['test'].status = JobStatus.SUCCEEDED
            self.assertEqual(group.poll_job(third, next_job, decision_id='d', after=0)[0].status, ExecutionStatus.COMPLETED)
            later = group.execute(OperationCall(first.name, first.arguments, call_id='after-success'))
            self.assertIn('violates once_per', later.error)
            self.assertEqual(effects, [first.call_id, third.call_id])

    def test_history_rejection_survives_measurement_and_journal_replay(self):
        effects = []
        class Lab:
            def describe(self):
                return tuple(OperationSpec(name, name, {}) for name in ('measure', 'select'))

            def invariants(self):
                return (Invariant.measured_selection(),)

            def evaluation_operations(self):
                return ('measure',)

            def candidate_identity_fields(self):
                return {'measure': ('candidate_id',), 'select': ('candidate_id',)}

            def execute(self, call):
                effects.append(call.call_id)
                return OperationResult('ok', (), False)

        def open_group():
            group = LabGroup(receipts=self.root)
            group.add('test', Lab())
            return group

        group = open_group()
        denied = OperationCall('test.select', {'candidate_id': 'x'}, call_id='choice')
        measure = OperationCall('test.measure', denied.arguments, call_id='score')
        failed = group.execute(denied)
        self.assertEqual((failed.status, group.stored_receipt(denied)), (ExecutionStatus.FAILED, failed))
        self.assertEqual(group.execute(measure).status, ExecutionStatus.COMPLETED)
        reopened = open_group()
        reopened.restore_journaled_calls((measure, denied))
        self.assertEqual(reopened.execute(denied), failed)
        self.assertEqual(reopened.execute(OperationCall(denied.name, denied.arguments, call_id='new')).status,
                         ExecutionStatus.COMPLETED)
        self.assertEqual(effects, ['score', 'new'])
