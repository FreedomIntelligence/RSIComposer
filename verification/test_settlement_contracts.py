"""Offline regressions for Controller/Lab settlement and receipt ownership."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from rsicomposer.application import Application
from rsicomposer.loops.contracts.coordination import OperationResult
from rsicomposer.loops.contracts.decision import Decision
from rsicomposer.loops.contracts.jobs import JobOutcome, JobRef, JobStatus
from rsicomposer.runtime.orchestration.run_policy import TimeRunPolicy
from rsicomposer.runtime.state.recovery import RecoveryRequired
from rsicomposer.runtime.storage.codec import decode, encode
from verification import test_run_control as run


class SettlementContractsTest(unittest.TestCase):
    setup_application = run.RunControlTest.setup_application
    step = run.RunControlTest.step

    def _submit(self):
        with patch('rsicomposer.host.execution.ThreadPoolExecutor', run.ImmediateExecutor):
            return self.step()

    def test_poll_only_lab_settles_on_cancel_without_reported_cancellation(self):
        class PollOnlyLab(run.Lab):
            cancel_job = None

        lab = PollOnlyLab(backend=True)
        app = self.setup_application(lab, run.Policy((run.operation(),)), TimeRunPolicy(1.))
        self.assertEqual(self._submit().job, lab.job)
        self.clock.now = 2.
        stop = self.step()
        self.assertEqual((stop.kind, stop.job), ('waiting', lab.job))
        self.assertFalse(stop.event['run_stop']['shutdown_complete'])
        lab.settled = True
        receipt = self.step().event['operation_result']
        self.assertEqual((receipt['call_id'], receipt['status']), ('first', 'completed'))
        self.assertEqual(lab.cancels, [])
        self.assertTrue(self.step().event['run_stop']['shutdown_complete'])
        self.assertFalse(app.runner.busy)

    def test_cancel_errors_propagate_and_successful_requests_are_not_repeated(self):
        class BrokenCancelLab(run.Lab):
            def cancel_job(self, job):
                self.cancels.append(job)
                if len(self.cancels) == 1:
                    raise TypeError('Lab does not implement cancel_job(job)')
                return JobOutcome(job, JobStatus.RUNNING)

        lab = BrokenCancelLab(backend=True)
        app = self.setup_application(lab, run.Policy((run.operation(),)), TimeRunPolicy(1.))
        self.assertEqual(self._submit().job, lab.job)
        self.clock.now = 2.
        with self.assertRaisesRegex(TypeError, 'does not implement'):
            self.step()
        self.assertTrue(app.runner.busy)
        self.assertEqual(self.step().job, lab.job)
        self.assertEqual(self.step().job, lab.job)
        self.assertEqual(lab.cancels, [lab.job, lab.job])
        lab.settled = True
        self.assertEqual(self.step().event['operation_result']['status'], 'cancelled')
        self.assertTrue(self.step().event['run_stop']['shutdown_complete'])

    def test_simultaneous_native_explicit_and_lab_work_have_independent_settlement(self):
        class MixedPolicy(run.Policy):
            explicit = JobRef('controller-explicit', 'fixture')
            native = JobRef('controller-native', 'fixture')

            def __init__(self):
                super().__init__()
                self.explicit_ready = self.native_ready = False
                self.actions = []

            def decide(self, window):
                self.turns += 1
                if self.turns == 1:
                    return Decision('invoke', operations=(run.operation(),), decision_id='lab-decision')
                return Decision('wait', job=self.explicit, decision_id='wait-decision')

            def inference_ready(self):
                return self.turns == 1

            def pending_research_job(self):
                return self.native if self.turns >= 2 and not self.native_ready else None

            def can_poll_job(self, job):
                return job == self.explicit

            def poll_job(self, job):
                self.actions.append(('poll', job))
                return (JobOutcome(job, JobStatus.SUCCEEDED, True) if self.explicit_ready
                        else JobOutcome(job, JobStatus.RUNNING))

            def cancel_job(self, job):
                self.actions.append(('cancel', job))
                return JobOutcome(job, JobStatus.RUNNING)

            def drain_research(self):
                self.actions.append(('native-drain', self.native))
                return False

            def cancel_research(self):
                self.actions.append(('native-cancel', self.native))
                return False

        for mode in ('drain', 'cancel'):
            with self.subTest(mode=mode):
                lab, policy = run.Lab(backend=True), MixedPolicy()
                app = self.setup_application(lab, policy, TimeRunPolicy(1., shutdown=mode))
                self._submit()
                self.assertEqual(self.step().job, policy.explicit)
                state = app.assembly.host_execution.state
                self.assertIsNone(state.pending_operation_decision_id)
                self.assertEqual(state.pending_wait_decision_id(), 'wait-decision')

                self.clock.now = 2.
                lab.settled = True
                receipt = self.step()
                self.assertEqual(receipt.event['operation_result']['call_id'], 'first')
                self.assertEqual(receipt.event['decision_id'], 'lab-decision')
                self.assertFalse(receipt.event['run_stop']['shutdown_complete'])
                self.assertEqual(policy.actions[:2], [(f'native-{mode}', policy.native),
                                                      ('poll' if mode == 'drain' else 'cancel', policy.explicit)])
                self.assertEqual(state.pending_wait_decision_id(), 'wait-decision')

                policy.explicit_ready = True
                only_native = self.step()
                self.assertEqual((only_native.kind, only_native.job), ('waiting', policy.native))
                self.assertIsNone(state.pending_wait_decision_id())
                self.assertEqual(state.decision_records['wait-decision']['status'], 'completed')
                policy.native_ready = True
                self.assertTrue(self.step().event['run_stop']['shutdown_complete'])
                self.assertEqual(policy.turns, 2)

    def test_native_inference_is_a_managed_run_wait(self):
        class AsyncPolicy(run.Policy):
            job = JobRef('native-proposal', 'fixture')
            ready = False

            def decide(self, window):
                self.turns += 1
                return Decision('wait') if self.turns == 1 else Decision('finish')

            def pending_research_job(self):
                return self.job if self.turns == 1 else None

            def can_decide(self):
                return self.turns != 1 or self.ready

        policy = AsyncPolicy()
        app = self.setup_application(run.Lab(), policy)
        waits = 0

        def release_inference():
            nonlocal waits
            waits += 1
            policy.ready = waits == 2

        with patch.object(app.runner, '_wait_for_progress', side_effect=release_inference):
            result = app.run(wait_for_jobs=True)
        self.assertEqual((result.status, result.decision_steps, waits), ('finished', 2, 2))
        self.assertEqual(len(result.events), 2)  # No extra events for idle native polls.

    def test_native_stop_only_hook_remains_managed_until_settlement(self):
        class NativeStopOnly(run.Policy):
            job = JobRef('native-stop-only', 'fixture')
            cancels = 0

            def decide(self, window):
                self.turns += 1
                return Decision('wait')

            def pending_research_job(self):
                return self.job if self.turns and self.cancels < 2 else None

            def cancel_research(self):
                self.cancels += 1
                return self.cancels == 2

        policy = NativeStopOnly()
        app = self.setup_application(run.Lab(), policy, TimeRunPolicy(1.))
        with patch.object(app.runner, '_wait_for_progress') as wait:
            parked = app.run(wait_for_jobs=True)
            wait.assert_not_called()
            self.assertEqual((parked.status, parked.decision_steps, policy.cancels), ('waiting', 1, 0))
            self.clock.now = 2.
            stopped = app.run(wait_for_jobs=True)
            wait.assert_called_once()
        self.assertEqual((stopped.status, stopped.shutdown_complete, policy.cancels), ('finished', True, 2))
        self.assertEqual(policy.turns, 1)

    def test_opaque_lab_wait_does_not_hide_manageable_native_shutdown(self):
        class OpaqueLab(run.Lab):
            poll_job = None
            cancel_job = None

        class NativeWithLab(run.Policy):
            job = JobRef('native-with-lab', 'fixture')
            drains = 0

            def pending_research_job(self):
                return self.job if self.turns and self.drains < 2 else None

            def drain_research(self):
                self.drains += 1
                return self.drains == 2

        lab, policy = OpaqueLab(backend=True), NativeWithLab((run.operation(),))
        app = self.setup_application(lab, policy, TimeRunPolicy(1., shutdown='drain'))
        self.assertEqual(self._submit().job, lab.job)
        self.clock.now = 2.
        with patch.object(app.runner, '_wait_for_progress') as wait:
            result = app.run(wait_for_jobs=True)
            wait.assert_called_once()
        self.assertEqual((result.status, result.waiting_job, policy.drains), ('waiting', lab.job, 2))
        self.assertIsNone(result.events[-1]['run_stop']['unresolved_research_job'])

    def test_wait_checkpoint_migrates_only_a_correlated_legacy_pointer(self):
        job = JobRef('legacy-job', 'fixture')

        class WaitingPolicy(run.Policy):
            def decide(self, window):
                return Decision('wait', job=job, decision_id='legacy-wait')

        app = self.setup_application(run.Lab(), WaitingPolicy())
        self.assertEqual(self.step().job, job)
        state = deepcopy(app.assembly.host_execution.state)
        state.pending_operation_decision_id = 'legacy-wait'
        recovered = decode(encode(state))
        self.assertEqual(recovered.pending_wait_decision_id(), 'legacy-wait')
        self.assertIsNone(recovered.pending_operation_decision_id)
        state.pending_job = JobRef(job.job_id, 'other-backend')
        with self.assertRaisesRegex(ValueError, 'match exactly one'):
            decode(encode(state))

    def test_lab_cannot_relabel_an_explicitly_foreign_receipt(self):
        class WrongIdLab(run.Lab):
            def execute(self, call):
                return super().execute(call) if self.backend else OperationResult('foreign', (), True, call_id='other')

            def poll_job(self, job):
                return JobOutcome(job, JobStatus.SUCCEEDED, OperationResult('result', (), True,
                                  call_id=None if self.settled else 'other'))

        lab = WrongIdLab()
        app = self.setup_application(lab, run.Policy())
        group, call = app.assembly.lab_execution, run.operation()
        with self.assertRaisesRegex(ValueError, 'receipt call_id does not match'):
            group.execute(call)
        self.assertIsNone(group.stored_receipt(call))
        lab.backend = True
        self.assertEqual(group.execute(call), lab.job)
        with self.assertRaisesRegex(ValueError, 'receipt call_id does not match'):
            group.poll_job(call, lab.job, decision_id='decision', after=0)
        self.assertEqual(group.stored_receipt(call), lab.job)
        lab.settled = True
        receipt, _ = group.poll_job(call, lab.job, decision_id='decision', after=0)
        self.assertEqual(receipt.call_id, 'first')
        self.assertEqual(group.stored_receipt(call), receipt)

    def test_recovery_matches_full_live_job_ref_before_reattach(self):
        class RecoverableLab(run.Lab):
            def __init__(self):
                super().__init__(backend=True)
                self.reattached = []

            def state(self):
                return {}

            def restore(self, state):
                if state != {}:
                    raise ValueError('unexpected fixture state')

            def reattach_job(self, job):
                self.reattached.append(job)

        for change in ('none', 'backend', 'job_id'):
            with self.subTest(change=change):
                lab = RecoverableLab()
                app = self.setup_application(lab, run.Policy((run.operation(),)), durable=True)
                self.assertEqual(self._submit().job, lab.job)
                # Exercise the previous in-flight summary shape as well: the
                # restored execution queue still has the owning full JobRef.
                summary = app.assembly.host_execution.in_flight_operations()
                for row in summary:
                    row.pop('job')
                app.recovery._checkpoint(app.recovery.generation, in_flight=summary)
                if change != 'none':
                    wrong = (JobRef(lab.job.job_id, 'other-backend') if change == 'backend'
                             else JobRef('other-job', lab.job.backend))
                    app.assembly.lab_execution._store(run.operation(), wrong, 0.)
                root, modules = app.config.data_dir, app.modules
                app.close()
                with Application.resume(root, modules=modules) as resumed:
                    if change == 'none':
                        self.assertEqual(lab.reattached, [lab.job])
                        self.assertEqual(resumed.runner.step(resumed.config.objective).job, lab.job)
                    else:
                        self.assertEqual(lab.reattached, [])
                        self.assertEqual(resumed.recovery.intervention['reason'], 'interrupted_operation')
                        with self.assertRaises(RecoveryRequired):
                            resumed.runner.step(resumed.config.objective)
                    self.assertEqual(lab.calls, ['first'])


if __name__ == '__main__':
    unittest.main()
