"""Observed job metrics reach review without models, sockets, or accelerator use."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.qwen_toy.adapters import WorkerJobs
from examples.qwen_toy.run import GuardedDecisionPolicy
from rsisys.config import DemoConfig, build_demo
from rsisys.loops.contracts.coordination import ExecutionView, ExplorationProgress, ReviewDecision
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.jobs import JobOutcome, JobProgress, JobRef, JobStatus
from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef, TokenUsage
from rsisys.loops.contracts.requests import SFTSpec
from rsisys.loops.decision_making.action_space import OperationCall
from rsisys.loops.decision_making.action_space.directives import DirectiveDefinitions
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.harness.state import DecisionContext
from rsisys.runtime.jobs import JobBook, JobHandle


class PayloadJobs(WorkerJobs):
    def validate(self, request):
        pass

    def convert(self, request, raw):
        return raw


class JobProgressContractTests(unittest.TestCase):
    def test_optional_progress_and_numeric_validation(self):
        job = JobRef('job', 'backend')
        self.assertIsNone(JobOutcome(job, JobStatus.RUNNING).progress)
        self.assertEqual(asdict(JobProgress()), {
            'step': None, 'total_steps': None, 'metrics': {},
            'elapsed_seconds': None, 'observed_at': None,
        })
        for fields in (
            {'step': True}, {'step': -1}, {'step': 1.5}, {'total_steps': 0},
            {'step': 3, 'total_steps': 2}, {'elapsed_seconds': float('inf')},
            {'observed_at': -1}, {'observed_at': True}, {'elapsed_seconds': -0.1},
            {'metrics': {'loss': float('nan')}}, {'metrics': {'loss': True}},
            {'metrics': {'': 1}}, {'metrics': {1: 1}}, {'metrics': []},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                JobProgress(**fields)
        with self.assertRaises(TypeError):
            JobOutcome(job, JobStatus.RUNNING, progress={'step': 1})
        values = {'loss': 0.0, 'learning_rate': 0.001}
        progress = JobProgress(0, 10, values, 0.0, 1730000000.0)
        values['loss'] = 99
        self.assertEqual(progress.metrics['loss'], 0.0)

    def test_handle_keeps_last_observation_on_status_only_and_terminal_polls(self):
        job = JobRef('job', 'backend')
        observed = JobProgress(2, 10, {'loss': 0.7}, 4.0, 1730000000.0)
        for status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
            with self.subTest(status=status):
                terminal = JobOutcome(job, status,
                    result={} if status == JobStatus.SUCCEEDED else None,
                    error='failure' if status == JobStatus.FAILED else None)
                poll = Mock(side_effect=[JobOutcome(job, JobStatus.RUNNING, progress=observed),
                                         JobOutcome(job, JobStatus.RUNNING), terminal])
                cancel = Mock()
                handle = JobHandle(job, poll, cancel)
                first = handle.poll()
                first.progress.metrics['loss'] = 99
                self.assertEqual(handle.poll().progress, observed)
                last = handle.cancel()
                self.assertEqual(last.progress, observed)
                self.assertEqual(last.status, status)
                last.progress.metrics['loss'] = 88
                self.assertEqual(handle.poll().progress, observed)
                self.assertEqual(poll.call_count, 3)
                handle.cancel()
                cancel.assert_called_once_with(job)

    def test_new_observation_replaces_old_and_new_job_does_not_inherit(self):
        job = JobRef('job', 'backend')
        latest = JobProgress(3, 5, {'loss': 0.4}, 5, 1730000001.0)
        poll = Mock(side_effect=[JobOutcome(job, JobStatus.RUNNING, progress=JobProgress(1, 5)),
                                 JobOutcome(job, JobStatus.RUNNING, progress=latest)])
        handle = JobHandle(job, poll, Mock())
        handle.poll()
        self.assertEqual(handle.poll().progress, latest)
        other = JobRef('other', 'backend')
        other_handle = JobHandle(other, lambda ref: JobOutcome(ref, JobStatus.RUNNING), Mock())
        self.assertIsNone(other_handle.poll().progress)


class ProgressBridgeTests(unittest.TestCase):
    def adapter(self, raw, *, kind='train'):
        events = []
        def call(path, payload=None):
            if path == '/train':
                return {'job_id': raw['job_id']}
            if path == '/cancel':
                raw.clear()
                raw.update(job_id='job', status='cancelled')
                return {}
            return deepcopy(raw)
        client = SimpleNamespace(call=call, deadline=SimpleNamespace(remaining=lambda: 100))
        adapter = PayloadJobs(client, None, kind=kind,
            notify=lambda event, **details: events.append((event, deepcopy(details))))
        return adapter, events

    def test_worker_payload_reaches_default_inspect_and_actual_review_prompt(self):
        runner, _ = build_demo(DemoConfig.load(ROOT / 'configs/demo.json'), with_learning=True)
        actor, dm = runner.exploration, runner.decision_making
        data = dm.harness.tools.artifacts.put(b'data', kind='training_data')
        directive = Directive('live', 'task', 'train', mode='sft', input_refs=(data,),
                              payload=SFTSpec('recipe', apply_result=False))
        raw = {'job_id': 'job', 'status': 'running', 'step': 2, 'total_steps': 8,
               'elapsed_seconds': 4.25, 'metrics': {'loss': 0.7, 'learning_rate': 0.0001},
               'observed_at': 1730000000.0}
        adapter, events = self.adapter(raw)
        actor.harness.learner = adapter
        actor.harness.training = JobBook(adapter)
        dm.harness.policy = SimpleNamespace(decide=lambda context: Decision('explore', directive=directive))
        self.assertEqual(runner.run('task', 1).status, 'waiting')
        self.assertIsNone(actor.inspect('live').job_progress)  # Submission is not observation.
        progress = actor.advance('live')
        self.assertFalse(progress.terminal)
        self.assertEqual(progress.metrics, raw['metrics'])
        self.assertEqual(asdict(progress.job_progress), {k: raw[k] for k in
            ('step', 'total_steps', 'metrics', 'elapsed_seconds', 'observed_at')})
        # Monitor fields remain identical; backend sample time is contract-only.
        self.assertEqual(events[-1], ('job', {k: raw[k] for k in
            ('job_id', 'status', 'step', 'total_steps', 'elapsed_seconds', 'metrics')} | {'kind': 'train'}))
        progress.metrics['loss'] = 99
        progress.job_progress.metrics['loss'] = 88
        self.assertEqual(actor.inspect('live').metrics['loss'], 0.7)
        raw.update(step=3, elapsed_seconds=6.0, metrics={'loss': 0.5}, observed_at=1730000002.0)
        progress = actor.advance('live')
        model = ModelRef('review', 'model', 'baseline')
        prompts = []
        def generate(request):
            prompts.append(json.loads(request.prompt))
            return GenerationResponse('{"action":"continue","next_review_seconds":null}', model, False)
        dm.harness.policy = ModelDecisionPolicy(SimpleNamespace(generate=generate), model)
        self.assertEqual(dm.review('task', progress).action, 'continue')
        evidence = prompts[-1]['execution']['progress']
        self.assertEqual(evidence['metrics'], {'loss': 0.5})
        self.assertEqual(evidence['job_progress'], asdict(progress.job_progress))
        # Inspect is a read of the last observation; it must not repoll the backend.
        raw['step'] = 4
        self.assertEqual(actor.inspect('live').job_progress.step, 3)
        actor.cancel('live')
        terminal = actor.advance('live')
        self.assertTrue(terminal.terminal)
        self.assertEqual(terminal.job_progress, progress.job_progress)
        self.assertEqual(terminal.metrics, {})  # Terminal feedback remains distinct.
        actor.acknowledge('live')
        following = Directive('execute', 'task', 'execute')
        self.assertIsNone(actor.accept(following).job_progress)

    def test_no_fabricated_progress_for_queued_eval_or_legacy_payloads(self):
        for status, kind, fields in (
            ('running', 'train', {}), ('succeeded', 'train', {}),
            ('queued', 'train', {'step': 0, 'total_steps': 4, 'observed_at': 123}),
            ('running', 'evaluate', {'step': 4, 'metrics': {'loss': 0.1}}),
        ):
            with self.subTest(status=status, kind=kind):
                raw = {'job_id': 'job', 'status': status, 'result': {}, **fields}
                adapter, _ = self.adapter(raw, kind=kind)
                job = JobRef('job', adapter.backend_id)
                adapter.requests[job] = None
                self.assertIsNone(adapter.poll(job).progress)

    def test_terminal_adapter_cache_keeps_progress_and_sample_time_unknown(self):
        raw = {'job_id': 'job', 'status': 'succeeded', 'result': {}, 'step': 4,
               'total_steps': 4, 'elapsed_seconds': 10, 'metrics': {'loss': 0.1}}
        adapter, events = self.adapter(raw)
        job = JobRef('job', adapter.backend_id)
        adapter.requests[job] = None
        result = adapter.poll(job)
        self.assertIsNone(result.progress.observed_at)
        result.progress.metrics['loss'] = 99
        raw['step'] = 500
        self.assertEqual(adapter.poll(job).progress.step, 4)
        self.assertEqual(adapter.poll(job).progress.metrics, {'loss': 0.1})
        self.assertEqual(len(events), 1)


class ToyReviewInvocationTests(unittest.TestCase):
    def fixture(self):
        model = ModelRef('review', 'model', 'baseline')
        monitor = Mock()
        policy = GuardedDecisionPolicy(Mock(), model, DirectiveDefinitions(), monitor, attempts=1)
        directive = Directive('live', 'task', 'train', mode='train')
        context = DecisionContext('task', (), (), (), execution=ExecutionView(
            model, ExplorationProgress('live', 'running', directive,
                job_progress=JobProgress(2, 8, {'loss': 0.7}))))
        return policy, context, monitor

    def test_guard_returns_invocation_intact_for_harness_validation(self):
        policy, context, monitor = self.fixture()
        invocation = Decision('invoke', operation=OperationCall('query_experiments', {}),
                              simulated=True, usage=TokenUsage(11, 7))
        with patch.object(ModelDecisionPolicy, 'review', return_value=invocation):
            self.assertIs(policy.review(context), invocation)
        self.assertEqual(policy.reviews, [])
        monitor.review.assert_not_called()
        final = ReviewDecision('continue', None, 'Evidence supports continuing')
        with patch.object(ModelDecisionPolicy, 'review', return_value=final):
            self.assertIs(policy.review(context), final)
        self.assertEqual(len(policy.reviews), 1)
        monitor.review.assert_called_once()

    def test_guard_rejects_non_review_decisions_and_invalid_final_action(self):
        for returned in (Decision('finish'), SimpleNamespace(action='apply')):
            policy, context, monitor = self.fixture()
            with patch.object(ModelDecisionPolicy, 'review', return_value=returned):
                with self.assertRaisesRegex(ValueError, 'bounded training review'):
                    policy.review(context)
            self.assertEqual(policy.reviews, [])
            monitor.review.assert_not_called()


if __name__ == '__main__':
    unittest.main()
