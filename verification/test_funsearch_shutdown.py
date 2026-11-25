"""FunSearch's real sampler and Host stop protocol, with a local mock provider."""
from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch

from rsicomposer.api import BackendBinding, GenerationResponse, ModelRef
from rsicomposer.application import Application, ApplicationConfig
from rsicomposer.registry import Registry
from rsicomposer.runtime.orchestration.run_policy import TimeRunPolicy


class ImmediateLabExecutor(ThreadPoolExecutor):
    """Make Lab receipts deterministic without replacing the sampler's executor."""

    def submit(self, fn, *args, **kwargs):
        future = super().submit(fn, *args, **kwargs)
        future.result(timeout=5)
        return future


class MockProvider:
    def __init__(self):
        self.entered = Event()
        self.release = Event()
        self.seeds = []
        self.closed = False

    def generate(self, request):
        self.seeds.append(request.seed)
        if len(self.seeds) == 1:
            self.entered.set()
            if not self.release.wait(10):
                raise TimeoutError('test must release the first provider call')
        return GenerationResponse('return -square(bins - item)', request.model, simulated=True)

    def close(self):
        self.closed = True


class FixturePopulation:
    """No external pinned checkout: the real Lab still scores real sampler output."""

    def __init__(self, **_):
        self.parents = []

    def register(self, source, version, receipt_id, decision_id, scores, island):
        self.parents = [{'version': version, 'score_decision_id': decision_id}]

    def prompt(self):
        return 'def priority(item, bins):\n  return 0', 0, self.parents

    def state(self):
        return {'parents': self.parents}

    def restore(self, state):
        self.parents = state['parents']


@unittest.skipUnless(find_spec('numpy') is not None, 'requires the optional FunSearch NumPy dependency')
class FunSearchShutdownTest(unittest.TestCase):
    def fixture(self, run_policy=None):
        from rsicomposer.algorithms import funsearch
        from rsicomposer.algorithms.funsearch import controller

        registry = Registry(builtins=False)
        funsearch.register(registry)
        method_config = {
            'controller': dict(seed=1, source_dir='unused', islands=1, reset_every=10,
                               samplers=1, evaluators=2, proposals=2, samples_per_prompt=2,
                               temperature=0.5, max_output_tokens=30, condition='frozen_parents',
                               development_count=1, heldout_count=1),
            'labs': {'funsearch': dict(development_count=1, heldout_count=1, max_concurrent=2)},
        }
        modules = registry.build({'method': {'name': 'funsearch', 'config': method_config}})
        workdir = self.enterContext(TemporaryDirectory())
        model = ModelRef('fixture', 'packing-model', 'mock-v1')
        self.provider = MockProvider()
        self.enterContext(patch('rsicomposer.host.execution.ThreadPoolExecutor', ImmediateLabExecutor))
        self.enterContext(patch.object(controller, 'Population', FixturePopulation))
        with patch('rsicomposer.backends.serving.build_serving_backend', return_value=self.provider):
            app = Application(ApplicationConfig('fixture objective', Path(workdir),
                {'fixture': BackendBinding('fixture', 'openai_chat_completions')}, {'decision_maker': model}),
                modules=modules, run_policy=run_policy, cleanup_timeout_seconds=3)
        self.app, self.now = app, 0.
        app.runner._clock = lambda: self.now
        self.policy = app.assembly.controller_host.active.policy
        self.addCleanup(self.close_fixture)
        return app

    def close_fixture(self):
        self.provider.release.set()
        try:
            for future in self.policy.sampler.jobs.values():
                if not future.cancelled():
                    future.result(timeout=5)
        finally:
            self.app.close()

    def step(self):
        return self.app.runner.step(self.app.config.objective)

    def start_batch(self):
        for _ in range(20):
            waiting = self.step()
            if waiting.job is not None and waiting.job.backend == 'funsearch-sampler':
                break
        else:
            self.fail('FunSearch did not yield a managed sampling wait')
        self.assertEqual(waiting.kind, 'waiting')
        self.assertTrue(self.provider.entered.wait(5))
        self.assertTrue(self.app.runner.decision_making.can_poll_job(waiting.job))
        self.assertEqual(self.app.runner.decision_making.pending_research_job(), waiting.job)
        future, = self.policy.sampler.jobs.values()
        self.assertFalse(future.done())
        self.assertEqual(len(self.provider.seeds), 1)
        return waiting.job, future

    def uncommitted_writes(self):
        return (self.app.artifacts.refs(), self.policy.emit.path.read_bytes())

    def assert_cooperative_stop(self, mode):
        self.fixture(TimeRunPolicy(1., shutdown=mode))
        job, future = self.start_batch()
        before = self.uncommitted_writes()
        self.now = 2.
        stopping = self.step()
        self.assertEqual((stopping.kind, stopping.job, stopping.decision_steps), ('waiting', job, 0))
        stop = stopping.event['run_stop']
        self.assertFalse(stop['shutdown_complete'])
        self.assertEqual(stop['unresolved_research_job'], {'job_id': job.job_id, 'backend': job.backend})
        self.assertFalse(future.done())
        self.assertFalse(self.policy.state()['sampling'])
        self.assertTrue(self.policy.state()['stopping'])
        self.assertEqual(self.uncommitted_writes(), before)

        self.provider.release.set()
        future.result(timeout=5)
        settled = self.step()
        self.assertEqual(settled.kind, 'finished')
        self.assertTrue(settled.event['run_stop']['shutdown_complete'])
        self.assertIsNone(settled.event['run_stop']['unresolved_research_job'])
        self.assertEqual(self.provider.seeds, [100003])
        self.assertEqual(self.uncommitted_writes(), before)

    def test_cancel_blocks_second_sample_and_waits_for_worker(self):
        self.assert_cooperative_stop('cancel')

    def test_drain_blocks_second_sample_and_waits_for_worker(self):
        self.assert_cooperative_stop('drain')

    def test_cleanup_timeout_keeps_provider_open_until_owned_close_retries(self):
        self.fixture(TimeRunPolicy(1.))
        job, future = self.start_batch()
        before = self.uncommitted_writes()
        self.now = 2.
        self.assertEqual(self.step().kind, 'waiting')
        self.now = 5.
        terminal = self.step()
        self.assertEqual(terminal.kind, 'finished')
        self.assertTrue(terminal.event['run_stop']['cleanup_timed_out'])
        self.assertFalse(terminal.event['run_stop']['shutdown_complete'])
        self.assertEqual(self.app.runner.decision_making.pending_research_job(), job)
        with self.assertRaisesRegex(RuntimeError, 'still owns running generation'):
            self.app.close()
        self.assertFalse(self.provider.closed)

        self.provider.release.set()
        future.result(timeout=5)
        self.assertEqual(self.provider.seeds, [100003])
        self.assertEqual(self.uncommitted_writes(), before)
        self.app.close()
        self.assertTrue(self.provider.closed)

    def test_normal_wait_resumes_and_owner_publishes_completed_batch(self):
        self.fixture()
        _, future = self.start_batch()
        before = self.uncommitted_writes()
        self.provider.release.set()
        future.result(timeout=5)
        self.assertEqual(self.provider.seeds, [100003, 100004])
        self.assertEqual(self.uncommitted_writes(), before)
        self.assertEqual(self.step().kind, 'progressed')  # Poll the sampler wait.
        for _ in range(20):
            result = self.step()
            if result.kind == 'finished':
                break
        else:
            self.fail('FunSearch did not complete normally')
        self.assertTrue(result.event['run_stop']['shutdown_complete'])
        self.assertEqual(self.policy.stats['proposals'], 2)
        self.assertEqual(sum(ref.kind == 'funsearch-propose' for ref in self.app.artifacts.refs()), 1)
        self.assertIsNone(self.app.runner.decision_making.pending_research_job())


if __name__ == '__main__':
    unittest.main()
