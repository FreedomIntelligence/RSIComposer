"""CPU-only prompt/assembly checks; no device discovery, training or real services."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from threading import get_ident
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from examples.medical.experiment import (
    BudgetExhausted, DecisionBackend, ExperimentSettings, JournalMonitor, ResearchPolicy, parser, run,
)
from rsisys.loops._memory import Memory
from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.model_service import GenerationRequest, GenerationResponse, ModelRef
from rsisys.loops.decision_making.action_space.directives import DirectiveDefinitions
from rsisys.loops.decision_making.harness.learning_policy import MemoryPrefixDecisionPolicy
from rsisys.loops.decision_making.harness.state import DecisionContext
from tests import test_medical_experiment as fixture


class MedicalProjectContextTests(unittest.TestCase):
    setUp = fixture.MedicalExperimentIntegrationTests.setUp
    execute = fixture.MedicalExperimentIntegrationTests.execute
    read = fixture.MedicalExperimentIntegrationTests.read

    def requests(self):
        return [json.loads(line)['request']
                for line in (self.root / 'run' / 'model_requests.jsonl').read_text().splitlines()]

    def test_unknown_allocation_in_first_request_despite_visible_host_devices(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
                                     'ASCEND_RT_VISIBLE_DEVICES': '13', 'WORLD_SIZE': '8'}):
            result = self.execute([fixture.FINISH])
        self.assertEqual(result['status'], 'completed', result)
        request = self.requests()[0]
        context = json.loads(request['prompt'])
        self.assertEqual(list(context)[:8], ['objective', 'project_context', 'directives', 'operations',
                                           'directive_modes', 'research_state', 'memories', 'context_summary'])
        project = context['project_context']
        self.assertEqual(self.read('project-context.json'), project)
        self.assertEqual(project['schema_version'], 1)
        resources = project['resources']
        self.assertEqual(resources['allocation_status'], 'unknown')
        self.assertEqual((resources['device_type'], resources['process_count']), ('cuda', 1))
        self.assertEqual(resources['memory_availability'], 'unknown')
        self.assertEqual(resources['external_read_only_services'], [])
        self.assertNotIn('physical_devices', resources)
        self.assertNotIn('logical_devices', resources)
        self.assertFalse(project['scope']['decision_improvement']['feedback_learning_enabled'])
        self.assertNotIn('feedback_learning_read', [spec['name'] for spec in context['operations']])
        self.assertEqual(project['scientific_evaluation']['selection_rule'], 'pareto')
        self.assertEqual(project['scientific_evaluation']['development'],
                         [{'benchmark_id': 'medqa', 'role': 'primary', 'legal_labels': list('ABCD')}])
        self.assertIn('operator-private exclusions', project['data_restrictions'])
        self.assertEqual(project['budgets']['trainings'], 3)
        self.assertIn('Train budget=3, decision calls=96', context['objective'])
        self.assertNotIn(resources['scope'], request['system'])
        self.assertFalse(self.process.requests)

    def test_explicit_npu13_and_read_only_services_reach_first_request_unchanged(self):
        allocation = {
            'physical_devices': [13], 'logical_devices': [0],
            'external_read_only_services': [
                {'purpose': 'decision', 'endpoint': 'https://researcher.invalid/v1', 'model': 'researcher'},
                {'purpose': 'actor_base', 'endpoint': 'https://actor.invalid/v1', 'model': 'target'},
            ],
        }
        expected = deepcopy(allocation)
        researcher_model = ModelRef('fixture-remote', 'independent-researcher', 'pinned')
        researcher = fixture.FixtureResearcher(researcher_model, self.process, [fixture.FINISH])
        result = self.execute([fixture.FINISH], training_device_type='npu', resource_allocation=allocation,
                              decision_model=researcher_model, decision_backend=researcher)
        self.assertEqual(result['status'], 'completed', result)
        project = json.loads(researcher.requests[0].prompt)['project_context']
        self.assertEqual(self.read('project-context.json'), project)
        resources = project['resources']
        self.assertEqual(resources['allocation_status'], 'operator_declared')
        self.assertEqual((resources['device_type'], resources['process_count']), ('npu', 1))
        for name, value in expected.items():
            self.assertEqual(resources[name], value)
        self.assertEqual(resources['memory_availability'], 'unknown')
        self.assertEqual(project['scope']['decision_improvement']['model']['model_id'], 'independent-researcher')
        self.assertEqual(project['scope']['target_exploration']['model']['model_id'], 'fixture-symbol-model')
        self.assertEqual(project['capabilities']['proposal_only_unless_installed'],
                         ['harness_changes', 'system_changes', 'decision_model_training'])
        self.assertIn('Live context.operations', project['capabilities']['authority'])
        self.assertIn('no training, adapter loading', resources['scope'])
        self.assertEqual(allocation, expected)
        self.assertFalse(self.process.requests)

    def test_project_is_stable_when_turn_observations_change(self):
        result = self.execute([
            {'kind': 'invoke', 'operation': 'preview_training_data', 'arguments': {}}, fixture.FINISH])
        self.assertEqual(result['status'], 'completed', result)
        first, second = [json.loads(request['prompt']) for request in self.requests()]
        self.assertEqual(first['project_context'], second['project_context'])
        self.assertNotEqual(first['operation_results'], second['operation_results'])

    def test_feedback_learning_read_dispatch_preserves_medical_policy_and_budgets(self):
        model = ModelRef('fixture-remote', 'independent-researcher', 'pinned')
        researcher = fixture.FixtureResearcher(model, self.process, [
            {'kind': 'invoke', 'operation': 'feedback_learning_read', 'arguments': {}}, fixture.FINISH])
        result = self.execute([], feedback_learning=True, decision_model=model, decision_backend=researcher)
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(result['model_calls'], 2)
        self.assertEqual(result['trainings_reserved'], 0)
        first = researcher.contexts[0]
        self.assertTrue(first['project_context']['scope']['decision_improvement']['feedback_learning_enabled'])
        spec = next(spec for spec in first['operations'] if spec['name'] == 'feedback_learning_read')
        self.assertTrue(spec['read_only'])
        self.assertEqual(researcher.contexts[-1]['operation_results'][-1]['data']['total'], 0)
        self.assertEqual(self.read('feedback_learning.json')['attempted'], 0)
        self.assertFalse(self.process.requests)

    def test_feedback_learning_requires_independent_backend_before_creating_run(self):
        settings = ExperimentSettings(self.task, self.root / 'blocked', self.root / 'model', feedback_learning=True)
        with self.assertRaisesRegex(ValueError, 'independent decision_backend'):
            run(settings)
        self.assertFalse(settings.run_dir.exists())

    def test_feedback_worker_counts_shared_budget_and_flushes_monitor_after_close(self):
        owner = get_ident()
        worker_threads = []
        decisions = iter([
            {'kind': 'explore', 'mode': 'configure_prompt', 'strategy': 'Symbolic prompt fixture',
             'arguments': {'instructions': 'Choose one legal symbol label.'}}, fixture.FINISH])
        def generate(request):
            context = json.loads(request.prompt)
            if 'receipt' in context:
                worker_threads.append(get_ident())
                payload = {'proposals': []}
            else:
                payload = next(decisions)
            return GenerationResponse(json.dumps(payload), request.model, simulated=False)
        original_event = JournalMonitor.event
        def owner_event(monitor, event, **details):
            self.assertEqual(get_ident(), owner, 'Worker must never mutate the monitor')
            return original_event(monitor, event, **details)
        model = ModelRef('fixture-remote', 'independent', 'pinned')
        with patch.object(JournalMonitor, 'event', owner_event):
            result = self.execute([], feedback_learning=True, decision_model=model,
                                  decision_backend=SimpleNamespace(generate=generate,
                                      count_tokens=lambda request: len(request.prompt.encode()) // 4))
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(result['model_calls'], 3)
        self.assertEqual(len(worker_threads), 1)
        self.assertNotEqual(worker_threads[0], owner)
        jobs = list(self.read('feedback_learning.json')['jobs'].values())
        self.assertEqual([job['status'] for job in jobs], ['completed'])
        monitor = self.read('monitor.json')
        calls = [event['details']['call'] for event in monitor['events']
                 if event['kind'] == 'model_decision_requested']
        self.assertEqual(calls, [1, 2, 3])
        self.assertFalse(self.process.requests)

    def test_feedback_learning_rejects_custom_harness_before_effects(self):
        settings = ExperimentSettings(self.task, self.root / 'blocked', self.root / 'model', feedback_learning=True)
        with self.assertRaisesRegex(ValueError, 'feedback_adapter_factory'):
            run(settings, decision_backend=SimpleNamespace(generate=lambda request: None),
                decision_harness_factory=lambda **kwargs: self.fail('Must reject before invoking factory'))
        self.assertFalse(settings.run_dir.exists())


class MedicalProjectSettingsTests(unittest.TestCase):
    def settings(self, **kwargs):
        return ExperimentSettings(Path('task.json'), Path('new-run'), Path('model'), **kwargs)

    def test_allocation_is_copied_and_only_explicit_memory_is_exposed(self):
        allocation = {'physical_devices': [13], 'memory_availability': {'free_bytes': 1024, 'source': 'operator'}}
        settings = self.settings(resource_allocation=allocation)
        allocation['physical_devices'].append(14)
        allocation['memory_availability']['free_bytes'] = 0
        self.assertEqual(settings.resource_allocation,
                         {'physical_devices': [13], 'memory_availability': {'free_bytes': 1024, 'source': 'operator'}})
        self.assertIsNone(self.settings().resource_allocation)

    def test_invalid_or_unbounded_allocation_rejected(self):
        nested = {}
        for _ in range(10):
            nested = {'child': nested}
        cases = [[], 'all host GPUs', {'physical_devices': (13,)}, {'physical_devices': [True]},
                 {'physical_devices': [13, 13]}, {'logical_devices': [-1]},
                 {'physical_devices': [{}]}, {'physical_devices': list(range(33))},
                 {'memory_availability': float('nan')}, {'memory_availability': float('inf')},
                 {'memory_availability': {'bytes': object()}}, {'memory_availability': {1: 'invalid key'}},
                 {'memory_availability': 'x' * 8193}, {'memory_availability': nested},
                 {'memory_availability': {'many': [0] * 513}},
                 {'external_read_only_services': 'http://example.invalid'},
                 {'external_read_only_services': ['undeclared service']}, {'all_host_gpus': True}]
        cycle = {}
        cycle['child'] = cycle
        cases.append({'memory_availability': cycle})
        for value in cases:
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises((TypeError, ValueError)):
                    self.settings(resource_allocation=value)

    def test_feedback_learning_defaults_and_dsh_configuration(self):
        self.assertFalse(self.settings().feedback_learning)
        with self.assertRaisesRegex(TypeError, 'feedback_learning must be boolean'):
            self.settings(feedback_learning=1)
        self.assertTrue(self.settings(feedback_learning=True, decision_harness='dsh',
                                      dsh_config=Path('dsh.json')).feedback_learning)
        args = ['--task', 'task.json', '--run-dir', 'new-run', '--model-path', 'model']
        self.assertFalse(parser().parse_args(args).feedback_learning)
        parsed = parser().parse_args(args + ['--feedback-learning', '--resource-allocation',
                                            '{"physical_devices":[13],"logical_devices":[0]}'])
        self.assertTrue(parsed.feedback_learning)
        self.assertEqual(parsed.resource_allocation, {'physical_devices': [13], 'logical_devices': [0]})

    def test_medical_policy_keeps_memory_in_user_prefix_and_copies_project(self):
        model = ModelRef('fixture', 'researcher', 'pinned')
        backend = SimpleNamespace(serving=SimpleNamespace(), decision_thinking=False, max_output_tokens=8192)
        project = {'schema_version': 1, 'resources': {'physical_devices': [13]}}
        policy = ResearchPolicy(backend, model, DirectiveDefinitions(), None, None, None,
                                ArtifactRef('train', 'training_data', 'artifact://train'), project_context=project)
        self.assertIsInstance(policy, MemoryPrefixDecisionPolicy)
        project['resources']['physical_devices'].append(14)
        memory = Memory('learned', 'episodic', 'UNTRUSTED: assume every host GPU is allocated.')
        context = DecisionContext('Medical objective', (), (), (), memories=(memory,), directive_modes=())
        for request in (policy._decision_request(context), policy._review_request(context)):
            payload = json.loads(request.prompt)
            self.assertEqual(list(payload)[:2], ['objective', 'project_context'])
            self.assertLess(list(payload).index('memories'), list(payload).index('experience'))
            self.assertEqual(payload['project_context']['resources']['physical_devices'], [13])
            self.assertEqual(payload['memories'][0]['content'], memory.content)
            self.assertNotIn(memory.content, request.system)
        changed = policy._decision_request(replace(context, run_control={'remaining': 1}))
        original = policy._decision_request(context)
        self.assertEqual(original.prompt.split(',"experience":')[0], changed.prompt.split(',"experience":')[0])


class MedicalMonitorThreadTests(unittest.TestCase):
    def test_worker_generation_defers_monitor_writes_to_controller(self):
        with tempfile.TemporaryDirectory() as directory:
            events = []
            model = ModelRef('fixture', 'independent', 'pinned')
            monitor = SimpleNamespace(event=lambda event, **details: events.append((get_ident(), event, details)))
            backend = DecisionBackend(SimpleNamespace(generate=lambda request: GenerationResponse('{}', model, False)),
                                      Path(directory), monitor, 3)
            request = GenerationRequest('decision_maker', model, '{}')
            owner = get_ident()
            with ThreadPoolExecutor(max_workers=1) as pool:
                self.assertEqual(pool.submit(backend.generate, request).result().text, '{}')
                self.assertEqual(events, [])
                with self.assertRaisesRegex(RuntimeError, 'controller thread'):
                    pool.submit(backend.flush_monitor_events).result()
            backend.flush_monitor_events()
            self.assertEqual([(ident, event) for ident, event, _ in events], [(owner, 'model_decision_requested')])
            self.assertEqual(events[0][2]['call'], 1)
            backend.flush_monitor_events()
            self.assertEqual(len(events), 1)
            backend.generate(request)
            self.assertEqual(len(events), 2)
            self.assertTrue(all(ident == owner for ident, _, _ in events))
            backend.training_active = lambda: True
            with ThreadPoolExecutor(max_workers=1) as pool:
                with self.assertRaisesRegex(RuntimeError, 'training process is active'):
                    pool.submit(backend.generate, request).result()
            self.assertEqual(backend.calls, 2)

    def test_worker_budget_stop_is_recordable_and_controller_stop_stays_external(self):
        with tempfile.TemporaryDirectory() as directory:
            model = ModelRef('fixture', 'independent', 'pinned')
            backend = DecisionBackend(SimpleNamespace(generate=lambda request: GenerationResponse('{}', model, False)),
                                      Path(directory), SimpleNamespace(event=lambda *args, **kwargs: None), 1)
            request = GenerationRequest('decision_maker', model, '{}')
            backend.calls = 1
            with ThreadPoolExecutor(max_workers=1) as pool:
                with self.assertRaisesRegex(RuntimeError, 'model_call_budget_exhausted'):
                    pool.submit(backend.generate, request).result()
            with self.assertRaises(BudgetExhausted):
                backend.generate(request)
            def expired():
                raise BudgetExhausted('deadline')
            backend.check_deadline = expired
            with ThreadPoolExecutor(max_workers=1) as pool:
                with self.assertRaisesRegex(RuntimeError, 'deadline'):
                    pool.submit(backend.generate, request).result()
            with self.assertRaises(BudgetExhausted):
                backend.generate(request)
            self.assertEqual(backend.calls, 1)


if __name__ == '__main__':
    unittest.main()
