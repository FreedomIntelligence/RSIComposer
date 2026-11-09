"""Research decisions cross the host boundary; test metrics never feed search."""
from copy import deepcopy
import importlib.util
import json
import os
import shutil
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest

from rsisys.api import Registry
from rsisys.controllers.ft_agent import FTAgentController, validate_config
from rsisys.controllers.ft_agent_upstream import FT_AGENT_REVISION, Scenario, load_policy
from rsisys.labs.ft_agent import LLMFinetuneLab, bounded_train_config, evaluation_answer
from rsisys.loops.contracts.coordination import OperationResult


class FTAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {'upstream_root': 'upstream', 'researcher': {'model': 'research', 'endpoints': ['http://localhost/v1']}}

    def test_registry_is_optional_dependency_free_and_selects_both_boundaries(self):
        registry = Registry().describe()
        self.assertIn('ft_agent', registry['controllers'])
        self.assertIn('llm_finetune', registry['labs'])
        for bad in ({}, {**self.config, 'iterations': True}, {**self.config, 'lab': '../x'},
                    {**self.config, 'unknown': 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_config(bad)

    def test_native_worker_waits_for_receipt_and_preserves_failed_experiment(self):
        events = [
            {'kind': 'execute', 'stage': 'prepare', 'iteration': 1, 'files': {'process_data.py': 'native'}},
            {'kind': 'execute', 'stage': 'prepare', 'iteration': 2, 'files': {'process_data.py': 'native-revision'}},
            {'kind': 'finished', 'selected_iteration': 0, 'iterations': 2},
        ]
        class Bridge:
            def __init__(self, *args, **kwargs):
                self.sent, self.closed = [], False
            def receive(self, timeout):
                return deepcopy(events.pop(0))
            def send(self, value):
                self.sent.append(deepcopy(value))
            def close(self):
                self.closed = True
        controller = FTAgentController(SimpleNamespace(data_dir=self.root), self.config, bridge_factory=Bridge)
        self.addCleanup(controller.close)
        initial = SimpleNamespace(last_operation=None, feedback=None)
        operation = controller.decide(initial).operation
        self.assertTrue(operation.name.endswith('.baseline'))
        receipt = OperationResult('baseline', (), False, data={'request': {}, 'ok': True,
            'benchmark': {'dev': 0.5}, 'policy_context': {'scenario': 'dev only'}})
        operation = controller.decide(SimpleNamespace(last_operation=operation, feedback=receipt)).operation
        bridge = controller.bridge
        for index in (1, 2):
            with self.assertRaisesRegex(RuntimeError, 'matching terminal'):
                controller.decide(initial)
            receipt = OperationResult('prepare failed', (), False, data={'request': operation.arguments,
                'ok': False, 'error': 'Generated script failed'})
            operation = controller.decide(SimpleNamespace(last_operation=operation, feedback=receipt)).operation
            self.assertEqual(bridge.sent[-1]['iteration'], index)
            self.assertFalse(bridge.sent[-1]['data']['ok'])
        self.assertTrue(operation.name.endswith('.final'))
        final = OperationResult('final', (), False, data={'request': operation.arguments, 'ok': True,
            'test_accuracy': 0.75, 'trained_experiments': []})
        self.assertEqual(controller.decide(SimpleNamespace(last_operation=operation, feedback=final)).kind, 'finish')
        self.assertTrue(bridge.closed)
        self.assertNotIn('test_accuracy', json.dumps(bridge.sent))

    def test_resource_limit_and_foreign_path_are_rejected_without_rewriting_strategy(self):
        directory = self.root / 'experiment'
        directory.mkdir()
        (directory / 'data.json').write_text(json.dumps([{'instruction': 'a', 'output': 'b'}] * 8))
        config = {'initial_model': str(self.root / 'base'), 'gpu_ids': [1, 2, 3], 'max_steps': 8}
        proposal = {'learning_rate': 2e-4, 'lora_rank': 16, 'num_train_epochs': 1}
        result = bounded_train_config(proposal, config, directory)
        self.assertEqual(result['learning_rate'], proposal['learning_rate'])
        self.assertEqual(result['lora_rank'], proposal['lora_rank'])
        result = bounded_train_config({'learning_rate': '1e-4'}, config, directory)
        self.assertEqual(result['learning_rate'], 0.0001)
        for bad in ({'cutoff_len': 4096}, {'num_train_epochs': 100}, {'max_steps': 1000},
                    {'model_name_or_path': '/other/model'}, {'dataset_dir': '/private'},
                    {'deepspeed': '/x'}, {'trust_remote_code': True}, {'report_to': ['wandb']},
                    {'finetuning_type': 'full'}, {'learning_rate': float('nan')}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                bounded_train_config(bad, config, directory)

    def _lab(self, **updates):
        model = self.root / 'base'
        model.mkdir(exist_ok=True)
        config = {'initial_model': str(model), 'llama_factory_root': 'upstream',
                  'verifier': 'strict_numeric', **updates}
        for name in ('train', 'dev', 'test'):
            path = self.root / (name + '.jsonl')
            path.write_text(json.dumps({'id': name, 'prompt': name + ' question', 'answer': '#### 2'}) + '\n')
            config[name + '_data'] = str(path)
        class Engine:
            def generate(self, model, prompts, directory, **kwargs):
                directory.mkdir()
                return [{'completion': '#### 2'} for p in prompts]
        return LLMFinetuneLab(SimpleNamespace(data_dir=self.root / 'run'), config, engine=Engine())

    def test_test_split_is_only_used_after_research(self):
        lab = self._lab()
        baseline = lab._baseline({})
        self.assertNotIn('test question', json.dumps(baseline))
        self.assertNotIn('test_accuracy', json.dumps(baseline))
        final = lab._final({'iteration': 0})
        self.assertEqual(final['baseline_test'], final['selected_test'])
        self.assertFalse(final['training_performed'])

    def test_gsm8k_gold_preserves_only_final_numeric_value_for_scoring(self):
        from rsisys.labs.math_verifiers import StrictNumericVerifier
        answer = 'Reasoning: 9*10=90, then 1080/18=60.\n#### 60'
        canonical = evaluation_answer(answer)
        self.assertEqual(canonical, '\\boxed{60}')
        verifier = StrictNumericVerifier()
        self.assertTrue(verifier.is_correct('The result is \\boxed{60}', canonical))
        self.assertFalse(verifier.is_correct('The result is \\boxed{90}', canonical))

    @unittest.skipUnless(importlib.util.find_spec('math_verify'), 'Optional Math-Verify dependency')
    def test_real_math_verify_accepts_gsm8k_gold_and_rejects_wrong_final_answer(self):
        from rsisys.labs.math_verifiers import MathVerifyVerifier
        verifier = MathVerifyVerifier()
        gold = evaluation_answer('9*10=90; 1080/18=60\n#### 60')
        self.assertTrue(verifier.is_correct('Reasoning leads to \\boxed{60}', gold))
        self.assertFalse(verifier.is_correct('Reasoning leads to \\boxed{90}', gold))

    @unittest.skipUnless(sys.platform.startswith('linux') and shutil.which('bwrap'), 'Linux bubblewrap required')
    def test_generated_data_code_cannot_escape_or_publish_a_host_symlink(self):
        lab = self._lab()
        lab._baseline({})
        source = (
            'import json, pathlib, socket\n'
            'assert not pathlib.Path("/home").exists()\n'
            'rows=json.load(open("/datasets/seed/train.json"))\n'
            'json.dump(rows,open("data.json","w"))\n'
        )
        data = lab._prepare({'iteration': 1, 'files': {'process_data.py': source}})
        self.assertEqual(data['data_stats']['total_samples'], 1)
        with self.assertRaisesRegex(ValueError, 'regular file'):
            lab._prepare({'iteration': 2, 'files': {'process_data.py':
                'import os\nos.symlink("/etc/passwd", "data.json")\n'}})


@unittest.skipUnless(importlib.util.find_spec('yaml') and importlib.util.find_spec('jinja2'), 'Optional FT controller dependencies')
class PinnedFTPolicyTests(unittest.TestCase):
    def setUp(self):
        self.upstream = Path(os.environ.get('RSISYS_FT_UPSTREAM',
            Path(__file__).resolve().parents[2] / '.local/upstream/rd-agent'))
        if not (self.upstream / '.git').exists():
            self.skipTest('Pinned optional FT-Agent checkout not installed')

    def test_original_policy_uses_feedback_to_select_parent_and_reuse_generated_files(self):
        requests = []
        class Backend:
            def build_chat_session(self, session_system_prompt):
                requests.append(session_system_prompt)
                class Session:
                    def build_chat_completion(self, user_prompt, **kwargs):
                        requests.append(user_prompt)
                        if '"reason"' in user_prompt[-500:]:
                            return '{"reason":"native reason"}'
                        if '"hypothesis"' in user_prompt[-500:]:
                            return '{"hypothesis":"native hypothesis"}'
                        return '{"task":"native task", "skip_data_processing":true}'
                return Session()
        api = load_policy(self.upstream, settings=SimpleNamespace(base_model='target',
            target_benchmark='gsm8k', force_think_token=False), backend=Backend(),
            parameter_info={'methods': ['lora']}, dataset_descriptor=object())
        trace = api.FTTrace(Scenario('frozen task', {}))
        generator = api.LLMFinetuneExpGen(trace.scen)
        first = generator.gen(trace)
        self.assertFalse(first.sub_tasks[0].skip_data_processing)
        first.experiment_workspace.file_dict['process_data.py'] = 'accepted original'
        trace.sync_dag_parent_and_hist((first, api.HypothesisFeedback(reason='measured', decision=True)), 1)
        second = generator.gen(trace)
        self.assertEqual(second.local_selection, (0,))
        self.assertTrue(second.sub_tasks[0].skip_data_processing)
        self.assertEqual(second.experiment_workspace.file_dict['process_data.py'], 'accepted original')
        self.assertTrue(any('native hypothesis' in r for r in requests))
        self.assertEqual(FT_AGENT_REVISION, '484776c211e4fbbeef03e0ec00d6bbee7362a4f4')
        self.assertTrue(api.convert2bool('ok'))
        with self.assertRaises(ValueError):
            api.convert2bool('uncertain')

    def test_native_configuration_generation_and_feedback_execute_all_dependencies(self):
        class Backend:
            def build_chat_session(self, **kwargs):
                class Session:
                    def build_chat_completion(self, user_prompt, **kwargs):
                        return 'learning_rate: 0.0002\n'
                return Session()
            def build_messages_and_create_chat_completion(self, **kwargs):
                return '{"Code Summary":"native", "Reason":"measured validation result", "Decision":"yes"}'
        with tempfile.TemporaryDirectory() as temporary:
            api = load_policy(self.upstream, settings=SimpleNamespace(base_model='target',
                target_benchmark='gsm8k', force_think_token=False), backend=Backend(),
                parameter_info={'methods': ['lora']}, dataset_descriptor=object())
            scen = Scenario('frozen task', {'accuracy_summary': []})
            strategy = api.LLMFinetuneEvolvingStrategy(scen, object())
            ws = api.FTWorkspace()
            generated = strategy._generate_llamafactory_config_with_llm('target',
                queried_former_failed_knowledge=([], []), workspace=ws)
            self.assertIn('train.yaml', generated)
            self.assertIn('test_params.yaml', generated)
            task = api.FTTask('target', 'native task', 'gsm8k')
            exp = api.FTExperiment([task], hypothesis=api.FTHypothesis('target', 'native hypothesis', 'reason'))
            exp.experiment_workspace.file_dict = generated
            exp.experiment_workspace.running_info = SimpleNamespace(
                result={'benchmark': {'accuracy_summary': [], 'error_samples': []}}, running_time=1)
            feedback = api.FTExperiment2Feedback(scen).generate_feedback(exp, api.FTTrace(scen))
            self.assertTrue(feedback.decision)


if __name__ == '__main__':
    unittest.main()
