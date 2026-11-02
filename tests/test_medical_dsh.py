"""Same medical task/registrations with replaceable harnesses; CPU fixtures only."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from examples.medical.experiment import DecisionBackend, ExperimentSettings, parser, run
from rsisys.loops.contracts.model_service import GenerationRequest, GenerationResponse, ModelRef
from rsisys.loops.decision_making.harness.context import IdentityContextCompactor
from rsisys.loops.decision_making.harness.dsh import DSHConfig, DeepSeekDecisionHarness, _sdk_runtime
from tests.test_dsh_harness import Runtime
from tests import test_medical_experiment as medical_fixture
from tests.test_medical_experiment import (
    Converter, FINISH, FixtureServing,
    apply, evaluate, train,
)


class MedicalRuntime(Runtime):
    def __init__(self, script, process):
        super().__init__(script)
        self.process, self.context, self.contexts = process, {}, []

    def run(self, prompt, on_notification=None):
        if self.process.active_job:
            raise AssertionError('DSH inference while training owns resources')
        self.prompts.append(prompt)
        delta = json.loads(prompt.split('Current research observations (data):\n')[1])
        for key, value in delta.items():
            if key in ('history', 'experience', 'memories', 'operation_results'):
                self.context.setdefault(key, []).extend(value)
            else:
                self.context[key] = value
        self.contexts.append(deepcopy(self.context))
        answer = next(self.answers)
        if callable(answer):
            answer = answer(self.context)
        return SimpleNamespace(final_response=json.dumps(answer), finish_reason='completed')


class MedicalDSHTests(unittest.TestCase):
    setUp = medical_fixture.MedicalExperimentIntegrationTests.setUp

    def execute(self, script, *, kind='dsh', max_model_calls=96, harness_factory=None, require_research_attempts=False,
                feedback_learning=False, feedback_backend=None):
        runtime = MedicalRuntime(script, self.process)
        config_path = self.root / ('dsh-'+str(len(list(self.root.glob('run-*'))))+'.json')
        config_path.write_text(json.dumps({'home': str(self.root/'dsh-home'), 'workspace': str(self.root),
            'model': 'fixture-independent-researcher', 'session_id': 'medical-test',
            'provider': 'openai-completions', 'base_url': 'http://127.0.0.1:1/v1'}))
        root = self.root / ('run-'+kind)
        settings = ExperimentSettings(self.task, root, self.root/'fixture-model', decision_harness=kind,
            dsh_config=config_path if kind == 'dsh' else None, max_model_calls=max_model_calls,
            require_research_attempts=require_research_attempts, feedback_learning=feedback_learning)
        serving_box = []
        def serve(path, **kwargs):
            serving = FixtureServing(path, process=self.process, script=script if kind == 'default' else [], **kwargs)
            serving_box.append(serving)
            return serving
        result = run(settings, serving_factory=serve, converter_factory=lambda *a, **kw: Converter(),
            process_backend=self.process, dsh_runtime_factory=lambda config: runtime,
            decision_harness_factory=harness_factory, feedback_backend=feedback_backend,
            identity_factory=lambda path: ('fixture-pin', {'model_id': 'fixture-actor', 'actual_weight_sha256': 'fixture'}))
        return result, runtime, serving_box[0], root

    def test_dsh_feedback_uses_independent_transport_and_shared_budget(self):
        from tests.test_feedback_learning_integration import Backend
        feedback = Backend()
        result, runtime, serving, root = self.execute([
            {'kind': 'explore', 'mode': 'configure_prompt', 'strategy': 'Test labels',
             'arguments': {'instructions': 'Return one legal label.'}}, FINISH],
            feedback_learning=True, feedback_backend=feedback)
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(len(feedback.requests), 1)
        self.assertEqual(result['model_calls'], 3)
        self.assertEqual(len(runtime.prompts), 2)
        jobs = json.loads((root / 'feedback_learning.json').read_text())['jobs']
        self.assertEqual([row['status'] for row in jobs.values()], ['completed'])
        self.assertTrue(runtime.closed)
        self.assertEqual(serving.model_calls, 0)
        self.assertNotIn('Feedback evidence (untrusted data)', ''.join(runtime.prompts))

    def test_dsh_preserves_explicit_attempt_admission(self):
        proposal = {'kind': 'explore', 'mode': 'configure_prompt', 'strategy': 'Synthetic comparison',
                    'arguments': {'instructions': 'Return one label.'},
                    'research_attempt_title': 'Short prompt hypothesis'}
        result, runtime, serving, root = self.execute([proposal, FINISH], require_research_attempts=True)
        self.assertEqual(result['status'], 'completed', result)
        rows = json.loads((root/'monitor.json').read_text())['decisions']
        research = [r for r in rows if r.get('mode') == 'configure_prompt']
        self.assertEqual(len(research), 1)
        self.assertTrue(research[0]['research_attempt_id'])

    def test_same_prompt_evaluate_apply_without_required_training(self):
        proposal = {'kind': 'explore', 'mode': 'configure_prompt', 'strategy': 'Synthetic comparison',
                    'arguments': {'template': 'Return one answer label.'}, 'input_ids': []}
        # Prompt parameter ownership stays with the registered medical directive.
        from rsisys.loops.decision_making.action_space.directives import builtin_directives
        spec = next(d.spec for d in builtin_directives() if d.spec.mode == 'configure_prompt')
        proposal['arguments'] = {next(iter(spec.input_schema['properties'])): 'Return one answer label.'}
        outcomes = []
        for kind in ('default', 'dsh'):
            result, runtime, serving, root = self.execute([proposal, evaluate, apply, FINISH], kind=kind)
            self.assertEqual(result['status'], 'completed', result)
            self.assertEqual(result['model_calls'], 4)
            self.assertEqual(result['trainings_reserved'], 0)
            self.assertEqual(len(serving.activated), 0)  # Fixed fixture changes accuracy only for trained weights.
            self.assertEqual(serving.model_calls, 4 if kind == 'default' else 0)
            contexts = serving.contexts if kind == 'default' else runtime.contexts
            self.assertEqual(contexts[-1]['last_feedback']['status'], 'failed')
            self.assertTrue(contexts[-1]['candidate_evaluations'])
            outcomes.append([spec['name'] for spec in contexts[0]['operations']])
            if kind == 'dsh':
                self.assertEqual(runtime.opened, [('start', 'medical-test')])
                self.assertTrue(runtime.closed)
                self.assertNotEqual(contexts[0]['decision_model']['model_id'], serving.base.model_id)
                self.assertNotIn('objective', json.loads(runtime.prompts[1].split('Current research observations (data):\n')[1]))
                self.assertTrue(json.loads((root/'decision_harness.json').read_text())['dsh'])
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(self.process.requests, {})

    def test_dsh_training_budget_and_native_input_scope_match_default(self):
        result, runtime, serving, root = self.execute([train, train, FINISH])
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(result['model_calls'], 3)
        self.assertEqual(len(self.process.requests), 1)
        feedback = runtime.contexts[-1]['last_feedback']
        self.assertEqual(feedback['status'], 'failed')
        self.assertIn('Exact duplicate', feedback['error'])
        self.assertEqual(serving.model_calls, 0)

    def test_model_budget_counts_native_turns_and_stops_without_finish(self):
        result, runtime, serving, root = self.execute([
            {'kind': 'invoke', 'operation': 'preview_training_data', 'arguments': {}}, FINISH], max_model_calls=1)
        self.assertEqual(result['status'], 'stopped', result)
        self.assertEqual(result['reason'], 'model_call_budget_exhausted')
        self.assertEqual(result['model_calls'], 1)
        self.assertEqual(len(runtime.prompts), 1)
        self.assertIsNone(result['researcher_finish_reason'])
        self.assertTrue(runtime.closed)

    def test_envelope_normalization_preserves_known_completed_lineage(self):
        selected = []
        def with_lineage(context):
            ident = next(row['decision_id'] for row in context['decision_graph'] if row['status'] == 'completed')
            selected.append(ident)
            return {'action': 'preview_training_data', 'arguments': {},
                    'depends_on_decision_ids': [ident], 'source_decision_ids': [ident]}
        for kind in ('default', 'dsh'):
            result, runtime, serving, root = self.execute([
                {'kind': 'invoke', 'operation': 'preview_training_data', 'arguments': {}}, with_lineage, FINISH], kind=kind)
            self.assertEqual(result['status'], 'completed', result)
            contexts = serving.contexts if kind == 'default' else runtime.contexts
            rows = contexts[-1]['decision_graph']
            linked = next(record for record in rows if record.get('depends_on_decision_ids'))
            self.assertEqual(linked['depends_on_decision_ids'], [selected[-1]])
            self.assertEqual(linked['source_decision_ids'], [selected[-1]])

    def test_both_harnesses_record_attempt_membership_through_medical_envelope_and_monitor(self):
        def continue_attempt(context):
            first = next(row for row in context['decision_graph'] if row.get('research_attempt_id'))
            return {'action': 'preview_training_data', 'arguments': {},
                    'research_attempt_id': first['research_attempt_id'],
                    'depends_on_decision_ids': [first['decision_id']]}
        for kind in ('default', 'dsh'):
            result, runtime, serving, root = self.execute([
                {'action': 'preview_training_data', 'arguments': {},
                 'research_attempt_title': 'Inspect training coverage before changing the model'},
                continue_attempt, FINISH], kind=kind)
            self.assertEqual(result['status'], 'completed', result)
            rows = json.loads((root/'monitor.json').read_text())['decisions']
            grouped = [r for r in rows if r.get('research_attempt_id')]
            self.assertEqual(len(grouped), 2)
            self.assertEqual(len({r['research_attempt_id'] for r in grouped}), 1)
            self.assertTrue(all(r['research_attempt_title'] ==
                'Inspect training coverage before changing the model' for r in grouped))
            self.assertEqual(grouped[1]['depends_on_decision_ids'], [grouped[0]['decision_id']])

    def test_factory_replaces_harness_and_forces_identity_on_both_layers(self):
        built = []
        runtime = Runtime([json.dumps(FINISH)])
        def factory(**deps):
            config = DSHConfig(self.root/'factory-home', self.root, 'fixture-researcher', 'factory')
            def policy(backend, model):
                deps['backend'].serving = backend
                return deps['policy_factory'](backend, model)
            harness = DeepSeekDecisionHarness(config, deps['tools'], deps['experience'],
                memory=deps['memory'], research_agenda=deps['research_agenda'],
                directives=deps['directives'], model=deps['model'], policy_factory=policy,
                runtime_factory=lambda config: runtime)
            built.append(harness)
            return harness
        result, _, _, _ = self.execute([], harness_factory=factory)
        self.assertEqual(result['status'], 'completed', result)
        self.assertIsInstance(built[0], DeepSeekDecisionHarness)
        self.assertIsInstance(built[0].context_compactor, IdentityContextCompactor)
        self.assertIsInstance(built[0].policy.parser.context_compactor, IdentityContextCompactor)

    def test_cli_requires_an_explicit_provider_configuration(self):
        args = parser().parse_args(['--task', str(self.task), '--run-dir', str(self.root/'out'),
            '--model-path', str(self.root/'model'), '--decision-harness', 'dsh', '--dsh-config', str(self.root/'dsh.json')])
        self.assertEqual(args.decision_harness, 'dsh')
        self.assertEqual(args.dsh_config, self.root/'dsh.json')
        with self.assertRaisesRegex(ValueError, 'explicit'):
            ExperimentSettings(self.task, self.root/'out', self.root/'model', decision_harness='dsh')

    def test_count_tokens_delegates_or_explicitly_estimates_and_summary_cap_is_respected(self):
        model = ModelRef('test', 'fixture', 'pin')
        serving = SimpleNamespace(token_count_mode='provider_tokenizer', count_tokens=lambda req: 3979,
                                  generate=lambda req: SimpleNamespace())
        backend = DecisionBackend(serving, self.root, None, 3)
        request = GenerationRequest('decision_maker', model, 'data', max_output_tokens=64)
        self.assertEqual(backend.count_tokens(request), 3979)
        self.assertEqual(backend.token_count_mode, 'provider_tokenizer')
        del serving.count_tokens
        del serving.token_count_mode
        self.assertGreater(backend.count_tokens(request), len('data'))
        self.assertEqual(backend.token_count_mode, 'utf8_conservative_estimate')
        received = []
        serving.generate = lambda req: (received.append(req), GenerationResponse('summary', req.model, False))[1]
        backend.monitor = SimpleNamespace(event=lambda *a, **kw: None)
        backend.generate(request)
        self.assertEqual(received[-1].max_output_tokens, 64)


@unittest.skipUnless(importlib.util.find_spec('deepseek_harness'), 'Optional SDK required')
class MedicalDSHProviderTests(unittest.TestCase):
    def test_real_rc2_chat_completions_provider_is_explicit_and_persistent(self):
        from examples.dsh_smoke import LocalModelFixture, MODEL, fixture_credential, API_KEY_ENV
        with tempfile.TemporaryDirectory() as directory, LocalModelFixture(['first', 'second']) as server, fixture_credential():
            root = Path(directory)
            config = DSHConfig(root/'home', root, MODEL, 'medical-sdk-chat', provider='openai-completions',
                base_url=server.base_url, api_key_env=API_KEY_ENV)
            runtime = _sdk_runtime(config)
            try:
                session = runtime.start_session(config.session_id)
                self.assertEqual(session.run('First fixture observation').final_response, 'first')
            finally:
                runtime.close()
            runtime = _sdk_runtime(config)
            try:
                session = runtime.resume_session(config.session_id)
                result = session.run('Second fixture observation')
                self.assertEqual(result.final_response, 'second')
                self.assertEqual(result.finish_reason, 'completed')
            finally:
                runtime.close()
            server.verify()
            self.assertEqual(set(server.paths), {'/v1/chat/completions'})
            self.assertIn('First fixture observation', json.dumps(server.requests[-1]))
            self.assertTrue(all(not request.get('tools') for request in server.requests))

    def test_real_chat_provider_native_compaction_preserves_latest_observation(self):
        from examples.dsh_smoke import LocalModelFixture, MODEL, fixture_credential, API_KEY_ENV
        marker = 'LATEST-RAW-RECEIPT-END'
        sections = ('Primary Request and Intent', 'Key Technical Concepts', 'Files and Code',
                    'Errors and Fixes', 'Pending Jobs', 'Current Work', 'Next Step', 'Critical Context')
        summary = '\n\n'.join('## '+name+'\n- Fixture archived history.' for name in sections)
        def respond(payload, compacting):
            return summary if compacting else 'native turn complete'
        with tempfile.TemporaryDirectory() as directory, LocalModelFixture(responder=respond) as server, fixture_credential():
            root = Path(directory)
            config = DSHConfig(root/'home', root, MODEL, 'medical-compact', provider='openai-completions',
                base_url=server.base_url, api_key_env=API_KEY_ENV, context_window=4096,
                max_tokens=512, trigger_tokens=3000, target_tokens=1600, summary_tokens=512)
            runtime = _sdk_runtime(config)
            events = []
            latest = ''
            try:
                session = runtime.start_session(config.session_id)
                for index in range(12):
                    latest = f'Raw observation {index}: '+('distinct medical fixture evidence '+str(index)+'; ')*48+marker
                    result = session.run(latest)
                    self.assertEqual(result.finish_reason, 'completed', str(result.events[-2:]))
                    events.extend(result.events)
                    if any(event['type'] == 'compaction/summary' for event in events):
                        break
            finally:
                runtime.close()
            server.verify()
            self.assertTrue(any(event['type'] == 'compaction/summary' for event in events))
            self.assertIn(latest, json.dumps(server.requests[-1]['messages']))
            self.assertEqual(set(server.paths), {'/v1/chat/completions'})
            self.assertTrue(all(not request.get('tools') for request in server.requests))


if __name__ == '__main__':
    unittest.main()
