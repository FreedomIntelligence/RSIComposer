"""New research admission is explicit; historical experiments remain ungrouped."""
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from experiments.medical.experiment import ExperimentSettings
from rsisys.adapters.serving import VLLMBackend
from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.model_service import BackendBinding, GenerationResponse, ModelRef
from rsisys.loops.decision_making.action_space import DemoDecisionActionSpace, OperationCall, OperationResult
from rsisys.loops.decision_making.harness.attempts import AttemptContract
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.harness.state import DecisionContext
from tests.experiments.medical import test_medical_experiment as fixture

MODEL = ModelRef('fixture', 'researcher', 'v1')
CONTRACT = AttemptContract(required=True, operations=('prepare_data',))
PROMPT = {'kind': 'explore', 'mode': 'configure_prompt', 'strategy': 'Test a shorter prompt',
          'arguments': {'instructions': 'Return the single best answer label.'}}


class RequiredAttemptTests(unittest.TestCase):
    def setUp(self):
        self.context = DecisionContext('task', (), DemoDecisionActionSpace().describe(), (),
            directive_modes=('configure_prompt',), decision_graph=(
                {'decision_id': 'origin', 'status': 'completed', 'research_attempt_id': 'known',
                 'research_attempt_title': 'Simplify prompt'},))

    def policy(self, *payloads):
        backend = Mock(supports_response_schema=True)
        backend.generate.side_effect = [GenerationResponse(json.dumps(p), MODEL, False) for p in payloads]
        return ModelDecisionPolicy(backend, MODEL, attempt_contract=CONTRACT)

    def test_missing_membership_rejected_but_auxiliary_finish_allowed(self):
        for payload in (PROMPT, {'kind': 'invoke', 'operation': 'prepare_data', 'arguments': {'text': 'x'}}):
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, 'requires an Attempt'):
                self.policy(payload).decide(self.context)
        self.assertEqual(self.policy({'kind': 'finish'}).decide(self.context).kind, 'finish')

    def test_new_and_continue_wire_branches_require_membership_even_in_portable_mode(self):
        policy = self.policy()
        serving = VLLMBackend(BackendBinding('fixture', 'http', 'http://127.0.0.1:1/v1'),
                              native_schema_mode='portable')
        for schema in (policy.decision_schema(self.context), policy.review_schema(self.context)):
            for wire in (schema, serving.native_response_schema(schema)):
                research = [b for b in wire['oneOf'] if b['properties'].get('kind', {}).get('const') == 'explore'
                    or b['properties'].get('operation', {}).get('const') == 'prepare_data']
                self.assertTrue(research)
                for branch in research:
                    required = set(branch['required'])
                    self.assertTrue(required & {'research_attempt_title', 'research_attempt_id'})
                    if 'research_attempt_id' in required:
                        self.assertEqual(branch['properties']['research_attempt_id']['enum'], ['known'])
                    else:
                        self.assertEqual(branch['properties']['research_attempt_id'], {'type': 'null'})
        self.assertIn('MUST belong to an Attempt', policy._decision_request(self.context).system)
        self.assertNotIn('optional for legacy', policy._decision_request(self.context).system)

    def test_unknown_id_empty_title_and_rename_do_not_satisfy_membership(self):
        for fields in ({'research_attempt_title': ''}, {'research_attempt_title': '  '},
                       {'research_attempt_id': 'invented'},
                       {'research_attempt_id': 'known', 'research_attempt_title': 'rename'}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.policy({**PROMPT, **fields}).decide(self.context)
        decision = self.policy({**PROMPT, 'research_attempt_id': 'known'}).decide(self.context)
        self.assertEqual(decision.research_attempt_id, 'known')

    def test_executor_rejects_bypass_without_creating_attempt_or_side_effect(self):
        policy, tools = Mock(), Mock()
        tools.execute.return_value = OperationResult('done', (), False)
        harness = DefaultDecisionHarness(policy, tools, BasicExperienceSystem(InMemoryExperienceStore()),
                                         attempt_contract=CONTRACT)
        policy.decide.return_value = Decision('invoke', decision_id='missing',
                                              operation=OperationCall('prepare_data', {'text': 'x'}))
        with self.assertRaisesRegex(ValueError, 'requires an Attempt'):
            harness.step('task', DemoDecisionActionSpace())
        tools.execute.assert_not_called()
        record = harness.state.decision_records['missing']
        self.assertEqual(record['status'], 'failed')
        self.assertIsNone(record['research_attempt_id'])

    def test_new_medical_settings_do_not_require_model_grouping(self):
        settings = ExperimentSettings(Path('task'), Path('run'), Path('model'))
        self.assertFalse(settings.require_research_attempts)


class RequiredAttemptIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.case = fixture.MedicalExperimentIntegrationTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)

    def test_prompt_change_evaluation_and_application_share_model_chosen_attempt(self):
        def grouped(action):
            def decide(context):
                prior = [row for row in context['decision_graph'] if row.get('research_attempt_id')]
                return {**action(context), 'research_attempt_id': prior[0]['research_attempt_id']}
            return decide
        result = self.case.execute([
            {**PROMPT, 'research_attempt_title': 'Simplify prompt to improve accuracy'},
            grouped(fixture.evaluate), grouped(fixture.apply), fixture.FINISH,
        ], require_research_attempts=True,
           actor_response=lambda request: 'B' if 'Return the single best answer label.' in request.system else 'A')
        self.assertEqual(result['status'], 'completed', result)
        rows = self.case.read('monitor.json')['decisions']
        research = [r for r in rows if r.get('mode') in ('configure_prompt', 'evaluate', 'apply_candidate')]
        self.assertEqual(len(research), 3)
        self.assertEqual(len({r['research_attempt_id'] for r in research}), 1)
        self.assertEqual({r['research_attempt_title'] for r in research}, {'Simplify prompt to improve accuracy'})
        self.assertTrue(all(r.get('research_attempt_id') for r in research))
        self.assertFalse(self.case.process.requests)  # Prompt research is an Attempt without SFT.
        self.assertTrue(self.case.read('attempt-contract.json')['required'])

    def test_missing_model_membership_is_corrected_before_any_prompt_executes(self):
        result = self.case.execute([PROMPT, {**PROMPT, 'research_attempt_title': 'Explicit hypothesis'}, fixture.FINISH],
                                   require_research_attempts=True)
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(result['model_calls'], 3)
        research = [r for r in self.case.read('monitor.json')['decisions'] if r.get('mode') == 'configure_prompt']
        self.assertEqual(len(research), 1)
        self.assertTrue(research[0]['research_attempt_id'])
        self.assertIn('requires an Attempt', self.case.serving.contexts[1]['objective'])

    def test_repeated_omission_exhausts_validation_without_execution(self):
        result = self.case.execute([PROMPT, PROMPT, PROMPT], require_research_attempts=True)
        self.assertEqual(result['status'], 'failed', result)
        self.assertEqual(result['model_calls'], 3)
        self.assertFalse(self.case.read('monitor.json')['decisions'])
        self.assertFalse(self.case.process.requests)

    def test_feedback_learning_keeps_attempt_scope_across_prompt_and_evaluation(self):
        from types import SimpleNamespace
        scopes = []
        def continue_evaluation(context):
            first = next(row for row in context['decision_graph'] if row.get('research_attempt_id'))
            return {**fixture.evaluate(context), 'research_attempt_id': first['research_attempt_id']}
        script = iter([{**PROMPT, 'research_attempt_title': 'Short prompt hypothesis'},
                       continue_evaluation, fixture.FINISH])
        def generate(request):
            context = json.loads(request.prompt)
            if 'receipt' in context:
                scopes.append(context['research_context'])
                payload = {'proposals': []}
            else:
                payload = next(script)
                if callable(payload):
                    payload = payload(context)
            return GenerationResponse(json.dumps(payload), request.model, False)
        backend = SimpleNamespace(generate=generate, supports_response_schema=True,
                                  count_tokens=lambda request: len(request.prompt.encode()) // 4)
        result = self.case.execute([], decision_model=MODEL, decision_backend=backend,
                                   memory_provider='journal', harness_proposals=True,
                                   require_research_attempts=True)
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(len(scopes), 2)
        self.assertTrue(scopes[0]['research_attempt_id'])
        self.assertEqual(scopes[0]['research_attempt_id'], scopes[1]['research_attempt_id'])
        self.assertEqual([len(s['member_decisions']) for s in scopes], [1, 2])
