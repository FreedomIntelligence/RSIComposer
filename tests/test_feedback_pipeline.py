"""Feedback products -> ordinary decision -> receipt -> extracted public trace."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from rsisys.config import build_demo
from rsisys.loops.contracts.model_service import GenerationResponse
from rsisys.loops.decision_making.action_space import OperationCall
from rsisys.loops.decision_making.feedback import (
    FeedbackPipeline, ModelHarnessProposer, enable_feedback_pipeline)
from rsisys.loops.decision_making.feedback.learning import FeedbackLearning, SerializedLearningBackend
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.runtime.files import atomic_write, json_bytes
from rsisys.runtime.json_tools import EvidencePathError
from rsisys.snapshot.codec import encode, decode
from tests.test_feedback_learning_integration import ImmediateExecutor
from tests.test_feedback_required_outputs import CONFIG, MODEL, observation, exact


class ProposalBackend:
    supports_response_schema = True

    def __init__(self):
        self.requests = []
        self.bad_source = self.wrong_model = self.too_large = self.native = False

    def count_tokens(self, request):
        return 1000000 if self.too_large else 100

    def generate(self, request):
        self.requests.append(request)
        packet = json.loads(request.prompt)
        proposal = {'kind': 'harness_change', 'subject': 'meta_agent',
            'content': 'Draft and test a clearer output parser.', 'applicability': 'This harness version.',
            'validation_plan': 'Run a parser reproduction before applying a version.',
            'source_refs': ['unknown'] if self.bad_source else [packet['source_refs'][0]]}
        if 'candidate_evidence' in packet:
            proposal['evaluation_claims'] = []
        payload = {'proposals': [proposal]}
        model = replace(request.model, revision='wrong') if self.wrong_model else request.model
        if request.tools:
            return GenerationResponse('', model, True, finish_reason='tool_calls', tool_calls=[{
                'id': 'proposal', 'type': 'function', 'function': {
                    'name': 'propose_harness_changes', 'arguments': json.dumps(payload)}}])
        return GenerationResponse(json.dumps(payload), model, True)


class RecordedMemory:
    """Durable fixture product; no inference and no custom memory algorithm."""
    max_calls_per_job = 2

    def __init__(self, path):
        self.path = path
        self.fail = False

    def configuration(self):
        return {'provider': 'fixture_memory', 'path': str(self.path)}

    def process(self, job, backend, limits):
        result = {'status': 'failed' if self.fail else 'completed', 'error': 'fixture' if self.fail else None,
                  'charged_calls': 0, 'proposals': [], 'response': {'fixture': True}}
        atomic_write(self.path, json_bytes(result))
        return result

    def recover(self, job):
        return json.loads(self.path.read_text()) if self.path.exists() else None


class Script:
    def __init__(self, model, decision):
        self.model, self.decision = model, decision

    def decide(self, context):
        return self.decision


class FeedbackPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.backend = ProposalBackend()

    def learning(self, *, memory=None, proposer=True, **options):
        component = (ModelHarnessProposer(self.root / 'proposals') if proposer is True
                     else None if proposer is False else proposer)
        pipeline = FeedbackPipeline(memory_processor=memory, harness_proposals=component)
        learning = FeedbackLearning(SerializedLearningBackend(self.backend),
            path=self.root / 'feedback.json', processor=pipeline,
            executor=ImmediateExecutor(), **options)
        self.addCleanup(lambda: None if learning._closed else learning.close())
        return learning, pipeline

    def capture(self, learning, ident='one'):
        ident = learning.capture_receipt(observation(ident), MODEL)
        learning.poll()
        learning.poll()
        return ident

    def test_memory_and_optional_proposals_have_separate_products_and_exchanges(self):
        learning, _ = self.learning()
        ident = self.capture(learning)
        row = learning.state['jobs'][ident]
        self.assertEqual(row['status'], 'completed')
        self.assertEqual(row['products']['context_memory']['charged_calls'], 0)
        self.assertEqual(row['products']['harness_proposals']['charged_calls'], 1)
        self.assertEqual([p['kind'] for p in row['proposals']], ['harness_change'])
        self.assertEqual(learning.state['attempted'], len(self.backend.requests))
        self.assertTrue(row['simulated'])
        self.assertEqual(row['request']['actual_exchanges_read_path'], '/feedback_response/products')
        request = self.backend.requests[0]
        self.assertEqual(request.model, MODEL)
        branches = request.response_schema['properties']['proposals']['items']['oneOf']
        self.assertEqual([b['properties']['kind']['const'] for b in branches], ['harness_change'])
        self.assertIn('main meta-agent', request.system)
        calls = exact(learning, job_id=ident,
            evidence_path='/feedback_response/products/harness_proposals/response/model_exchanges')
        self.assertEqual(len(calls), 1)
        learning.capture_receipt(observation(), MODEL)
        learning.poll()
        self.assertEqual(len(self.backend.requests), 1)

    def test_disabling_proposals_keeps_memory_and_extraction_without_inference(self):
        learning, _ = self.learning(proposer=False)
        ident = self.capture(learning)
        self.assertEqual(learning.state['jobs'][ident]['status'], 'completed')
        self.assertEqual(learning.state['attempted'], 0)
        self.assertEqual(self.backend.requests, [])
        self.assertEqual(learning.context_record(ident)['facts']['status'], 'completed')
        self.assertEqual(learning.extract_trajectory(ident)['execution_receipt'],
                         json.loads(json.dumps(observation().payload())))

    def test_optional_failure_preserves_required_memory_and_failed_model_output(self):
        memory = RecordedMemory(self.root / 'memory.json')
        self.backend.bad_source = True
        learning, _ = self.learning(memory=memory)
        ident = self.capture(learning)
        row = learning.state['jobs'][ident]
        self.assertEqual(row['status'], 'completed')
        self.assertEqual(row['products']['harness_proposals']['status'], 'failed')
        self.assertEqual(learning.read(ident)['products']['harness_proposals']['status'], 'failed')
        self.assertEqual(learning.read()['items'][0]['products']['context_memory']['status'], 'completed')
        self.assertEqual(row['proposals'], [])
        self.assertTrue(memory.path.exists())
        self.assertIn('unknown', json.dumps(row['response']))

    def test_optional_plugins_cannot_emit_training_targets_or_untargeted_changes(self):
        class WrongProduct(RecordedMemory):
            kind = 'training_data'

            def process(self, job, backend, limits):
                proposal = {'kind': self.kind, 'subject': 'unspecified', 'content': 'Unaccepted fixture output',
                    'applicability': 'Fixture', 'source_refs': [job['input']['source_refs'][0]],
                    'validation_plan': 'Fixture'}
                if self.kind == 'training_data':
                    proposal['training_example'] = {'input': 'fixture', 'target': 'fixture',
                                                    'target_status': 'unverified'}
                return {'status': 'completed', 'charged_calls': 0, 'proposals': [proposal]}

        for kind in ('training_data', 'harness_change'):
            with self.subTest(kind=kind):
                self.root = self.root / kind
                component = WrongProduct(self.root / 'component.json')
                component.kind = kind
                learning, _ = self.learning(proposer=component)
                ident = self.capture(learning)
                self.assertEqual(learning.read(ident)['products']['harness_proposals']['status'], 'failed')
                self.assertEqual(learning.read(ident)['proposals'], [])
                self.assertEqual(self.backend.requests, [])

    def test_memory_failure_does_not_block_independent_optional_model_call(self):
        memory = RecordedMemory(self.root / 'memory.json')
        memory.fail = True
        learning, _ = self.learning(memory=memory)
        ident = self.capture(learning)
        self.assertEqual(learning.state['jobs'][ident]['status'], 'completed')
        self.assertEqual(learning.state['jobs'][ident]['products']['context_memory']['status'], 'failed')
        self.assertEqual(learning.state['jobs'][ident]['products']['harness_proposals']['status'], 'completed')
        self.assertEqual(len(self.backend.requests), 1)
        self.assertEqual(len(learning.trajectories()), 1)

    def test_native_proposal_call_is_recorded_without_executing_a_tool(self):
        self.backend.supports_tool_calling = True
        learning, _ = self.learning()
        ident = self.capture(learning)
        self.assertEqual(learning.state['jobs'][ident]['products']['harness_proposals']['status'], 'completed')
        self.assertEqual(self.backend.requests[0].tools[0]['function']['name'], 'propose_harness_changes')
        self.assertEqual(learning.state['attempted'], 1)

    def test_identity_and_complete_request_admission_reject_proposals(self):
        for field in ('wrong_model', 'too_large'):
            with self.subTest(field=field):
                self.root = self.root / field
                self.backend = ProposalBackend()
                setattr(self.backend, field, True)
                learning, _ = self.learning()
                ident = self.capture(learning)
                self.assertEqual(learning.state['jobs'][ident]['products']['harness_proposals']['status'], 'failed')
                self.assertEqual(learning.state['jobs'][ident]['proposals'], [])
                self.assertEqual(learning.state['attempted'], 1)
                self.assertEqual(len(self.backend.requests), int(field == 'wrong_model'))

    def test_committed_products_recover_without_new_inference_or_budget_reset(self):
        memory = RecordedMemory(self.root / 'memory.json')
        learning, pipeline = self.learning(memory=memory)
        ident = learning.capture_receipt(observation(), MODEL)
        learning.poll()  # Worker completed, owner still has durable running reservation.
        reserved = learning.state['attempted']
        with patch.object(learning, '_commit', side_effect=OSError('owner journal unavailable')):
            with self.assertRaises(OSError):
                learning.poll()
        restored = FeedbackLearning(SerializedLearningBackend(self.backend), path=learning.path,
            processor=pipeline, executor=ImmediateExecutor())
        try:
            self.assertEqual(restored.state['jobs'][ident]['status'], 'completed')
            self.assertEqual(restored.state['attempted'], 1)
            self.assertGreater(reserved, restored.state['attempted'])
            self.assertEqual(len(self.backend.requests), 1)
        finally:
            restored.close()

    def test_unknown_optional_call_preserves_memory_and_reservation_without_replay(self):
        memory = RecordedMemory(self.root / 'memory.json')
        learning, pipeline = self.learning(memory=memory)
        ident = learning.capture_receipt(observation(), MODEL)
        # Crash after inference: upstream memory committed, proposal result did not.
        with patch('rsisys.loops.decision_making.feedback.proposals.atomic_write', side_effect=OSError('crash')):
            learning.poll()
        restored = FeedbackLearning(SerializedLearningBackend(self.backend), path=learning.path,
            processor=pipeline, executor=ImmediateExecutor())
        try:
            row = restored.state['jobs'][ident]
            self.assertEqual(row['status'], 'completed')
            self.assertEqual(row['products']['harness_proposals']['status'], 'interrupted')
            self.assertEqual(restored.state['attempted'], learning.state['attempted'])
            self.assertEqual(len(self.backend.requests), 1)
        finally:
            restored.close()

    def test_trajectory_extraction_is_exact_detached_and_not_training_data(self):
        learning, _ = self.learning()
        ident = self.capture(learning)
        segment = learning.extract_trajectory(ident)
        self.assertEqual(segment, exact(learning, job_id=ident, evidence_path='/trajectory_segment'))
        self.assertEqual(segment['feedback']['request'],
                         json.loads(json.dumps(learning.state['jobs'][ident]['request'])))
        self.assertFalse(segment['training_ready'])
        self.assertFalse(segment['complete_conversation'])
        self.assertEqual(segment['original_meta_agent_exchange'], 'not_captured_by_this_observer')
        segment['execution_receipt'].clear()
        self.assertTrue(learning.extract_trajectory(ident)['execution_receipt'])
        self.assertEqual(len(self.backend.requests), 1)

    def test_disabled_archive_rejects_extraction(self):
        learning, _ = self.learning(trajectory_archive=None)
        ident = self.capture(learning)
        with self.assertRaisesRegex(EvidencePathError, 'disabled'):
            learning.extract_trajectory(ident)

    def test_proposal_selection_executes_through_main_decision_and_returns_feedback(self):
        runner, _ = build_demo(CONFIG)
        with runner:
            loop = runner.decision_making
            model = loop.harness.policy.model
            enable_feedback_pipeline(loop, backend=self.backend, path=self.root / 'feedback.json',
                harness_proposals=ModelHarnessProposer(self.root / 'proposals'), executor=ImmediateExecutor())
            service = loop.feedback_service
            service.observe(observation())
            service.refresh()
            proposal, = service.proposals()
            context = loop.harness._context('objective', loop.action_space)
            inbox = context.research_agenda['feedback_learning']['harness_proposals']
            self.assertEqual(inbox['recent'][0]['proposal_id'], proposal['proposal_id'])
            self.assertEqual(loop.harness.state.decision_records, {})  # Proposing executes nothing.
            payload = {'kind': 'invoke', 'operation': 'prepare_data', 'arguments': {'text': 'fixture build output'},
                       'feedback_proposal_ids': [proposal['proposal_id']]}
            policy = ModelDecisionPolicy(self.backend, model)
            decision = policy._parse_decision(GenerationResponse(json.dumps(payload), model, True), context)
            loop.harness.policy = Script(model, decision)
            result = loop.step('objective')
            self.assertEqual(result.kind, 'progressed')
            record = loop.harness.state.decision_records[decision.decision_id]
            self.assertEqual(record['status'], 'completed')
            self.assertEqual(record['feedback_proposals'][0]['proposal_id'], proposal['proposal_id'])
            linked = service.proposals()[0]
            self.assertEqual(linked['decision_links'][0]['decision_id'], decision.decision_id)
            self.assertFalse(linked['applied'])  # Generic success never claims harness activation.
            trace = service.extract_trajectory(service.learning.ordered_jobs()[-1]['job_id'])
            self.assertEqual(trace['research_context']['feedback_proposal_ids'], [proposal['proposal_id']])
            self.assertEqual(trace['execution_receipt']['task']['decision_id'], decision.decision_id)
            state = decode(encode(loop.harness.state))
            self.assertEqual(state.decision_records[decision.decision_id]['feedback_proposal_ids'],
                             (proposal['proposal_id'],))

    def test_unknown_proposal_reference_is_rejected_before_any_execution(self):
        runner, _ = build_demo(CONFIG)
        with runner:
            loop = runner.decision_making
            enable_feedback_pipeline(loop, backend=self.backend, executor=ImmediateExecutor())
            model = loop.harness.policy.model
            decision = Decision('invoke', operation=OperationCall('prepare_data', {'text': 'must not execute'}),
                                feedback_proposal_ids=('unknown:0',))
            loop.harness.policy = Script(model, decision)
            with self.assertRaisesRegex(ValueError, 'Unknown.*improvement proposal'):
                loop.step('objective')
            self.assertEqual(loop.harness.state.operation_results, [])

    def test_new_decision_metadata_roundtrips_and_old_checkpoint_defaults(self):
        decision = Decision('finish', feedback_proposal_ids=('learning:0',))
        self.assertEqual(decode(encode(decision)), decision)
        old = encode(Decision('finish'))
        del old['fields']['feedback_proposal_ids']
        self.assertEqual(decode(old).feedback_proposal_ids, ())


if __name__ == '__main__':
    unittest.main()
