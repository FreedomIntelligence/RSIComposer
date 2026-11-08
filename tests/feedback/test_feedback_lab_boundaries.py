"""Real receipts, deferred memory and pinned raw sources are separate products."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

from rsisys.adapters.trajectory import FeedbackTrajectorySource
from rsisys.config import build_demo
from rsisys.loops.decision_making.action_space import OperationCall
from rsisys.loops.decision_making.feedback import ModelImprovementProposer
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.research import ResearchAssembly, assemble_research
from rsisys.runtime.json_tools import EvidencePathError
from tests.feedback.test_feedback_required_outputs import CONFIG, EmptyBackend, observation, exact
from tests.feedback.test_feedback_learning_integration import ImmediateExecutor
from tests.regressions.test_m01_contract_boundaries import Script


class BlockingMemory:
    max_calls_per_job = 1

    def __init__(self):
        self.started, self.release = Event(), Event()

    def configuration(self):
        return {'provider': 'blocking-fixture'}

    def process(self, job, backend, limits):
        self.started.set()
        if not self.release.wait(5):
            raise TimeoutError('Fixture was not released')
        return {'status': 'completed', 'charged_calls': 0, 'proposals': []}

    def recover(self, job):
        return None


class ImprovementBackend(EmptyBackend):
    def __init__(self):
        super().__init__()
        self.kind, self.subject, self.example = 'policy_training', 'meta_agent', False

    def generate(self, request):
        from rsisys.loops.contracts.model_service import GenerationResponse
        self.requests.append(request)
        payload = json.loads(request.prompt)
        item = {'kind': self.kind, 'subject': self.subject,
            'content': 'Investigate a data recipe and evaluate a decision-policy candidate.',
            'applicability': 'This captured research context.',
            'source_refs': [payload['source_refs'][0]],
            'validation_plan': 'Construct data in the improvement Lab, then train and evaluate separately.'}
        if 'candidate_evidence' in payload:
            item['evaluation_claims'] = []
        if self.example:
            item['training_example'] = {'input': 'x', 'target': 'y', 'target_status': 'unverified'}
        if request.tools:
            return GenerationResponse('', request.model, simulated=True, finish_reason='tool_calls', tool_calls=[{
                'id': 'proposal', 'type': 'function', 'function': {
                    'name': request.tools[0]['function']['name'],
                    'arguments': json.dumps({'proposals': [item]})}}])
        return GenerationResponse(json.dumps({'proposals': [item]}), request.model, simulated=True)


class FeedbackLabBoundaryTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def install(self, assembly=None, *, backend=None, **options):
        runner, _ = build_demo(CONFIG)
        self.addCleanup(runner.close)
        backend = backend or EmptyBackend()
        service = assemble_research(runner.decision_making, assembly=assembly or ResearchAssembly(),
            path=self.root / 'feedback.json', backend=backend, owner=runner.assembly, **options)
        return runner, service, backend

    def test_receipt_is_durable_before_any_memory_or_proposal_work_is_dispatched(self):
        runner, service, backend = self.install(executor=ImmediateExecutor())
        loop = runner.decision_making
        loop.harness.policy = Script(Decision('invoke', decision_id='real-result',
            operation=OperationCall('prepare_data', {'text': 'Exact real result'})))
        step = loop.step('objective')
        row, = service.learning.state['jobs'].values()
        saved = json.loads(service.learning.path.read_text())['jobs'][row['job_id']]
        self.assertEqual(saved['raw_input']['receipt']['result'], json.loads(json.dumps(step.operation_result.payload())))
        self.assertNotIn('feedback_delivery', saved['raw_input']['receipt']['result'])
        self.assertEqual(row['status'], 'queued')
        self.assertEqual(service.learning.state['attempted'], 0)
        self.assertFalse(service.learning._futures)
        self.assertFalse(backend.requests)
        self.assertEqual(loop.harness.state.decision_records['real-result']['status'], 'completed')

    def test_pending_memory_does_not_hide_the_real_result_from_the_next_decision(self):
        memory = BlockingMemory()
        runner, service, _ = self.install(ResearchAssembly(memory_processor=memory))
        self.addCleanup(memory.release.set)
        loop = runner.decision_making
        seen = []

        class Policy:
            model = loop.harness.policy.model

            def decide(self, context):
                seen.append(context)
                return (Decision('invoke', operation=OperationCall('prepare_data', {'text': 'ready'}))
                        if len(seen) == 1 else Decision('finish'))

        loop.harness.policy = Policy()
        result = loop.step('objective').operation_result
        self.assertEqual(loop.step('objective').kind, 'finished')
        self.assertTrue(memory.started.wait(1))
        self.assertFalse(memory.release.is_set())
        self.assertIn(result, seen[1].operation_results)
        self.assertEqual(next(iter(service.learning.state['jobs'].values()))['status'], 'running')

    def test_query_is_archived_without_recursive_feedback_inference(self):
        runner, service, backend = self.install(executor=ImmediateExecutor())
        loop = runner.decision_making
        loop.harness.policy = Script(Decision('invoke', decision_id='query',
            operation=OperationCall('feedback_learning_read', {})))
        result = loop.step('objective').operation_result
        row, = service.learning.state['jobs'].values()
        self.assertEqual(row['status'], 'archived')
        self.assertEqual(row['raw_input']['receipt']['result'], result.payload())
        service.refresh()
        self.assertFalse(backend.requests)
        self.assertEqual(service.learning.state['attempted'], 0)
        self.assertEqual(json.loads(service.context())['provider'], 'agent_managed')
        self.assertEqual(json.loads(service.context())['total'], 0)
        self.assertEqual(len(service.trajectories()), 1)

    def test_pinned_raw_snapshot_does_not_change_when_memory_finishes_or_is_read(self):
        _, service, _ = self.install(executor=ImmediateExecutor())
        ident = service.observe(observation())
        source = FeedbackTrajectorySource(service, archive_id='fixture-archive')
        ref = source.reference(ident)
        before = source.read(ref)
        self.assertEqual(before['dataset_construction_owner'], 'lab')
        self.assertFalse(before['training_ready'])
        self.assertEqual(before['coverage']['original_meta_agent_exchange'], 'not_captured')
        service.refresh()
        self.assertEqual(source.reference(ident), ref)
        self.assertEqual(source.read(ref), before)
        self.assertEqual(exact(service.learning, job_id=ident, evidence_path='/trajectory_snapshot'), before)
        detached = source.read(ref)
        detached['capture'].clear()
        self.assertEqual(source.read(ref), before)

    def test_archive_or_digest_mismatch_is_rejected(self):
        _, service, _ = self.install()
        ident = service.observe(observation())
        source = FeedbackTrajectorySource(service, archive_id='one')
        ref = source.reference(ident)
        for bad in (replace(ref, archive_id='other'), replace(ref, digest='0' * 64)):
            with self.subTest(ref=bad), self.assertRaises(ValueError):
                source.read(bad)

    def test_disabled_or_index_only_archive_cannot_silently_supply_training_sources(self):
        from tests.feedback.test_feedback_components import ExternalArchive
        for archive in (None, ExternalArchive()):
            with self.subTest(archive=archive):
                runner, _ = build_demo(CONFIG)
                with runner:
                    service = assemble_research(runner.decision_making,
                        assembly=ResearchAssembly(trajectory_archive=archive), backend=EmptyBackend())
                    ident = service.observe(observation())
                    with self.assertRaises(EvidencePathError):
                        FeedbackTrajectorySource(service, archive_id='test').reference(ident)

    def test_improvement_proposal_is_a_task_without_generated_training_examples(self):
        backend = ImprovementBackend()
        assembly = ResearchAssembly(improvement_proposals=ModelImprovementProposer(self.root / 'proposals'))
        runner, service, _ = self.install(assembly, backend=backend, executor=ImmediateExecutor())
        ident = service.observe(observation())
        self.assertFalse(backend.requests)
        service.refresh()
        row = service.learning.state['jobs'][ident]
        self.assertEqual(row['products']['improvement_proposals']['status'], 'completed', row)
        proposal, = service.proposals(target='weights')
        self.assertEqual(proposal['kind'], 'policy_training')
        self.assertNotIn('training_example', proposal['proposal'])
        self.assertFalse(proposal['applied'])
        self.assertEqual(service.resolve_improvement_proposals([proposal['proposal_id']])[0]['kind'], 'policy_training')
        self.assertIs(runner.decision_making.harness.feedback_proposal_resolver.__self__, service)
        loop = runner.decision_making
        loop.harness.policy = Script(Decision('invoke', operation=OperationCall('prepare_data', {'text': 'build plan'}),
            feedback_proposal_ids=(proposal['proposal_id'],)))
        step = loop.step('Choose an ordinary follow-up action')
        self.assertEqual(step.kind, 'progressed')
        linked, = service.proposals(target='weights')
        self.assertEqual(linked['decision_links'][0]['decision_id'], step.decision_id)
        self.assertFalse(linked['applied'])

    def test_improvement_proposer_supports_native_tool_calling_without_executing_tasks(self):
        backend = ImprovementBackend()
        backend.supports_tool_calling = True
        _, service, _ = self.install(ResearchAssembly(
            improvement_proposals=ModelImprovementProposer(self.root / 'native')),
            backend=backend, executor=ImmediateExecutor())
        ident = service.observe(observation())
        service.refresh()
        self.assertEqual(backend.requests[0].tools[0]['function']['name'], 'propose_improvements')
        self.assertEqual(service.learning.state['jobs'][ident]['products']['improvement_proposals']['charged_calls'], 1)
        self.assertEqual(service.learning.state['attempted'], 1)

    def test_only_one_optional_proposal_stage_can_be_installed(self):
        from rsisys.loops.decision_making.feedback import ModelHarnessProposer
        with self.assertRaisesRegex(ValueError, 'one improvement proposal stage'):
            ResearchAssembly(harness_proposals=ModelHarnessProposer(self.root / 'old'),
                             improvement_proposals=ModelImprovementProposer(self.root / 'new'))

    def test_feedback_proposal_cannot_construct_examples_or_train_the_lab_actor(self):
        for case in ('example', 'lab'):
            with self.subTest(case=case):
                backend = ImprovementBackend()
                backend.example, backend.subject = case == 'example', 'lab' if case == 'lab' else 'meta_agent'
                runner, _ = build_demo(CONFIG)
                with runner:
                    service = assemble_research(runner.decision_making,
                        assembly=ResearchAssembly(improvement_proposals=ModelImprovementProposer(self.root / case)),
                        backend=backend, executor=ImmediateExecutor())
                    ident = service.observe(observation())
                    service.refresh()
                    self.assertEqual(service.learning.state['jobs'][ident]['products']['improvement_proposals']['status'], 'failed')
                    self.assertFalse(service.proposals(target=None))


if __name__ == '__main__':
    unittest.main()
