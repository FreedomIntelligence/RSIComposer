"""A controlled outer-loop wiring scenario; this does not measure model quality."""
from concurrent.futures import Future
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rsisys.assembly import AssemblyContext, build_default_exploration
from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef
from rsisys.loops.decision_making.harness.learning_harness import feedback_learning_decision_factory
from rsisys.runtime.artifacts import FileArtifactStore
from rsisys.runtime.model import MockModelBackend
from rsisys.runtime.sandbox import MockSandboxBackend


class ImmediateExecutor:
    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future


class EvidenceFixture(MockModelBackend):
    def __init__(self):
        self.used_memory = False

    def generate(self, request):
        packet = json.loads(request.prompt)
        if 'receipt' in packet:
            if 'feedback' in packet['receipt']:
                assert packet['receipt']['feedback']['quality_metrics']['accuracy'] == 0
                payload = {'proposals': [{'kind': 'memory', 'content': 'Fixture receipt: wrong output format.',
                    'applicability': 'This simulated format task only.', 'source_refs': [packet['source_refs'][0]],
                    'validation_plan': 'Check corrected format on a fresh simulated task.'}]}
            else:
                assert packet['receipt']['operation']['name'] == 'prepare_data'
                payload = {'proposals': []}
        elif packet['research_state']['findings']:
            content = packet['research_state']['findings'][0]
            assert packet['last_feedback']['quality_metrics']['accuracy'] == 0
            assert content['authority'] == 'unverified_interpretation'
            assert content['proposals'][0]['content'] == 'Fixture receipt: wrong output format.'
            assert content['source_refs']
            self.used_memory = True
            scope = content['research_context']
            payload = {'kind': 'invoke', 'operation': 'prepare_data',
                       'arguments': {'text': 'candidate correction: one legal label'},
                       'research_attempt_id': scope['research_attempt_id'],
                       'reason': 'Test a format correction suggested by the negative fixture receipt.'}
        else:
            payload = {'kind': 'explore', 'mode': 'execute', 'strategy': 'Try initial fixture format',
                       'research_attempt_title': 'Check output format'}
        return GenerationResponse(json.dumps(payload), request.model, simulated=True)


class OuterLoopFeedbackTests(unittest.TestCase):
    def test_negative_feedback_changes_next_action_via_memory_with_same_attempt(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            backend = EvidenceFixture()
            assembly = AssemblyContext(BasicExperienceSystem(InMemoryExperienceStore()),
                FileArtifactStore(root / 'artifacts'), {'fixture': backend}, MockSandboxBackend(),
                {role: ModelRef('fixture', role, 'v1') for role in ('decision_maker', 'actor')}, data_dir=root)
            try:
                exploration = build_default_exploration(assembly)
                loop = feedback_learning_decision_factory(executor=ImmediateExecutor())(assembly, exploration)
                first = loop.harness.step('Improve the simulated format task', loop.action_space)
                directive = first.directive
                self.assertIsNotNone(directive)
                feedback = Feedback(directive.directive_id, ExecutionStatus.COMPLETED,
                    'Wrong format in simulated experiment', True, quality_metrics={'accuracy': 0})
                loop.harness.receive_feedback(directive, feedback)
                second = loop.harness.step(directive.objective, loop.action_space)
                self.assertTrue(backend.used_memory)
                records = loop.harness.state.decision_records
                self.assertEqual(records[first.decision_id]['research_attempt_id'],
                                 records[second.decision_id]['research_attempt_id'])
                self.assertEqual(records[second.decision_id]['operation'], 'prepare_data')
                latest = loop.harness.state.operation_results[-1]
                self.assertEqual(assembly.artifacts.get(latest.artifacts[0]).decode(),
                                 'candidate correction: one legal label')
                self.assertEqual(loop.harness.state.last_feedback, feedback)
                self.assertEqual(loop.harness.feedback_learning.state['attempted'], 2)
                self.assertEqual(loop.harness.memory.recall()[0].status, 'unverified')
                archived = json.loads(loop.harness.memory.recall()[0].content)
                self.assertEqual(archived['receipt_facts']['quality_metrics']['accuracy'], 0)
            finally:
                assembly.close()
