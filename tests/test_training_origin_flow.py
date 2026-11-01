from dataclasses import asdict, replace
from unittest import TestCase
from unittest.mock import Mock

from rsisys.loops.contracts.backend import BackendCapabilities
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.contracts.training import TrainingOrigin, ResumeMode, TrainingResult
from rsisys.loops.decision_making.action_space.directives import training_directive
from rsisys.loops.exploration.harness.learner import ActorTrainingRequest, LearningExecution
from rsisys.loops.exploration.harness.learner.command import TrainingCommandHandler
from rsisys.loops.exploration.harness.commands import CommandState
from rsisys.loops.exploration.harness.state import ExplorationState
from rsisys.loops.exploration.harness.memory import ExplorationMemory
from rsisys.runtime.jobs import JobBook, MockJobQueue, InvalidJobResult
from rsisys.snapshot.codec import encode, decode


class TrainingOriginFlowTests(TestCase):
    def setUp(self):
        self.base = ModelRef('native', 'base', 'original')
        self.origin = TrainingOrigin('system-best', self.base)

    def test_model_arguments_to_handler_request_preserve_origin(self):
        definition = training_directive(BackendCapabilities(methods=('sft',)))
        args = {'method': 'sft', 'recipe_ref': 'native.json', 'origin': asdict(self.origin)}
        args['origin']['resume_mode'] = 'restart'
        payload = definition.parser(args, ())
        directive = Directive('id', 'medical', 'train', mode='train', payload=payload)
        harness = Mock()
        harness.policy.model = ModelRef('serving', 'incumbent', 'best-weights')
        state = CommandState(directive)
        TrainingCommandHandler().prepare(harness, state)
        self.assertEqual(state.request.base, self.base)
        self.assertEqual(state.request.origin, self.origin)
        self.assertEqual(decode(encode(state.request)), state.request)
        self.assertEqual(definition.spec.input_schema['properties']['method']['enum'], ['sft'])
        args['method'] = 'rl'
        with self.assertRaises(NotImplementedError): definition.parser(args, ())

    def test_initial_weights_must_match_request_base(self):
        with self.assertRaisesRegex(ValueError, 'initial_model'):
            ActorTrainingRequest(replace(self.base, revision='different'), (), 'recipe', origin=self.origin)

    def test_unsupported_training_rejected_before_submit_and_result_parent_validated(self):
        class Backend:
            def __init__(self, base): self.queue = MockJobQueue('native'); self.base = base; self.calls = 0
            def capabilities(self): return BackendCapabilities(methods=('sft',))
            def submit(self, request, *, idempotency_key):
                self.calls += 1
                return self.queue.submit(request, lambda: TrainingResult(self.base,
                    replace(self.base, revision='new'),
                    Candidate('candidate', self.base.revision, {}, ('actor',), 'wrong-parent')),
                    idempotency_key=idempotency_key)
            def poll(self, job): return self.queue.poll(job)
            def cancel(self, job): return self.queue.cancel(job)
        backend = Backend(self.base)
        learning = LearningExecution(Mock(), ExplorationMemory(), Mock(), ExplorationState(), JobBook(backend))
        request = ActorTrainingRequest(self.base, (), 'recipe', method='rl', origin=self.origin)
        with self.assertRaises(NotImplementedError): learning.submit(request, idempotency_key='reject')
        self.assertEqual(backend.calls, 0)
        job = learning.submit(replace(request, method='sft'), idempotency_key='accept')
        with self.assertRaises(InvalidJobResult):
            for _ in range(3): learning.poll(job)
