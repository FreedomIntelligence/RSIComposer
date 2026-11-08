"""Explicit research cycles preserve evidence and isolate external running hints."""

from copy import deepcopy
from dataclasses import replace
import json
import unittest
from unittest.mock import Mock, patch

from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops._memory import Memory
from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.coordination import ExecutionView, ExplorationProgress, ReviewDecision
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.jobs import JobRef
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef, TokenUsage
from rsisys.loops.decision_making.action_space import (
    DemoDecisionActionSpace, ModularDecisionActionSpace, OperationCall, OperationResult, OperationSpec,
)
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.harness.tools import BoundDecisionTools
from rsisys.loops.decision_making.harness.state import (
    CandidateEvaluation, DecisionContext, DecisionState,
    _PendingOperationRecord, _PendingReviewRecord, _ReviewSession,
)
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.snapshot.codec import decode, encode


MODEL = ModelRef('test', 'decision', 'v2')
CONTROL = {'research_round': 2, 'mode': 'continuous', 'budget': {'remaining': 10}}


class ResearchCycleTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryExperienceStore()
        self.policy = Mock(model=MODEL, decide=Mock(return_value=Decision('finish')))
        self.harness = DefaultDecisionHarness(self.policy, Mock(), BasicExperienceSystem(self.store))
        self.loop = DecisionMakingLoop(self.harness, DemoDecisionActionSpace())

    def dispatch(self):
        directive = Directive('experiment', 'original objective', 'test')
        self.policy.decide.return_value = Decision('explore', directive=directive)
        self.assertEqual(self.loop.step(directive.objective).directive, directive)
        return directive

    def finish(self):
        self.policy.decide.return_value = Decision('finish')
        self.assertEqual(self.loop.step('original objective').kind, 'finished')

    def test_restart_preserves_all_evidence_models_and_counters(self):
        directive = self.dispatch()
        candidate = Candidate('candidate', 'v1', {
            'model': ArtifactRef('weights', 'model', 'memory://weights'),
        }, ('actor',))
        feedback = Feedback(directive.directive_id, ExecutionStatus.COMPLETED, 'trained', True,
                            candidates=(candidate,))
        self.loop.receive_feedback(directive, feedback)
        self.harness.memory.remember(Memory('lesson', 'semantic', 'Keep the evidence'))
        state = self.harness.state
        state.execution = ExecutionView(ModelRef('test', 'actor', 'v3'))
        state.candidate_bases[candidate.candidate_id] = ModelRef('test', 'actor', 'v1')
        state.candidate_evaluations[candidate.candidate_id] = CandidateEvaluation(
            candidate.candidate_id, directive.directive_id, 'protocol', (),
            ExecutionStatus.COMPLETED, {'score': 0.9}, True)
        state.operation_results.append(OperationResult('prepared', (), True))
        job = JobRef('learned', 'test')
        state.recorded_training.add(job)
        state.applied_training.add(job)
        state.training_record_ids[job] = 'learning-receipt'
        state.measured_input_tokens, state.measured_output_tokens = 100, 20
        self.finish()
        before, memory, records = deepcopy(state), self.harness.memory.export(), self.store.records()
        self.loop.restart_research(CONTROL)
        self.assertIs(self.harness.state, state)
        self.assertEqual(state, replace(before, finished=False, run_control=CONTROL,
                                        research_cycles={2: CONTROL}))
        self.assertEqual(self.harness.memory.export(), memory)
        self.assertEqual(self.store.records(), records)
        self.assertEqual(self.policy.model, MODEL)
        context = self.harness._context('original objective', self.loop.action_space)
        self.assertEqual(context.candidates, (candidate,))
        self.assertEqual(context.last_feedback, feedback)
        self.assertEqual(context.decision_model, MODEL)
        self.assertEqual(context.run_control, CONTROL)
        self.finish()
        self.assertEqual(self.policy.decide.call_count, 3)
        self.assertEqual(self.policy.decide.call_args.args[0].objective, 'original objective')

    def test_restart_requires_finished_without_mutation(self):
        before = deepcopy(self.harness.state)
        with self.assertRaisesRegex(RuntimeError, 'finished'):
            self.loop.restart_research(CONTROL)
        self.assertEqual(self.harness.state, before)

    def test_each_pending_state_blocks_new_cycle_without_mutation(self):
        directive = Directive('pending', 'task', 'test')
        job = JobRef('pending', 'test')
        operation = OperationCall('prepare', {})
        record = _PendingOperationRecord('op', operation, OperationResult('ready', (), True), None, job)
        review = _PendingReviewRecord('review', 'task', 'pending', ReviewDecision(), {})
        session = _ReviewSession('task', 'pending', DecisionContext('task', (), (), ()), 1)
        pending = {
            'pending_directive_id': 'pending', 'pending_directive': directive,
            'outstanding': {'pending': directive}, 'feedback_record_id': 'receipt',
            'feedback_record_ids': {'pending': 'receipt'}, 'pending_job': job,
            'pending_operation': operation, 'pending_operation_record': record,
            'pending_review_record': review, 'review_session': session,
            'review_sessions': {'pending': session}, 'review_records': {'pending': review},
        }
        for field, value in pending.items():
            with self.subTest(field=field):
                self.harness.state = replace(DecisionState(finished=True), **{field: value})
                before = deepcopy(self.harness.state)
                with self.assertRaisesRegex(RuntimeError, 'pending'):
                    self.loop.restart_research(CONTROL)
                self.assertEqual(self.harness.state, before)

    def test_parallel_finished_cycle_must_drain_feedback_before_restart(self):
        self.harness.max_outstanding = 2
        directive = self.dispatch()
        self.finish()
        with self.assertRaisesRegex(RuntimeError, 'pending'):
            self.loop.restart_research(CONTROL)
        self.loop.receive_feedback(directive, Feedback(directive.directive_id,
                                   ExecutionStatus.COMPLETED, 'done', True))
        self.loop.restart_research(CONTROL)
        self.assertFalse(self.harness.state.finished)
        self.assertEqual(self.harness.state.completed, 1)

    def test_retries_are_noops_while_active_finished_and_in_later_rounds(self):
        self.finish()
        self.loop.restart_research(CONTROL)
        # Current hints may evolve without changing the original idempotency receipt.
        self.loop.update_run_control({'remaining': 5})
        for phase in ('active', 'in_flight', 'finished', 'later_round'):
            with self.subTest(phase=phase):
                if phase == 'in_flight':
                    directive = self.dispatch()
                elif phase == 'finished':
                    self.loop.receive_feedback(directive, Feedback(directive.directive_id,
                                               ExecutionStatus.COMPLETED, 'done', True))
                    self.finish()
                elif phase == 'later_round':
                    self.loop.restart_research({'research_round': 3})
                before = deepcopy(self.harness.state)
                self.loop.restart_research(deepcopy(CONTROL))
                self.assertEqual(self.harness.state, before)
                with self.assertRaisesRegex(ValueError, 'context differs'):
                    self.loop.restart_research({**CONTROL, 'mode': 'different'})
                self.assertEqual(self.harness.state, before)

    def test_context_requires_positive_integer_round_and_string_keys(self):
        self.finish()
        for context in ({}, {'research_round': None}, {'research_round': True},
                        {'research_round': 0}, {'research_round': -1},
                        {'research_round': 1.0}, {'research_round': '2'}):
            with self.subTest(context=context):
                before = deepcopy(self.harness.state)
                with self.assertRaisesRegex(ValueError, 'positive integer'):
                    self.loop.restart_research(context)
                self.assertEqual(self.harness.state, before)
        for method in (self.loop.restart_research, self.loop.update_run_control):
            for context in (None, [], {1: 'invalid key'}):
                with self.subTest(method=method.__name__, context=context):
                    before = deepcopy(self.harness.state)
                    with self.assertRaisesRegex(TypeError, 'dict with string keys'):
                        method(context)
                    self.assertEqual(self.harness.state, before)

    def test_update_replaces_only_hints_and_copies_nested_values(self):
        self.finish()
        control = deepcopy(CONTROL)
        self.loop.restart_research(control)
        control['budget']['remaining'] = -1
        self.assertEqual(self.harness.state.run_control, CONTROL)
        self.assertEqual(self.harness.state.research_cycles, {2: CONTROL})
        before, memory = deepcopy(self.harness.state), self.harness.memory.export()
        hints = {'objective': 'hint only', 'history': ['hint only'], 'budget': [4]}
        self.loop.update_run_control(hints)
        self.assertEqual(self.harness.state, replace(before, run_control=hints))
        hints['budget'].append(0)
        context = self.harness._context('original objective')
        self.assertEqual(context.objective, 'original objective')
        self.assertEqual(context.history, self.harness.memory.history())
        self.assertEqual(context.run_control['budget'], [4])
        context.run_control['budget'].append(99)
        self.assertEqual(self.harness.state.run_control['budget'], [4])
        self.assertEqual(self.harness.memory.export(), memory)
        self.loop.update_run_control({})
        self.assertEqual(self.harness.state, replace(before, run_control={}))

    def test_default_fields_are_empty_independent_and_codec_round_trips(self):
        first, second = DecisionState(), DecisionState()
        first.run_control['mode'] = 'continuous'
        first.research_cycles[2] = deepcopy(CONTROL)
        self.assertEqual(second.run_control, {})
        self.assertEqual(second.research_cycles, {})
        first_context, second_context = (DecisionContext('task', (), (), ()) for _ in range(2))
        first_context.run_control['mode'] = 'continuous'
        self.assertEqual(second_context.run_control, {})
        self.finish()
        self.loop.restart_research(CONTROL)
        for state in (DecisionState(), self.harness.state):
            self.assertEqual(decode(json.loads(json.dumps(encode(state)))), state)
        self.harness.state = decode(json.loads(json.dumps(encode(self.harness.state))))
        self.finish()
        before = deepcopy(self.harness.state)
        self.loop.restart_research(CONTROL)
        self.assertEqual(self.harness.state, before)

    def test_model_decision_and_resumed_review_prompts_include_current_run_control(self):
        backend = Mock(generate=Mock(return_value=GenerationResponse(
            '{"kind":"finish"}', MODEL, True, TokenUsage(1, 1))))
        self.harness.policy = ModelDecisionPolicy(backend, MODEL)
        self.finish()
        self.assertEqual(json.loads(backend.generate.call_args.args[0].prompt)['run_control'], {})
        self.loop.restart_research(CONTROL)
        self.finish()
        self.assertEqual(json.loads(backend.generate.call_args.args[0].prompt)['run_control'], CONTROL)
        self.loop.restart_research({'research_round': 3})
        backend.generate.return_value = GenerationResponse(
            '{"kind":"explore","strategy":"test","mode":"execute","arguments":{}}', MODEL, True)
        directive = self.loop.step('original objective').directive
        progress = ExplorationProgress(directive.directive_id, 'running', directive)
        # A review resumed after a recording failure must also see updated constraints.
        self.harness.state.review_session = _ReviewSession(
            'original objective', directive.directive_id, self.harness._context('original objective'), 1)
        self.loop.update_run_control({'remaining_seconds': 2})
        backend.generate.return_value = GenerationResponse('{"action":"continue"}', MODEL, True)
        self.loop.review('original objective', progress)
        prompt = json.loads(backend.generate.call_args.args[0].prompt)
        self.assertEqual(prompt['run_control'], {'remaining_seconds': 2})
        self.assertEqual(prompt['objective'], 'original objective')

    def test_custom_harness_methods_are_delegated_or_explicitly_unsupported(self):
        for harness in (object(), type('NoCycles', (), {
                'restart_research': None, 'update_run_control': None})()):
            loop = DecisionMakingLoop(harness, self.loop.action_space)
            for name in ('restart_research', 'update_run_control'):
                with self.subTest(method=name), self.assertRaisesRegex(NotImplementedError, name):
                    getattr(loop, name)(CONTROL)
        harness = Mock(spec=['restart_research', 'update_run_control'])
        loop = DecisionMakingLoop(harness, self.loop.action_space)
        for name in ('restart_research', 'update_run_control'):
            getattr(loop, name)(CONTROL)
            getattr(harness, name).assert_called_once_with(CONTROL)
            self.assertIs(getattr(harness, name).call_args.args[0], CONTROL)

    def test_pending_research_job_reports_external_resume_without_side_effects(self):
        self.assertIsNone(self.loop.pending_research_job())
        job = JobRef('external-tool', 'test')
        self.policy.decide.return_value = Decision('wait', job=job)
        self.assertEqual(self.loop.step('task').job, job)
        before = deepcopy(self.harness.state)
        self.assertEqual(self.loop.pending_research_job(), job)
        self.assertEqual(self.harness.state, before)
        self.harness.tools.assert_not_called()
        self.harness.resume(job, OperationResult('completed externally', (), True))
        self.assertIsNone(self.loop.pending_research_job())
        for harness in (object(), type('NoPendingJob', (), {'pending_research_job': None})()):
            self.assertIsNone(DecisionMakingLoop(harness, self.loop.action_space).pending_research_job())
        harness = Mock(spec=['pending_research_job'])
        harness.pending_research_job.return_value = job
        self.assertEqual(DecisionMakingLoop(harness, self.loop.action_space).pending_research_job(), job)
        harness.pending_research_job.assert_called_once_with()


class PendingRecordFlushTests(unittest.TestCase):
    def setUp(self):
        class Inspect:
            spec = OperationSpec('inspect', 'Inspect evidence', {'type': 'object'}, read_only=True)

            def validate(self, arguments):
                return OperationCall('inspect', arguments)

        self.result = OperationResult('observed once', (), True)
        self.execute = Mock(return_value=self.result)
        self.policy = Mock(model=MODEL, decide=Mock(return_value=Decision(
            'invoke', operation=OperationCall('inspect', {}))))
        self.store = InMemoryExperienceStore()
        self.harness = DefaultDecisionHarness(self.policy, BoundDecisionTools({'inspect': self.execute}),
                                               BasicExperienceSystem(self.store))
        self.space = ModularDecisionActionSpace((Inspect(),))

    def fail_writes(self, target, after_commit):
        owner, method = ((self.harness.memory, 'append') if target == 'memory'
                         else (self.harness.experience, 'record'))
        original = getattr(owner, method)

        def failed(record):
            if record.kind != 'decision' or record.details.get('event') == 'review':
                if after_commit:
                    original(record)
                raise OSError('record response lost')
            original(record)

        return patch.object(owner, method, side_effect=failed)

    def forbid_execution(self):
        self.policy.decide.side_effect = AssertionError('flush must not decide')
        self.policy.review.side_effect = AssertionError('flush must not review')
        self.execute.side_effect = AssertionError('flush must not execute tools')

    def test_flush_operation_record_only_retries_writes_after_execution(self):
        for target in ('memory', 'experience'):
            for after_commit in (False, True):
                with self.subTest(target=target, after_commit=after_commit):
                    self.setUp()
                    with self.fail_writes(target, after_commit):
                        with self.assertRaisesRegex(OSError, 'record response lost'):
                            self.harness.step('task', self.space)
                        pending = deepcopy(self.harness.state.pending_operation_record)
                        self.assertTrue(self.harness.pending_records())
                        self.forbid_execution()
                        # A further failed write preserves exactly the same retry receipt.
                        with self.assertRaises(OSError):
                            self.harness.flush_pending_records()
                        self.assertEqual(self.harness.state.pending_operation_record, pending)
                    self.harness.flush_pending_records()
                    self.harness.flush_pending_records()
                    self.assertFalse(self.harness.pending_records())
                    self.assertEqual(self.harness.state.operation_results, [self.result])
                    self.assertEqual(len(self.store.records()), 1)
                    self.assertEqual(self.store.records()[0].record_id, pending.record_id)
                    self.assertEqual(len(self.harness.memory.history(kind='operation')), 1)
                    self.policy.decide.assert_called_once()
                    self.policy.review.assert_not_called()
                    self.execute.assert_called_once_with({})

    def prepare_review(self, ident):
        directive = Directive(ident, 'task', 'test')
        self.harness.state.outstanding[ident] = directive
        self.harness.state.pending_directive = directive
        self.harness.state.pending_directive_id = ident
        return ExplorationProgress(ident, 'running', directive)

    def test_flush_final_reviews_records_without_applying_controls(self):
        for parallel in (False, True):
            for target in ('memory', 'experience'):
                with self.subTest(parallel=parallel, target=target):
                    self.setUp()
                    self.harness.max_outstanding = 2 if parallel else 1
                    ids = ('first', 'second') if parallel else ('first',)
                    self.policy.review.return_value = ReviewDecision('cancel', reason='record only')
                    for ident in ids:
                        progress = self.prepare_review(ident)
                        with self.fail_writes(target, True), self.assertRaises(OSError):
                            self.harness.review_with_tools('task', progress, self.space)
                    outstanding = deepcopy(self.harness.state.outstanding)
                    self.assertTrue(self.harness.pending_records())
                    self.forbid_execution()
                    self.harness.flush_pending_records()
                    self.harness.flush_pending_records()
                    self.assertFalse(self.harness.pending_records())
                    self.assertEqual(self.harness.state.outstanding, outstanding)
                    self.assertEqual(len(self.store.records()), len(ids))
                    self.assertEqual(len(self.harness.memory.history(kind='decision')), len(ids))
                    self.assertEqual(self.policy.review.call_count, len(ids))
                    self.policy.decide.assert_not_called()
                    self.execute.assert_not_called()

    def test_flush_review_queries_keeps_unfinished_sessions_pending(self):
        for parallel in (False, True):
            for after_commit in (False, True):
                with self.subTest(parallel=parallel, after_commit=after_commit):
                    self.setUp()
                    self.harness.max_outstanding = 2 if parallel else 1
                    ids = ('first', 'second') if parallel else ('first',)
                    self.policy.review.return_value = Decision('invoke', operation=OperationCall('inspect', {}))
                    for ident in ids:
                        progress = self.prepare_review(ident)
                        with self.fail_writes('memory', after_commit), self.assertRaises(OSError):
                            self.harness.review_with_tools('task', progress, self.space)
                    sessions = (tuple(self.harness.state.review_sessions.values()) if parallel
                                else (self.harness.state.review_session,))
                    record_ids = [session.pending_record.record_id for session in sessions]
                    before = deepcopy(self.harness.state)
                    self.assertTrue(self.harness.pending_records())
                    self.assertEqual(self.harness.state, before)
                    self.forbid_execution()
                    self.harness.flush_pending_records()
                    self.harness.flush_pending_records()
                    self.assertTrue(self.harness.pending_records())
                    for session, record_id in zip(sessions, record_ids):
                        self.assertIsNone(session.pending_record)
                        self.assertEqual(session.queries_used, 1)
                        self.assertEqual(session.query_refs, [f'history:decision_maker:{record_id}'])
                        self.assertEqual(session.context.operation_results, (self.result,))
                    self.assertEqual(len(self.harness.memory.history(kind='operation')), len(ids))
                    self.assertEqual(self.store.records(), ())
                    self.assertEqual(self.execute.call_count, len(ids))
                    self.assertEqual(self.policy.review.call_count, len(ids))
                    self.policy.decide.assert_not_called()

    def test_no_records_does_not_resume_jobs_or_finish_review_sessions(self):
        self.assertFalse(self.harness.pending_records())
        self.harness.state.pending_job = JobRef('external', 'test')
        before = deepcopy(self.harness.state)
        self.forbid_execution()
        self.harness.flush_pending_records()
        self.assertFalse(self.harness.pending_records())
        self.assertEqual(self.harness.state, before)
        # Model failure can leave a session with no result or tool receipt at all.
        self.harness.state.review_session = _ReviewSession('task', 'active',
                                                          DecisionContext('task', (), (), ()), 1)
        before = deepcopy(self.harness.state)
        self.harness.flush_pending_records()
        self.assertTrue(self.harness.pending_records())
        self.assertEqual(self.harness.state, before)
        self.policy.decide.assert_not_called()
        self.policy.review.assert_not_called()
        self.execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
