"""Decision dependencies and executor-confirmed provenance, independent of history order."""
from copy import deepcopy
from dataclasses import fields, replace
import json
import unittest
from unittest.mock import Mock

from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.candidate import Candidate, SystemVersion
from rsisys.loops.contracts.coordination import ExplorationProgress, ReviewDecision
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.jobs import JobRef
from rsisys.loops.contracts.lifecycle import ExecutionStatus as Status
from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef
from rsisys.loops.contracts.requests import ApplyCandidateSpec, EvaluationSpec
from rsisys.loops.decision_making.action_space import (
    DemoDecisionActionSpace, ModularDecisionActionSpace, OperationCall, OperationResult, OperationSpec,
)
from rsisys.loops.decision_making.action_space.access import ActionPermissions
from rsisys.loops.decision_making.action_space.capabilities import Capabilities
from rsisys.loops.decision_making.action_space.directives import (
    DirectiveDefinitions, apply_candidate_directive, builtin_directives,
)
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.memory import DecisionMemory
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.harness.state import DecisionContext, DecisionState
from rsisys.loops.decision_making.harness.tools import BoundDecisionTools
from rsisys.runtime.systems import ReferenceSystemActivator, SystemRegistry, system_adoption_capability
from rsisys.snapshot.codec import LINEAGE_DEFAULTS, decode, encode


MODEL = ModelRef('test', 'model', 'v1')


class DecisionLineageTests(unittest.TestCase):
    def setUp(self):
        self.policy, self.tools = Mock(), Mock()
        self.store = InMemoryExperienceStore()
        self.harness = DefaultDecisionHarness(self.policy, self.tools, BasicExperienceSystem(self.store),
                                             max_outstanding=4, memory_limit=1)
        self.space = DemoDecisionActionSpace()
        self.tools.execute.return_value = OperationResult('done', (), False)

    def step(self, decision):
        self.policy.decide.return_value = decision
        return self.harness.step('task', self.space)

    def invoke(self, ident, **kwargs):
        return Decision('invoke', operation=OperationCall('prepare_data', {'text': ident}), decision_id=ident, **kwargs)

    def producer(self, ident, *candidate_ids):
        directive = Directive(f'directive-{ident}', 'task', 'produce')
        decision = Decision('explore', directive=directive, decision_id=ident)
        self.step(decision)
        candidates = tuple(Candidate(c, 'v1', {}, ('actor',)) for c in candidate_ids)
        receipt = Feedback(directive.directive_id, Status.COMPLETED, 'produced', False, candidates=candidates)
        return decision, receipt

    def complete(self, decision, receipt):
        self.harness.receive_feedback(decision.directive, receipt)

    def test_ids_are_system_generated_and_reference_and_receipt_contracts_are_strict(self):
        self.assertNotEqual(Decision('wait').decision_id, Decision('wait').decision_id)
        for name in ('depends_on_decision_ids', 'source_decision_ids'):
            for refs in (['a'], ('',), ('a', 'a'), ('self',)):
                with self.subTest(name=name, refs=refs), self.assertRaises(ValueError):
                    Decision('wait', decision_id='self', **{name: refs})
        for factory in (lambda **kw: OperationResult('receipt', (), False, **kw),
                        lambda **kw: Feedback('d', Status.COMPLETED, 'receipt', False, **kw)):
            for ids in (['a'], ('',), ('a', 'a')):
                with self.subTest(ids=ids), self.assertRaises(ValueError):
                    factory(adopted_candidate_ids=ids)
        for status in (Status.FAILED, Status.CANCELLED, Status.PARTIAL):
            with self.subTest(status=status), self.assertRaises(ValueError):
                Feedback('d', status, 'not success', False, error='failure', adopted_candidate_ids=('a',))
            with self.assertRaises(ValueError):
                OperationResult('not success', (), False, status, 'failure', adopted_candidate_ids=('a',))

    def test_parallel_sources_join_only_after_completion_and_adopt_multiple_origins(self):
        first, first_receipt = self.producer('first', 'a', 'a2')
        second, second_receipt = self.producer('second', 'b')
        self.assertEqual(len(self.harness.pending_all()), 2)
        blocked = self.invoke('too-early', depends_on_decision_ids=('first', 'second'))
        with self.assertRaises(ValueError):
            self.step(blocked)
        self.tools.execute.assert_not_called()
        self.assertEqual(self.harness.state.decision_records['too-early']['status'], 'failed')
        history = self.harness.memory.history(kind='decision')[-1]
        self.assertEqual(history.details['status'], 'failed')
        self.assertTrue(history.details['validation_error'])
        self.complete(second, second_receipt)
        self.complete(first, first_receipt)
        adoption = self.invoke('join', depends_on_decision_ids=('first', 'second'), source_decision_ids=('first', 'second'))
        self.tools.execute.return_value = OperationResult('adopted', (), False, adopted_candidate_ids=('a', 'a2', 'b'))
        step = self.step(adoption)
        self.assertEqual(step.decision_id, 'join')
        ledger = self.harness.state.decision_records['join']
        self.assertEqual(ledger['adopted_decision_ids'], ('first', 'second'))
        self.assertEqual(ledger['unresolved_adopted_candidate_ids'], ())
        self.assertEqual([e['from'] for e in ledger['confirmed_edges'] if e['kind'] == 'adopted'], ['first', 'second'])
        receipt = self.harness.memory.history(kind='operation')[-1]
        self.assertEqual(receipt.details['decision_id'], 'join')
        self.assertEqual(receipt.details['adopted_candidate_ids'], ['a', 'a2', 'b'])
        self.assertEqual(self.step(adoption), step)
        with self.assertRaisesRegex(ValueError, 'different content'):
            self.step(replace(adoption, reason='Changed proposal using the same ID'))
        self.tools.execute.assert_called_once()
        self.assertEqual(decode(encode(self.harness.state)), self.harness.state)

    def test_failed_unknown_and_forward_dependencies_never_execute_but_sources_may_be_pending_or_failed(self):
        producer, receipt = self.producer('producer', 'candidate')
        self.step(Decision('wait', decision_id='cite-pending', source_decision_ids=('producer',)))
        failed = replace(receipt, status=Status.FAILED, error='producer failed')
        self.complete(producer, failed)
        self.assertEqual(self.harness.state.candidate_origins, {})
        self.step(Decision('wait', decision_id='cite-failed', source_decision_ids=('producer',)))
        for ref in ('producer', 'unknown', 'future'):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                self.step(self.invoke(f'blocked-{ref}', depends_on_decision_ids=(ref,)))
        with self.assertRaises(ValueError):
            self.step(self.invoke('bad-source', source_decision_ids=('unknown',)))
        self.tools.execute.assert_not_called()
        self.step(Decision('wait', decision_id='future'))
        self.step(self.invoke('new-proposal', depends_on_decision_ids=('future',)))
        self.tools.execute.assert_called_once()

    def test_evaluate_and_apply_feedback_preserve_the_producer_origin(self):
        self.space = DemoDecisionActionSpace(directives=DirectiveDefinitions((*builtin_directives(),
            apply_candidate_directive())))
        producer, receipt = self.producer('producer', 'candidate')
        self.complete(producer, receipt)
        candidate, = receipt.candidates
        for mode, payload in (('evaluate', EvaluationSpec(candidate, 'protocol')), ('apply_candidate', ApplyCandidateSpec('candidate'))):
            directive = Directive(f'{mode}-directive', 'task', mode, mode=mode, payload=payload)
            self.step(Decision('explore', directive=directive, decision_id=mode))
            confirmed = ('candidate',) if mode == 'apply_candidate' else ()
            feedback = Feedback(directive.directive_id, Status.COMPLETED, 'done', False,
                                candidates=(candidate,), adopted_candidate_ids=confirmed)
            self.harness.receive_feedback(directive, feedback)
            self.harness.receive_feedback(directive, deepcopy(feedback))
            self.assertEqual(self.harness.state.candidate_origins, {'candidate': 'producer'})
        self.assertEqual(self.harness.state.decision_records['apply_candidate']['adopted_decision_ids'], ('producer',))
        context = self.harness._context('task', self.space)
        self.assertEqual(context.candidate_origins, {'candidate': 'producer'})
        self.assertNotIn('reason', context.decision_graph[0])
        self.assertNotIn('confirmed_edges', context.decision_graph[0])
        context.candidate_origins.clear()
        self.assertEqual(self.harness.state.candidate_origins, {'candidate': 'producer'})

    def test_unknown_origin_keeps_real_success_and_candidate_receipt_without_inventing_edges(self):
        candidate = Candidate('external', 'v1', {'prompt': ArtifactRef('p', 'prompt', 'memory://p')}, ('actor',), 'base')
        registry = SystemRegistry(SystemVersion('base', {}, MODEL), ReferenceSystemActivator(MODEL))
        caps = Capabilities((system_adoption_capability(registry, lambda: [candidate]),))
        self.space = ModularDecisionActionSpace(caps.definitions, directive_modes=())
        self.harness.tools = BoundDecisionTools(caps.bindings)
        self.harness.state.candidates[candidate.candidate_id] = candidate
        decision = Decision('invoke', decision_id='real-adoption', operation=OperationCall('system_adopt',
            {'candidate_ids': ['external'], 'version': 'active', 'idempotency_key': 'once'}))
        result = self.step(decision)
        self.assertEqual(registry.current.version, 'active')
        self.assertEqual(result.operation_result.status, Status.COMPLETED)
        ledger = self.harness.state.decision_records[decision.decision_id]
        self.assertEqual(ledger['status'], 'completed')
        self.assertEqual(ledger['adopted_candidate_ids'], ('external',))
        self.assertEqual(ledger['unresolved_adopted_candidate_ids'], ('external',))
        self.assertEqual(ledger['adopted_decision_ids'], ())
        self.assertEqual(ledger['confirmed_edges'], ())
        self.assertEqual(self.step(decision), result)
        self.assertEqual(decode(encode(self.harness.state)), self.harness.state)

    def test_history_commit_response_loss_then_json_reload_retries_one_decision_and_executes_once(self):
        decision = self.invoke('stable')
        append = self.harness.memory.append
        def lost(entry):
            append(entry)
            raise OSError('committed response lost')
        self.harness.memory.append = lost
        with self.assertRaises(OSError):
            self.step(decision)
        self.tools.execute.assert_not_called()
        state = decode(encode(self.harness.state))
        loaded = DecisionMemory()
        loaded._load(json.loads(json.dumps(self.harness.memory.export())))
        self.harness.memory, self.harness.state = loaded, state
        self.step(decision)
        self.step(decision)
        self.tools.execute.assert_called_once()
        self.assertEqual(len(loaded.history(kind='decision')), 1)
        self.assertEqual(len(loaded.history(kind='operation')), 1)

    def test_operation_receipt_loss_and_json_reload_preserve_confirmed_edges_without_reexecution(self):
        producer, receipt = self.producer('producer', 'candidate')
        self.complete(producer, receipt)
        self.tools.execute.return_value = OperationResult('adopted', (), False, adopted_candidate_ids=('candidate',))
        append = self.harness.memory.append
        def lost(entry):
            append(entry)
            if entry.kind == 'operation':
                raise OSError('committed receipt response lost')
        self.harness.memory.append = lost
        decision = self.invoke('adopt', depends_on_decision_ids=('producer',))
        with self.assertRaises(OSError):
            self.step(decision)
        loaded = DecisionMemory()
        loaded._load(json.loads(json.dumps(self.harness.memory.export())))
        self.harness.state = decode(encode(self.harness.state))
        self.harness.memory = loaded
        result = self.harness.step('task', self.space)
        self.assertEqual(result.decision_id, 'adopt')
        self.assertEqual(self.harness.state.decision_records['adopt']['adopted_decision_ids'], ('producer',))
        self.tools.execute.assert_called_once()
        self.assertEqual(len(loaded.history(kind='operation')), 1)

    def test_async_resume_correlates_reused_jobref_to_current_invoke_or_wait_decision(self):
        job = JobRef('reused', 'backend')
        self.tools.execute.return_value = job
        receipt = OperationResult('done', (), False)
        for decision in (self.invoke('first'), self.invoke('second'), Decision('wait', job=job, decision_id='wait')):
            self.assertEqual(self.step(decision).decision_id, decision.decision_id)
            self.harness.resume(job, receipt)
            self.assertIsNone(self.harness.state.pending_job)
            self.assertEqual(self.harness.state.decision_records[decision.decision_id]['status'], 'completed')
        self.assertEqual(len(self.harness.state.operation_receipts), 3)
        self.assertEqual(self.tools.execute.call_count, 2)
        with self.assertRaises(ValueError):
            self.harness.resume(job, receipt)
        self.assertEqual(decode(encode(self.harness.state)), self.harness.state)

    def test_review_write_operation_uses_the_same_lineage_and_idempotent_receipt_path(self):
        producer, receipt = self.producer('producer', 'candidate')
        self.complete(producer, receipt)
        active, _ = self.producer('running')
        operation = self.invoke('review-adopt', depends_on_decision_ids=('producer',))
        self.space.permissions = ActionPermissions(review_operations='all')
        self.policy.review = Mock(side_effect=[operation, ReviewDecision()])
        self.tools.execute.return_value = OperationResult('adopted', (), False, adopted_candidate_ids=('candidate',))
        progress = ExplorationProgress(active.directive.directive_id, 'running', active.directive)
        append = self.harness.memory.append
        def lost(entry):
            append(entry)
            if entry.kind == 'operation':
                raise OSError('lost review receipt')
        self.harness.memory.append = lost
        with self.assertRaises(OSError):
            self.harness.review_with_tools('task', progress, self.space)
        self.harness.memory.append = append
        self.assertEqual(self.harness.review_with_tools('task', progress, self.space), ReviewDecision())
        self.tools.execute.assert_called_once()
        self.assertEqual(self.harness.state.decision_records['review-adopt']['adopted_decision_ids'], ('producer',))
        record, = self.harness.memory.history(kind='operation')
        self.assertEqual(record.details['decision_id'], 'review-adopt')
        self.assertEqual(record.details['event'], 'review_operation')

    def test_model_schema_parser_accept_refs_but_never_model_identity_or_confirmation(self):
        known = ({'decision_id': 'ok', 'status': 'completed'}, {'decision_id': 'pending', 'status': 'pending'})
        context = DecisionContext('task', (), self.space.describe(), (), decision_graph=known)
        backend = Mock()
        policy = ModelDecisionPolicy(backend, MODEL, structured_output=False)
        payload = {'kind': 'invoke', 'operation': 'prepare_data', 'arguments': {'text': 'data'},
                   'depends_on_decision_ids': ['ok'], 'source_decision_ids': ['pending']}
        backend.generate.return_value = GenerationResponse(json.dumps(payload), MODEL, False)
        decision = policy.decide(context)
        self.assertEqual(decision.depends_on_decision_ids, ('ok',))
        self.assertEqual(decision.source_decision_ids, ('pending',))
        for extra in ({'decision_id': 'fake'}, {'adopted_candidate_ids': ['c']}, {'adopted_decision_ids': ['ok']},
                      {'confirmed_edges': []}, {'depends_on_decision_ids': ['pending']},
                      {'source_decision_ids': ['future']}, {'source_decision_ids': ['ok', 'ok']}):
            backend.generate.return_value = GenerationResponse(json.dumps({**payload, **extra}), MODEL, False)
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                policy.decide(context)
        for branch in policy.decision_schema(context)['oneOf']:
            self.assertFalse(branch['additionalProperties'])
            self.assertTrue(branch['properties']['depends_on_decision_ids']['uniqueItems'])
            self.assertNotIn('adopted_candidate_ids', branch['properties'])

    def test_review_async_submission_stays_pending_until_correlated_resume(self):
        active, _ = self.producer('active')
        job = JobRef('review-job', 'backend')
        self.space.permissions = ActionPermissions(review_operations='all')
        self.policy.review = Mock(side_effect=[self.invoke('async-review'), ReviewDecision()])
        self.tools.execute.return_value = job
        progress = ExplorationProgress(active.directive.directive_id, 'running', active.directive)
        self.harness.review_with_tools('task', progress, self.space)
        self.assertEqual(self.harness.state.decision_records['async-review']['status'], 'pending')
        self.assertEqual(self.harness.state.decision_records['async-review']['review_job'], job)
        self.assertIsNone(self.harness.state.pending_job)
        self.assertIsNone(self.harness.state.pending_operation_decision_id)
        self.assertEqual(self.harness.pending(), active.directive)
        calls = self.policy.decide.call_count
        self.step(Decision('wait', decision_id='independent-turn'))
        self.assertEqual(self.policy.decide.call_count, calls + 1)
        with self.assertRaises(ValueError):
            self.step(self.invoke('blocked-review-job', depends_on_decision_ids=('async-review',)))
        self.tools.execute.assert_called_once()
        # Even the same backend job identity can coexist with the main wait slot.
        self.step(self.invoke('main-async'))
        self.harness.resume(job, OperationResult('review done', (), False), decision_id='async-review')
        self.assertEqual(self.harness.state.decision_records['async-review']['status'], 'completed')
        self.assertEqual(self.harness.state.pending_job, job)
        self.assertEqual(self.harness.state.pending_operation_decision_id, 'main-async')
        self.assertEqual(self.harness.state.decision_records['main-async']['status'], 'pending')
        self.harness.resume(job, OperationResult('main done', (), False))
        self.assertIsNone(self.harness.state.pending_job)
        self.assertEqual(len(self.harness.memory.history(kind='operation')), 3)
        self.assertEqual(self.tools.execute.call_count, 2)

    def test_independent_review_completion_retries_json_restore_without_touching_main_slot(self):
        active, _ = self.producer('active')
        job = JobRef('review-job', 'backend')
        self.space.permissions = ActionPermissions(review_operations='all')
        self.policy.review = Mock(side_effect=[self.invoke('review-job'), ReviewDecision()])
        self.tools.execute.return_value = job
        self.harness.review_with_tools('task', ExplorationProgress(active.directive.directive_id, 'running', active.directive), self.space)
        self.assertFalse(self.harness.pending_records())
        append = self.harness.memory.append
        def lost(entry):
            append(entry)
            raise OSError('committed review completion response lost')
        self.harness.memory.append = lost
        receipt = OperationResult('done', (), False)
        with self.assertRaises(OSError):
            self.harness.resume(job, receipt)
        self.assertTrue(self.harness.pending_records())
        self.assertIsNone(self.harness.state.pending_job)
        loaded = DecisionMemory()
        loaded._load(json.loads(json.dumps(self.harness.memory.export())))
        self.harness.memory = loaded
        self.harness.state = decode(encode(self.harness.state))
        self.harness.flush_pending_records()
        self.assertFalse(self.harness.pending_records())
        self.assertEqual(self.harness.state.decision_records['review-job']['status'], 'completed')
        self.assertIsNone(self.harness.state.pending_job)
        self.assertEqual(self.harness.pending(), active.directive)
        self.assertEqual(len(loaded.history(kind='operation')), 2)
        self.tools.execute.assert_called_once()
        with self.assertRaises(ValueError):
            self.harness.resume(job, receipt, decision_id='review-job')

    def test_snapshot_additive_defaults_remain_strict_and_lineage_metadata_is_validated(self):
        for value in (DecisionState(), Decision('wait'), DecisionContext('task', (), (), ()),
                      OperationResult('done', (), False), Feedback('d', Status.COMPLETED, 'done', False)):
            encoded = encode(value)
            for key in LINEAGE_DEFAULTS[encoded['$type']]:
                encoded['fields'].pop(key, None)
            restored = decode(encoded)
            self.assertEqual(type(restored), type(value))
            encoded['fields']['unknown_field'] = None
            with self.assertRaises(ValueError):
                decode(encoded)
        producer, receipt = self.producer('producer', 'candidate')
        self.complete(producer, receipt)
        corrupted = deepcopy(self.harness.state)
        corrupted.candidate_origins['candidate'] = 'unknown'
        with self.assertRaises(ValueError):
            decode(encode(corrupted))
        corrupted = deepcopy(self.harness.state)
        corrupted.decision_records['producer']['confirmed_edges'] = ({'from': 'unknown', 'to': 'producer',
                                                                    'kind': 'adopted', 'confirmed': True},)
        with self.assertRaises(ValueError):
            decode(encode(corrupted))
        self.assertEqual([f.name for f in fields(Decision)][-3:],
                         ['decision_id', 'depends_on_decision_ids', 'source_decision_ids'])


if __name__ == '__main__':
    unittest.main()
