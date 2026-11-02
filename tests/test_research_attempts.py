"""Attempt membership is explicit, executor-independent and replay-stable."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import unittest
from unittest.mock import Mock
from uuid import UUID

from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.coordination import ExplorationProgress, ReviewDecision
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.lifecycle import ExecutionStatus as Status
from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef
from rsisys.loops.decision_making.action_space import DemoDecisionActionSpace, OperationCall, OperationResult
from rsisys.loops.decision_making.action_space.access import ActionPermissions
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.memory import DecisionMemory
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.harness.state import DecisionContext
from rsisys.snapshot.codec import decode, encode


MODEL = ModelRef('fixture', 'researcher', 'v1')


class ResearchAttemptTests(unittest.TestCase):
    def setUp(self):
        self.policy, self.tools = Mock(), Mock()
        self.tools.execute.return_value = OperationResult('done', (), False)
        self.harness = DefaultDecisionHarness(self.policy, self.tools,
            BasicExperienceSystem(InMemoryExperienceStore()), max_outstanding=4, memory_limit=1)
        self.space = DemoDecisionActionSpace()

    def step(self, decision):
        self.policy.decide.return_value = decision
        return self.harness.step('task', self.space)

    def invoke(self, ident, **fields):
        return Decision('invoke', operation=OperationCall('prepare_data', {'text': ident}),
                        decision_id=ident, **fields)

    def membership(self, ident):
        row = self.harness.state.decision_records[ident]
        return {key: row[key] for key in ('research_attempt_id', 'research_attempt_title')}

    def restore(self):
        self.harness.state = decode(json.loads(json.dumps(encode(self.harness.state))))
        memory = DecisionMemory()
        memory._load(json.loads(json.dumps(self.harness.memory.export())))
        self.harness.memory = memory

    def test_new_and_continued_attempts_resolve_canonical_metadata_without_rewriting_proposals(self):
        original = self.invoke('new', research_attempt_title='Compare retrieval strategies')
        self.step(original)
        membership = self.membership('new')
        attempt_id = membership['research_attempt_id']
        self.assertEqual(str(UUID(attempt_id)), attempt_id)
        self.assertEqual(original.research_attempt_id, None)
        self.assertEqual(self.harness.state.decision_records['new']['proposal'], asdict(original))
        for ident, title in (('continued', ''), ('continued-with-title', original.research_attempt_title)):
            request = self.invoke(ident, research_attempt_id=attempt_id, research_attempt_title=title)
            self.step(request)
            self.assertEqual(self.membership(ident), membership)
            self.assertEqual(self.harness.state.decision_records[ident]['proposal'], asdict(request))
        for history in self.harness.memory.history():
            self.assertEqual({key: history.details[key] for key in membership}, membership)
        graph = self.harness._context('task', self.space).decision_graph
        self.assertEqual(len(graph), 3)
        self.assertTrue(all(row['research_attempt_id'] == attempt_id for row in graph))
        self.assertTrue(all(row['research_attempt_title'] == original.research_attempt_title for row in graph))
        graph[0]['research_attempt_title'] = 'mutable copy'
        self.assertEqual(self.membership('new'), membership)
        self.assertEqual(decode(encode(self.harness.state)), self.harness.state)

    def test_title_only_always_starts_a_distinct_attempt_and_omission_never_infers_membership(self):
        for ident in ('first', 'same-title'):
            self.step(self.invoke(ident, research_attempt_title='Identical title'))
        first, second = (self.membership(ident)['research_attempt_id'] for ident in ('first', 'same-title'))
        self.assertNotEqual(first, second)
        self.step(self.invoke('legacy', depends_on_decision_ids=('first', 'same-title'),
                              source_decision_ids=('first',)))
        self.assertEqual(self.membership('legacy'), {'research_attempt_id': None, 'research_attempt_title': ''})
        self.assertEqual(self.harness.state.decision_records['legacy']['depends_on_decision_ids'],
                         ('first', 'same-title'))

    def test_parallel_attempts_with_multiple_results_join_and_adopt_without_losing_origins(self):
        for ident in ('a', 'b'):
            self.step(Decision('explore', decision_id=ident, directive=Directive(f'd-{ident}', 'task', ident),
                               research_attempt_title=f'Approach {ident}'))
        self.assertEqual(len(self.harness.pending_all()), 2)
        for ident, candidates in (('b', ('b1',)), ('a', ('a1', 'a2'))):
            directive = self.harness.state.outstanding[f'd-{ident}']
            self.harness.receive_feedback(directive, Feedback(directive.directive_id, Status.COMPLETED,
                'produced', False, candidates=tuple(Candidate(c, 'v1', {}, ('actor',)) for c in candidates)))
        self.tools.execute.return_value = OperationResult('joined', (), False,
                                                          adopted_candidate_ids=('a1', 'a2', 'b1'))
        self.step(self.invoke('join', depends_on_decision_ids=('a', 'b'), research_attempt_title='Combine results'))
        row = self.harness.state.decision_records['join']
        self.assertEqual(row['adopted_decision_ids'], ('a', 'b'))
        self.assertEqual(self.harness.state.candidate_origins, {'b1': 'b', 'a1': 'a', 'a2': 'a'})
        self.assertEqual(len({self.membership(ident)['research_attempt_id'] for ident in ('a', 'b', 'join')}), 3)
        self.assertEqual([(edge['kind'], edge['from']) for edge in row['confirmed_edges']],
                         [('depends_on', 'a'), ('depends_on', 'b'), ('adopted', 'a'), ('adopted', 'b')])
        for history in self.harness.memory.history():
            ident = history.details['decision_id']
            self.assertEqual({key: history.details[key] for key in self.membership(ident)}, self.membership(ident))
        self.restore()
        # Even actual adoption is not an implicit attempt declaration.
        self.step(self.invoke('ungrouped-adopt'))
        self.assertIsNone(self.membership('ungrouped-adopt')['research_attempt_id'])
        self.assertEqual(self.harness.state.decision_records['ungrouped-adopt']['adopted_decision_ids'], ('a', 'b'))

    def test_continuation_may_return_to_an_attempt_across_other_attempts_and_source_references(self):
        self.step(self.invoke('a1', research_attempt_title='A'))
        attempt_a = self.membership('a1')['research_attempt_id']
        self.step(self.invoke('b1', research_attempt_title='B', depends_on_decision_ids=('a1',)))
        self.step(self.invoke('a2', research_attempt_id=attempt_a, depends_on_decision_ids=('b1',),
                              source_decision_ids=('a1',)))
        self.assertEqual(self.membership('a2'), self.membership('a1'))
        self.assertEqual(self.harness.state.decision_records['a2']['confirmed_edges'], (
            {'from': 'b1', 'to': 'a2', 'kind': 'depends_on', 'confirmed': True},
            {'from': 'a1', 'to': 'a2', 'kind': 'source', 'confirmed': True}))
        self.restore()
        self.assertEqual(self.tools.execute.call_count, 3)

    def test_failed_validation_does_not_establish_an_origin_and_corrected_proposal_can_retry(self):
        invalid = self.invoke('invalid', research_attempt_title='Unaccepted attempt',
                              depends_on_decision_ids=('unknown',))
        with self.assertRaisesRegex(ValueError, 'dependency/reference'):
            self.step(invalid)
        self.assertEqual(self.membership('invalid'), {'research_attempt_id': None, 'research_attempt_title': ''})
        self.assertIsNone(self.harness.memory.history(kind='decision')[-1].details['research_attempt_id'])
        self.assertEqual(self.harness.memory.history(kind='decision')[-1].details['research_attempt_title'], '')
        self.tools.execute.assert_not_called()
        with self.assertRaises(ValueError):
            self.step(invalid)
        with self.assertRaisesRegex(ValueError, 'Unknown research_attempt_id'):
            self.step(self.invoke('fake-origin', research_attempt_id='invalid'))
        self.restore()
        self.step(replace(invalid, decision_id='corrected', depends_on_decision_ids=()))
        self.assertIsNotNone(self.membership('corrected')['research_attempt_id'])
        self.tools.execute.assert_called_once()

    def test_unknown_identity_and_rename_requests_fail_before_execution_and_record_no_membership(self):
        self.step(self.invoke('origin', research_attempt_title='Canonical title'))
        attempt_id = self.membership('origin')['research_attempt_id']
        for ident, fields, error in (
            ('unknown', {'research_attempt_id': 'invented', 'research_attempt_title': 'Supplied title'}, 'Unknown'),
            ('rename', {'research_attempt_id': attempt_id, 'research_attempt_title': 'Changed title'}, 'rename'),
        ):
            with self.subTest(ident=ident), self.assertRaisesRegex(ValueError, error):
                self.step(self.invoke(ident, **fields))
            self.assertEqual(self.membership(ident), {'research_attempt_id': None, 'research_attempt_title': ''})
            self.assertTrue(self.harness.state.decision_records[ident]['validation_error'])
        self.tools.execute.assert_called_once()
        self.restore()
        self.step(self.invoke('continued', research_attempt_id=attempt_id))
        self.assertEqual(self.membership('continued'), self.membership('origin'))

    def test_commit_response_loss_and_json_reload_keep_uuid_original_proposal_and_single_execution(self):
        for committed in (False, True):
            with self.subTest(committed=committed):
                self.setUp()
                request = self.invoke('stable', research_attempt_title='Stable attempt')
                append = self.harness.memory.append
                def lost(entry):
                    if committed:
                        append(entry)
                    raise OSError('history write response lost')
                self.harness.memory.append = lost
                with self.assertRaises(OSError):
                    self.step(request)
                membership = self.membership('stable')
                self.tools.execute.assert_not_called()
                self.restore()
                result = self.step(request)
                self.assertEqual(self.step(request), result)
                self.assertEqual(self.membership('stable'), membership)
                self.assertEqual(self.harness.state.decision_records['stable']['proposal'], asdict(request))
                self.assertEqual(len(self.harness.memory.history(kind='decision')), 1)
                self.tools.execute.assert_called_once()
                with self.assertRaisesRegex(ValueError, 'different content'):
                    self.step(replace(request, research_attempt_id=membership['research_attempt_id']))

    def test_operation_receipt_loss_retries_resolved_membership_without_reexecution(self):
        request = self.invoke('stable', research_attempt_title='Stable receipt')
        append = self.harness.memory.append
        def lost(entry):
            append(entry)
            if entry.kind == 'operation':
                raise OSError('operation response lost')
        self.harness.memory.append = lost
        with self.assertRaises(OSError):
            self.step(request)
        membership = self.membership('stable')
        self.restore()
        self.harness.step('task', self.space)
        receipt, = self.harness.memory.history(kind='operation')
        self.assertEqual({key: receipt.details[key] for key in membership}, membership)
        self.tools.execute.assert_called_once()

    def test_review_invocations_can_start_and_continue_attempts_in_the_same_session(self):
        directive = Directive('active', 'task', 'active')
        self.step(Decision('explore', directive=directive, decision_id='active'))
        self.space.permissions = ActionPermissions(review_operations='all')
        contexts = []
        def review(context):
            contexts.append(context)
            if len(contexts) == 1:
                return self.invoke('review-start', research_attempt_title='Review evidence')
            if len(contexts) == 2:
                row = next(row for row in context.decision_graph if row['decision_id'] == 'review-start')
                return self.invoke('review-continue', research_attempt_id=row['research_attempt_id'])
            return ReviewDecision()
        self.policy.review = review
        self.harness.review_with_tools('task', ExplorationProgress('active', 'running', directive), self.space)
        self.assertEqual(len(contexts), 3)
        self.assertEqual(self.membership('review-start'), self.membership('review-continue'))
        self.assertIsNone(self.membership('active')['research_attempt_id'])
        for receipt in self.harness.memory.history(kind='operation'):
            self.assertEqual(receipt.details['research_attempt_title'], 'Review evidence')
            self.assertEqual(receipt.details['research_attempt_id'], self.membership('review-start')['research_attempt_id'])
        self.restore()

    def test_old_snapshots_default_to_ungrouped_and_can_replay_unchanged_old_history(self):
        self.step(self.invoke('old'))
        for row in self.harness.state.decision_records.values():
            for key in ('research_attempt_id', 'research_attempt_title'):
                row.pop(key)
                row['proposal'].pop(key)
        payload = self.harness.memory.export()
        for history in payload['history']:
            for key in ('research_attempt_id', 'research_attempt_title'):
                history['details'].pop(key, None)
        self.harness.memory._load(json.loads(json.dumps(payload)))
        self.restore()
        self.step(self.invoke('old'))
        self.tools.execute.assert_called_once()
        self.assertNotIn('research_attempt_id', self.harness._context('task', self.space).decision_graph[0])
        value = encode(self.invoke('old'))
        for key in ('research_attempt_id', 'research_attempt_title'):
            value['fields'].pop(key)
        restored = decode(value)
        self.assertIsNone(restored.research_attempt_id)
        self.assertEqual(restored.research_attempt_title, '')

    def test_snapshot_rejects_unpaired_forged_or_rejected_membership_and_changed_canonical_title(self):
        self.step(self.invoke('origin', research_attempt_title='Canonical title'))
        self.step(self.invoke('continued', research_attempt_id=self.membership('origin')['research_attempt_id']))
        for ident, changes in (
            ('origin', {'research_attempt_title': ''}),
            ('origin', {'research_attempt_id': None}),
            ('origin', {'validation_error': 'rejected'}),
            ('origin', {'research_attempt_title': 'renamed'}),
            ('continued', {'research_attempt_title': 'renamed'}),
            ('continued', {'research_attempt_id': 'forged'}),
        ):
            state = deepcopy(self.harness.state)
            state.decision_records[ident].update(changes)
            with self.subTest(ident=ident, changes=changes), self.assertRaises(ValueError):
                decode(encode(state))
        state = deepcopy(self.harness.state)
        for row in state.decision_records.values():
            row['proposal']['research_attempt_id'] = row['research_attempt_id']
        with self.assertRaisesRegex(ValueError, 'validated ledger origin'):
            decode(encode(state))


class AttemptPolicyTests(unittest.TestCase):
    def setUp(self):
        self.space = DemoDecisionActionSpace()
        self.context = DecisionContext('task', (), self.space.describe(), (), decision_graph=(
            {'decision_id': 'origin', 'status': 'completed', 'research_attempt_id': 'known',
             'research_attempt_title': 'Canonical title'},
            {'decision_id': 'rejected', 'status': 'failed', 'research_attempt_id': None, 'research_attempt_title': ''},
        ))

    def policy(self, *payloads, **settings):
        backend = Mock(supports_response_schema=True)
        backend.generate.side_effect = [GenerationResponse(json.dumps(payload), MODEL, False) for payload in payloads]
        return ModelDecisionPolicy(backend, MODEL, **settings)

    def test_both_parsers_preserve_optional_original_fields_and_schemas_bound_them(self):
        invocation = {'kind': 'invoke', 'operation': 'prepare_data', 'arguments': {'text': 'data'}}
        for fields in ({}, {'research_attempt_title': 'New title'}, {'research_attempt_id': 'known'},
                       {'research_attempt_id': 'known', 'research_attempt_title': 'Canonical title'}):
            for method in ('decide', 'review'):
                with self.subTest(fields=fields, method=method):
                    policy = self.policy({**invocation, **fields})
                    decision = getattr(policy, method)(self.context)
                    self.assertEqual(decision.research_attempt_id, fields.get('research_attempt_id'))
                    self.assertEqual(decision.research_attempt_title, fields.get('research_attempt_title', ''))
                    request = policy.backend.generate.call_args.args[0]
                    self.assertIn('starts a NEW research attempt', request.system)
                    self.assertIn('never inferred', request.system)
                    branches = [b for b in request.response_schema['oneOf'] if 'kind' in b['properties']]
                    for branch in branches:
                        self.assertEqual(branch['properties']['research_attempt_id']['maxLength'], 200)
                        self.assertEqual(branch['properties']['research_attempt_title']['maxLength'], 500)
                        self.assertNotIn('research_attempt_id', branch['required'])
                        self.assertNotIn('research_attempt_title', branch['required'])

    def test_explore_attempt_metadata_stays_out_of_nested_and_legacy_directive_parameters(self):
        for arguments in ({'arguments': {'instructions': 'Be concise'}}, {'instructions': 'Be concise'}):
            payload = {'kind': 'explore', 'mode': 'configure_prompt', 'strategy': 'Check answer length',
                       'research_attempt_title': 'Prompt length', **arguments}
            decision = self.policy(payload).decide(replace(self.context, directive_modes=('configure_prompt',)))
            self.assertEqual(decision.research_attempt_title, 'Prompt length')
            self.assertEqual(decision.directive.payload.instructions, 'Be concise')

    def test_unknown_id_rename_invalid_types_and_oversized_fields_are_validation_errors(self):
        for fields in (
            {'research_attempt_id': 'unknown'}, {'research_attempt_id': 'rejected'},
            {'research_attempt_id': 'known', 'research_attempt_title': 'Renamed'},
            {'research_attempt_id': ''}, {'research_attempt_id': ' '}, {'research_attempt_id': 42},
            {'research_attempt_id': 'i' * 201}, {'research_attempt_title': 't' * 501},
            {'research_attempt_title': ' '}, {'research_attempt_title': None},
        ):
            for method in ('decide', 'review'):
                payload = {'kind': 'invoke', 'operation': 'prepare_data', 'arguments': {'text': 'data'}, **fields}
                with self.subTest(fields=fields, method=method), self.assertRaises(ValueError):
                    getattr(self.policy(payload), method)(self.context)
        # Boundaries are inclusive; identity verification is a separate check.
        self.assertEqual(Decision('wait', research_attempt_id='i' * 200).research_attempt_id, 'i' * 200)
        self.assertEqual(Decision('wait', research_attempt_title='t' * 500).research_attempt_title, 't' * 500)
        for fields in ({'research_attempt_id': 'i' * 201}, {'research_attempt_title': 't' * 501},
                       {'research_attempt_id': 42}, {'research_attempt_title': None}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                Decision('wait', **fields)

    def test_unknown_membership_uses_bounded_policy_retry_without_minting_identity(self):
        policy = self.policy({'kind': 'wait', 'research_attempt_id': 'invented'},
                             {'kind': 'wait', 'research_attempt_title': 'Corrected new attempt'},
                             max_validation_attempts=2)
        result = policy.decide(self.context)
        self.assertIsNone(result.research_attempt_id)
        self.assertEqual(result.research_attempt_title, 'Corrected new attempt')
        self.assertEqual(policy.backend.generate.call_count, 2)
        self.assertIn('Unknown research_attempt_id', policy.backend.generate.call_args.args[0].prompt)


if __name__ == '__main__':
    unittest.main()
