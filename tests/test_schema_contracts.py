"""Check public JSON schemas against pure validators, never execute model actions.

Install the optional test extra to run with the standard JSON Schema validator.
State-dependent checks (known candidates, revisions, paths, running jobs) remain
the executor's responsibility, not assertions of static schema sufficiency.
"""
from dataclasses import asdict
import tempfile
import unittest

try:
    from jsonschema import Draft202012Validator
except ImportError:
    Draft202012Validator = None

from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.backend import BackendCapabilities
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.contracts.queries import QuerySpec
from rsisys.loops.contracts.training import ResumeMode, TrainingOrigin
from rsisys.loops.decision_making.action_space import LocalDecisionActionSpace, OperationCall
from rsisys.loops.decision_making.action_space.directives import builtin_directives, training_directive
from rsisys.loops.decision_making.action_space.queries import QueryActionDefinition
from rsisys.loops.decision_making.harness.memory import DecisionMemory
from rsisys.loops.decision_making.harness.research import ResearchAgenda
from rsisys.loops.decision_making.harness.tools.research import ResearchTools
from rsisys.loops.decision_making.harness.tools.workspace import WorkspaceTools
from rsisys.loops.exploration.action_space import ActionProposal
from rsisys.loops.exploration.action_space.local import CommandActionSpace


@unittest.skipUnless(Draft202012Validator, 'Install rsisys[test] for JSON Schema contract checks')
class SchemaContractTests(unittest.TestCase):
    def agrees(self, schema, parser, arguments, expected):
        Draft202012Validator.check_schema(schema)
        self.assertEqual(Draft202012Validator(schema).is_valid(arguments), expected, arguments)
        try:
            parser(arguments)
            accepted = True
        except (ValueError, TypeError, KeyError, NotImplementedError):
            accepted = False
        self.assertEqual(accepted, expected, arguments)

    def test_training_initialization_modes_and_null_defaults(self):
        initial = ModelRef('native', 'base', 'v1')
        checkpoint = asdict(ArtifactRef('state', 'training_state', 'file:///checkpoint'))
        for modes in (tuple(ResumeMode), (ResumeMode.RESTART,), (ResumeMode.WEIGHTS_ONLY,),
                      (ResumeMode.FULL_STATE,), ()):
            definition = training_directive(BackendCapabilities(methods=('sft',), resume_modes=modes))
            parser = lambda args: definition.parser(args, ())
            for mode in ResumeMode:
                for cp in ('omit', None, checkpoint):
                    origin = {**asdict(TrainingOrigin('parent', initial)), 'resume_mode': mode.value}
                    if cp == 'omit':
                        origin.pop('checkpoint')
                    else:
                        origin['checkpoint'] = cp
                    expected = (mode in modes and (mode != ResumeMode.RESTART or cp in ('omit', None))
                                and (mode != ResumeMode.FULL_STATE or isinstance(cp, dict)))
                    self.agrees(definition.spec.input_schema, parser,
                        {'method': 'sft', 'recipe_ref': 'recipe', 'origin': origin}, expected)
            for args in ({'method': 'sft', 'recipe_ref': 'recipe'},
                         {'method': 'sft', 'recipe_ref': 'recipe', 'origin': None}):
                self.agrees(definition.spec.input_schema, parser, args, ResumeMode.RESTART in modes)

    def test_builtin_recipe_types_and_empty_values(self):
        candidate = Candidate('c', 'v1', {}, ('actor',))
        for definition in (*builtin_directives(), training_directive(BackendCapabilities(methods=('sft',)))):
            mode = definition.spec.mode
            if mode not in ('sft', 'rl', 'train', 'evaluate'):
                continue
            key = 'protocol_ref' if mode == 'evaluate' else 'recipe_ref'
            for value in ('recipe', '', '  ', 1, True, None):
                args = {key: value}
                if mode == 'train':
                    args['method'] = 'sft'
                if mode == 'evaluate':
                    args['candidate_id'] = candidate.candidate_id
                self.agrees(definition.spec.input_schema, lambda a: definition.parser(a, (candidate,)),
                            args, value == 'recipe')

    def test_query_cursor_and_filter_bounds(self):
        definition = QueryActionDefinition(QuerySpec('rows', 'Read rows', tuple(f'f{i}' for i in range(9))))
        for cursor in (None, '0', '9999', '10000', '10001', '99999', '00', '-1', '', '١'):
            self.agrees(definition.spec.input_schema, definition.validate, {'cursor': cursor},
                        cursor in (None, '0', '9999', '10000'))
        for count in (8, 9):
            self.agrees(definition.spec.input_schema, definition.validate,
                        {'filters': {f'f{i}': 'v' for i in range(count)}}, count == 8)

    def test_research_read_and_combined_patch_bounds(self):
        tools = ResearchTools(ResearchAgenda(), DecisionMemory())
        definitions = {d.spec.name: d for d in tools.definitions}
        read = definitions['research_read']
        for args, valid in (({}, True), ({'view': 'revisions'}, True),
                            ({'view': 'revisions', 'kind': 'todo'}, False)):
            self.agrees(read.spec.input_schema, read.validate, args, valid)
        patch = definitions['research_patch']
        for upserts, removals in ((0, 0), (1, 0), (0, 32), (16, 16), (33, 0), (0, 33)):
            args = {'expected_revision': 0,
                'upserts': [{'id': f'u{i}', 'kind': 'todo', 'text': 'question'} for i in range(upserts)],
                'remove_ids': [f'r{i}' for i in range(removals)], 'reason': 'update notes', 'evidence_refs': []}
            self.agrees(patch.spec.input_schema, patch.validate, args, 1 <= upserts + removals <= 32)
        archive = definitions['research_archive']
        self.agrees(archive.spec.input_schema, archive.validate, {'collection': 'experience'}, False)

    def test_workspace_command_and_nonempty_text_parameters(self):
        with tempfile.TemporaryDirectory() as root:
            definition = next(d for d in WorkspaceTools({'run': root}).definitions if d.spec.name == 'workspace_search')
            for query in ('token', '', 'two\nlines', 'two\rlines'):
                self.agrees(definition.spec.input_schema, definition.validate,
                            {'root': 'run', 'query': query}, query == 'token')
        commands = CommandActionSpace()
        spec = next(s for s in commands.describe() if s.name == 'execute')
        for argv in ([], ['python', '-V'], ['bad\0argument']):
            self.agrees(spec.input_schema,
                lambda a: commands.validate(ActionProposal('execute', a, True), ('execute',)),
                {'argv': argv}, bool(argv) and '\0' not in argv[0])
        space = LocalDecisionActionSpace()
        spec = next(s for s in space.describe() if s.name == 'search_literature')
        for query in ('question', '', '   '):
            self.agrees(spec.input_schema, lambda a: space.validate(OperationCall(spec.name, a)),
                        {'query': query}, query == 'question')
