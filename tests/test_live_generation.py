"""Read-only generation canaries, NOT autonomous research or formal SFT evidence.

Opt in with BOTH RSISYS_LIVE_BASE_URL (API root including /v1) and
RSISYS_LIVE_MODEL (served alias), then run:
    PYTHONPATH=src python -m unittest tests.test_live_generation -v
Optional: RSISYS_LIVE_API_KEY (bearer token), RSISYS_LIVE_TIMEOUT (seconds,
180 by default), RSISYS_LIVE_EVIDENCE (JSONL evidence path), and
RSISYS_LIVE_NATIVE_PREFLIGHT=1 (compile with the installed xgrammar before HTTP).
Use RSISYS_LIVE_SCHEMA_MODE=portable for the tested modelmate service.
Default discovery skips every network test without constructing
an HTTP client. An opted-in run requires jsonschema (pip install -e '.[test]').

Six live tests make SEVEN chat/completions calls in total. No probes, retries,
SSH, deployment, runner.run(), actor inference, commands, or training submission.
Expected values appear ONLY in prompts, never added const/enum constraints.
Schemas come from production definitions or an actual CPU medical assembly.
The medical canary defines a real local recipe and prepares a request, then stops.

Coverage: builtin recipe/protocol/candidate refs; generic train method/origin,
checkpoint IDs/kind/URI/version/digest (strings and null); research IDs/status,
Chinese/multiline/quotes/backslashes/reason/evidence arrays; query filters,
cursor null/0/9999/10000; workspace fields/single-line query; argv/output paths; system
candidate_ids/version/idempotency object; inspect/cancel/update/compare control
parameters; medical numeric recipe and full train
policy envelope through parser, action space, CheckedTraining.prepare.

System adoption uses the complete production spec and a registered candidate;
only definition.validate runs, never the adoption handler. Controls likewise
validate parameters without executing inspect/cancel/update/compare.
Limits: no exhaustive schema/action coverage, scientific decision quality, optimization, submission,
checkpoint loading, URI resolution, deployment or weight identity attestation.
Business string guards are tested separately; grammar need not enforce strip/NUL
semantics. Replies are never stripped, decoded as HTML, repaired or normalized.
"""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

try:
    from jsonschema import Draft202012Validator
except ImportError:
    Draft202012Validator = None

from rsisys.adapters.serving import VLLMBackend
from rsisys.loops.contracts.candidate import Candidate, SystemVersion
from rsisys.loops.contracts.model_service import BackendBinding, GenerationRequest, ModelRef
from rsisys.loops.contracts.queries import QuerySpec
from rsisys.loops.decision_making.action_space import (
    LocalDecisionActionSpace, ModularDecisionActionSpace, OperationCall,
)
from rsisys.loops.decision_making.action_space.directives import builtin_directives, training_directive
from rsisys.loops.decision_making.action_space.queries import QueryActionDefinition
from rsisys.loops.decision_making.harness.memory import DecisionMemory
from rsisys.loops.decision_making.harness.research import ResearchAgenda
from rsisys.loops.decision_making.harness.tools.research import ResearchTools
from rsisys.loops.decision_making.harness.tools.workspace import WorkspaceTools
from rsisys.loops.exploration.action_space import ActionProposal
from rsisys.loops.exploration.action_space.local import CommandActionSpace
from rsisys.loops.exploration.control import ExperimentControls
from rsisys.runtime.systems import ReferenceSystemActivator, SystemRegistry, system_adoption_capability


LIVE = bool(os.environ.get('RSISYS_LIVE_BASE_URL') and os.environ.get('RSISYS_LIVE_MODEL'))
COPY_INSTRUCTION = (
    'This is a deterministic read-only transport canary, not a research decision. '
    'Return exactly the supplied target JSON object. Copy every string byte-for-byte '
    'after JSON decoding: preserve all spaces, Chinese, punctuation, quotes, '
    'backslashes and actual newline characters. Serialize actual newlines as JSON '
    'escape sequences \\n, never HTML entities (such as &#10;) or spaces. '
    'Do not abbreviate IDs or infer alternative values. No actions will execute. '
    'Do not add fields, markdown, commentary or reasoning. Target JSON follows:\n'
)


def object_schema(properties):
    """Packing wrapper only; all child constraints are untouched production schemas."""
    return {'type': 'object', 'properties': deepcopy(properties),
            'required': list(properties), 'additionalProperties': False}


def prompt_order(schema, value):
    """Present unchanged target values in the production grammar's field order."""
    if isinstance(value, dict):
        properties = deepcopy(schema.get('properties', {}))
        for keyword in ('allOf', 'oneOf', 'anyOf'):
            for branch in schema.get(keyword, ()):
                if Draft202012Validator(branch).is_valid(value):
                    for name, field in branch.get('properties', {}).items():
                        properties[name] = {**properties.get(name, {}), **field}
        names = [name for name in properties if name in value]
        names += [name for name in value if name not in properties]
        return {name: prompt_order(properties.get(name, {}), value[name]) for name in names}
    if isinstance(value, list):
        return [prompt_order(schema.get('items', {}), item) for item in value]
    return value


def research_definitions():
    tools = ResearchTools(ResearchAgenda(), DecisionMemory())
    return {d.spec.name: d for d in tools.definitions}


def adoption_definition():
    candidate = Candidate('candidate-live-canary-0123456789', 'base-revision', {}, ('actor',), 'initial-canary')
    registry = SystemRegistry(SystemVersion('initial-canary', {}), ReferenceSystemActivator())
    registry.register(candidate)
    return system_adoption_capability(registry, lambda: (candidate,)).definition, candidate, registry


def control_definitions():
    guards = {name: Mock(side_effect=AssertionError(f'Control execution forbidden: {name}'))
              for name in ('inspect', 'cancel', 'update', 'compare_candidates')}
    controls = ExperimentControls(SimpleNamespace(**{name: guards[name] for name in ('inspect', 'cancel', 'update')}),
                                  SimpleNamespace(compare_candidates=guards['compare_candidates']))
    return {d.spec.name: d for d in controls.definitions}, guards


@unittest.skipUnless(LIVE, 'Set BOTH RSISYS_LIVE_BASE_URL and RSISYS_LIVE_MODEL to allow real inference')
class LiveGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if Draft202012Validator is None:
            raise RuntimeError("Opted-in live canaries require jsonschema: pip install -e '.[test]'")
        cls.model = ModelRef('live-canary-vllm', os.environ['RSISYS_LIVE_MODEL'], 'provider-managed-canary')
        cls.backend = VLLMBackend(BackendBinding(
            cls.model.backend_id, 'chat_completions', os.environ['RSISYS_LIVE_BASE_URL'],
            credential_env='RSISYS_LIVE_API_KEY' if os.environ.get('RSISYS_LIVE_API_KEY') else None,
            timeout_seconds=float(os.environ.get('RSISYS_LIVE_TIMEOUT', '180'))),
            generation_options={'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}},
            native_schema_mode=os.environ.get('RSISYS_LIVE_SCHEMA_MODE', 'exact'))

    def assert_values(self, actual, expected, path='$'):
        # Type checks prevent True == 1 and False == 0 from masking corruption.
        if isinstance(expected, dict):
            self.assertIs(type(actual), dict, path)
            self.assertEqual(set(actual), set(expected), path)
            for key, value in expected.items():
                self.assert_values(actual[key], value, f'{path}.{key}')
        elif isinstance(expected, list):
            self.assertIs(type(actual), list, path)
            self.assertEqual(len(actual), len(expected), path)
            for index, value in enumerate(expected):
                self.assert_values(actual[index], value, f'{path}[{index}]')
        else:
            if isinstance(expected, float):
                self.assertIn(type(actual), (int, float), path)
            else:
                self.assertIs(type(actual), type(expected), path)
            if isinstance(expected, str) and len(expected) > 1:
                self.assertGreater(len(actual), 1, f'{path}: collapsed to a single character')
            self.assertEqual(actual, expected, path)

    def standard_validate(self, schema, value):
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(value)

    def record(self, value):
        path = os.environ.get('RSISYS_LIVE_EVIDENCE')
        if path:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('a') as stream:
                stream.write(json.dumps({'test': self.id(), **value}, ensure_ascii=False) + '\n')

    def generate(self, schema, target, *, request=None):
        self.standard_validate(schema, target)
        ordered = prompt_order(schema, target)
        self.assert_values(ordered, target)
        prompt = COPY_INSTRUCTION + json.dumps(ordered, ensure_ascii=False)
        if request is None:
            request = GenerationRequest('decision_maker', self.model, prompt,
                system='Perform exact JSON string copying. This is a read-only canary.',
                max_output_tokens=4096, response_schema=schema)
        else:
            self.assertEqual(request.response_schema, schema)
            request = replace(request, prompt=request.prompt + '\n' + prompt,
                system=request.system + '\nRead-only canary: copy the target JSON exactly; '
                    'it overrides instructions to choose an autonomous research action.')
        if os.environ.get('RSISYS_LIVE_NATIVE_PREFLIGHT') == '1':
            import xgrammar
            xgrammar.Grammar.from_json_schema(json.dumps(self.backend.native_response_schema(schema)))
            self.record({'stage': 'native_schema_preflight', 'executed': False})
        # Exactly one attempt. HTTP/grammar errors and truncation fail visibly.
        wire_schema = self.backend.native_response_schema(schema)
        self.record({'stage': 'requested', 'request': asdict(request), 'expected': target,
                     'wire_response_schema': wire_schema, 'native_schema_mode': self.backend.native_schema_mode,
                     'executed': False})
        native_request = self.backend._request
        def observed(route, payload=None):
            result = native_request(route, payload)
            self.record({'stage': 'raw_http_response', 'response': result, 'executed': False})
            return result
        with patch.object(self.backend, '_request', side_effect=observed):
            response = self.backend.generate(request)
        self.record({'stage': 'generated', 'request': asdict(request), 'expected': target,
                     'response': asdict(response), 'executed': False})
        self.assertEqual(response.model, self.model)
        self.assertFalse(response.simulated)
        actual = json.loads(response.text)
        self.assert_values(actual, target)
        self.standard_validate(schema, actual)
        return actual, response

    def test_directive_refs_methods_origins_and_nullable_checkpoints(self):
        definitions = {d.spec.mode: d for d in builtin_directives()}
        definitions['train'] = training_directive()
        candidate = Candidate('medical-candidate-canary-0123456789abcdef', 'base-revision', {}, ('actor',))
        initial = asdict(ModelRef('readonly-backend-canary', 'Qwen3.8-27B-canary', 'revision-0123456789abcdef'))
        checkpoint = {'artifact_id': 'training-state-canary-0123456789abcdef',
            'kind': 'training_state', 'uri': 'file:///readonly/canary/checkpoints/step-000123/state.json',
            'version': 'checkpoint-version-000123', 'digest': 'sha256:' + 'ab' * 32}
        train = {'recipe_ref': 'medical-recipe://' + '0123456789abcdef' * 4,
            'method': 'supervised_finetuning_canary', 'apply_result': False,
            'origin': {'parent_system': 'system-canary-version-0000123', 'initial_model': initial,
                       'resume_mode': 'full_state', 'checkpoint': checkpoint}}
        target = {
            'train_full': train,
            'train_nullable': {**train, 'origin': {**train['origin'], 'resume_mode': 'weights_only',
                'checkpoint': {**checkpoint, 'version': None, 'digest': None}}},
            'train_restart': {**train, 'origin': None},
            'sft': {'recipe_ref': 'recipe://readonly/long-sft-ref-0123456789', 'apply_result': True},
            'rl': {'recipe_ref': 'recipe://readonly/long-rl-ref-0123456789', 'updates': 12},
            'evaluate': {'candidate_id': candidate.candidate_id, 'protocol_ref': 'medical-protocol-0123456789abcdef'},
            'configure_prompt': {'instructions': ''},
        }
        modes = {key: 'train' if key.startswith('train_') else key for key in target}
        schema = object_schema({key: definitions[mode].spec.input_schema for key, mode in modes.items()})
        actual, _ = self.generate(schema, target)
        for key, mode in modes.items():
            parsed = definitions[mode].parser(actual[key], (candidate,))
            if mode == 'train':
                self.assertEqual(parsed.recipe_ref, train['recipe_ref'])
                self.assertEqual(parsed.method, train['method'])
                if actual[key]['origin'] is not None:
                    self.assertEqual(asdict(parsed.origin.checkpoint), actual[key]['origin']['checkpoint'])
            elif mode in ('sft', 'rl'):
                self.assertEqual(parsed.recipe_ref, target[key]['recipe_ref'])
            elif mode == 'evaluate':
                self.assertEqual(parsed.candidate.candidate_id, candidate.candidate_id)
                self.assertEqual(parsed.protocol_ref, target[key]['protocol_ref'])

    def test_research_notes_exact_chinese_newlines_quotes_and_arrays(self):
        definition = research_definitions()['research_patch']
        target = {'expected_revision': 0, 'upserts': [{
            'id': 'note-canary-transport-0123456789', 'kind': 'todo',
            'text': '  医疗回归 canary：保留完整文字 "alpha"。\nbeta 第二行\\literal\\path\n第三行：仅验证传输。  ',
            'status': 'awaiting_readonly_review',
            'evidence_refs': ['artifact:readonly-evidence-0123456789', 'protocol:medical-canary-abcdef']}],
            'remove_ids': [], 'reason': '只读测试 "引号" 和换行\n不执行训练、不构成科研结论。', 'evidence_refs': []}
        actual, _ = self.generate(definition.spec.input_schema, target)
        checked = definition.validate(actual)
        self.assert_values(checked.arguments, target)

    def test_queries_workspace_and_multiline_argv(self):
        with tempfile.TemporaryDirectory() as root:
            workspace = next(d for d in WorkspaceTools({'canary_workspace': root}).definitions
                             if d.spec.name == 'workspace_search')
            query = QueryActionDefinition(QuerySpec('live_canary_rows', 'Read-only query canary', ('status',)))
            read = research_definitions()['research_read']
            literature = LocalDecisionActionSpace()
            lit_spec = next(s for s in literature.describe() if s.name == 'search_literature')
            commands = CommandActionSpace()
            command_spec = next(s for s in commands.describe() if s.name == 'execute')
            schema = object_schema({'workspace': workspace.spec.input_schema, 'query': query.spec.input_schema,
                'query_zero': query.spec.input_schema, 'query_9999': query.spec.input_schema,
                'query_null': query.spec.input_schema, 'read': read.spec.input_schema,
                'literature': lit_spec.input_schema, 'command': command_spec.input_schema})
            target = {'workspace': {'root': 'canary_workspace', 'path': 'notes/canary.txt',
                'query': '医疗 "quoted" \\literal\\path', 'offset': 12, 'limit': 7, 'max_bytes': 4096,
                'scan_budget_bytes': 8192, 'max_entries': 32, 'max_depth': 3},
                'query': {'filters': {'status': 'awaiting_readonly_review'}, 'limit': 17, 'cursor': '10000'},
                'query_zero': {'filters': {}, 'limit': 2, 'cursor': '0'},
                'query_9999': {'filters': {}, 'limit': 9, 'cursor': '9999'},
                'query_null': {'filters': {}, 'limit': 3, 'cursor': None},
                'read': {'view': 'items', 'kind': None, 'limit': 12, 'cursor': None},
                'literature': {'query': '医疗问题 "canary"\nsecond line \\literal'},
                'command': {'argv': ['python3', '-c', 'print("医疗 canary")\nprint("second line")', ''],
                            'output_paths': ['outputs/canary-report.json']}}
            actual, _ = self.generate(schema, target)
            for key, definition in (('workspace', workspace), ('query', query),
                    ('query_zero', query), ('query_9999', query), ('query_null', query), ('read', read)):
                self.assert_values(definition.validate(actual[key]).arguments, target[key])
            literature.validate(OperationCall('search_literature', actual['literature']))
            commands.validate(ActionProposal('execute', actual['command'], True), ('execute',))
            # No workspace binding, command executor or query provider is called.

    def test_system_adopt_complete_production_spec_validate_only(self):
        definition, candidate, registry = adoption_definition()
        original = deepcopy(definition.spec.input_schema)
        current = registry.current
        target = {'candidate_ids': [candidate.candidate_id], 'version': 'system-canary-version-0123456789',
                  'idempotency_key': 'readonly-adoption-canary:0123456789abcdef'}
        with patch.object(registry, 'adopt', side_effect=AssertionError('Adoption is forbidden')) as adopt, \
                patch.object(registry.activator, 'activate', side_effect=AssertionError('Activation is forbidden')) as activate:
            actual, _ = self.generate(original, target)
            self.assert_values(definition.validate(actual).arguments, target)
            adopt.assert_not_called()
            activate.assert_not_called()
        self.assertEqual(definition.spec.input_schema, original)
        self.assertEqual(registry.current, current)
        self.assertEqual(registry.get(candidate.candidate_id).parent_version, current.version)
        # Never invoke capability.handler (it would adopt a system).

    def test_control_complete_schemas_validate_parameters_only(self):
        definitions, guards = control_definitions()
        target = {
            'experiment_inspect': {'directive_id': 'directive-inspect-canary-0123456789'},
            'experiment_cancel': {'directive_id': 'directive-cancel-canary-0123456789'},
            'experiment_update': {'directive_id': 'directive-update-canary-0123456789',
                'expected_version': 12, 'idempotency_key': 'readonly-update-canary:0123456789abcdef',
                'changes': {'remaining_polls': 17, 'paused': False, 'labels': ['medical', 'canary'], 'note': None}},
            'experiment_compare': {'candidate_ids': ['candidate-canary-0123456789', 'candidate-canary-abcdef0123'],
                'metric': 'medical_development_accuracy', 'higher_is_better': True},
        }
        schema = object_schema({name: definitions[name].spec.input_schema for name in target})
        actual, _ = self.generate(schema, target)
        for name, definition in definitions.items():
            self.assert_values(definition.validate(actual[name]).arguments, target[name])
        for guard in guards.values():
            guard.assert_not_called()

    def test_medical_numeric_recipe_and_full_train_envelope_prepare_only(self):
        from examples.medical.experiment import CheckedTraining, DecisionBackend, ExperimentOperations, ResearchPolicy
        from examples.medical.run import build
        from rsisys.loops.exploration.harness.commands import CommandState
        if __package__:
            from .test_medical_experiment import Converter, MedicalOperationContractTests
        else:
            from test_medical_experiment import Converter, MedicalOperationContractTests

        fixture = MedicalOperationContractTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        learner = fixture.learner
        with ExitStack() as stack:
            guards = [stack.enter_context(patch.object(owner, 'submit', side_effect=AssertionError(
                'Read-only canaries must NEVER submit'))) for owner in (learner, fixture.process)]
            assembly = build(fixture.assembly.task, artifacts=fixture.assembly.artifacts,
                decision_backend=self.backend, decision_model=self.model, initial_model=learner.base,
                inference=lambda request: self.fail('Actor inference is forbidden'),
                system_activator=ReferenceSystemActivator(learner.base), learner=learner,
                bind_training_inputs=learner.bind_inputs)
            guards.append(stack.enter_context(patch.object(assembly.evaluator, 'submit',
                side_effect=AssertionError('Evaluation submission is forbidden'))))
            guards.append(stack.enter_context(patch.object(assembly.systems.activator, 'activate',
                side_effect=AssertionError('System activation is forbidden'))))
            operations = ExperimentOperations(assembly, learner, Converter(), ())
            definitions = {d.spec.name: d for d in operations.definitions}
            parameters = {'steps': 5, 'lr': 0.0002, 'seed': 17, 'weight_decay': 0.03, 'warmup_ratio': 0.12}
            actual, _ = self.generate(definitions['define_recipe'].spec.input_schema, parameters)
            call = definitions['define_recipe'].validate(actual)
            receipt = operations.bindings['define_recipe'](call.arguments)
            self.assertEqual(receipt.status.value, 'completed')
            recipe = receipt.data['recipe_ref']
            self.assertGreater(len(recipe), 32)
            self.assert_values(learner.recipes[recipe]['parameters'], parameters)
            template = learner.recipes[recipe]['template']
            self.assertEqual(template['seed'], parameters['seed'])
            for native, parameter in (('max_steps', 'steps'), ('learning_rate', 'lr'),
                    ('seed', 'seed'), ('data_seed', 'seed'), ('weight_decay', 'weight_decay'),
                    ('warmup_ratio', 'warmup_ratio')):
                self.assertEqual(template['training_args'][native], parameters[parameter])
            decision_harness = assembly.runner.decision_making.harness
            research = decision_harness.tools.research_tools
            space = ModularDecisionActionSpace((*research.definitions, *operations.definitions),
                directives=assembly.runner.decision_making.action_space.directives)
            # This is the production context builder, with real frozen data and catalogs.
            context = decision_harness._context('Read-only medical generation canary; no action execution.', space)
            backend = DecisionBackend(self.backend, Path(learner.root), SimpleNamespace(), 1, decision_thinking=False)
            policy = ResearchPolicy(backend, self.model, space.directives, space,
                SimpleNamespace(event=lambda *a, **kw: None), learner, operations.train_ref)
            request = policy._decision_request(context)
            target = {'kind': 'explore', 'mode': 'train',
                'strategy': '只读医疗 canary "完整参数"\n仅 prepare，不 submit。',
                'input_ids': [operations.train_ref.artifact_id],
                'arguments': {'recipe_ref': recipe, 'method': 'sft', 'apply_result': False},
                'reason': 'Verify actual recipe, frozen data and initialization.\n不执行模型动作。',
                'depends_on_decision_ids': [], 'source_decision_ids': []}
            _, response = self.generate(request.response_schema, target, request=request)
            parsed = policy._parse_decision(response, context)
            self.assertEqual(parsed.kind, 'explore')
            self.assertEqual(parsed.reason, target['reason'])
            directive = parsed.directive
            space.validate_directive(directive)
            self.assertEqual(directive.strategy, target['strategy'])
            self.assertEqual(directive.payload.recipe_ref, recipe)
            self.assertEqual(directive.input_refs, (operations.train_ref,))
            exploration = assembly.runner.exploration.harness
            handler = CheckedTraining(exploration.commands.handlers['train'], learner)
            state = CommandState(directive)
            handler.prepare(exploration, state)
            self.assertNotIn('business_rejection', state.data, state.data)
            expected_request = learner.bind_inputs(state.request, assembly.artifacts)
            self.assertEqual(state.request, expected_request)
            self.assertEqual(state.request.method, 'sft')
            self.assertEqual(state.request.base, learner.base)
            self.assertEqual(state.request.data_refs, (operations.train_ref,))
            self.assertEqual(state.request.origin.parent_system, assembly.systems.current.version)
            self.assertEqual(state.request.origin.initial_model, learner.base)
            self.assertEqual(state.request.origin.resume_mode.value, 'restart')
            self.assertIsNone(state.request.origin.checkpoint)
            self.assertEqual(state.request.recipe_ref, learner.bind_inputs(
                replace(state.request, recipe_ref=recipe), assembly.artifacts).recipe_ref)
            self.assertEqual(learner.check_request(state.request, state.idempotency_key),
                             learner.signature(state.request))
            self.assertFalse(learner.ledger)
            self.assertFalse(fixture.process.requests)
            self.assertFalse(state.started)
            self.assertIsNone(state.job)
            for guard in guards:
                guard.assert_not_called()
            self.record({'stage': 'prepared_not_submitted', 'request': asdict(state.request),
                'recipe_parameters': parameters, 'training_args': template['training_args'],
                'submissions': 0, 'executed': False})


class GenerationParserGuardTests(unittest.TestCase):
    """Offline business validation remains mandatory even when grammar is looser."""
    def test_nonempty_nul_cursor_and_singleline_guards(self):
        definitions = {d.spec.mode: d for d in builtin_directives()}
        definitions['train'] = training_directive()
        candidate = Candidate('canary-candidate', 'revision', {}, ('actor',))
        valid = {'sft': {'recipe_ref': 'long-recipe'}, 'rl': {'recipe_ref': 'long-recipe'},
            'train': {'recipe_ref': 'long-recipe', 'method': 'long-method'},
            'evaluate': {'candidate_id': candidate.candidate_id, 'protocol_ref': 'long-protocol'}}
        for mode, arguments in valid.items():
            for name in arguments:
                for blank in ('', ' ', '\t\n\r'):
                    with self.subTest(mode=mode, field=name, blank=repr(blank)), self.assertRaises((ValueError, TypeError)):
                        definitions[mode].parser({**arguments, name: blank}, (candidate,))
        origin = {'parent_system': 'canary-parent',
            'initial_model': {'backend_id': 'canary-backend', 'model_id': 'canary-model', 'revision': 'canary-revision'},
            'resume_mode': 'full_state', 'checkpoint': {'artifact_id': 'canary-state',
                'kind': 'training_state', 'uri': 'file:///canary/state.json',
                'version': 'canary-version', 'digest': 'canary-digest'}}
        for group, fields in (('checkpoint', tuple(origin['checkpoint'])),
                              ('initial_model', tuple(origin['initial_model'])),
                              (None, ('parent_system',))):
            for name in fields:
                for blank in ('', ' \n\t '):
                    bad_origin = deepcopy(origin)
                    (bad_origin[group] if group else bad_origin)[name] = blank
                    with self.subTest(origin_group=group, field=name, blank=repr(blank)), self.assertRaises(ValueError):
                        definitions['train'].parser({**valid['train'], 'origin': bad_origin}, ())
        research = research_definitions()['research_patch']
        patch_args = {'expected_revision': 0, 'upserts': [{'id': 'canary-note', 'kind': 'todo', 'text': 'valid note'}],
                      'remove_ids': [], 'reason': 'canary reason', 'evidence_refs': []}
        for location in ('id', 'text', 'reason', 'status', 'evidence_refs'):
            args = deepcopy(patch_args)
            if location == 'reason':
                args['reason'] = ' \n '
            elif location == 'evidence_refs':
                args['evidence_refs'] = [' \n ']
            else:
                args['upserts'][0][location] = ' \n '
            with self.subTest(research_field=location), self.assertRaises(ValueError):
                research.validate(args)
        definition, candidate, _ = adoption_definition()
        for name in ('candidate_ids', 'version', 'idempotency_key'):
            args = {'candidate_ids': [candidate.candidate_id], 'version': 'long-version', 'idempotency_key': 'long-key'}
            args[name] = [' \n '] if name == 'candidate_ids' else ' \n '
            with self.subTest(system_field=name), self.assertRaises(ValueError):
                definition.validate(args)
        with self.assertRaises(ValueError):
            definition.validate({'candidate_ids': [candidate.candidate_id, candidate.candidate_id],
                                 'version': 'long-version', 'idempotency_key': 'long-key'})
        controls, guards = control_definitions()
        with self.assertRaises(ValueError):
            controls['experiment_compare'].validate({'candidate_ids': ['duplicate-id', 'duplicate-id'],
                'metric': 'medical_development_accuracy', 'higher_is_better': True})
        for guard in guards.values():
            guard.assert_not_called()
        commands = CommandActionSpace()
        with self.assertRaises(ValueError):
            commands.validate(ActionProposal('execute', {'argv': ['python3', 'bad\0argument']}, True), ('execute',))
        query = QueryActionDefinition(QuerySpec('canary_rows', 'Read-only guard'))
        for cursor in ('10001', '00', '-1', '12\n', '١٢'):
            with self.subTest(cursor=repr(cursor)), self.assertRaises(ValueError):
                query.validate({'cursor': cursor})
        with tempfile.TemporaryDirectory() as root:
            workspace_definitions = WorkspaceTools({'canary': root}).definitions
            workspace = next(d for d in workspace_definitions if d.spec.name == 'workspace_search')
            for text in ('', 'alpha\nbeta', 'alpha\rbeta'):
                with self.subTest(query=repr(text)), self.assertRaises(ValueError):
                    workspace.validate({'root': 'canary', 'query': text})
        with self.assertRaises(ValueError):
            LocalDecisionActionSpace().validate(OperationCall('search_literature', {'query': ' \n '}))
        controls, _ = control_definitions()
        specs = [*(d.spec for d in definitions.values()), *(d.spec for d in research_definitions().values()),
                 *(d.spec for d in workspace_definitions), *(d.spec for d in controls.values()), definition.spec,
                 *commands.describe(), *LocalDecisionActionSpace().describe(), query.spec]
        portable = VLLMBackend(BackendBinding('canary', 'http', 'http://127.0.0.1:1/v1'),
                               native_schema_mode='portable')
        def check_native_keywords(node):
            if isinstance(node, dict):
                for key in ('pattern', 'minLength', 'maxLength', 'uniqueItems'):
                    self.assertNotIn(key, node)
                for value in node.values():
                    check_native_keywords(value)
            elif isinstance(node, list):
                for value in node:
                    check_native_keywords(value)
        for spec in specs:
            source = deepcopy(spec.input_schema)
            check_native_keywords(portable.native_response_schema(source))
            self.assertEqual(source, spec.input_schema)


if __name__ == '__main__':
    unittest.main()
