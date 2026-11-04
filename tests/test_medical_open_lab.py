"""Deployment boundaries: timed decisions, exact admission, allocated child devices."""
import json
from collections import UserDict
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from examples.medical.open_lab_experiment import (
    AllocatedProcessBackend, HostAdmissionStop, RecordedBackend, TimedOpenPolicy,
    AbsoluteHostPolicy,
)
from rsisys.loops.contracts.model_service import ModelRef, GenerationRequest, GenerationResponse
from rsisys.loops.decision_making.action_space.directives import DirectiveDefinitions
from rsisys.loops.decision_making.harness.state import DecisionContext
from rsisys.runtime.sandbox import SandboxSpec, ExecRequest
from rsisys.runtime.run_policy import RunContext


MODEL = ModelRef('meta', 'fixture', 'revision')


class Tokenizer:
    def apply_chat_template(self, messages, **options):
        self.seen = messages, options
        return list(range(100 + len(options.get('tools', []))))


class Transport:
    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail
    def _generation_payload(self, request):
        return {'model': request.model.model_id, 'messages': request.chat_messages(), **request.chat_tools()}
    def generate_with_response_observer(self, request, observer):
        self.calls += 1
        observer({'body': '{"provider":"fixture"}'})
        if self.fail:
            raise ValueError('Rejected model identity')
        return GenerationResponse('{"kind":"finish"}', request.model, True)


class MedicalOpenLabTests(unittest.TestCase):
    def test_no_idle_envelopes_and_no_host_budget_in_context(self):
        policy = TimedOpenPolicy(Transport(), MODEL, directives=DirectiveDefinitions(()))
        context = DecisionContext('objective', (), (), (), directive_modes=(),
            run_control={'remaining_seconds': 10, 'research_round': 4})
        self.assertEqual(policy._context_inputs(context)['run_control'], {'stopping_owner': 'host'})
        schema = policy.decision_schema(context)
        self.assertFalse(any(b['properties']['kind']['const'] in ('wait', 'finish') for b in schema['oneOf']))
        with self.assertRaises(ValueError):
            policy._parse_decision(GenerationResponse('{"kind":"finish"}', MODEL, True), context)

    def test_actual_native_messages_and_tools_are_counted(self):
        with TemporaryDirectory() as directory:
            tokenizer = Tokenizer()
            backend = RecordedBackend(Transport(), tokenizer, directory, max_calls=1)
            request = GenerationRequest('decision_maker', MODEL, '', messages=(
                {'role': 'user', 'content': 'observation'},), tools=(
                {'type': 'function', 'function': {'name': 'record', 'parameters': {'type': 'object'}}},))
            self.assertEqual(backend.count_tokens(request), 101)
            self.assertEqual(tokenizer.seen[0], request.chat_messages())
            self.assertEqual(tokenizer.seen[1]['tools'], list(request.tools))
            self.assertFalse(tokenizer.seen[1]['enable_thinking'])

    def test_transformers_batch_encoding_mapping_counts_tokens_not_fields(self):
        class BatchTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return UserDict(input_ids=list(range(54075)), attention_mask=[1] * 54075)
        with TemporaryDirectory() as directory:
            backend = RecordedBackend(Transport(), BatchTokenizer(), directory, max_calls=1)
            self.assertEqual(backend.count_tokens(GenerationRequest('decision_maker', MODEL, 'long')), 54075)

    def test_host_signal_closes_admission_before_another_request(self):
        with TemporaryDirectory() as directory:
            transport = Transport()
            backend = RecordedBackend(transport, Tokenizer(), directory, max_calls=10)
            backend.stop_requested = True
            with self.assertRaises(HostAdmissionStop):
                backend.generate(GenerationRequest('decision_maker', MODEL, 'choose'))
            self.assertEqual(transport.calls, 0)

    def test_absolute_deadline_cannot_be_extended_by_a_new_run(self):
        backend = type('Backend', (), {'stop_requested': False})()
        policy = AbsoluteHostPolicy(backend, 36000, deadline_epoch=1)
        outcome = policy.evaluate(RunContext('objective', 0, 1, False))
        self.assertEqual((outcome.action, outcome.shutdown), ('stop', 'cancel'))

    def test_rejected_response_is_accounted_and_not_replayed_after_limit(self):
        with TemporaryDirectory() as directory:
            transport = Transport(fail=True)
            backend = RecordedBackend(transport, Tokenizer(), directory, max_calls=1)
            request = GenerationRequest('decision_maker', MODEL, 'choose')
            with self.assertRaisesRegex(ValueError, 'identity'):
                backend.generate(request)
            with self.assertRaises(HostAdmissionStop):
                backend.generate(request)
            self.assertEqual(transport.calls, 1)
            errors = (Path(directory) / 'model_errors.jsonl').read_text().splitlines()
            self.assertEqual(json.loads(errors[0])['call'], 1)
            self.assertEqual(len((Path(directory) / 'model_wire.jsonl').read_text().splitlines()), 1)

    def test_real_child_inherits_only_explicit_host_environment(self):
        with TemporaryDirectory() as directory:
            backend = AllocatedProcessBackend(directory, {'CUDA_VISIBLE_DEVICES': '3,4',
                'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'HF_HUB_OFFLINE': '1'})
            sandbox = backend.create(SandboxSpec('local', 10))
            self.addCleanup(backend.close_all)
            ref = backend.exec(sandbox, ExecRequest((sys.executable, '-c',
                'import os,json;print(json.dumps({k:os.environ.get(k) for k in '
                '["CUDA_VISIBLE_DEVICES","HF_HUB_OFFLINE","PRIVATE_CREDENTIAL"]}))'), 10))
            result = backend.wait(ref)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(json.loads(result.stdout), {'CUDA_VISIBLE_DEVICES': '3,4',
                'HF_HUB_OFFLINE': '1', 'PRIVATE_CREDENTIAL': None})


if __name__ == '__main__':
    unittest.main()
