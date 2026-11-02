"""Local HTTP fixtures: wire audit, bounded recovery and no execution of rejected replies."""
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock

from examples.medical.experiment import BudgetExhausted, DecisionBackend
from rsisys.adapters.serving import VLLMBackend
from rsisys.loops.contracts.model_service import BackendBinding, GenerationRequest, ModelRef
from rsisys.runtime.model import ModelResponseError

MODEL = ModelRef('vllm', 'model', 'revision')
SCHEMA = {'type': 'object', 'properties': {'kind': {'const': 'finish'}, 'reason': {'type': 'string'}},
          'required': ['kind', 'reason'], 'additionalProperties': False}


def reply(content='{"kind":"finish","reason":"fixture"}', finish='stop', **extra):
    return {'model': 'model', 'choices': [{'message': {'content': content}, 'finish_reason': finish}], **extra}


class MedicalResponseRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.responses, self.requests = [], []
        case = self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path == '/tokenize':
                    value = {'count': 42}
                else:
                    case.requests.append(payload)
                    value = case.responses.pop(0)
                body = value if isinstance(value, bytes) else json.dumps(value).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)
        binding = BackendBinding('vllm', 'http', f'http://127.0.0.1:{self.server.server_port}/v1')
        self.serving = VLLMBackend(binding)
        self.backend = DecisionBackend(self.serving, self.root, Mock(), 10)
        self.request = GenerationRequest('decision_maker', MODEL, 'choose', response_schema=SCHEMA,
                                         max_output_tokens=8192)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def rows(self, name):
        return [json.loads(line) for line in (self.root/name).read_text().splitlines()]

    def test_truncation_recovery_preserves_wire_identity_schema_and_budget(self):
        self.responses = [reply('{"kind":"invoke",', 'length'), reply()]
        result = self.backend.generate(self.request)
        self.assertEqual(json.loads(result.text)['kind'], 'finish')
        self.assertEqual(self.backend.calls, 2)
        for key in ('model', 'response_format', 'max_tokens'):
            self.assertEqual(self.requests[0][key], self.requests[1][key])
        self.assertEqual(self.requests[0]['messages'][-1], self.requests[1]['messages'][-1])
        self.assertIn('shorter, complete JSON object', self.requests[1]['messages'][0]['content'])
        self.assertEqual(self.requests[1]['max_tokens'], 8192)
        errors = self.rows('model_response_errors.jsonl')
        self.assertEqual((errors[0]['reason'], errors[0]['finish_reason']), ('output_truncated', 'length'))
        self.assertTrue(errors[0]['retryable'])
        wire = self.rows('model_wire_responses.jsonl')
        self.assertEqual(len(wire), 2)
        self.assertEqual(json.loads(wire[0]['body'])['choices'][0]['message']['content'], '{"kind":"invoke",')
        self.assertEqual([r['call'] for r in self.rows('model_responses.jsonl')], [2])

    def test_format_errors_exhaust_after_three_charged_calls(self):
        self.responses = [reply('not JSON'), reply('[]'), reply('{"kind":"a","kind":"b"}')]
        with self.assertRaises(ModelResponseError) as error:
            self.backend.generate(self.request)
        self.assertEqual(error.exception.reason, 'duplicate_json_key')
        self.assertEqual(self.backend.calls, 3)
        self.assertEqual([r['reason'] for r in self.rows('model_response_errors.jsonl')],
                         ['invalid_json', 'nonobject_json', 'duplicate_json_key'])
        self.assertFalse((self.root/'model_responses.jsonl').exists())

    def test_identity_mismatch_wins_over_truncation_and_never_retries(self):
        self.responses = [reply('unfinished', 'length', model='different')]
        with self.assertRaises(ModelResponseError) as error:
            self.backend.generate(self.request)
        self.assertEqual(error.exception.reason, 'model_identity_mismatch')
        self.assertFalse(error.exception.retryable)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.rows('model_response_errors.jsonl')[0]['finish_reason'], 'length')

    def test_revision_mismatch_never_retries(self):
        self.responses = [reply(model_revision='different')]
        with self.assertRaises(ModelResponseError) as error:
            self.backend.generate(self.request)
        self.assertEqual(error.exception.reason, 'model_revision_mismatch')
        self.assertEqual(len(self.requests), 1)

    def test_recovery_cannot_exceed_original_call_budget(self):
        self.backend.limit = 1
        self.responses = [reply('invalid')]
        with self.assertRaises(BudgetExhausted):
            self.backend.generate(self.request)
        self.assertEqual(self.backend.calls, 1)
        self.assertEqual(len(self.requests), 1)

    def test_recovery_checks_original_deadline(self):
        def deadline():
            if self.backend.calls:
                raise BudgetExhausted('deadline')
        self.backend.check_deadline = deadline
        self.responses = [reply('invalid')]
        with self.assertRaises(BudgetExhausted):
            self.backend.generate(self.request)
        self.assertEqual(len(self.requests), 1)

    def test_wire_is_delivered_before_validation_including_malformed_envelope(self):
        self.responses = [b'not a JSON response']
        observed = []
        with self.assertRaises(ModelResponseError) as error:
            self.serving.generate_with_response_observer(self.request, observed.append)
        self.assertEqual(observed[0]['body'], 'not a JSON response')
        self.assertEqual(error.exception.reason, 'invalid_response_envelope')

    def test_failed_evidence_write_prevents_returning_a_decision(self):
        self.responses = [reply()]
        def failed_write(record):
            raise OSError('audit disk unavailable')
        with self.assertRaisesRegex(OSError, 'audit disk unavailable'):
            self.serving.generate_with_response_observer(self.request, failed_write)
        self.assertEqual(len(self.requests), 1)

    def test_run_executes_only_validated_response_after_recovery(self):
        from tests.test_medical_experiment import MedicalExperimentIntegrationTests
        case = MedicalExperimentIntegrationTests()
        case.setUp()
        try:
            self.responses = [reply('{"kind":"explore","mode":"train",', 'length'), reply()]
            result = case.execute([], decision_model=MODEL, decision_backend=self.serving)
            self.assertEqual(result['status'], 'completed', result)
            self.assertEqual(result['model_calls'], 2)
            self.assertEqual(result['trainings_reserved'], 0)
            self.assertFalse(case.process.requests)
            self.assertEqual(len(self.requests), 2)
        finally:
            case.doCleanups()
