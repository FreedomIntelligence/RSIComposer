"""Provider constraints are transmitted unchanged; invalid replies never execute."""
import unittest
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dataclasses import replace
from unittest.mock import patch
from rsisys.adapters.serving import VLLMBackend
from rsisys.runtime.model import RemoteModelBackend, ModelResponseError, ModelTransportError
from rsisys.loops.contracts.model_service import BackendBinding, GenerationRequest, ModelRef

MODEL = ModelRef('vllm', 'model', 'revision')
SCHEMA = {'type': 'object', 'properties': {'kind': {'const': 'finish'}},
          'required': ['kind'], 'additionalProperties': False}

class StructuredTransportTests(unittest.TestCase):
    def setUp(self):
        self.binding = BackendBinding('vllm', 'http', 'http://127.0.0.1:1/v1')
        self.backend = VLLMBackend(self.binding)
        self.request = GenerationRequest('decision_maker', MODEL, 'choose', response_schema=SCHEMA)

    def reply(self, content='{"kind":"finish"}', finish='stop'):
        return {'model': 'model', 'choices': [{'message': {'content': content}, 'finish_reason': finish}]}

    def test_schema_transmitted_with_native_constraint_and_identity_unchanged(self):
        with patch.object(self.backend, '_request', return_value=self.reply()) as call:
            response = self.backend.generate(self.request)
        payload = call.call_args.args[1]
        self.assertEqual(payload['response_format'], {'type': 'json_schema', 'json_schema': {
            'name': 'rsisys_decision', 'strict': True, 'schema': SCHEMA}})
        self.assertEqual(response.model, MODEL)
        self.assertFalse(response.revision_verified)
        self.assertNotIn('tools', payload)

    def test_portable_schema_is_explicit_preserves_structure_and_literal_data(self):
        schema = {'type': 'object', 'properties': {
            'text': {'type': ['string', 'null'], 'pattern': r'\S', 'minLength': 1, 'maxLength': 256},
            'ids': {'type': 'array', 'minItems': 1, 'uniqueItems': True,
                    'items': {'type': 'string', 'minLength': 1}},
            'steps': {'type': 'integer', 'minimum': 1, 'maximum': 64},
            'literal': {'const': {'pattern': r'\S', 'minLength': 1}},
        }, 'required': ['text', 'ids', 'steps'], 'additionalProperties': False,
           'anyOf': [{'required': ['literal']}, {'properties': {'steps': {'minimum': 2}}}]}
        request = replace(self.request, response_schema=schema)
        portable = VLLMBackend(self.binding, native_schema_mode='portable')
        with patch.object(portable, '_request', return_value=self.reply()) as call:
            portable.generate(request)
        wire = call.call_args.args[1]['response_format']['json_schema']['schema']
        self.assertEqual(wire['properties']['text'], {'type': ['string', 'null']})
        self.assertEqual(wire['properties']['ids'], {'type': 'array', 'minItems': 1, 'items': {'type': 'string'}})
        self.assertEqual(wire['properties']['steps'], schema['properties']['steps'])
        self.assertEqual(wire['properties']['literal'], schema['properties']['literal'])
        self.assertEqual(wire['required'], schema['required'])
        self.assertFalse(wire['additionalProperties'])
        self.assertNotIn('anyOf', wire)
        self.assertEqual(list(wire['properties']), list(schema['properties']))
        self.assertEqual(request.response_schema, schema)
        self.assertEqual(self.backend.native_response_schema(schema), schema)
        envelope = {'oneOf': [schema, SCHEMA]}
        self.assertEqual(len(portable.native_response_schema(envelope)['oneOf']), 2)
        for value in ('fallback', None, True):
            with self.assertRaises(ValueError):
                VLLMBackend(self.binding, native_schema_mode=value)

    def test_real_http_preserves_schema_order_in_requests_and_native_options(self):
        received = []
        reply = self.reply()
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(self.rfile.read(int(self.headers['Content-Length'])))
                body = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        schema = {'oneOf': [
            {'type': 'object', 'properties': {
                'kind': {'const': 'invoke'}, 'operation': {'const': 'inspect'},
                'arguments': {'type': 'object', 'properties': {
                    'zeta': {'type': 'string'}, 'alpha': {'type': 'integer'}}},
                'reason': {'type': 'string'}}, 'required': ['kind', 'operation', 'arguments']},
            SCHEMA]}
        binding = replace(self.binding, endpoint=f'http://127.0.0.1:{server.server_port}/v1')
        try:
            VLLMBackend(binding).generate(replace(self.request, response_schema=schema))
            backend = VLLMBackend(binding, generation_options={'structured_outputs': {'json': schema}})
            backend.generate(replace(self.request, response_schema=None))
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)
        self.assertEqual(len(received), 2)
        for raw, path in zip(received, ('response_format', 'structured_outputs')):
            payload = json.loads(raw)
            wire = (payload[path]['json_schema']['schema'] if path == 'response_format'
                    else payload[path]['json'])
            self.assertEqual(wire, schema)
            self.assertEqual(list(wire['oneOf'][0]['properties']),
                             ['kind', 'operation', 'arguments', 'reason'])
            self.assertEqual(list(wire['oneOf'][0]['properties']['arguments']['properties']),
                             ['zeta', 'alpha'])

    def test_token_counter_uses_root_endpoint_and_the_same_chat_template(self):
        received = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                body = b'{"count": 42}'
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args): pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        backend = VLLMBackend(replace(self.binding, endpoint=f'http://127.0.0.1:{server.server_port}/v1'),
            generation_options={'chat_template_kwargs': {'enable_thinking': False}})
        try:
            self.assertEqual(backend.count_tokens(replace(self.request, system='Research instructions')), 42)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=5)
        path, payload = received[0]
        self.assertEqual(path, '/tokenize')
        self.assertEqual(payload['messages'], [{'role':'system','content':'Research instructions'},
                                               {'role':'user','content':'choose'}])
        self.assertTrue(payload['add_generation_prompt'])
        self.assertEqual(payload['chat_template_kwargs'], {'enable_thinking':False})
        for bad in (True, -1, '42', None):
            with patch.object(backend, '_request', return_value={'count': bad}), self.assertRaises(ModelResponseError):
                backend.count_tokens(self.request)

    def test_token_count_response_has_independent_bounded_capacity(self):
        received = []
        reply = json.dumps({'count': 300, 'tokens': [123456] * 300}).encode()
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                self.send_response(200)
                self.send_header('Content-Length', str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)
            def log_message(self, *args): pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        binding = replace(self.binding, endpoint=f'http://127.0.0.1:{server.server_port}/v1',
                          max_response_bytes=1024)
        try:
            backend = VLLMBackend(binding, max_tokenize_response_bytes=4096)
            self.assertEqual(backend.count_tokens(self.request), 300)
            self.assertFalse(received[-1]['return_token_strs'])
            self.assertNotIn('return_tokens', received[-1])
            with self.assertRaisesRegex(ModelResponseError, 'size limit'):
                backend.generate(self.request)
            with self.assertRaisesRegex(ModelResponseError, 'size limit'):
                VLLMBackend(binding, max_tokenize_response_bytes=1024).count_tokens(self.request)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=5)
        for value in (True, 0, -1, '4096'):
            with self.assertRaises(ValueError):
                VLLMBackend(binding, max_tokenize_response_bytes=value)

    def test_unstructured_requests_retain_existing_json_mode(self):
        with patch.object(self.backend, '_request', return_value=self.reply()) as call:
            self.backend.generate(replace(self.request, response_schema=None))
        self.assertEqual(call.call_args.args[1]['response_format'], {'type': 'json_object'})

    def test_server_rejection_never_retries_without_schema(self):
        with patch.object(self.backend, '_request', side_effect=ModelTransportError('HTTP 400')) as call:
            with self.assertRaises(ModelTransportError): self.backend.generate(self.request)
        self.assertEqual(call.call_count, 1)

    def test_native_options_cannot_override_request_constraint(self):
        for key in ('response_format','structured_outputs','tools','tool_choice','guided_json'):
            backend = VLLMBackend(self.binding, generation_options={key: {}})
            with self.subTest(key=key), patch.object(backend, '_request') as call:
                with self.assertRaisesRegex(ValueError, 'conflicts'): backend.generate(self.request)
                call.assert_not_called()

    def test_truncation_duplicate_fields_invalid_json_and_nonobjects_rejected(self):
        for content, finish in [('{"kind":"finish"}', 'length'),
                ('{"kind":"invoke","kind":"finish"}', 'stop'),
                ('{"arguments":{"kind":"a","kind":"b"}}', 'stop'),
                ('[]', 'stop'), ('garbage', 'stop')]:
            with self.subTest(content=content, finish=finish), patch.object(self.backend, '_request',
                    return_value=self.reply(content, finish)):
                with self.assertRaises(ModelResponseError): self.backend.generate(self.request)

    def test_request_copies_schema_and_generic_backend_fails_explicitly(self):
        schema={'type':'object','properties':{'kind':{'const':'finish'}}}
        request=replace(self.request,response_schema=schema)
        schema['properties']['kind']['const']='altered'
        self.assertEqual(request.response_schema['properties']['kind']['const'],'finish')
        with self.assertRaises(NotImplementedError): RemoteModelBackend(self.binding).generate(request)
        with self.assertRaises(TypeError): replace(request,response_schema=[])
        with self.assertRaises(ValueError): replace(request,response_schema={'x':float('nan')})

    def test_medical_wrapper_preserves_schema_and_serial_rules(self):
        from tests.test_medical_experiment import MedicalExperimentIntegrationTests, FixtureResearcher, FINISH
        import json
        case = MedicalExperimentIntegrationTests()
        case.setUp()
        try:
            backend = FixtureResearcher(MODEL, case.process, [FINISH])
            backend.supports_response_schema = True
            result = case.execute([], decision_model=MODEL, decision_backend=backend)
            self.assertEqual(result['status'], 'completed', result)
            request = backend.requests[0]
            branches = request.response_schema['oneOf']
            self.assertFalse(any(b['properties']['kind']['const'] == 'wait' for b in branches))
            train = next(b for b in branches if b['properties'].get('mode', {}).get('const') == 'train')
            self.assertIn('input_ids', train['required'])
            self.assertEqual(train['properties']['input_ids']['minItems'], 1)
            self.assertNotIn('review_after_seconds', train['properties'])
            recorded = json.loads((case.root/'run/model_requests.jsonl').read_text())
            self.assertEqual(recorded['request']['response_schema'], request.response_schema)
            for logged, original in zip(recorded['request']['response_schema']['oneOf'], branches):
                self.assertEqual(list(logged['properties']), list(original['properties']))
            self.assertEqual(recorded['request']['max_output_tokens'], request.max_output_tokens)
            self.assertFalse(case.process.requests)
        finally:
            case.doCleanups()

    def test_medical_normalization_preserves_public_rationale(self):
        from tests.test_medical_experiment import MedicalExperimentIntegrationTests, FINISH
        case = MedicalExperimentIntegrationTests()
        case.setUp()
        try:
            result = case.execute([{'kind': 'invoke', 'operation': 'preview_training_data',
                'arguments': {'offset': 0, 'limit': 1}, 'reason': 'Inspect a sample before choosing training.'}, FINISH])
            self.assertEqual(result['status'], 'completed', result)
            self.assertEqual(result['model_calls'], 2)
            self.assertFalse(case.process.requests)
        finally:
            case.doCleanups()

if __name__ == '__main__': unittest.main()
