"""Provider constraints are transmitted unchanged; invalid replies never execute."""
import unittest
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
