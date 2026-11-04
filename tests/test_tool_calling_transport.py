"""Native function calls cross HTTP as data; the transport never runs tools."""
from copy import deepcopy
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest

from rsisys.adapters.serving import VLLMBackend
from rsisys.loops.contracts.model_service import BackendBinding, GenerationRequest, ModelRef
from rsisys.loops.decision_making.feedback.learning import SerializedLearningBackend
from rsisys.runtime.model import RemoteModelBackend, ModelResponseError


MODEL = ModelRef('native', 'meta-agent', 'weights-v1')
TOOLS = [{'type': 'function', 'function': {'name': 'remember', 'description': 'Record a memory',
    'parameters': {'type': 'object', 'properties': {'content': {'type': 'string'}},
                   'required': ['content']}}}]


def tool_call(ident='call-1', arguments='{"content":"new evidence"}', name='remember'):
    return {'id': ident, 'type': 'function', 'function': {'name': name, 'arguments': arguments}}


def reply(calls=None, *, text=None, finish='tool_calls'):
    message = {'role': 'assistant', 'content': text}
    if calls is not None:
        message['tool_calls'] = calls
    return {'model': MODEL.model_id, 'model_revision': MODEL.revision,
            'choices': [{'message': message, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 41, 'completion_tokens': 12}}


class NativeToolTransportTests(unittest.TestCase):
    def setUp(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.server.requests.append((self.path, payload))
                body = json.dumps({'count': 99} if self.path == '/tokenize' else self.server.reply).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.requests = []
        self.server.reply = reply([tool_call()])
        self.worker = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.01), daemon=True)
        self.worker.start()
        self.binding = BackendBinding('native', 'http', f'http://127.0.0.1:{self.server.server_port}/v1')
        self.backends = [RemoteModelBackend(self.binding), VLLMBackend(self.binding)]
        self.request = GenerationRequest('decision_maker', MODEL, 'Organize this feedback.',
                                         tools=TOOLS, tool_choice='auto', parallel_tool_calls=False)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(timeout=5)

    def test_native_request_and_null_content_response_through_both_http_adapters(self):
        for backend in self.backends:
            with self.subTest(backend=type(backend).__name__):
                response = backend.generate(self.request)
                path, payload = self.server.requests[-1]
                self.assertEqual(path, '/v1/chat/completions')
                self.assertEqual(payload['tools'], TOOLS)
                self.assertEqual(payload['tool_choice'], 'auto')
                self.assertFalse(payload['parallel_tool_calls'])
                self.assertNotIn('response_format', payload)
                self.assertEqual(response.text, '')
                self.assertEqual(response.tool_calls, (tool_call(),))
                self.assertEqual(response.finish_reason, 'tool_calls')
                self.assertEqual(response.model, MODEL)
                self.assertTrue(response.revision_verified)
                self.assertEqual(response.usage.input_tokens, 41)
        self.assertEqual(len(self.server.requests), 2)  # No execution or implicit model loop.

    def test_parallel_calls_then_tool_results_round_trip_without_losing_ids_or_arguments(self):
        calls = [tool_call(arguments=' { "content": "line\\nquoted \\\"value\\\"" } '), tool_call('call-2')]
        initial = [{'role': 'system', 'content': 'Organize memory.'},
                   {'role': 'user', 'content': 'New feedback.'}]
        request = replace(self.request, prompt='', messages=initial, parallel_tool_calls=True)
        for backend in self.backends:
            with self.subTest(backend=type(backend).__name__):
                self.server.reply = reply(calls)
                first = backend.generate(request)
                self.assertEqual(list(first.tool_calls), calls)
                history = initial + [{'role': 'assistant', 'content': None, 'tool_calls': list(first.tool_calls)},
                    {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'recorded'},
                    {'role': 'tool', 'tool_call_id': 'call-2', 'content': 'merged'}]
                self.server.reply = reply(text='Memory organized.', finish='stop')
                final = backend.generate(replace(request, messages=history))
                self.assertEqual(self.server.requests[-1][1]['messages'], history)
                self.assertEqual(final.text, 'Memory organized.')
                self.assertFalse(final.tool_calls)
                self.assertEqual(final.finish_reason, 'stop')

    def test_required_named_and_none_choices_are_transmitted(self):
        for backend in self.backends:
            for choice in ('required', {'type': 'function', 'function': {'name': 'remember'}}, 'none'):
                with self.subTest(backend=type(backend).__name__, choice=choice):
                    self.server.reply = (reply(text='No call.', finish='stop') if choice == 'none'
                                         else reply([tool_call()]))
                    backend.generate(replace(self.request, tool_choice=choice))
                    self.assertEqual(self.server.requests[-1][1]['tool_choice'], choice)

    def test_unknown_malformed_truncated_or_disallowed_calls_are_not_executed_or_retried(self):
        cases = [
            (reply([tool_call(name='run_shell')]), {}, 'unexpected_tool_name'),
            (reply([tool_call(arguments='not json')]), {}, 'invalid_tool_arguments'),
            (reply([tool_call(arguments='[]')]), {}, 'invalid_tool_arguments'),
            (reply([tool_call(arguments='{"content":1,"content":2}')]), {}, 'invalid_tool_arguments'),
            (reply([tool_call(arguments='{"content":NaN}')]), {}, 'invalid_tool_arguments'),
            (reply([tool_call(), tool_call()]), {'parallel_tool_calls': True}, 'malformed_tool_call'),
            (reply([tool_call(), tool_call('call-2')]), {}, 'unexpected_parallel_tool_calls'),
            (reply([tool_call()]), {'tool_choice': 'none'}, 'unexpected_tool_calls'),
            (reply(text='No call.', finish='stop'), {'tool_choice': 'required'}, 'missing_tool_calls'),
            (reply(text='No call.', finish='stop'),
             {'tool_choice': {'type': 'function', 'function': {'name': 'remember'}}}, 'missing_tool_calls'),
            (reply([], text='No calls.'), {}, 'missing_tool_calls'),
            (reply({}, text='Invalid calls.', finish='stop'), {}, 'malformed_tool_calls'),
            (reply([tool_call()], finish='length'), {}, 'output_truncated'),
        ]
        for backend in self.backends:
            for response, options, reason in cases:
                with self.subTest(backend=type(backend).__name__, reason=reason):
                    self.server.reply = response
                    count = len(self.server.requests)
                    with self.assertRaises(ModelResponseError) as caught:
                        backend.generate(replace(self.request, **options))
                    self.assertEqual(caught.exception.reason, reason)
                    self.assertEqual(len(self.server.requests), count + 1)

    def test_identity_checks_precede_native_call_parsing(self):
        for backend in self.backends:
            for overrides in ({'model': 'other-model'}, {'model_revision': 'other-weights'}):
                with self.subTest(backend=type(backend).__name__, overrides=overrides):
                    self.server.reply = reply([tool_call(arguments='{')], finish='length') | overrides
                    with self.assertRaises(ModelResponseError) as caught:
                        backend.generate(self.request)
                    self.assertFalse(caught.exception.retryable)

    def test_legacy_text_request_still_rejects_unrequested_calls(self):
        for backend in self.backends:
            with self.subTest(backend=type(backend).__name__):
                with self.assertRaises(ModelResponseError):
                    backend.generate(GenerationRequest('decision_maker', MODEL, 'Return a JSON decision.'))
                self.assertEqual(self.server.requests[-1][1]['response_format'], {'type': 'json_object'})
                self.assertNotIn('tools', self.server.requests[-1][1])

    def test_tokenization_includes_tools_and_the_same_multi_turn_history(self):
        history = [{'role': 'system', 'content': 'Organize memory.'},
                   {'role': 'user', 'content': 'New feedback.'}]
        request = replace(self.request, prompt='', messages=history)
        backend = VLLMBackend(self.binding, generation_options={
            'chat_template_kwargs': {'enable_thinking': False}, 'chat_template': 'fixture-template'})
        self.assertEqual(backend.count_tokens(request), 99)
        backend.generate(request)
        path, token_payload = self.server.requests[-2]
        generation = self.server.requests[-1][1]
        self.assertEqual(path, '/tokenize')
        for key in ('model', 'messages', 'tools', 'chat_template', 'chat_template_kwargs'):
            self.assertEqual(token_payload[key], generation[key])
        self.assertNotIn('response_format', token_payload)
        self.assertTrue(token_payload['add_generation_prompt'])

    def test_schema_or_native_option_conflicts_fail_before_http(self):
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            replace(self.request, response_schema={'type': 'object'})
        for name in ('tools', 'tool_choice', 'parallel_tool_calls', 'response_format', 'structured_outputs'):
            with self.subTest(name=name):
                backend = VLLMBackend(self.binding, generation_options={name: {}})
                with self.assertRaisesRegex(ValueError, 'conflict'):
                    backend.generate(self.request)
        self.assertFalse(self.server.requests)

    def test_inputs_and_responses_are_detached_and_request_can_round_trip_json(self):
        tools, history = deepcopy(TOOLS), [{'role': 'user', 'content': 'feedback'}]
        choice = {'type': 'function', 'function': {'name': 'remember'}}
        request = replace(self.request, prompt='', tools=tools, messages=history, tool_choice=choice)
        tools[0]['function']['name'] = 'changed'
        history[0]['content'] = 'changed'
        choice['function']['name'] = 'changed'
        self.assertEqual(request.tools[0]['function']['name'], 'remember')
        self.assertEqual(request.messages[0]['content'], 'feedback')
        self.assertEqual(request.tool_choice['function']['name'], 'remember')
        wire = json.loads(json.dumps(asdict(request)))
        wire['model'] = ModelRef(**wire['model'])
        self.assertEqual(GenerationRequest(**wire), request)
        response = self.backends[0].generate(request)
        self.server.reply['choices'][0]['message']['tool_calls'][0]['function']['name'] = 'changed'
        self.assertEqual(response.tool_calls[0]['function']['name'], 'remember')

    def test_invalid_request_inputs_are_rejected(self):
        cases = [
            {'messages': []}, {'messages': [{'role': 'user', 'content': 'ambiguous'}]},
            {'prompt': '', 'messages': [{'role': 'unknown', 'content': 'text'}]},
            {'prompt': '', 'messages': [{'role': 'tool', 'content': 'result'}]},
            {'tools': [TOOLS[0], TOOLS[0]]}, {'tools': [{'type': 'shell'}]},
            {'tool_choice': 'remember'},
            {'tool_choice': {'type': 'function', 'function': {'name': 'unknown'}}},
            {'tools': (), 'tool_choice': None, 'parallel_tool_calls': True},
        ]
        for options in cases:
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                replace(self.request, **options)

    def test_shared_feedback_gateway_retains_native_calls_and_capability(self):
        backend = SerializedLearningBackend(self.backends[1])
        self.assertTrue(backend.supports_tool_calling)
        self.assertEqual(backend.count_tokens(self.request), 99)
        self.assertEqual(backend.generate(self.request).tool_calls, (tool_call(),))
        self.assertEqual([path for path, _ in self.server.requests], ['/tokenize', '/v1/chat/completions'])

    def test_wire_observer_still_captures_native_response(self):
        evidence = []
        response = self.backends[1].generate_with_response_observer(self.request, evidence.append)
        self.assertEqual(len(evidence), 1)
        captured = json.loads(evidence[0]['body'])
        self.assertEqual(tuple(captured['choices'][0]['message']['tool_calls']), response.tool_calls)


if __name__ == '__main__':
    unittest.main()
