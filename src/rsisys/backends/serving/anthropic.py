"""Non-streaming Anthropic Messages transport; tool execution stays with the caller."""

import json

from rsisys.loops.contracts.model_service import GenerationResponse, ModelResponseError, TokenUsage

from .chat_completions import RemoteModelBackend
from .responses import parse_tool_response


class AnthropicMessagesBackend(RemoteModelBackend):
    kinds = ('anthropic_messages',)
    supports_response_schema = True

    def _auth_headers(self, key):
        return {'anthropic-version': '2023-06-01', **({'x-api-key': key} if key else {})}

    def _payload(self, request):
        self._validate_request(request)
        if self.binding.json_mode and not request.tools and request.response_schema is None:
            raise ValueError('Anthropic Messages requires json_mode=false or an explicit response_schema')
        system, messages = [], []
        for message in request.chat_messages():
            role, content = message['role'], message.get('content')
            if role in ('system', 'developer'):
                if messages:
                    raise ValueError('Anthropic system instructions must precede conversation turns')
                system.append({'type': 'text', 'text': content})
                continue
            if role == 'tool':
                role = 'user'
                blocks = [{'type': 'tool_result', 'tool_use_id': message['tool_call_id'], 'content': content}]
            else:
                blocks = [{'type': 'text', 'text': content}] if content else []
                for call in message.get('tool_calls', ()):
                    if role != 'assistant' or call['type'] != 'function':
                        raise ValueError('Anthropic tool use must be an assistant function call')
                    function = call['function']
                    arguments = json.loads(function['arguments'])
                    if not isinstance(arguments, dict):
                        raise ValueError('Anthropic tool arguments must be an object')
                    blocks.append({'type': 'tool_use', 'id': call['id'], 'name': function['name'], 'input': arguments})
            if not blocks:
                raise ValueError('Anthropic conversation turns must contain text or tool calls')
            if messages and messages[-1]['role'] == role:
                messages[-1]['content'].extend(blocks)
            else:
                messages.append({'role': role, 'content': blocks})
        payload = {'model': request.model.model_id, 'messages': messages}
        if system:
            payload['system'] = system
        if request.tools:
            payload['tools'] = [{
                'name': fn['name'], 'input_schema': fn.get('parameters') or {'type': 'object'},
                **{key: fn[key] for key in ('description', 'strict') if key in fn},
            } for fn in (tool['function'] for tool in request.tools)]
            choice = request.tool_choice
            if choice is not None or request.parallel_tool_calls is not None:
                native = ({'type': 'tool', 'name': choice['function']['name']} if isinstance(choice, dict)
                          else {'type': 'any' if choice == 'required' else choice or 'auto'})
                if native['type'] != 'none' and request.parallel_tool_calls is not None:
                    native['disable_parallel_tool_use'] = not request.parallel_tool_calls
                payload['tool_choice'] = native
        if request.response_schema is not None:
            payload['output_config'] = {'format': {'type': 'json_schema', 'schema': request.response_schema}}
        return payload

    def count_tokens(self, request):
        payload = self._payload(request)
        payload.pop('tool_choice', None)
        data = self._post('messages/count_tokens', payload)
        if not isinstance(data, dict) or type(data.get('input_tokens')) is not int or data['input_tokens'] < 0:
            raise ModelResponseError('Malformed Anthropic token count')
        return data['input_tokens']

    def generate(self, request):
        payload = self._payload(request)
        options = request.sampling_options()
        if 'seed' in options or options.get('temperature', 0) > 1:
            raise ValueError('Anthropic Messages does not support seed or temperature greater than 1')
        if 'stop' in options:
            options['stop_sequences'] = options.pop('stop')
        data = self._post('messages', {**payload, **options, 'max_tokens': request.max_output_tokens})
        try:
            if data['model'] != request.model.model_id:
                raise ModelResponseError('Provider model identity differs from the requested model')
            counts = data['usage']
            usage = TokenUsage(sum(counts.get(key, 0) for key in (
                'input_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens')), counts['output_tokens'])
            finish = {'end_turn': 'stop', 'stop_sequence': 'stop', 'max_tokens': 'length',
                      'tool_use': 'tool_calls'}.get(data['stop_reason'], data['stop_reason'])
            blocks = data['content']
            text, calls = [], []
            for block in blocks:
                if block['type'] == 'text':
                    text.append(block['text'])
                elif block['type'] == 'tool_use':
                    if not isinstance(block['input'], dict):
                        raise ValueError('Invalid tool input')
                    calls.append({'type': 'function', 'id': block['id'], 'function': {
                        'name': block['name'], 'arguments': json.dumps(block['input'], allow_nan=False)}})
                else:
                    raise ModelResponseError('Unsupported Anthropic response content block')
            text = ''.join(text)
            if finish == 'length':
                raise ModelResponseError('Anthropic response reached the output token limit', reason='output_truncated',
                                         finish_reason=finish, provider_model=data['model'], usage=usage)
            if request.tools:
                text, calls = parse_tool_response(request, {'content': text, 'tool_calls': calls}, finish, data['model'])
            elif finish != 'stop' or not text.strip() or calls:
                raise ModelResponseError('Anthropic did not return a complete text response', finish_reason=finish,
                                         provider_model=data['model'], usage=usage)
        except ModelResponseError:
            raise
        except (KeyError, TypeError, ValueError, AttributeError):
            raise ModelResponseError('Malformed Anthropic Messages response') from None
        return GenerationResponse(text, request.model, False, usage, data['model'], tool_calls=tuple(calls),
                                  finish_reason=finish)
