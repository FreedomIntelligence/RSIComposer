"""Native tool-call envelope validation shared by model transports and SDK adapters."""
import json

from rsicomposer.loops.contracts.model_service import ModelResponseError


def parse_tool_response(request, message, finish_reason, provider_model):
    """Validate a native function-call envelope without executing its contents.

    Argument schemas and tool effects belong to the consuming library/harness.
    Keep argument strings verbatim for native SDKs and request/response evidence.
    """
    def reject(reason, *, retryable=False):
        raise ModelResponseError('Rejected native tool response: ' + reason,
            reason=reason, retryable=retryable, finish_reason=finish_reason,
            provider_model=provider_model)

    if not isinstance(message, dict):
        reject('malformed_message', retryable=True)
    if message.get('refusal'):
        reject('refusal')
    if finish_reason == 'length':
        reject('output_truncated', retryable=True)
    if finish_reason not in ('stop', 'tool_calls'):
        reject('unexpected_finish_reason')
    calls = message.get('tool_calls')
    calls = [] if calls is None else calls
    if not isinstance(calls, list):
        reject('malformed_tool_calls', retryable=True)
    text = message.get('content')
    if text is not None and not isinstance(text, str):
        reject('empty_or_invalid_content', retryable=True)
    if not calls:
        if (finish_reason == 'tool_calls' or request.tool_choice == 'required'
                or isinstance(request.tool_choice, dict)):
            reject('missing_tool_calls', retryable=True)
        if not isinstance(text, str) or not text.strip():
            reject('empty_or_invalid_content', retryable=True)
        return text, ()
    if request.tool_choice == 'none':
        reject('unexpected_tool_calls')
    if request.parallel_tool_calls is False and len(calls) > 1:
        reject('unexpected_parallel_tool_calls')
    names = {tool['function']['name'] for tool in request.tools}
    selected = (request.tool_choice['function']['name']
                if isinstance(request.tool_choice, dict) else None)
    seen = set()

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate argument field')
            value[key] = item
        return value

    def invalid_constant(value):
        raise ValueError('Nonfinite JSON argument')

    for call in calls:
        function = call.get('function') if isinstance(call, dict) else None
        if (not isinstance(function, dict) or call.get('type') != 'function'
                or not isinstance(call.get('id'), str) or not call['id'].strip()
                or call['id'] in seen or not isinstance(function.get('name'), str)
                or not isinstance(function.get('arguments'), str)):
            reject('malformed_tool_call', retryable=True)
        seen.add(call['id'])
        if function.get('name') not in names or (selected and function['name'] != selected):
            reject('unexpected_tool_name')
        try:
            arguments = json.loads(function['arguments'], object_pairs_hook=unique_object,
                                   parse_constant=invalid_constant)
            if not isinstance(arguments, dict):
                raise ValueError('Function arguments must be an object')
        except ValueError:
            reject('invalid_tool_arguments', retryable=True)
    return text or '', tuple(calls)
