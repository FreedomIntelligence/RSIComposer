"""Optional LangChain chat transport over the product-v1 FeedbackModel capability.

No provider is constructed here. Importing this module does not import LangChain;
only the explicit factory loads the optional SDK. Tool execution remains with the
calling SDK: this bridge transports declarations, messages and proposed calls.
"""
from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache

from rsisys.loops.contracts.model_service import GenerationRequest, GenerationResponse, ModelRef


def _tool_choice(value):
    if value is None or value is False:
        return None
    if value is True or value == 'any':
        return 'required'
    if isinstance(value, str) and value not in ('auto', 'none', 'required'):
        return {'type': 'function', 'function': {'name': value}}
    return deepcopy(value)


def _json_schema(value):
    """Accept the explicit response-format envelope or a native JSON Schema."""
    if not isinstance(value, dict):
        raise TypeError('response_format requires a JSON Schema object')
    if value.get('type') == 'json_schema':
        envelope = value.get('json_schema')
        if not isinstance(envelope, dict) or not isinstance(envelope.get('schema'), dict):
            raise ValueError('Invalid json_schema response format')
        return deepcopy(envelope['schema'])
    if value.get('type') == 'json_object':
        raise ValueError('Use an explicit JSON Schema instead of provider JSON mode')
    return deepcopy(value)


@lru_cache(maxsize=1)
def _chat_model_type():
    try:
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.messages.utils import convert_to_openai_messages
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.utils.function_calling import convert_to_openai_tool
        from pydantic import Field
    except ImportError as error:
        raise ImportError("LangChain feedback bridge requires pip install 'rsisys[langmem]'") from error
    from typing import Any

    class FeedbackChatModel(BaseChatModel):
        gateway: Any = Field(exclude=True, repr=False)
        model_ref: Any = Field(exclude=True)
        max_output_tokens: int
        default_parallel_tool_calls: bool | None = None

        @property
        def _llm_type(self):
            return 'rsisys-product-v1-feedback'

        @property
        def _identifying_params(self):
            return {'model': asdict(self.model_ref), 'max_output_tokens': self.max_output_tokens}

        def bind_tools(self, tools, *, tool_choice=None, strict=None,
                       parallel_tool_calls=None, **kwargs):
            if strict is not None and type(strict) is not bool:
                raise TypeError('strict must be a boolean')
            if tools and self.gateway.supports_tool_calling is not True:
                raise ValueError('FeedbackModel does not support native tool calling')
            declarations = [convert_to_openai_tool(deepcopy(tool['function'])
                if isinstance(tool, dict) and tool.get('type') == 'function' else tool, strict=strict)
                for tool in tools]
            options = {'tools': declarations, **kwargs}
            choice = _tool_choice(tool_choice)
            if choice is not None:
                options['tool_choice'] = choice
            parallel = (self.default_parallel_tool_calls if parallel_tool_calls is None
                        else parallel_tool_calls)
            if parallel is not None:
                if type(parallel) is not bool:
                    raise TypeError('parallel_tool_calls must be a boolean')
                options['parallel_tool_calls'] = parallel
            return self.bind(**options)

        def with_structured_output(self, schema, *, method='function_calling', include_raw=False, **kwargs):
            if method == 'function_calling':
                return super().with_structured_output(schema, include_raw=include_raw, **kwargs)
            if method != 'json_schema':
                raise ValueError('Feedback bridge supports function_calling or json_schema structured output')
            if kwargs:
                raise ValueError('Unsupported JSON Schema options: ' + ', '.join(sorted(kwargs)))
            if self.gateway.supports_response_schema is not True:
                raise ValueError('FeedbackModel does not support native response schemas')
            from operator import itemgetter
            from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
            from langchain_core.runnables import RunnableMap, RunnablePassthrough
            from langchain_core.utils.pydantic import is_basemodel_subclass
            if isinstance(schema, type) and is_basemodel_subclass(schema):
                native = schema.model_json_schema()
                parser = PydanticOutputParser(pydantic_object=schema)
            else:
                if isinstance(schema, dict):
                    native = (schema['function']['parameters'] if schema.get('type') == 'function'
                              else _json_schema(schema))
                else:
                    native = convert_to_openai_tool(schema)['function']['parameters']
                parser = JsonOutputParser()
            model = self.bind(response_format=deepcopy(native))
            if not include_raw:
                return model | parser
            assign = RunnablePassthrough.assign(parsed=itemgetter('raw') | parser, parsing_error=lambda _: None)
            fallback = RunnablePassthrough.assign(parsed=lambda _: None)
            return RunnableMap(raw=model) | assign.with_fallbacks([fallback], exception_key='parsing_error')

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if self.gateway.model != self.model_ref:
                raise ValueError('FeedbackModel identity changed after LangChain binding')
            declarations = kwargs.pop('tools', ())
            if declarations and self.gateway.supports_tool_calling is not True:
                raise ValueError('FeedbackModel does not support native tool calling')
            response_format = kwargs.pop('response_format', None)
            if response_format is not None and self.gateway.supports_response_schema is not True:
                raise ValueError('FeedbackModel does not support native response schemas')
            limits = [kwargs.pop(name) for name in ('max_tokens', 'max_completion_tokens', 'max_output_tokens')
                      if name in kwargs]
            if len(limits) > 1:
                raise ValueError('Use exactly one output token limit')
            output_limit = limits[0] if limits else self.max_output_tokens
            if type(output_limit) is not int or not 0 < output_limit <= self.max_output_tokens:
                raise ValueError('LangChain request exceeds the FeedbackModel output limit')
            choice = _tool_choice(kwargs.pop('tool_choice', None))
            parallel = kwargs.pop('parallel_tool_calls',
                                  self.default_parallel_tool_calls if declarations else None)
            sampling = {name: kwargs.pop(name) for name in ('temperature', 'top_p', 'seed') if name in kwargs}
            if kwargs:
                raise ValueError('Unsupported FeedbackModel options: ' + ', '.join(sorted(kwargs)))
            wire = convert_to_openai_messages(messages, text_format='string')
            # LangChain's standard converter omits invalid_tool_calls. Trustcall
            # needs those original argument strings in repair conversations too.
            for message, row in zip(messages, wire, strict=True):
                invalid = getattr(message, 'invalid_tool_calls', ())
                if row['role'] == 'assistant' and invalid:
                    calls = list(row.get('tool_calls', ()))
                    calls.extend({'id': call['id'], 'type': 'function',
                        'function': {'name': call['name'], 'arguments': call['args']}}
                        for call in invalid)
                    original = getattr(message, 'additional_kwargs', {}).get('tool_calls', ())
                    positions = {call.get('id'): index for index, call in enumerate(original)}
                    row['tool_calls'] = sorted(calls, key=lambda call: positions.get(call.get('id'), len(positions)))
            request = GenerationRequest('decision_maker', self.model_ref, '',
                max_output_tokens=output_limit, messages=tuple(wire), tools=tuple(declarations),
                tool_choice=choice, parallel_tool_calls=parallel,
                response_schema=_json_schema(response_format) if response_format is not None else None,
                stop=tuple(stop) if stop is not None else None, **sampling)
            response = self.gateway.generate(request)
            if not isinstance(response, GenerationResponse):
                raise TypeError('FeedbackModel.generate must return GenerationResponse')
            if response.model != self.model_ref:
                raise ValueError('FeedbackModel returned a different model identity')
            metadata = {'model': asdict(response.model), 'model_name': response.model.model_id,
                'provider_model': response.provider_model, 'revision_verified': response.revision_verified,
                'simulated': response.simulated, 'finish_reason': response.finish_reason}
            usage_metadata = None
            token_usage = None
            if response.usage is not None:
                usage = response.usage
                usage.__post_init__()
                token_usage = {'prompt_tokens': usage.input_tokens, 'completion_tokens': usage.output_tokens}
                if usage.input_tokens is not None and usage.output_tokens is not None:
                    total = usage.input_tokens + usage.output_tokens
                    token_usage['total_tokens'] = total
                    usage_metadata = {'input_tokens': usage.input_tokens,
                                      'output_tokens': usage.output_tokens, 'total_tokens': total}
                metadata['token_usage'] = deepcopy(token_usage)
            additional = {'tool_calls': deepcopy(list(response.tool_calls))} if response.tool_calls else {}
            message = AIMessage(content=response.text, additional_kwargs=additional,
                                response_metadata=metadata, usage_metadata=usage_metadata)
            output = {'model': asdict(response.model)}
            if token_usage is not None:
                output['token_usage'] = token_usage
            return ChatResult(generations=[ChatGeneration(message=message,
                generation_info={'finish_reason': response.finish_reason})], llm_output=output)

    return FeedbackChatModel


def build_feedback_chat_model(model, *, max_output_tokens=None, parallel_tool_calls=None):
    """Bind a host-issued FeedbackModel; no provider or accounting owner is created."""
    if not isinstance(getattr(model, 'model', None), ModelRef) or not callable(getattr(model, 'generate', None)):
        raise TypeError('LangChain feedback bridge requires a product-v1 FeedbackModel')
    for name in ('supports_tool_calling', 'supports_response_schema'):
        if type(getattr(model, name, None)) is not bool:
            raise TypeError('FeedbackModel must declare boolean ' + name)
    host_limit = getattr(model, 'max_output_tokens', None)
    if type(host_limit) is not int or host_limit < 1:
        raise ValueError('FeedbackModel must declare a positive output limit')
    if max_output_tokens is not None and (type(max_output_tokens) is not int or max_output_tokens < 1):
        raise ValueError('max_output_tokens must be positive')
    if parallel_tool_calls is not None and type(parallel_tool_calls) is not bool:
        raise TypeError('parallel_tool_calls must be a boolean')
    limit = host_limit if max_output_tokens is None else min(host_limit, max_output_tokens)
    return _chat_model_type()(gateway=model, model_ref=model.model, max_output_tokens=limit,
                              default_parallel_tool_calls=parallel_tool_calls, cache=False)
