"""Optional LangChain model bridge; inference stays behind the host backend.

No provider client, model selection, cache, implicit retries, or tool execution
lives here. LangMem/Trustcall own their tools and validation loop.
"""
import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, convert_to_openai_messages
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from rsisys.loops.contracts.model_service import GenerationRequest, ModelRef
from rsisys.runtime.model import ModelResponseError, parse_tool_response


class BackendChatModel(BaseChatModel):
    """Use the captured active meta-agent identity in an independent context."""

    backend: Any = Field(exclude=True)
    model_ref: ModelRef
    max_output_tokens: int = Field(default=4096, gt=0)

    @property
    def _llm_type(self):
        return 'rsisys-host-backend'

    @property
    def _identifying_params(self):
        return {'model': self.model_ref.model_id, 'revision': self.model_ref.revision,
                'backend_id': self.model_ref.backend_id}

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        declarations = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice == 'any' or tool_choice is True:
            tool_choice = 'required'
        elif tool_choice is False:
            tool_choice = 'none'
        elif isinstance(tool_choice, str) and tool_choice not in ('auto', 'required', 'none'):
            tool_choice = {'type': 'function', 'function': {'name': tool_choice}}
        return self.bind(tools=declarations, tool_choice=tool_choice, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if stop:
            raise ValueError('The host model contract does not support stop sequences')
        unknown = kwargs.keys() - {'tools', 'tool_choice', 'parallel_tool_calls'}
        if unknown:
            raise ValueError('Unsupported host model options: ' + ', '.join(sorted(unknown)))
        request = GenerationRequest('decision_maker', self.model_ref, '',
            messages=convert_to_openai_messages(messages),
            tools=kwargs.get('tools', ()), tool_choice=kwargs.get('tool_choice'),
            parallel_tool_calls=kwargs.get('parallel_tool_calls'),
            max_output_tokens=self.max_output_tokens)
        # The backend wrapper owns admission, exact-request counting and audit.
        # Never unwrap it to call a provider SDK or a forwarded observer method.
        response = self.backend.generate(request)
        if response.model != request.model:
            raise ModelResponseError('Memory model identity mismatch', reason='model_identity_mismatch')
        calls = response.tool_calls
        if calls and not request.tools:
            raise ModelResponseError('Memory response called an undeclared tool', reason='invalid_tool_call')
        if request.tools:
            _, calls = parse_tool_response(request,
                {'content': response.text, 'tool_calls': list(calls)},
                response.finish_reason, response.provider_model)
        message = AIMessage(content=response.text,
            tool_calls=[{'id': call['id'], 'name': call['function']['name'],
                         'args': json.loads(call['function']['arguments']), 'type': 'tool_call'}
                        for call in calls],
            response_metadata={'provider_model': response.provider_model,
                'revision_verified': response.revision_verified, 'simulated': response.simulated,
                'finish_reason': response.finish_reason})
        usage = response.usage
        llm_output = {'model_name': response.provider_model or self.model_ref.model_id}
        if usage is not None:
            llm_output['token_usage'] = {'prompt_tokens': usage.input_tokens,
                                         'completion_tokens': usage.output_tokens}
            if usage.input_tokens is not None and usage.output_tokens is not None:
                message.usage_metadata = {'input_tokens': usage.input_tokens,
                    'output_tokens': usage.output_tokens,
                    'total_tokens': usage.input_tokens + usage.output_tokens}
        return ChatResult(generations=[ChatGeneration(message=message)], llm_output=llm_output)
