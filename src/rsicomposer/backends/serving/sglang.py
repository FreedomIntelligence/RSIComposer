"""SGLang's native tokenizer and version-checked Chat Completions transport."""

from threading import Lock

from rsicomposer.loops.contracts.model_service import ModelResponseError

from .native_chat import NativeChatBackend


class SGLangBackend(NativeChatBackend):
    provider = 'SGLang'
    root_routes = {'tokenize', 'server_info'}
    adapter_version = '1'

    def __init__(self, binding, *args, **kwargs):
        if not binding.engine_version:
            raise ValueError('SGLang requires an exact engine_version')
        super().__init__(binding, *args, **kwargs)
        self._version_lock = Lock()
        self._version_verified = False

    def _request(self, route, payload=None, *, response_observer=None):
        with self._version_lock:
            if not self._version_verified:
                version = super()._request('server_info').get('version')
                if version != self.binding.engine_version:
                    raise ModelResponseError(f'SGLang version mismatch; expected {self.binding.engine_version}',
                                             reason='engine_version_mismatch')
                self._version_verified = True
        return super()._request(route, payload, response_observer=response_observer)

    def _tokenize_payload(self, request):
        payload = super()._tokenize_payload(request)
        payload.pop('return_token_strs')
        if request.tool_choice is not None:
            payload['tool_choice'] = request.tool_choice
        return payload
