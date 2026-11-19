"""Loopback Chat Completions gateway with host-issued component capabilities.

Tokens select a host session, never a client-supplied provider or component.
In-process plugins remain trusted Python code, not an isolation boundary.
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import secrets
import threading
import time
from uuid import uuid4

from rsisys.loops.contracts.model_service import GenerationRequest, ModelTransportError, ModelResponseError
from rsisys.runtime.models.accounting import HostAdmissionStop
from rsisys.runtime.models.model_session import HostModelSession
from rsisys.runtime.execution.budget import BudgetExhausted


class ComponentEndpoint:
    def __init__(self, server, component, session, token, attribution_provider=None):
        self._server, self.component, self._session, self._token = server, component, session, token
        self._lock = threading.RLock()
        self._active = {}
        self._revoked = False
        self._children = []
        self._attribution_provider = attribution_provider

    @property
    def base_url(self):
        return self._server.base_url

    def environment(self):
        """Ephemeral worker environment; callers must not persist these values."""
        return {'OPENAI_BASE_URL': self.base_url, 'OPENAI_API_KEY': self._token}

    @contextmanager
    def scope(self, *, decision_id=None, call_id=None):
        scope = {'component': self.component, 'decision_id': decision_id, 'call_id': call_id}
        if any(v is not None and (not isinstance(v, str) or not v) for v in scope.values()):
            raise ValueError('Invalid component inference scope')
        ident = uuid4().hex
        with self._lock:
            if self._revoked:
                raise PermissionError('Component model capability was revoked')
            self._active[ident] = scope
        try:
            yield self
        finally:
            with self._lock:
                self._active.pop(ident, None)

    def _attribution(self):
        with self._lock:
            if self._revoked:
                raise PermissionError('Component model capability was revoked')
            if self._attribution_provider is not None:
                return self._attribution_provider()
            scopes = list(self._active.values())
            if len(scopes) != 1:
                raise PermissionError('Component requires one active host inference scope')
            return dict(scopes[0])

    def generate(self, request):
        return self._session.generate(request, attribution=self._attribution())

    def _tokenize(self, raw):
        self._attribution()
        if (not isinstance(raw, dict) or set(raw) - {'model', 'messages', 'tools', 'add_generation_prompt',
                'return_token_strs', 'chat_template_kwargs'} or raw.get('add_generation_prompt', True) is not True
                or raw.get('return_token_strs', False) is not False
                or raw.get('chat_template_kwargs', {}) not in ({}, {'enable_thinking': False})):
            raise ValueError('Unsupported tokenization request')
        models = [model for model in self._session.models if model.model_id == raw.get('model')]
        if len(models) != 1:
            raise PermissionError('Model is not uniquely registered in this component session')
        request = GenerationRequest('decision_maker' if self.component == 'controller' else 'actor',
            models[0], '', messages=raw['messages'], tools=raw.get('tools', ()))
        count = self._session.count_tokens(request)
        if type(count) is not int or count < 0:
            raise ValueError('Invalid host tokenizer response')
        return {'count': count}

    def revoke(self):
        with self._lock:
            self._revoked = True
            children, self._children = self._children, []
        for child in children:
            child.revoke()
        self._server._revoke(self._token)

    def _delegate(self, raw):
        self._attribution()
        from rsisys.runtime.models.delegation import delegate_session
        session, attribution = delegate_session(self._session, raw)
        with self._lock:
            if self._revoked:
                raise PermissionError('Component model capability was revoked')
            child = self._server.issue(attribution['component'], session,
                                        attribution_provider=lambda: dict(attribution))
            self._children.append(child)
        return {'endpoint': child.base_url, 'token': child.environment()['OPENAI_API_KEY']}

    def _revoke_self(self, raw):
        if raw != {}:
            raise ValueError('Revocation takes no parameters')
        self._attribution()
        self.revoke()
        return {'revoked': True}

    def _complete(self, raw):
        allowed = {'model', 'messages', 'stream', 'n', 'max_tokens', 'max_completion_tokens',
                   'temperature', 'top_p', 'seed', 'stop', 'tools', 'tool_choice', 'parallel_tool_calls',
                   'response_format', 'chat_template_kwargs'}
        if not isinstance(raw, dict) or set(raw) - allowed:
            raise ValueError('Unsupported Chat Completions fields')
        if raw.get('stream', False) is not False or raw.get('n', 1) != 1:
            raise ValueError('Only single non-streaming completions are supported')
        # This provider extension has one supported meaning; never silently drop options.
        if raw.get('chat_template_kwargs', {'enable_thinking': False}) != {'enable_thinking': False}:
            raise ValueError('Unsupported chat template options')
        models = [model for model in self._session.models if model.model_id == raw.get('model')]
        if len(models) != 1:
            raise PermissionError('Model is not uniquely registered in this component session')
        if 'max_tokens' in raw and 'max_completion_tokens' in raw:
            raise ValueError('Choose one output token limit')
        schema = None
        fmt = raw.get('response_format')
        if fmt is not None:
            if not isinstance(fmt, dict):
                raise ValueError('Invalid response_format')
            if fmt.get('type') == 'json_schema':
                schema = fmt['json_schema']['schema']
            elif fmt == {'type': 'json_object'}:
                schema = {'type': 'object', 'additionalProperties': True}
            else:
                raise ValueError('Unsupported response_format')
        options = {key: raw[key] for key in ('temperature', 'top_p', 'seed', 'stop', 'tools',
                   'tool_choice', 'parallel_tool_calls') if key in raw}
        request = GenerationRequest('decision_maker' if self.component == 'controller' else 'actor',
            models[0], '', messages=raw['messages'], response_schema=schema,
            max_output_tokens=raw.get('max_tokens', raw.get('max_completion_tokens', 2048)), **options)
        response = self.generate(request)
        message = {'role': 'assistant', 'content': response.text}
        if response.tool_calls:
            message['tool_calls'] = list(response.tool_calls)
        usage = response.usage
        counts = {'prompt_tokens': usage.input_tokens if usage else None,
                  'completion_tokens': usage.output_tokens if usage else None}
        counts['total_tokens'] = (sum(counts.values()) if all(v is not None for v in counts.values()) else None)
        return {'id': 'chatcmpl-' + uuid4().hex, 'object': 'chat.completion', 'created': int(time.time()),
                'model': response.model.model_id, 'model_revision': response.model.revision,
                'choices': [{'index': 0, 'message': message, 'finish_reason': response.finish_reason or 'stop'}],
                'usage': counts, 'rsisys_simulated': response.simulated}


    def _respond(self, raw):
        """Text/function-call Responses subset used by the pinned upstream workers."""
        allowed = {'model', 'input', 'max_output_tokens', 'temperature', 'top_p', 'tools',
                   'tool_choice', 'parallel_tool_calls', 'stream', 'chat_template_kwargs'}
        if not isinstance(raw, dict) or set(raw) - allowed:
            raise ValueError('Unsupported Responses fields')
        inputs = raw['input']
        if isinstance(inputs, str):
            messages = [{'role': 'user', 'content': inputs}]
        elif isinstance(inputs, list):
            messages = []
            for item in inputs:
                if not isinstance(item, dict) or set(item) - {'type', 'role', 'content'}:
                    raise ValueError('Unsupported Responses input item')
                content = item['content']
                if isinstance(content, list):
                    if any(not isinstance(part, dict) or set(part) != {'type', 'text'}
                           or part['type'] != 'input_text' or not isinstance(part['text'], str) for part in content):
                        raise ValueError('Only text Responses input is supported')
                    content = ''.join(part['text'] for part in content)
                messages.append({'role': item['role'], 'content': content})
        else:
            raise ValueError('Unsupported Responses input')
        chat = {key: value for key, value in raw.items() if key not in
                ('input', 'max_output_tokens', 'tools', 'tool_choice')}
        chat['messages'] = messages
        if 'max_output_tokens' in raw:
            chat['max_tokens'] = raw['max_output_tokens']
        if 'tools' in raw:
            tools = []
            for tool in raw['tools']:
                if not isinstance(tool, dict) or tool.get('type') != 'function':
                    raise ValueError('Only function tools are supported')
                tools.append({'type': 'function', 'function': {k: v for k, v in tool.items() if k != 'type'}})
            chat['tools'] = tools
        choice = raw.get('tool_choice')
        if choice is not None:
            chat['tool_choice'] = ({'type': 'function', 'function': {'name': choice['name']}}
                                   if isinstance(choice, dict) and choice.get('type') == 'function' else choice)
        completion = self._complete(chat)
        message = completion['choices'][0]['message']
        output = []
        if message['content']:
            output.append({'type': 'message', 'id': 'msg-' + uuid4().hex, 'role': 'assistant',
                'status': 'completed', 'content': [{'type': 'output_text', 'text': message['content'], 'annotations': []}]})
        for call in message.get('tool_calls', []):
            output.append({'type': 'function_call', 'id': 'fc-' + uuid4().hex, 'status': 'completed',
                'call_id': call['id'], 'name': call['function']['name'], 'arguments': call['function']['arguments']})
        usage = completion['usage']
        return {'id': 'resp-' + uuid4().hex, 'object': 'response', 'created_at': completion['created'],
                'status': 'completed', 'error': None, 'incomplete_details': None, 'output': output,
                'model': completion['model'], 'usage': {'input_tokens': usage['prompt_tokens'],
                'output_tokens': usage['completion_tokens'], 'total_tokens': usage['total_tokens']}}


class ComponentModelServer:
    def __init__(self, *, max_request_bytes=2 * 1024 * 1024, max_connections=32,
                 bind_host="127.0.0.1", advertise_host="127.0.0.1"):
        if any(type(v) is not int or v < 1 for v in (max_request_bytes, max_connections)):
            raise ValueError('Endpoint limits must be positive integers')
        self._lock, self._tokens, self._closed = threading.RLock(), {}, False
        self._issued_tokens = set()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass  # Neither bearer credentials nor request bodies belong in HTTP logs.

            def reply(self, status, result):
                body = json.dumps(result, allow_nan=False).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                self.connection.settimeout(30)
                if self.path not in ('/v1/chat/completions', '/v1/responses', '/tokenize', '/v1/tokenize',
                                     '/v1/delegate', '/v1/revoke'):
                    self.reply(404, {'error': {'message': 'Unknown model endpoint'}})
                    return
                with owner._lock:
                    authorization = self.headers.get('Authorization', '')
                    credential = owner._tokens.get(authorization[7:]) if authorization.startswith('Bearer ') else None
                if credential is None:
                    self.reply(401, {'error': {'message': 'Invalid component credential'}})
                    return
                try:
                    if self.headers.get('Transfer-Encoding') is not None:
                        raise ValueError('Transfer encoding is not supported')
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= max_request_bytes:
                        self.reply(413, {'error': {'message': 'Invalid request size'}})
                        return
                    raw = json.loads(self.rfile.read(size))
                    result = (credential._delegate(raw) if self.path == '/v1/delegate' else
                              credential._revoke_self(raw) if self.path == '/v1/revoke' else
                              credential._tokenize(raw) if self.path in ('/tokenize', '/v1/tokenize') else
                              credential._respond(raw) if self.path == '/v1/responses' else credential._complete(raw))
                except PermissionError:
                    self.reply(403, {'error': {'message': 'Component inference access denied'}})
                except (HostAdmissionStop, BudgetExhausted):
                    self.reply(429, {'error': {'message': 'Host inference admission closed'}})
                except ModelResponseError as error:
                    import re
                    reason = error.reason if isinstance(error.reason, str) and re.fullmatch(r'[a-z_]{1,64}', error.reason) else 'provider_rejected'
                    self.reply(502, {'error': {'message': 'Host model provider rejected its response',
                        'type': 'ModelResponseError', 'reason': reason, 'retryable': bool(error.retryable)}})
                except ModelTransportError:
                    self.reply(502, {'error': {'message': 'Host model provider failed'}})
                except (ValueError, TypeError, KeyError):
                    self.reply(400, {'error': {'message': 'Invalid Chat Completions request'}})
                except Exception:
                    self.reply(500, {'error': {'message': 'Host model request failed'}})
                else:
                    self.reply(200, result)

        class Server(ThreadingHTTPServer):
            daemon_threads = False
            slots = threading.BoundedSemaphore(max_connections)

            def process_request(self, request, client_address):
                if not self.slots.acquire(blocking=False):
                    self.shutdown_request(request)
                    return
                try:
                    super().process_request(request, client_address)
                except BaseException:
                    self.slots.release()
                    raise

            def process_request_thread(self, request, client_address):
                try:
                    super().process_request_thread(request, client_address)
                finally:
                    self.slots.release()

        if (not isinstance(bind_host, str) or not bind_host or not isinstance(advertise_host, str)
                or not advertise_host or any(c in advertise_host for c in '/:@?#')):
            raise ValueError('Model service requires bind and advertised host names')
        self.server = Server((bind_host, 0), Handler)
        self.base_url = f'http://{advertise_host}:{self.server.server_port}/v1'
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=.05), daemon=True)
        self.thread.start()

    @property
    def secret_values(self):
        """Issued credentials stay restricted even after access is revoked."""
        with self._lock:
            return tuple(self._issued_tokens)

    def issue(self, component, session, *, attribution_provider=None):
        if not isinstance(component, str) or not component or not isinstance(session, HostModelSession):
            raise TypeError('Component endpoint requires a named component and HostModelSession')
        with self._lock:
            if self._closed:
                raise RuntimeError('Component model server is closed')
            token = secrets.token_urlsafe(32)
            self._issued_tokens.add(token)
            credential = ComponentEndpoint(self, component, session, token, attribution_provider)
            self._tokens[token] = credential
            return credential

    def worker_environment(self, component, supplied):
        """Forward only a live capability issued for this worker's component."""
        if not supplied or supplied.get('OPENAI_BASE_URL') != self.base_url:
            return {}
        token = supplied.get('OPENAI_API_KEY')
        with self._lock:
            credential = self._tokens.get(token) if isinstance(token, str) else None
            if self._closed or credential is None or credential.component != component:
                return {}
        try:
            credential._attribution()
        except PermissionError:
            return {}
        return {'OPENAI_BASE_URL': self.base_url, 'OPENAI_API_KEY': token}

    def _revoke(self, token):
        with self._lock:
            self._tokens.pop(token, None)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for credential in self._tokens.values():
                credential._revoked = True
            self._tokens.clear()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
