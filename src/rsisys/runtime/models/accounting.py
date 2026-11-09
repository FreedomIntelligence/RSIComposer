"""Shared host model session. Reservations precede every provider attempt.

Only host-owned transports carry credentials. Untrusted workers use the broker
and an executor without direct network access; in-process plugins are trusted
host code and are not a security boundary.
"""
import fcntl
import json
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path

from rsisys.loops.contracts.model_service import ModelResponseError
from rsisys.runtime.storage.files import atomic_write, json_bytes


class HostAdmissionStop(RuntimeError):
    pass


def write(path, value):
    atomic_write(Path(path), json_bytes(value))


def append(path, value):
    with Path(path).open('ab') as stream:
        stream.write(json_bytes(value) + b'\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())


class AccountedModelBackend:
    """One account for decisions, feedback, summaries and rejected responses."""
    supports_tool_calling = True
    supports_response_schema = True
    token_count_mode = 'local_official_chat_template'

    @property
    def calls(self):
        path = self.root / 'model_calls.json'
        return json.loads(path.read_text())['attempted'] if path.exists() else self._calls

    @calls.setter
    def calls(self, value):
        self._calls = value

    def __init__(self, transport, tokenizer, root, *, max_calls, max_input_tokens=56000,
                 template_tool_normalizer=None, provider_token_counter=False, resume=False, allowed_models=None,
                 token_counter=None, response_attempts=1):
        self.transport, self.tokenizer, self.root = transport, tokenizer, Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.allowed_models = tuple(allowed_models) if allowed_models is not None else None
        self.template_tool_normalizer = template_tool_normalizer
        self.provider_token_counter = provider_token_counter
        if token_counter is not None and not callable(token_counter):
            raise TypeError('token_counter must be callable')
        if type(response_attempts) is not int or not 1 <= response_attempts <= 8:
            raise ValueError('response_attempts must be between 1 and 8')
        self.token_counter, self.response_attempts = token_counter, response_attempts
        if provider_token_counter:
            self.token_count_mode = 'provider_tokenizer'
        self.max_calls, self.max_input_tokens = max_calls, max_input_tokens
        if any(type(value) is not int or value < 1 for value in (max_calls, max_input_tokens)):
            raise ValueError('Host model call and input budgets must be positive integers')
        self.calls, self.deadline = 0, None
        self.stop_requested = False
        self.pause_requested = False
        self.lock = threading.RLock()
        with (self.root / 'model-account.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if resume:
                ledger = json.loads((self.root / 'model_calls.json').read_text())
                if (ledger['limit'] != max_calls or type(ledger['attempted']) is not int
                        or not 0 <= ledger['attempted'] <= max_calls):
                    raise ValueError('Cannot change or reset the recovered model-call budget')
                self.calls = ledger['attempted']
            else:
                if (self.root / 'model_calls.json').exists():
                    raise ValueError('Existing model account cannot be reset; resume it explicitly')
                write(self.root / 'model_calls.json', {'attempted': 0, 'limit': max_calls})

    def count_tokens(self, request):
        if self.token_counter is not None:
            return self.token_counter(request)
        if self.provider_token_counter:
            return self.transport.count_tokens(request)
        tools = (self.template_tool_normalizer(request) if self.template_tool_normalizer
                 else list(request.tools))
        kwargs = {'tools': tools} if tools else {}
        tokens = self.tokenizer.apply_chat_template(request.chat_messages(), tokenize=True,
            add_generation_prompt=True, enable_thinking=False, **kwargs)
        # Transformers 5 returns BatchEncoding (a Mapping, not a dict). Its
        # length counts fields, usually 2, rather than prompt tokens.
        return len(tokens['input_ids'] if isinstance(tokens, Mapping) else tokens)

    def generate(self, request):
        with self.lock:
            original = request
            for attempt in range(self.response_attempts):
                try:
                    return self._generate(request, response_attempt=attempt + 1)
                except ModelResponseError as exc:
                    # Native memory tools own their validation/recovery lifecycle.
                    # Only retry rejected decision JSON, before any action exists.
                    if (not exc.retryable or attempt == self.response_attempts - 1 or original.tools
                            or original.role != 'decision_maker' or original.response_schema is None):
                        raise
                    correction = ('The previous response exceeded the output limit. Return a shorter, complete JSON object.'
                                  if exc.reason == 'output_truncated' else
                                  'The previous response failed format validation. Return one complete valid JSON object.')
                    text = correction + ' Preserve the required schema and evidence; do not execute or assume any previous action.'
                    if original.messages is not None:
                        messages = original.chat_messages()
                        if messages[0]['role'] == 'system':
                            messages[0]['content'] += '\n' + text
                        else:
                            messages.insert(0, {'role': 'system', 'content': text})
                        request = replace(original, messages=messages)
                    else:
                        request = replace(original, system=original.system + '\n' + text)
            raise AssertionError('Unreachable response recovery loop')

    def _generate(self, request, *, response_attempt):
        with self.lock:
            if self.allowed_models is not None and request.model not in self.allowed_models:
                raise PermissionError("Model is outside the host session bindings")
            if self.stop_requested or self.calls >= self.max_calls or (self.deadline is not None and time.monotonic() >= self.deadline):
                raise HostAdmissionStop('Host inference admission closed')
            count = self.count_tokens(request)
            if type(count) is not int or count < 0:
                raise ValueError('Token counter must return a nonnegative integer')
            if count > self.max_input_tokens:
                raise ValueError('Complete request exceeds configured model input limit')
            # Different accounted clients sharing this host account reserve
            # under the same OS lock. A restart never reuses a call number.
            with (self.root / 'model-account.lock').open('a+b') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if self.stop_requested or (self.deadline is not None and time.monotonic() >= self.deadline):
                    raise HostAdmissionStop('Host inference admission closed')
                ledger = json.loads((self.root / 'model_calls.json').read_text())
                if ledger['limit'] != self.max_calls:
                    raise ValueError('Host model budget identity changed')
                if ledger['attempted'] >= self.max_calls:
                    raise HostAdmissionStop('Host inference admission closed')
                ident = ledger['attempted'] + 1
                self.calls = ident
                write(self.root / 'model_calls.json', {'attempted': ident, 'limit': self.max_calls})
            append(self.root / 'model_requests.jsonl', {'call': ident, 'at': time.time(),
                'response_attempt': response_attempt,
                'input_tokens': count, 'request': asdict(request),
                'wire_request': (self.transport._generation_payload(request) if callable(getattr(self.transport, '_generation_payload', None)) else None)})
            try:
                observer = getattr(self.transport, 'generate_with_response_observer', None)
                response = (observer(request, lambda wire: append(self.root / 'model_wire.jsonl', {
                        'call': ident, 'response_attempt': response_attempt, **wire}))
                    if callable(observer) else self.transport.generate(request))
                append(self.root / 'model_responses.jsonl', {'call': ident, 'at': time.time(),
                    'response': asdict(response)})
                return response
            except BaseException as exc:
                append(self.root / 'model_errors.jsonl', {'call': ident, 'at': time.time(),
                    'response_attempt': response_attempt,
                    **({'reason': exc.reason, 'retryable': exc.retryable,
                        'finish_reason': exc.finish_reason, 'provider_model': exc.provider_model}
                       if isinstance(exc, ModelResponseError) else {}),
                    'error': f'{type(exc).__name__}: {exc}'})
                raise
