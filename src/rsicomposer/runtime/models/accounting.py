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

from rsicomposer.loops.contracts.model_service import ModelRef, ModelResponseError
from rsicomposer.runtime.models.activation import validate_activation_route
from rsicomposer.runtime.storage.files import atomic_write, json_bytes


_ACCOUNT_BUDGET = object()


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
                 token_counter=None, response_attempts=1, max_concurrent=1):
        self.transport, self.tokenizer, self.root = transport, tokenizer, Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.allowed_models = tuple(allowed_models) if allowed_models is not None else None
        self.configured_models = self.allowed_models
        self.template_tool_normalizer = template_tool_normalizer
        self.provider_token_counter = provider_token_counter
        if token_counter is not None and not callable(token_counter):
            raise TypeError('token_counter must be callable')
        if type(response_attempts) is not int or not 1 <= response_attempts <= 8:
            raise ValueError('response_attempts must be between 1 and 8')
        self.token_counter, self.response_attempts = token_counter, response_attempts
        if provider_token_counter:
            self.token_count_mode = 'provider_tokenizer'
        if max_input_tokens is None:
            self.token_count_mode = 'unavailable_no_input_admission'
        self.max_calls, self.max_input_tokens = max_calls, max_input_tokens
        if (type(max_calls) is not int or max_calls < 1 or (max_input_tokens is not None
                and (type(max_input_tokens) is not int or max_input_tokens < 1))):
            raise ValueError('Host model call and input budgets must be positive integers')
        self.calls, self.deadline = 0, None
        self.host_budget = None
        self.stop_requested = False
        self.pause_requested = False
        if type(max_concurrent) is not int or max_concurrent < 1:
            raise ValueError('max_concurrent must be a positive integer')
        self.max_concurrent = max_concurrent
        self._slots = threading.BoundedSemaphore(max_concurrent)
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

    def is_model_authorized(self, model):
        if (not isinstance(model, ModelRef) or any(not isinstance(value, str) or not value.strip()
                for value in (model.backend_id, model.model_id, model.revision))):
            return False
        binding = getattr(self.transport, 'binding', None)
        if binding is not None and model.backend_id != binding.backend_id:
            return False
        return self.allowed_models is None or model in self.allowed_models

    def validate_activation_route(self, base, model, *, prepared):
        """Checking addressability does not grant admission or debit a call."""
        validate_activation_route(self.transport, base, model, prepared=prepared)

    def authorize_prepared_model(self, model):
        """Host checkpoint port; call only after successful model preparation.

        Grants are reconstructed from the ControllerHost checkpoint. They do
        not alter the original account/assembly configuration identity.
        """
        if (not isinstance(model, ModelRef) or any(not isinstance(value, str) or not value.strip()
                for value in (model.backend_id, model.model_id, model.revision))):
            raise ValueError('Prepared model authorization requires an explicit ModelRef')
        with self.lock:
            backends = ({item.backend_id for item in self.configured_models} if self.configured_models is not None else set())
            binding = getattr(self.transport, 'binding', None)
            if binding is not None:
                backends.add(binding.backend_id)
            if backends and model.backend_id not in backends:
                raise PermissionError('Prepared model cannot introduce an unregistered backend')
            if self.allowed_models is not None and model not in self.allowed_models:
                self.allowed_models += (model,)

    def generate(self, request, *, attribution=None, budget=_ACCOUNT_BUDGET):
        with self._slots:
            original = request
            for attempt in range(self.response_attempts):
                try:
                    return self._generate(request, response_attempt=attempt + 1, attribution=attribution, budget=budget)
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

    def _generate(self, request, *, response_attempt, attribution, budget=_ACCOUNT_BUDGET):
        attribution = dict(attribution or {})
        if set(attribution) - {'component', 'decision_id', 'call_id'}:
            raise ValueError('Unknown host model attribution field')
        if any(value is not None and (not isinstance(value, str) or not value) for value in attribution.values()):
            raise ValueError('Invalid host model attribution')
        if self.allowed_models is not None and request.model not in self.allowed_models:
            raise PermissionError("Model is outside the host session bindings")
        if self.stop_requested or self.calls >= self.max_calls or (self.deadline is not None and time.monotonic() >= self.deadline):
            raise HostAdmissionStop('Host inference admission closed')
        count = (self.count_tokens(request) if self.max_input_tokens is not None else None)
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError('Token counter must return a nonnegative integer')
        if self.max_input_tokens is not None and count > self.max_input_tokens:
            raise ValueError('Complete request exceeds configured model input limit')
        with self.lock:
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
                admission = self.host_budget if budget is _ACCOUNT_BUDGET else budget
                if admission is not None:
                    admission.reserve_model_call()
                ident = ledger['attempted'] + 1
                self.calls = ident
                write(self.root / 'model_calls.json', {'attempted': ident, 'limit': self.max_calls})
            append(self.root / 'model_requests.jsonl', {'call': ident, 'at': time.time(), **attribution,
                'response_attempt': response_attempt,
                'input_tokens': count, 'token_count_mode': self.token_count_mode, 'request': asdict(request),
                'wire_request': (self.transport._generation_payload(request) if callable(getattr(self.transport, '_generation_payload', None)) else None)})
        try:
            observer = getattr(self.transport, 'generate_with_response_observer', None)
            response = (observer(request, lambda wire: self._append('model_wire.jsonl', {
                    'call': ident, 'response_attempt': response_attempt, **wire, **attribution}))
                if callable(observer) else self.transport.generate(request))
            self._append('model_responses.jsonl', {'call': ident, 'at': time.time(), **attribution,
                'response': asdict(response)})
            return response
        except BaseException as exc:
            self._append('model_errors.jsonl', {'call': ident, 'at': time.time(), **attribution,
                'response_attempt': response_attempt,
                **({'reason': exc.reason, 'retryable': exc.retryable,
                    'finish_reason': exc.finish_reason, 'provider_model': exc.provider_model}
                   if isinstance(exc, ModelResponseError) else {}),
                'error': f'{type(exc).__name__}: {exc}'})
            raise

    def _append(self, filename, value):
        # Response observers may run on transport threads while other calls reserve.
        with self.lock:
            append(self.root / filename, value)
