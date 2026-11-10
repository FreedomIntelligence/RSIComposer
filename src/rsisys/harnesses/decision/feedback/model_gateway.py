"""Per-job inference admission and durable exchanges for library-owned workflows."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from inspect import getattr_static
from threading import RLock

from rsisys.runtime.storage.files import atomic_write, json_bytes


class FeedbackCapacityError(ValueError):
    """Complete native request cannot fit; patch repair cannot fix admission."""


class NativeAdmissionAbort(BaseException):
    """Private boundary signal escaping upstream blanket Exception repairs.

    Caught only by the owning LangMem adapter; never crosses the worker boundary.
    KeyboardInterrupt/SystemExit and ordinary provider failures are not converted.
    """


class FeedbackModelGateway:
    """Record every upstream model call before dispatch, including corrections.

    The owning feedback journal reserves max_calls before starting this worker.
    The original backend still enforces the host's shared limits and identity.
    This gateway is single-use; interrupted jobs are never silently replayed.
    """

    def __init__(self, backend, model, *, max_calls, max_input_tokens, path,
                 request_candidates=None):
        self.backend, self.model = backend, model
        self.max_calls, self.max_input_tokens = max_calls, max_input_tokens
        self.path = Path(path) if path is not None else None
        self.exchanges, self._lock = [], RLock()
        self.request_candidates = request_candidates
        if self.path is not None and self.path.exists():
            raise ValueError('Memory exchanges already exist; do not replay an uncertain job')

    @property
    def attempted(self):
        return len(self.exchanges)

    def _save(self):
        if self.path is not None:
            atomic_write(self.path, json_bytes({'version': 1, 'exchanges': self.exchanges}))

    def generate(self, request):
        with self._lock:
            if request.model != self.model:
                raise ValueError('Memory request must use the captured active meta-agent model')
            if self.attempted >= self.max_calls:
                raise RuntimeError('Feedback model call limit reached')
            candidates = self.request_candidates(request) if self.request_candidates else (request,)
            selected = False
            for candidate in candidates:
                if candidate.model != self.model or candidate.tools != request.tools:
                    raise ValueError('Memory projection changed model identity or native tools')
                tokens = self.backend.count_tokens(candidate)
                if type(tokens) is not int or tokens < 0:
                    raise ValueError('Memory token counter must return a nonnegative integer')
                request, selected = candidate, True
                if tokens <= self.max_input_tokens:
                    break
            if not selected:
                raise ValueError('Memory admission requires at least one request candidate')
            entry = {'request': asdict(request), 'input_tokens': tokens, 'status': 'reserved'}
            if tokens > self.max_input_tokens:
                entry.update(status='input_too_large', error='Complete native request exceeds input token limit')
                self.exchanges.append(entry)
                self._save()
                raise FeedbackCapacityError(entry['error'])
            self.exchanges.append(entry)
            self._save()
            try:
                response = self.backend.generate(request)
                entry.update(status='completed', response=asdict(response))
                if response.model != self.model:
                    raise ValueError('Memory response model identity mismatch')
            except Exception as exc:
                entry.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                self._save()
                raise
            self._save()
            return response

    def snapshot(self):
        with self._lock:
            return deepcopy(self.exchanges)


class SerializedLearningBackend:
    """Share one backend, including its admission/logging, across both callers.

    Generation is serialized because arbitrary backend clients and their counters
    are not thread safe. Jobs are asynchronous to the experiment, not a promise of
    parallel model inference. The wrapped transport must have a finite timeout
    and permit calls from a worker thread (serialization cannot remove affinity).
    """
    def __init__(self, backend):
        self.backend, self._lock = backend, RLock()

    def __getattr__(self, name):
        # Domain policies may inspect the same admission counters/capabilities.
        # All generate/tokenize calls still pass through the serialized methods.
        return getattr(self.backend, name)

    @property
    def supports_response_schema(self):
        return getattr(self.backend, 'supports_response_schema', False) is True

    @property
    def token_count_mode(self):
        return (getattr(self.backend, 'token_count_mode', 'provider_tokenizer')
                if getattr_static(self.backend, 'count_tokens', None) is not None
                else 'utf8_conservative_estimate')

    def count_tokens(self, request):
        with self._lock:
            counter = (getattr(self.backend, 'count_tokens')
                       if getattr_static(self.backend, 'count_tokens', None) is not None else None)
            if callable(counter):
                tokens = counter(request)
            else:
                tokens = len(json_bytes(asdict(request)))
            if type(tokens) is not int or tokens < 0:
                raise ValueError('Feedback token counter must return a nonnegative integer')
            return tokens

    def generate(self, request):
        with self._lock:
            return self.backend.generate(request)
