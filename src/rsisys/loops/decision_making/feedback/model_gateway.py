"""Per-job inference admission and durable exchanges for library-owned workflows."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from threading import RLock

from rsisys.runtime.files import atomic_write, json_bytes


class FeedbackModelGateway:
    """Record every upstream model call before dispatch, including corrections.

    The owning feedback journal reserves max_calls before starting this worker.
    The original backend still enforces the host's shared limits and identity.
    This gateway is single-use; interrupted jobs are never silently replayed.
    """

    def __init__(self, backend, model, *, max_calls, max_input_tokens, path):
        self.backend, self.model = backend, model
        self.max_calls, self.max_input_tokens = max_calls, max_input_tokens
        self.path = Path(path)
        self.exchanges, self._lock = [], RLock()
        if self.path.exists():
            raise ValueError('Memory exchanges already exist; do not replay an uncertain job')

    @property
    def attempted(self):
        return len(self.exchanges)

    def _save(self):
        atomic_write(self.path, json_bytes({'version': 1, 'exchanges': self.exchanges}))

    def generate(self, request):
        with self._lock:
            if request.model != self.model:
                raise ValueError('Memory request must use the captured active meta-agent model')
            if self.attempted >= self.max_calls:
                raise RuntimeError('Feedback model call limit reached')
            tokens = self.backend.count_tokens(request)
            if type(tokens) is not int or tokens < 0:
                raise ValueError('Memory token counter must return a nonnegative integer')
            entry = {'request': asdict(request), 'input_tokens': tokens, 'status': 'reserved'}
            if tokens > self.max_input_tokens:
                entry.update(status='input_too_large', error='Complete native request exceeds input token limit')
                self.exchanges.append(entry)
                self._save()
                raise ValueError(entry['error'])
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
