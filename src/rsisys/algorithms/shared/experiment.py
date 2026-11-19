"""Run artifacts and host model routing shared by the classic method packages."""

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import RLock

from rsisys.api import ModelResponseError, report_model_calls


class MethodEvents:
    """Preserve method-specific evidence without writing the host monitor snapshot."""

    def __init__(self, context, method):
        self.path = Path(context.data_dir) / f'{method}-events.jsonl'
        self.summary = {'method': method, 'status': 'running'}

    def __call__(self, kind, data):
        row = {'kind': kind, 'timestamp': datetime.now(timezone.utc).isoformat(), 'data': data}
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        if kind == 'finish':
            self.summary = {**data, 'status': 'completed'}

    def report(self):
        return self.summary


class ModelCalls:
    """Resolve explicitly configured roles and count every inference attempt."""

    def __init__(self, context, max_calls):
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError('max_model_calls must be a positive integer')
        self.session = context.model_session
        self.backends = context.backends
        self.roles = context.roles
        self.max_calls = max_calls
        self.count = 0
        self._lock = RLock()

    def generate(self, request):
        with self._lock:
            if self.count >= self.max_calls:
                raise RuntimeError('Method model-call budget exhausted')
            self.count += 1
        service = self.session or self.backends[request.model.backend_id]
        try:
            return service.generate(request)
        finally:
            report_model_calls(self.count, self.max_calls)

    def text(self, request):
        try:
            return self.generate(request).text
        except ModelResponseError as error:
            # The core text transport refuses truncated completions. Preserve
            # the attempted call and score it as a missing answer or proposal.
            if str(error) != 'Model did not finish a complete text response':
                raise
            return ''


def random_state(rng):
    """random.Random state as codec-compatible lists."""
    version, internal, gauss = rng.getstate()
    return [version, list(internal), gauss]


def set_random_state(rng, state):
    version, internal, gauss = state
    rng.setstate((version, tuple(internal), gauss))


def author_checkout(source_dir, revision, package):
    """Require the declared pinned checkout without changing import resolution."""
    import subprocess

    root = Path(source_dir).resolve(strict=True)
    commit = subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True, timeout=15).strip()
    clean = subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD', '--', package],
                           check=False, timeout=15)
    if commit != revision or clean.returncode:
        raise ValueError(f'Expected a clean {package} checkout at {revision}')
    return root
