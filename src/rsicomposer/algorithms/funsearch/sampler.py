"""FunSearch samplers: the Controller's research model writes priority programs concurrently.

Each sampling request asks the decision_maker model for `count` programs from one
population prompt. Up to `workers` requests run at once in the Controller's own
threads, so sampling overlaps the Lab's evaluations; the Controller polls
finished requests each decision. Programs that parse are stored as
funsearch-priority artifacts, which the Lab registers and scores.

In-flight requests are not checkpointed: the Controller keeps each request in
its state and resubmits it after a resume. Seeds depend only on the proposal
index, so a resubmitted request asks the model exactly the same questions.
"""
from concurrent.futures import ThreadPoolExecutor
import ast
from contextvars import copy_context
from dataclasses import asdict
import json
from threading import Event

from rsicomposer.api import GenerationRequest, ModelResponseError

from ..shared.experiment import ModelCalls
from .packing import InvalidPriority, digest, normalize

SYSTEM = """Improve the online bin-packing priority function. Each item is immediately
packed in the feasible bin with maximum priority, with ties going to the first.
item is the incoming size; bins is a vector of residual capacities in creation
order. The last feasible bin may be an empty bin of capacity 100. Sizes are
integers from 1 through 100. Use only item, bins, finite numeric literals,
arithmetic + - * /, one comparison at a time, and bare calls where(condition,
yes, no), minimum(a, b), maximum(a, b), clip(x, low, high), abs(x), square(x).
Aim to minimize total bins. You may account for the size of residual gaps.
Respond with one Python return statement on one line. Use no imports, indexing,
assignments or prose. Example: return -(bins - item)"""


def proposal_prompt(program):
    # The upstream database emits a completion stub; chat needs a standalone answer.
    trimmed = program.rstrip()
    has_bare_stub = trimmed.split('\n')[-1].startswith('def ') and trimmed.endswith(':')
    nodes = ast.parse(trimmed + ('\n    pass' if has_bare_stub else '')).body
    last = nodes[-1] if nodes else None
    if isinstance(last, ast.FunctionDef) and (has_bare_stub or
            (len(last.body) == 1 and ast.get_docstring(last) is not None)):
        program = '\n'.join(program.splitlines()[:last.lineno - 1])
    return ('Previous scoring functions:\n' + program.strip() +
            '\nWrite one new return statement that may improve the previous scores. Output only that line.')


class FunSearchSampler:
    def __init__(self, context, config, *, workers):
        self.config = config
        self.model = context.roles['decision_maker']
        self.artifacts = context.artifacts
        # One model call per proposal; the Controller restores the count from its state.
        self.calls = ModelCalls(context, config['proposals'])
        # The Controller's model access admits one call at a time unless widened before first use.
        session = getattr(context, 'model_session', None)
        if session is not None and hasattr(session, 'max_concurrent'):
            session.max_concurrent = max(session.max_concurrent, workers)
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='rsicomposer-funsearch-sampler')
        self.jobs = {}  # Only the Controller's owning thread changes this table.
        self._stopping = Event()

    def submit(self, key, request):
        if self._stopping.is_set():
            raise RuntimeError('FunSearch sampler is stopping')
        if key in self.jobs:
            raise ValueError('FunSearch sampling request is already owned')
        self.jobs[key] = self.pool.submit(copy_context().run, self.sample, request)

    def finished(self):
        """Keys of finished requests, in submission order."""
        return [key for key, job in self.jobs.items() if job.done()]

    def take(self, key):
        """Publish a completed batch only on the Controller's owning thread."""
        if self._stopping.is_set():
            raise RuntimeError('Cannot publish stopped FunSearch research')
        request, staged, samples = self.jobs[key].result()
        refs, proposals = [], []
        for row in staged:
            proposal = dict(row)
            source = proposal.pop('source', None)
            if source is not None:
                ref = self.artifacts.put(source.encode(), kind='funsearch-priority')
                proposal['version'] = asdict(ref)
                refs.append(ref)
            proposals.append(proposal)
        payload = {'operation': 'propose', 'request': request, 'evidence': {'samples': samples}}
        evidence = self.artifacts.put(json.dumps(payload, allow_nan=False).encode(), kind='funsearch-propose')
        del self.jobs[key]
        return {'proposals': proposals, 'receipt_id': digest(payload), 'artifacts': (*refs, evidence)}

    def stop(self):
        """Close admission, discard uncommitted results and confirm actual settlement."""
        if not self._stopping.is_set():
            self._stopping.set()
            self.pool.shutdown(wait=False, cancel_futures=True)
        if any(not job.done() for job in self.jobs.values()):
            return False
        self.pool.shutdown(wait=True)  # Only join threads once no provider call is running.
        self.jobs.clear()
        return True

    def close(self):
        if not self.stop():
            # The Host retains owned cleanup and its dependencies for retry.
            raise RuntimeError('FunSearch sampler still owns running generation')

    def sample(self, request):
        """Generate and parse one batch without publishing artifacts from a worker."""
        proposals, samples = [], []
        prompt = proposal_prompt(request['prompt'])
        for index in range(request['start'], request['start'] + request['count']):
            seed = (self.config['seed'] * 100003 + index) % 2**32
            sample = {'index': index, 'seed': seed, 'response': None}
            generation = GenerationRequest('decision_maker', self.model, prompt, system=SYSTEM,
                max_output_tokens=self.config['max_output_tokens'], temperature=self.config['temperature'],
                top_p=0.95, seed=seed)
            # This is the admission boundary for each proposal, not just each batch.
            # The accepted provider call may not be interruptible, but no further
            # proposals may be admitted after stop has been signalled.
            if self._stopping.is_set():
                return None
            try:
                response = self.calls.generate(generation)
                if self._stopping.is_set():
                    return None
                sample['response'] = response.text
                if response.finish_reason == 'length':
                    raise ModelResponseError('Candidate output reached its token limit', reason='output_truncated',
                                             finish_reason='length')
            except ModelResponseError as exc:
                if exc.reason not in ('output_truncated', 'empty_response', 'empty_or_invalid_content'):
                    raise
                reason = 'output_truncated' if exc.finish_reason == 'length' else exc.reason
                sample.update(rejection=str(exc), rejection_kind='generation', reason=reason,
                              finish_reason=exc.finish_reason)
                proposals.append({'index': index, 'version': None, 'generation_error': reason})
                samples.append(sample)
                continue
            try:
                source, _ = normalize(response.text)
            except InvalidPriority as exc:
                source = None
                sample.update(rejection=str(exc), rejection_kind='parse')
            proposals.append({'index': index, 'version': None, 'source': source})
            samples.append(sample)
        return request, proposals, samples
