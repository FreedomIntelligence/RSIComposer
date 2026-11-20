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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextvars import copy_context
from dataclasses import asdict
import json

from rsisys.api import GenerationRequest

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
Respond with just one return expression, or one def priority(item, bins) with
exactly one return statement. Use no imports, indexing, assignments or prose."""


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
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='rsisys-funsearch-sampler')
        self.jobs = {}

    def submit(self, key, request):
        self.jobs[key] = self.pool.submit(copy_context().run, self.sample, request)

    def finished(self):
        """Keys of finished requests, in submission order."""
        return [key for key, job in self.jobs.items() if job.done()]

    def wait(self):
        """Block until at least one request finishes."""
        wait(tuple(self.jobs.values()), return_when=FIRST_COMPLETED)

    def take(self, key):
        return self.jobs.pop(key).result()

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)

    def sample(self, request):
        """Sample, parse and seal one request's programs: {'proposals', 'receipt_id', 'artifacts'}."""
        proposals, samples, refs = [], [], []
        for index in range(request['start'], request['start'] + request['count']):
            seed = (self.config['seed'] * 100003 + index) % 2**32
            response = self.calls.text(GenerationRequest('decision_maker', self.model, request['prompt'],
                system=SYSTEM, max_output_tokens=self.config['max_output_tokens'],
                temperature=self.config['temperature'], top_p=1.0, seed=seed))
            sample = {'index': index, 'seed': seed, 'response': response}
            try:
                source, _ = normalize(response)
                ref = self.artifacts.put(source.encode(), kind='funsearch-priority')
                version = asdict(ref)
                refs.append(ref)
            except InvalidPriority as exc:
                version = None
                sample['rejection'] = str(exc)
            proposals.append({'index': index, 'version': version})
            samples.append(sample)
        payload = {'operation': 'propose', 'request': request, 'evidence': {'samples': samples}}
        evidence = self.artifacts.put(json.dumps(payload, allow_nan=False).encode(), kind='funsearch-propose')
        return {'proposals': proposals, 'receipt_id': digest(payload), 'artifacts': (*refs, evidence)}
