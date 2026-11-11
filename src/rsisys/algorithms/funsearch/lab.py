"""Generate sealed priorities and score them with the trusted packing evaluator."""

from dataclasses import asdict
import json
import time

from rsisys.api import OperationSpec
from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.model_service import GenerationRequest

from ..shared.experiment import ModelCalls
from .packing import BASES, InvalidPriority, digest, evaluate, make_cases, normalize

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


class FunSearchLab:
    def __init__(self, context, config):
        self.config = config
        self.artifacts = context.artifacts
        self.session = ModelCalls(context, config['max_model_calls'])
        self.models = context.roles
        self.cases = {split: make_cases(split, config[split + '_count'])
                      for split in ('development', 'heldout')}
        self.dataset_ids = {split: digest(rows) for split, rows in self.cases.items()}

    def describe(self):
        return (
            OperationSpec('seed', 'Seal the baseline online bin-packing priorities', {'type': 'object'}),
            OperationSpec('generate', 'Generate programs from one Controller-selected prompt', {'type': 'object'}),
            OperationSpec('score', 'Score a priority with the isolated numeric evaluator', {'type': 'object'}),
        )

    def execute(self, call):
        handler = {'seed': self.seed, 'generate': self.generate, 'score': self.score}.get(call.name)
        if handler is None:
            raise ValueError(f'Unknown FunSearch operation: {call.name}')
        if time.monotonic() >= getattr(self.session, 'deadline', float('inf')):
            raise TimeoutError('FunSearch experiment deadline exhausted')
        request = call.arguments
        values, evidence, artifact_refs = handler(request)
        payload = {'operation': call.name, 'request': request, 'evidence': evidence}
        ref = self.artifacts.put(json.dumps(payload, allow_nan=False).encode(), kind=f'funsearch-{call.name}')
        return OperationResult(f"FunSearch {call.name}: {request['request_id']}", (*artifact_refs, ref), False,
                               data={'request': request, 'receipt_id': digest(payload), **values})

    def seal(self, source):
        ref = self.artifacts.put(source.encode(), kind='funsearch-priority')
        return asdict(ref), ref

    def seed(self, request):
        versions, refs = {}, []
        for label, source in BASES.items():
            version, ref = self.seal(normalize(source)[0])
            versions[label] = {'version': version}
            refs.append(ref)
        return {'versions': versions, 'datasets': self.dataset_ids}, {'datasets': self.dataset_ids}, refs

    def generate(self, request):
        proposals, samples, refs = [], [], []
        role = 'decision_maker'
        for index in range(request['start'], request['start'] + request['count']):
            seed = (self.config['seed'] * 100003 + index) % 2**32
            response = self.session.text(GenerationRequest(role, self.models[role], request['prompt'],
                system=SYSTEM, max_output_tokens=self.config['max_output_tokens'],
                temperature=self.config['temperature'], top_p=1.0, seed=seed))
            sample = {'index': index, 'seed': seed, 'response': response}
            try:
                source, _ = normalize(response)
                version, ref = self.seal(source)
                refs.append(ref)
            except InvalidPriority as exc:
                version = None
                sample['rejection'] = str(exc)
            proposals.append({'index': index, 'version': version})
            samples.append(sample)
        return {'proposals': proposals}, {'samples': samples}, refs

    def score(self, request):
        ref = ArtifactRef(**request['version'])
        if ref.kind != 'funsearch-priority':
            raise ValueError('Expected a sealed FunSearch priority')
        source = self.artifacts.get(ref).decode()
        split = request['split']
        evidence = {'dataset_sha256': self.dataset_ids[split], 'source': source,
                    'evaluator': 'trusted-ast-bin-packing-v1'}
        try:
            result = evaluate(source, self.cases[split])
        except InvalidPriority as exc:
            return {'score': None, 'source': source}, {**evidence, 'rejection': str(exc)}, (ref,)
        evidence['score'] = result
        return {'score': {key: value for key, value in result.items() if key != 'rows'},
                'source': source,
                'quality_metrics': {'bins': result['bins'], 'instances': result['count']}}, evidence, (ref,)
