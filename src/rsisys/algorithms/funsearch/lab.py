"""Generate sealed priorities and score them with the trusted packing evaluator."""

from dataclasses import asdict
import json
import time

from rsisys.api import ArtifactRef, GenerationRequest, MetricSpec, OperationCall, OperationResult

from ..shared.experiment import ModelCalls
from ..shared.search_registry import SearchRegistry, operation
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


class FunSearchLab(SearchRegistry):
    # The registry uses complete, idempotent updates. Attempted model calls
    # remain chargeable even when a proposal is interrupted.
    concurrent_state_safe = True
    search_splits = ('development',)
    sealed_split = 'heldout'
    def __init__(self, context, config):
        self._search_init()
        self.config = config
        self.artifacts = context.artifacts
        self.session = ModelCalls(context, config['max_model_calls'])
        self.models = context.roles
        self.cases = {split: make_cases(split, config[split + '_count'])
                      for split in ('development', 'heldout')}
        self.dataset_ids = {split: digest(rows) for split, rows in self.cases.items()}

    def state(self):
        # Cases are regenerated from config; only the model-call budget moves.
        return {'model_calls': self.session.count, 'search': self._search_state()}

    def restore(self, state):
        self.session.count = state['model_calls']
        self._restore_search(state['search'])

    def primary_metric(self):
        return MetricSpec('bins', 'minimize', 'trusted-ast-bin-packing-v1')

    def _candidate_id(self, candidate):
        if not isinstance(candidate, dict) or set(candidate) != {'version'}:
            raise ValueError('Expected a sealed priority artifact')
        ref = ArtifactRef(**candidate['version'])
        if ref.kind != 'funsearch-priority':
            raise ValueError('Expected a sealed FunSearch priority')
        self.artifacts.read(ref)  # Validate ownership and digest before registration.
        return ref.digest

    def describe(self):
        return (*self._search_specs(), operation('task_spec'),
                operation('propose', prompt='string', start='integer', count='integer'))

    def execute(self, call):
        if time.monotonic() >= getattr(self.session, 'deadline', float('inf')):
            raise TimeoutError('FunSearch experiment deadline exhausted')
        request = call.arguments
        if call.name == 'register':
            values, evidence, artifact_refs = self._register(request), {}, ()
        elif call.name == 'select':
            values, evidence, artifact_refs = self._select(request), {}, ()
        elif call.name == 'evaluate':
            values, evidence, artifact_refs = self.evaluate({**request, **self._candidate(request)})
            if values['scored']:
                self._record_baseline(request)
        elif call.name in ('task_spec', 'propose'):
            values, evidence, artifact_refs = getattr(self, call.name)(request)
        else:
            raise ValueError(f'Unknown FunSearch operation: {call.name}')
        payload = {'operation': call.name, 'request': request, 'evidence': evidence}
        ref = self.artifacts.put(json.dumps(payload, allow_nan=False).encode(), kind=f'funsearch-{call.name}')
        if call.name == 'evaluate':
            values['trace_refs'] = [asdict(ref)] * len(values['scores'])
        return OperationResult(f"FunSearch {call.name}", (*artifact_refs, ref), False,
                               data={'request': request, 'receipt_id': digest(payload), **values})

    def final_evaluations(self):
        if self.selected_id is None:
            return ()
        return tuple(OperationCall('evaluate', {'candidate_id': key, 'split': 'heldout', 'indices': None,
            'label': label, 'request_id': f'host-final-{label}'})
            for label, key in (*self.baseline_ids.items(), ('selected', self.selected_id)))

    def seal(self, source):
        ref = self.artifacts.put(source.encode(), kind='funsearch-priority')
        return asdict(ref), ref

    def task_spec(self, request):
        versions, refs = {}, []
        for label, source in BASES.items():
            version, ref = self.seal(normalize(source)[0])
            versions[label] = {'version': version}
            refs.append(ref)
        return {'versions': versions, 'datasets': self.dataset_ids}, {'datasets': self.dataset_ids}, refs

    def propose(self, request):
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

    def evaluate(self, request):
        ref = ArtifactRef(**request['version'])
        if ref.kind != 'funsearch-priority':
            raise ValueError('Expected a sealed FunSearch priority')
        source = self.artifacts.read(ref).decode()
        split = request['split']
        evidence = {'dataset_sha256': self.dataset_ids[split], 'source': source,
                    'evaluator': 'trusted-ast-bin-packing-v1'}
        try:
            indices = self._indices(request, len(self.cases[split]))
            result = evaluate(source, [self.cases[split][i] for i in indices])
        except InvalidPriority as exc:
            return {'score': None, 'source': source, 'valid': False, 'scored': False,
                    'scores': [], 'trace_refs': [], 'feedback': [str(exc)], 'quality_metrics': {}}, {**evidence, 'rejection': str(exc)}, (ref,)
        evidence['score'] = result
        return {'score': {key: value for key, value in result.items() if key != 'rows'},
                'source': source, 'valid': True, 'scored': True, 'scores': result['signature'],
                'feedback': [f"Used {row['bins']} bins" for row in result['rows']],
                'quality_metrics': {'bins': result['bins'], 'instances': result['count']}}, evidence, (ref,)
