"""MathLab over the artifact-search protocol, for prompt optimization on math tasks.

The data, verifiers, scoring and evidence are MathLab's: the Lab config names a
MathLab dataset manifest with search and validation (development) and test
(sealed) splits, and each split selects its verifier there. This layer adds the
candidate registry (register, evaluate, select), evaluation by integer offsets
into a split, a model-call budget and the per-example fields artifact-search
Controllers read: scores, feedback, trace_refs and receipt_id (the digest of the
stored evaluation trace).

A method Lab subclasses this, implements solve() with its own prompt (calling
ask()), and _candidate_id() for its candidate shape. Config:

    {"manifest": "data/manifest.json", "seed": 17, "max_model_calls": 256,
     "model": {"max_output_tokens": 2048, "scorer_temperature": 0.0}}
"""
from rsisys.api import GenerationRequest, MathLab, MetricSpec, OperationResult

from .experiment import ModelCalls
from .search_registry import SearchRegistry


class MathSearchLab(SearchRegistry, MathLab):
    # Prefix of the receipts stored for register, select and other bookkeeping operations.
    receipt_kind = 'search'

    def __init__(self, context, config):
        self._search_init()
        self.config = config
        self.calls = ModelCalls(context, config['max_model_calls'])
        MathLab.__init__(self, context, {
            'manifest': config['manifest'],
            'split_roles': {**dict.fromkeys(self.search_splits, 'dev'), self.sealed_split: 'sealed'},
            'max_output_tokens': config['model']['max_output_tokens']})

    def state(self):
        return {'binding': self.binding(), 'model_calls': self.calls.count, 'search': self._search_state()}

    def restore(self, state):
        if state['binding'] != self.binding():
            raise ValueError('Math search Lab input binding changed since the checkpoint')
        self.calls.count = state['model_calls']
        self._restore_search(state['search'])

    def primary_metric(self):
        # The sealed split's verifier defines what the reported accuracy means.
        return MetricSpec('accuracy', 'maximize', self.datasets[self.sealed_split][0].verifier.name)

    def describe(self):
        return self._search_specs()

    def execute(self, call):
        request = call.arguments
        if call.name == 'evaluate':
            if request.get('split') not in self.datasets:
                raise ValueError('Unknown evaluation split')
            result = self.evaluate(call.call_id, {'split': request['split'], 'request': request,
                                                  'candidate': self._candidate(request)})
            self._record_baseline(request)
            return result
        if call.name == 'register':
            return self.receipt(call, self._register(request))
        if call.name == 'select':
            return self.receipt(call, self._select(request))
        raise ValueError(f'Unknown {self.receipt_kind} operation: {call.name}')

    def receipt(self, call, values):
        """Store a bookkeeping operation's request and result; its digest is the receipt_id."""
        import json

        payload = {'operation': call.name, 'request': call.arguments, 'result': values}
        ref = self.artifacts.put(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
                                 kind=f'{self.receipt_kind}-{call.name}')
        return OperationResult(f'{self.receipt_kind} {call.name}', (ref,), False,
            data={'request': call.arguments, 'receipt_id': ref.digest, **values}, call_id=call.call_id)

    def offsets(self, request, size):
        """Example offsets an evaluation answers, in order; all of the split when indices is null."""
        return self._indices(request, size)

    def problems(self, arguments):
        rows = self.datasets[arguments['split']]
        return [rows[i] for i in self.offsets(arguments['request'], len(rows))]

    def ask(self, prompt, system, *, seed):
        """One budgeted actor answer at the configured scorer temperature."""
        return self.attempt(GenerationRequest('actor', self.actor, prompt, system=system,
            temperature=self.config['model']['scorer_temperature'], top_p=1,
            max_output_tokens=self.max_output_tokens, seed=seed), self.calls.generate)

    def feedback(self, record):
        return 'Correct answer' if record['correct'] else f"Incorrect answer; reference: {record['expected']}"

    def result_data(self, records, ref, arguments):
        from dataclasses import asdict

        return {'request': arguments['request'], 'receipt_id': ref.digest, 'valid': True,
                'scores': [int(row['correct']) for row in records],
                'trace_refs': [asdict(ref)] * len(records),
                'feedback': [self.feedback(row) for row in records]}
