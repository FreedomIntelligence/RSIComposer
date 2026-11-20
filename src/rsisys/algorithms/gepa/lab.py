"""Evaluate one registered GEPA prompt on one split; mutation belongs to the Controller."""
from dataclasses import asdict
from hashlib import sha256
import json

from rsisys.api import (
    GenerationRequest,
    OperationResult,
    MarkedNumericVerifier,
    MetricSpec,
    report_samples,
    score_record,
)

from ..shared.experiment import ModelCalls
from ..shared.prompt_data import load_splits
from .protocol import valid_instruction
from .controller import candidate_id
from ..shared.search_registry import SearchRegistry


ACTOR_SYSTEM = ('Solve each math word problem. End with one line containing #### followed by the numeric answer only.')


class GEPAGSM8KLab(SearchRegistry):
    def __init__(self, context, config):
        self._search_init()
        self.verifier = MarkedNumericVerifier()
        self.config = config
        self.session = ModelCalls(context, config['max_model_calls'])
        self.models = context.roles
        self.artifacts = context.artifacts
        self.groups = load_splits(config['files'], {k: config['data'][k + '_count']
                                                  for k in ('search', 'validation', 'test')})

    def state(self):
        # Splits are rebuilt from config; only the budget moves.
        return {'model_calls': self.session.count, 'search': self._search_state()}

    def restore(self, state):
        self.session.count = state['model_calls']
        self._restore_search(state['search'])

    def primary_metric(self):
        return MetricSpec('accuracy', 'maximize', self.verifier.name)

    def _candidate_id(self, candidate):
        if not isinstance(candidate, dict) or set(candidate) != {'instruction'} or not valid_instruction(candidate['instruction']):
            raise ValueError('Expected one valid GEPA instruction')
        return candidate_id(candidate['instruction'])

    def describe(self):
        return self._search_specs()

    def execute(self, call):
        request = call.arguments
        if call.name == 'register':
            values, evidence = self._register(request), {}
        elif call.name == 'select':
            values, evidence = self._select(request), {}
        elif call.name == 'evaluate':
            values, evidence = self.evaluate({**request, **self._candidate(request)})
            self._record_baseline(request)
        else:
            raise ValueError(f'Unknown GEPA operation: {call.name}')
        payload = json.dumps({'operation': call.name, 'request': request,
                              'evidence': evidence}, ensure_ascii=False, allow_nan=False).encode()
        artifact = self.artifacts.put(payload, kind='gepa-' + call.name)
        if call.name == 'evaluate':
            values.update(valid=True, scored=True,
                trace_refs=[asdict(artifact)] * len(values['scores']),
                feedback=[('Correct answer' if row['correct'] else
                    f"Incorrect answer; reference: {row['expected']}") for row in evidence['records']])
            report_samples(evidence.get('records', ()))
        return OperationResult(f"GEPA {call.name}", (artifact,), False,
            data={'request': request, 'receipt_id': sha256(payload).hexdigest(), **values})

    def complete(self, prompt):
        return self.session.text(GenerationRequest('actor', self.models['actor'], prompt, system=ACTOR_SYSTEM,
            temperature=self.config['model']['scorer_temperature'], top_p=1,
            max_output_tokens=self.config['model']['max_output_tokens'], seed=self.config['seed']))

    def evaluate(self, request):
        instruction, split = request['instruction'], request['split']
        indices = self._indices(request, len(self.groups[split]))
        return self._evaluate_rows(request, indices,
            lambda row: f"Problem:\n{row['question']}\n\nInstruction:\n{instruction}")

    def _evaluate_rows(self, request, indices, render):
        split = request['split']
        rows = [self.groups[split][i] for i in indices]
        records = []
        for row in rows:
            response = self.complete(render(row))
            result = score_record(self.verifier, response, row['answer'])
            records.append({'id': row['id'], 'question': row['question'], 'response': response, **result})
        scores = [int(record['correct']) for record in records]
        return {'scores': scores, 'count': len(records),
                'quality_metrics': {'accuracy': sum(scores) / len(scores), 'examples': len(scores)}}, {'records': records}
