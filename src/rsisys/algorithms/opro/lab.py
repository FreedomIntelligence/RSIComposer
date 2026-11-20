"""GSM8K scoring with the pinned OPRO evaluator; proposing instructions belongs to the Controller."""
from dataclasses import asdict, replace
import hashlib
from rsisys.api import GenerationRequest, MetricSpec, OperationResult, report_samples
import json


from ..shared.experiment import ModelCalls
from ..shared.prompt_data import load_splits
from .protocol import SCORER_SYSTEM_V1, upstream, candidate_id
from ..shared.search_registry import SearchRegistry, operation


class OPROGSM8KLab(SearchRegistry):
    def __init__(self, context, config):
        self._search_init()
        self.config = config
        self.session = ModelCalls(context, config['max_model_calls'])
        self.artifacts = context.artifacts
        self.models = context.roles
        self.groups = load_splits(config['files'], {k: config['data'][k + '_count']
                                                  for k in ('search', 'validation', 'test')})
        self.opt = upstream(config['source_dir'])

    def state(self):
        # Data splits and the pinned upstream are rebuilt from config; only the budget moves.
        return {'model_calls': self.session.count, 'search': self._search_state()}

    def restore(self, state):
        self.session.count = state['model_calls']
        self._restore_search(state['search'])

    def primary_metric(self):
        return MetricSpec('accuracy', 'maximize', 'opro-official-gsm8k')

    def _candidate_id(self, candidate):
        if (not isinstance(candidate, dict) or set(candidate) != {'instruction'}
                or not isinstance(candidate['instruction'], str) or not 0 < len(candidate['instruction']) <= 500):
            raise ValueError('Expected one nonempty OPRO instruction')
        return candidate_id(candidate['instruction'])

    def describe(self):
        exemplars = operation('exemplars', indices='array')
        return (*self._search_specs(), replace(exemplars, description='Search-split problems shown to the optimizer',
                                               read_only=True))

    def execute(self, call):
        request = call.arguments
        if call.name == 'register':
            values, evidence = self._register(request), {}
        elif call.name == 'select':
            values, evidence = self._select(request), {}
        elif call.name == 'exemplars':
            values, evidence = self.exemplars(request), {}
        elif call.name == 'evaluate':
            candidate = self._candidate(request)
            values, evidence = self.evaluate({**request, **candidate})
            self._record_baseline(request)
        else:
            raise ValueError(f'Unknown OPRO Lab operation: {call.name}')
        payload = json.dumps({'operation': call.name, 'request': request,
                              'evidence': evidence}, ensure_ascii=False, allow_nan=False).encode()
        artifact = self.artifacts.put(payload, kind='opro-' + call.name)
        if call.name == 'evaluate':
            records = evidence['records']
            values.update(valid=True, scored=True, scores=[float(row['accuracy']) for row in records],
                          trace_refs=[asdict(artifact)] * len(records),
                          feedback=[f"Accuracy: {row['accuracy']}" for row in records])
            report_samples(records)
        return OperationResult(f"OPRO {call.name}", (artifact,), False,
            data={'request': request, 'receipt_id': hashlib.sha256(payload).hexdigest(), **values})

    def complete(self, prompt, *, seed):
        model, params = self.config['model'], self.config['parameters']
        adapted = params.get('local_adapter') == 'qwen-instructions-v1'
        return self.session.text(GenerationRequest('actor', self.models['actor'], prompt,
            system=SCORER_SYSTEM_V1 if adapted else '', temperature=model['scorer_temperature'], top_p=1,
            max_output_tokens=model['max_output_tokens'], seed=seed if adapted else None))

    def dataframe(self, split):
        import pandas as pd
        return pd.DataFrame([(row['question'], row['answer']) for row in self.groups[split]])

    def exemplars(self, request):
        rows = self.groups['search']
        indices = self._indices(request, len(rows))
        return {'exemplars': [{'question': rows[i]['question'], 'answer': rows[i]['answer']} for i in indices]}

    def evaluate(self, request):
        frame = self.dataframe(request['split'])
        indices = self._indices(request, len(frame))

        def scorer(prompt, **_):
            prompts = [prompt] if isinstance(prompt, str) else prompt
            return [self.complete(item, seed=self.config['seed']) for item in prompts]

        result = self.opt.eval_utils.evaluate_single_instruction(
            data=frame, instruction=request['instruction'], eval_index_all=indices,
            batch_size=1, call_server_func=scorer, dataset_name='gsm8k', num_servers=1,
            extract_final_answer_by_prompting_again=False, instruction_pos=self.config['parameters']['instruction_pos'],
            is_multiple_choice=False, include_qa=False, evaluate_in_parallel=False,
            prediction_treat_as_number=True)
        score = float(result['accuracy'].mean())
        return {'score': score, 'count': len(result),
                'quality_metrics': {'accuracy': score, 'examples': len(result)}}, {
            'records': json.loads(result.reset_index().to_json(orient='records'))}
