"""Evaluate one prompt on one split, or render and run one reflective mutation."""
from dataclasses import asdict
from hashlib import sha256
import json

from rsisys.api import (
    ArtifactRef,
    GenerationRequest,
    OperationResult,
    MarkedNumericVerifier,
    MetricSpec,
    report_samples,
    score_record,
)

from ..shared.experiment import ModelCalls
from ..shared.prompt_data import load_splits
from .protocol import upstream, valid_instruction
from .controller import candidate_id
from ..shared.search_registry import SearchRegistry, operation


ACTOR_SYSTEM = ('Solve each math word problem. End with one line containing #### followed by the numeric answer only.')
MUTATOR_SYSTEM = ('Write a single general-purpose instruction for solving unseen math word problems. '
                  'Keep the instruction concise and return it in a fenced text block.')
SCORE_ONLY_TEMPLATE = """I provided an assistant with the following instructions to perform a task for me:
```
<curr_instructions>
```

The following are task inputs and the assistant's correctness score for each one (1 correct, 0 incorrect):
```
<inputs_outputs_feedback>
```

Your task is to write a new instruction for the assistant. Read the inputs and scores carefully to infer the
task format and propose a generalizable strategy for solving the task more accurately.

Provide the new instructions within ``` blocks."""


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
        _, _, _, self.signature = upstream(config['source_dir'])

    def state(self):
        # Splits and the pinned upstream signature are rebuilt from config; only the budget moves.
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
        return (*self._search_specs(), operation('propose', instruction='string',
            parent_artifact='object', parent_receipt_id='string', indices='array'))

    def execute(self, call):
        request = call.arguments
        if call.name == 'register':
            values, evidence = self._register(request), {}
        elif call.name == 'select':
            values, evidence = self._select(request), {}
        elif call.name == 'evaluate':
            values, evidence = self.evaluate({**request, **self._candidate(request)})
            self._record_baseline(request)
        elif call.name == 'propose':
            values, evidence = self.propose(request)
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

    def complete(self, role, prompt, *, round_index):
        model, parameters = self.config['model'], self.config['parameters']
        optimizer = role == 'decision_maker'
        cap = parameters['optimizer_max_output_tokens'] if optimizer else model['max_output_tokens']
        temperature = model['optimizer_temperature' if optimizer else 'scorer_temperature']
        seed = (self.config['seed'] + round_index) % 2**32 if optimizer else self.config['seed']
        return self.session.text(GenerationRequest(role, self.models[role], prompt,
            system=MUTATOR_SYSTEM if optimizer else ACTOR_SYSTEM,
            temperature=temperature, top_p=1, max_output_tokens=cap, seed=seed))

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
            response = self.complete('actor', render(row), round_index=request['round'])
            result = score_record(self.verifier, response, row['answer'])
            records.append({'id': row['id'], 'question': row['question'], 'response': response, **result})
        scores = [int(record['correct']) for record in records]
        return {'scores': scores, 'count': len(records),
                'quality_metrics': {'accuracy': sum(scores) / len(scores), 'examples': len(scores)}}, {'records': records}

    def propose(self, request):
        artifact = ArtifactRef(**request['parent_artifact'])
        raw_artifact = self.artifacts.read(artifact)
        if artifact.kind != 'gepa-evaluate' or sha256(raw_artifact).hexdigest() != request['parent_receipt_id']:
            raise ValueError('GEPA mutation requires a matching immutable parent score receipt')
        parent = json.loads(raw_artifact)
        parent_request = parent['request']
        if (parent['operation'] != 'evaluate' or parent_request['label'] != 'parent'
                or parent_request['split'] != 'search'
                or parent_request['candidate_id'] != candidate_id(request['instruction'])
                or any(parent_request[key] != request[key] for key in ('round', 'indices'))):
            raise ValueError('GEPA mutation receipt does not match the chosen parent minibatch')
        reflective = self.config['parameters']['condition'] == 'reflective'
        examples = []
        for row in parent['evidence']['records']:
            example = {'Inputs': row['question']}
            if reflective:
                observed = row['response']
                if len(observed) > 2400:
                    observed = observed[:1600] + '\n[middle excerpt omitted]\n' + observed[-800:]
                example['Generated Outputs'] = observed
                if row['correct']:
                    example['Feedback'] = 'Score 1. The final answer matches the reference answer.'
                else:
                    issue = 'The final answer is incorrect.' if row['parsed'] else 'No valid numeric final answer was found.'
                    example['Feedback'] = f"Score 0. {issue} Reference answer: {row['expected']}."
            else:
                example['Feedback'] = int(row['correct'])
            examples.append(example)
        prompt = self.signature.prompt_renderer({'current_instruction_doc': request['instruction'],
            'dataset_with_feedback': examples, 'prompt_template': None if reflective else SCORE_ONLY_TEMPLATE})
        raw = self.complete('decision_maker', prompt, round_index=request['round'])
        instruction = self.signature.output_extractor(raw)['new_instruction']
        candidate = instruction if valid_instruction(instruction) else None
        digest = sha256(prompt.encode()).hexdigest()
        return {'candidate': candidate, 'meta_prompt_sha256': digest}, {
            'meta_prompt': prompt, 'response': raw, 'parent_receipt_id': request['parent_receipt_id']}
