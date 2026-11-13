"""Evaluate one prompt on one split, or render and run one reflective mutation."""
from hashlib import sha256
import json

from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.runtime.orchestration.monitor_snapshot import active_monitor
from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.decision import OperationSpec
from rsisys.loops.contracts.model_service import GenerationRequest

from ..shared.experiment import ModelCalls
from ..shared.numeric_answers import score_math
from ..shared.prompt_data import load_splits
from .protocol import upstream, valid_instruction


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


class GEPAGSM8KLab:
    def __init__(self, context, config):
        self.config = config
        self.session = ModelCalls(context, config['max_model_calls'])
        self.models = context.roles
        self.artifacts = context.artifacts
        self.groups = load_splits(config['files'], {k: config['data'][k + '_count']
                                                  for k in ('search', 'validation', 'test')})
        _, _, _, self.signature = upstream(config['source_dir'])

    def describe(self):
        return (
            OperationSpec('score', 'Run the specified prompt on a frozen GSM8K batch', {'type': 'object'}),
            OperationSpec('mutate', 'Generate one prompt using the specified parent scoring receipt', {'type': 'object'}),
        )

    def execute(self, call):
        if call.name == 'score':
            values, evidence = self.score(call.arguments)
        elif call.name == 'mutate':
            values, evidence = self.mutate(call.arguments)
        else:
            raise ValueError(f'Unknown GEPA operation: {call.name}')
        payload = json.dumps({'operation': call.name, 'request': call.arguments,
                              'evidence': evidence}, ensure_ascii=False, allow_nan=False).encode()
        artifact = self.artifacts.put(payload, kind='gepa-' + call.name)
        monitor = active_monitor.get()
        if monitor is not None and monitor.current_id and call.name == "score":
            monitor.add_samples(evidence.get("records", ()), monitor.current_id)
        return OperationResult(f"GEPA {call.name}: {call.arguments['request_id']}", (artifact,), False,
            data={'request': call.arguments, 'receipt_id': sha256(payload).hexdigest(), **values})

    def complete(self, role, prompt, *, round_index):
        model, parameters = self.config['model'], self.config['parameters']
        optimizer = role == 'decision_maker'
        cap = parameters['optimizer_max_output_tokens'] if optimizer else model['max_output_tokens']
        temperature = model['optimizer_temperature' if optimizer else 'scorer_temperature']
        seed = (self.config['seed'] + round_index) % 2**32 if optimizer else self.config['seed']
        return self.session.text(GenerationRequest(role, self.models[role], prompt,
            system=MUTATOR_SYSTEM if optimizer else ACTOR_SYSTEM,
            temperature=temperature, top_p=1, max_output_tokens=cap, seed=seed))

    def score(self, request):
        instruction, split, label, indices = (request[key] for key in ('instruction', 'split', 'label', 'indices'))
        if not valid_instruction(instruction):
            raise ValueError('Invalid GEPA instruction')
        if split == 'search':
            if (label not in ('parent', 'child') or not isinstance(indices, list)
                    or len(indices) != self.config['parameters']['minibatch_size']
                    or any(type(i) is not int or not 0 <= i < len(self.groups[split]) for i in indices)
                    or len(set(indices)) != len(indices)):
                raise ValueError('Expected a fixed-size search minibatch')
            rows = [self.groups[split][i] for i in indices]
        else:
            labels = {'validation': ('validation_initial', 'validation_child'),
                      'test': ('test_initial', 'test_selected')}
            if split not in labels or label not in labels[split] or indices is not None:
                raise ValueError('GEPA validation and test only score whole frozen splits')
            rows = self.groups[split]
        records = []
        for row in rows:
            prompt = f"Problem:\n{row['question']}\n\nInstruction:\n{instruction}"
            response = self.complete('actor', prompt, round_index=request['round'])
            result = score_math(response, row['answer'])
            records.append({'id': row['id'], 'question': row['question'], 'response': response, **result})
        scores = [int(record['correct']) for record in records]
        return {'scores': scores, 'count': len(records),
                'quality_metrics': {'accuracy': sum(scores) / len(scores), 'examples': len(scores)}}, {'records': records}

    def mutate(self, request):
        artifact = ArtifactRef(**request['parent_artifact'])
        raw_artifact = self.artifacts.get(artifact)
        if artifact.kind != 'gepa-score' or sha256(raw_artifact).hexdigest() != request['parent_receipt_id']:
            raise ValueError('GEPA mutation requires a matching immutable parent score receipt')
        parent = json.loads(raw_artifact)
        parent_request = parent['request']
        if (parent['operation'] != 'score' or parent_request['label'] != 'parent'
                or parent_request['split'] != 'search'
                or any(parent_request[key] != request[key] for key in ('instruction', 'round', 'indices'))):
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
