"""A dataset-backed Lab that scores actual answers from the actor model."""
import hashlib
import json
from pathlib import Path

from rsisys.api import (
    DEV, TRAIN, DatasetBinding, GenerationRequest, MetricSpec, OperationResult, OperationSpec,
)


SPLITS = {'train': TRAIN, 'dev': DEV}


def validate_config(config):
    datasets = config.get('datasets')
    if (set(config) != {'datasets'} or not isinstance(datasets, dict)
            or set(datasets) != set(SPLITS)
            or any(not isinstance(path, str) or not path.strip() for path in datasets.values())):
        raise ValueError('datasets must map train and dev to JSONL paths')


def read_dataset(path):
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict)
                       or any(not isinstance(row.get(key), str) or not row[key].strip()
                              for key in ('id', 'question'))
                       or type(row.get('answer')) is not int for row in rows):
        raise ValueError(f'{path}: expected JSONL rows with id, question and an integer answer')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError(f'{path}: duplicate row IDs')
    return rows, hashlib.sha256(raw).hexdigest()


class ArithmeticLab:
    def __init__(self, context, config):
        validate_config(config)
        self.models = context.models
        self.actor = context.roles.get('actor')
        if self.actor is None or context.backends[self.actor.backend_id].binding.kind != 'chat_completions':
            raise ValueError('The template Lab requires an actor on a GPU-backed chat_completions endpoint')
        self.rows, self.hashes = {}, {}
        self.paths = {split: str(Path(path).resolve()) for split, path in config['datasets'].items()}
        for split, path in config['datasets'].items():
            self.rows[split], self.hashes[split] = read_dataset(path)
        for field in ('id', 'question'):
            train_values = {row[field] for row in self.rows['train']}
            if any(row[field] in train_values for row in self.rows['dev']):
                raise ValueError(f'Training and benchmark rows overlap on {field}')

    def split_roles(self):
        return dict(SPLITS)

    def data_bindings(self):
        # The host records these in the Lab identity and rejects a resume with changed inputs.
        return tuple(DatasetBinding(
            name='arithmetic/' + split, split=split, role=split, source=self.paths[split],
            schema_version='arithmetic-jsonl/v1', content_digest=self.hashes[split],
            row_ids=tuple(row['id'] for row in self.rows[split]),
        ) for split in sorted(SPLITS))

    def primary_metric(self):
        return MetricSpec('accuracy', 'maximize', 'integer-text-exact-match/v1')

    def describe(self):
        return (OperationSpec('evaluate', 'Evaluate a prompt on the dev split', {
            'type': 'object', 'properties': {
                'split': {'type': 'string', 'enum': ['dev']},
                'instruction': {'type': 'string', 'minLength': 1},
                'few_shot': {'type': 'integer', 'minimum': 0, 'maximum': len(self.rows['train'])},
            }, 'required': ['split', 'instruction', 'few_shot'], 'additionalProperties': False,
        }, read_only=True),)

    def execute(self, call):
        args = call.arguments
        count = args.get('few_shot')
        instruction = args.get('instruction')
        if (call.name != 'evaluate' or set(args) != {'split', 'instruction', 'few_shot'}
                or args['split'] != 'dev' or type(count) is not int or not 0 <= count <= len(self.rows['train'])
                or not isinstance(instruction, str) or not instruction.strip()):
            raise ValueError('Expected evaluate(split=dev, instruction, few_shot)')

        examples = ''.join(f"Question: {row['question']}\nAnswer: {row['answer']}\n\n"
                           for row in self.rows['train'][:count])
        records = []
        for row in self.rows['dev']:
            response = self.models.generate(GenerationRequest(
                'actor', self.actor, f"{examples}Question: {row['question']}\nAnswer:",
                system=f'{instruction}\nReturn only the final integer.',
                temperature=0.0, max_output_tokens=64,
            ))
            answer = response.text.strip()
            records.append({'id': row['id'], 'answer': answer, 'correct': answer == str(row['answer'])})
        return OperationResult('Development benchmark evaluated', (), False, data={
            'quality_metrics': {'accuracy': sum(row['correct'] for row in records) / len(records)},
            'few_shot': count,
            'dataset_sha256': dict(self.hashes),
            'records': records,
        })

    def state(self):
        return {'dataset_sha256': dict(self.hashes)}

    def restore(self, state):
        if state != self.state():
            raise ValueError('Lab dataset contents changed since the checkpoint')
