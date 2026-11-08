"""Self-contained numeric data for library boundary tests."""
import json
from pathlib import Path


def make_math_selection(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in {
        'train': [{'id': 'train-1', 'prompt': 'What is 3 + 5?', 'answer': '#### 8'},
                  {'id': 'train-2', 'prompt': 'What is 4 * 3?', 'answer': '#### 12'}],
        'eval': [{'id': 'eval-1', 'prompt': 'What is 7 + 6?', 'answer': '#### 13'}],
    }.items():
        (directory / (name + '.jsonl')).write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return {'controller': {'name': 'rest_em', 'config': {
        'initial_model': 'fixture-initial', 'train_data': 'train.jsonl', 'eval_data': 'eval.jsonl',
        'iterations': 2, 'samples_per_prompt': 4, 'max_accepted_per_prompt': 1}},
        'labs': {'math': {'name': 'math', 'config': {'engine': 'fixture', 'data_root': str(directory)}}}}
