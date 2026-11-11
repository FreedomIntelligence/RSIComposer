"""Read named, disjoint JSONL cohorts for prompt optimization."""

import json
from pathlib import Path


SPLITS = ('search', 'validation', 'test')


def load_splits(files, counts, *, splits=SPLITS):
    groups = {}
    ids, questions = set(), set()
    if set(files) != set(splits) or set(counts) != set(splits):
        raise ValueError(f'Expected splits: {splits}')
    for name in splits:
        with Path(files[name]).open(encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if type(counts[name]) is not int or counts[name] < 1 or len(rows) != counts[name]:
            raise ValueError(f'Unexpected {name} split size')
        for row in rows:
            if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip()
                                                for key in ('id', 'question', 'answer')):
                raise ValueError(f'Invalid {name} row; expected id, question and numeric answer')
            normalized = ' '.join(row['question'].split()).casefold()
            if row['id'] in ids or normalized in questions:
                raise ValueError('Duplicate question or identifier across prompt splits')
            ids.add(row['id'])
            questions.add(normalized)
        groups[name] = rows
    return groups
