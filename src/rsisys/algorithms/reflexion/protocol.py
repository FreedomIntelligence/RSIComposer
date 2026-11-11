"""Pinned author generator adapter and official HumanEval data."""

from dataclasses import asdict
import gzip
from hashlib import sha256
import json

from ..shared.experiment import author_checkout


AUTHOR_REVISION = '218cf0ef1df84b05ce379dd4a8e47f17766733a0'
DATA_SHA256 = 'b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef'


def load_tasks(config):
    source = author_checkout(config['source_dir'], AUTHOR_REVISION, 'programming_runs') / 'programming_runs'
    dataset = source / 'human-eval/data/HumanEval.jsonl.gz'
    if sha256(dataset.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError('Unexpected HumanEval release digest')
    with gzip.open(dataset, 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    count = config['data']['count']
    if len(rows) != 164 or len({row['task_id'] for row in rows}) != 164 or not 1 <= count <= len(rows):
        raise ValueError('Invalid official HumanEval cohort')
    return source, rows[:count]


class BrokerModel:
    is_chat = True
    def __init__(self, client):
        self.client = client
        self.name = client.model.model_id
    def generate_chat(self, messages, max_tokens=1024, temperature=.2, num_comps=1):
        values = [self.client.generate('actor', messages=[asdict(m) for m in messages],
                     temperature=temperature, max_tokens=max_tokens) for _ in range(num_comps)]
        return values[0] if num_comps == 1 else values
    def generate(self, prompt, max_tokens=1024, stop_strs=None, temperature=0, num_comps=1):
        values = [self.client.generate('actor', prompt, max_tokens=max_tokens,
                     temperature=temperature, stop=stop_strs or None) for _ in range(num_comps)]
        return values[0] if num_comps == 1 else values

