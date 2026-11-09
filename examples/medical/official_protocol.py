"""Pinned MedXpertQA official zero-shot answer-only medical evaluation rules.

Source: TsinghuaC3I/MedXpertQA@5f1db162d0d2b03d62da2d4fe086076a36105e18,
eval/config/prompt_templates.py and eval/utils.py (MIT). This is the official
release's common MedQA/MedXpertQA answer-only path, not a claim that all
published model scores share one prompting/decoding setup.
"""
import hashlib
import json
from pathlib import Path
import re


SCORE_PROTOCOL = 'medxpertqa-official-zero-shot-ao-v1'
LABELS = {'medqa': 'ABCD', 'medxpertqa': 'ABCDEFGHIJ'}


def extract_answer(text, benchmark):
    labels = LABELS[benchmark]
    trigger = f'Among A through {labels[-1]}, the answer is'
    try:
        text = text.split(trigger)[1].strip()
    except IndexError:
        try:
            text = text.split(' answer is ')[1].strip()
        except IndexError:
            pass
    for unwanted in ('I understand', 'A through J', 'A through E', 'A through D'):
        text = text.replace(unwanted, '')
    matches = re.findall(r'\b(' + '|'.join(labels) + r')\b', text)
    return matches[0] if matches else None


def official_inputs(groups):
    items, gold, datasets = [], {}, {}
    for group in groups:
        content = Path(group['path']).read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        benchmark = group['benchmark_id']
        if sha != group['sha256'] or benchmark not in LABELS or benchmark in datasets:
            raise ValueError('Official dataset identity or benchmark changed')
        datasets[benchmark] = sha
        for line in content.decode().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (benchmark, row['id'])
            if (key in gold or row['gold'] not in LABELS[benchmark]
                    or set(row['options']) != set(LABELS[benchmark])
                    or not isinstance(row['question'], str) or not row['question'].strip()):
                raise ValueError('Invalid official evaluation row')
            # Prepared question preserves the upstream full question, including
            # its original Answer Choices suffix. Never append choices twice.
            if '\nAnswer Choices:' not in row['question']:
                raise ValueError('Official evaluation requires the complete upstream question')
            gold[key] = row['gold']
            trigger = f'Among A through {LABELS[benchmark][-1]}, the answer is'
            items.append({'benchmark_id': benchmark, 'id': row['id'], 'messages': [
                {'role': 'system', 'content': 'Please answer only.'},
                {'role': 'user', 'content': f"Q: {row['question'].strip()}\nA: {trigger}"}]})
    if not gold:
        raise ValueError('Official evaluation requires nonempty data')
    return {'protocol': SCORE_PROTOCOL, 'items': items}, gold, datasets


def official_score_predictions(predictions, gold):
    if not isinstance(predictions, list) or len(predictions) != len(gold):
        raise ValueError('Return one raw response for every fixed sample')
    seen, scored, counts = set(), [], {}
    for row in predictions:
        if not isinstance(row, dict) or set(row) != {'benchmark_id', 'id', 'text'}:
            raise ValueError('Prediction requires benchmark_id, id and text only')
        key = (row['benchmark_id'], row['id'])
        if key not in gold or key in seen or not isinstance(row['text'], str):
            raise ValueError('Unknown, duplicate or malformed prediction')
        seen.add(key)
        answer = extract_answer(row['text'], key[0])
        correct = answer == gold[key]
        scored.append({**row, 'extracted_answer': answer, 'gold': gold[key], 'correct': correct})
        count = counts.setdefault(key[0], [0, 0])
        count[0] += int(correct)
        count[1] += 1
    return {name: {'accuracy': c / n, 'n': n} for name, (c, n) in counts.items()}, scored


def scoring_functions(protocol):
    if protocol == SCORE_PROTOCOL:
        return official_inputs, official_score_predictions
    if protocol == 'exact-option-letter-v1':
        try:
            from .fixed_evaluation import fixed_inputs, score_predictions
        except ImportError:
            from fixed_evaluation import fixed_inputs, score_predictions
        return fixed_inputs, score_predictions
    raise ValueError('Unsupported host scoring protocol')
