"""Derived evaluation facts. These never change scores or prove an improvement."""
from collections import Counter
import re


def prediction_diagnostics(scored):
    groups = {}
    for row in scored:
        group = groups.setdefault(row['benchmark_id'], {'texts': [], 'answers': []})
        group['texts'].append(row['text'])
        if 'extracted_answer' in row:
            answer = row['extracted_answer']
        else:
            match = re.fullmatch(r'([A-J])[.\s]*', row['text'].strip())
            answer = match.group(1) if match else None
        group['answers'].append(answer)
    result = {}
    for name, group in groups.items():
        counts = Counter(answer for answer in group['answers'] if answer is not None)
        n = len(group['answers'])
        invalid = n - sum(counts.values())
        dominant = max(counts.values(), default=0) / n
        result[name] = {'n': n, 'invalid': invalid, 'answer_counts': dict(sorted(counts.items())),
            'dominant_answer_fraction': dominant,
            'empty_texts': sum(not text.strip() for text in group['texts']),
            'mean_returned_characters': sum(map(len, group['texts'])) / n,
            'warnings': (['invalid_answers'] if invalid else []) +
                        (['answer_distribution_collapsed'] if n >= 20 and dominant >= .9 else [])}
    return {'benchmarks': result,
            'scope': 'candidate_returned_text_only; backend_generation_not_attested',
            'affects_score': False}


def compare_evaluations(before, after):
    """Separate end-to-end comparison from attribution to changed weights alone."""
    fields = ('protocol_sha256', 'dataset_sha256', 'runtime_sha256', 'execution_budget',
              'inference_sha256', 'weight_sha256')
    missing = [key for key in fields if key not in before or key not in after]
    changed = [key for key in fields if key in before and key in after and before[key] != after[key]]
    comparable = not missing and not any(key in changed for key in fields[:4])
    metrics_before, metrics_after = before.get('metrics', {}), after.get('metrics', {})
    comparable = (comparable and before.get('status') == after.get('status') == 'succeeded'
                  and bool(metrics_before) and set(metrics_before) == set(metrics_after)
                  and all(metrics_before[k]['n'] == metrics_after[k]['n'] for k in metrics_before))
    return {'comparable': comparable, 'missing_identity_fields': missing, 'changed_axes': changed,
        'weight_effect_isolated': comparable and changed == ['weight_sha256'],
        'attribution_scope': 'identity_check_only; backend_weight_consumption_not_attested',
        'accuracy_delta': {key: metrics_after[key]['accuracy'] - metrics_before[key]['accuracy']
                           for key in metrics_before} if comparable else {},
        'adoption_verified': False, 'statistical_improvement_verified': False}
