"""Isolate upstream evaluator imports and send only grade metrics back to the Lab."""

from contextlib import redirect_stdout
from dataclasses import asdict
import importlib
from io import StringIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace


def checkout_path(settings):
    root = Path(settings['checkout']).resolve(strict=True)
    if subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() != settings['revision']:
        raise ValueError('Evaluator checkout revision does not match its binding')
    subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD', '--'], check=True)
    return root


def browsecomp(root, task, judge):
    sys.path.insert(0, str(root.parent))
    upstream = importlib.import_module(f'{root.name}.browsecomp_eval')
    prompt = upstream.GRADER_TEMPLATE.format(question=task['input']['question'],
        correct_answer=task['reference']['answer'], response=task['response'])
    # The pinned upstream checkout returns the whole "correct: yes" match;
    # parse the decision separately and distinguish malformed judge output.
    matches = re.findall(r'^\s*correct:\s*(yes|no)\s*\.?\s*$', judge(prompt), flags=re.M | re.I)
    if len(matches) != 1:
        raise ValueError('BrowseComp judge did not produce a single correct: yes/no decision')
    return {'accuracy': int(matches[0].lower() == 'yes')}


def widesearch(root, task, judge):
    sys.path.insert(0, str(root))
    import pandas as pd
    from loguru import logger
    from src.evaluation import evaluation, metric_utils
    from src.evaluation.data_loader import WideSearchQuery, WideSearchResponse
    from src.utils.utils import norm_column

    logger.remove()  # The upstream evaluator logs both gold cells and judge prompts.

    def completion(messages, model_config_name):
        text = judge(messages)
        values = metric_utils.parse_markdown_json(text)
        if not isinstance(values, dict):
            raise ValueError('WideSearch judge did not return a JSON mapping')
        if 'specific grading criteria section' in messages:
            keys = set(re.findall(r"'idx_(\d+)':\s*\{(?:'response'|\"response\")", messages))
            if set(values) != {f'idx_{key}' for key in keys} or any(
                    type(value) not in (int, float) or not 0 <= value <= 1 for value in values.values()):
                raise ValueError('WideSearch judge did not return a bounded score for every cell')
        return SimpleNamespace(content=text)

    metric_utils.llm_completion = completion
    ref = task['reference']
    gold = pd.read_csv(StringIO(ref['answer_csv']))
    gold.columns = [norm_column(col.strip()) for col in gold.columns]
    rules = ref['evaluation']
    query = WideSearchQuery(task['id'], task['input']['question'], rules,
                            gold[rules['required']], ref['language'])
    response = WideSearchResponse(task['id'], task['response'])
    result = asdict(evaluation.evaluate_single_query(query, response, eval_model_config_name='rsicomposer-judge'))
    if result['msg'].startswith('evaluator error:'):
        raise RuntimeError('WideSearch evaluator failed')
    return {key: float(value) for key, value in result.items() if key not in {'msg', 'instance_id'}}


def main():
    from openai import OpenAI

    task = json.load(sys.stdin)
    settings = task['settings']
    root = checkout_path(settings)
    key_env = settings['judge_key_env']
    key = 'EMPTY' if key_env == '-' else os.environ[key_env]
    judge = OpenAI(base_url=settings['judge_endpoint'], api_key=key, timeout=120, max_retries=0)

    def complete(prompt):
        result = judge.chat.completions.create(model=settings['judge_model'], temperature=0,
            messages=[{'role': 'user', 'content': prompt}]).choices[0]
        if not result.message.content or result.finish_reason != 'stop':
            raise ValueError('Judge response was incomplete')
        return result.message.content

    with redirect_stdout(sys.stderr):
        metrics = {'browsecomp': browsecomp, 'widesearch': widesearch}[task['benchmark']](root, task, complete)
    print(json.dumps(metrics, allow_nan=False))


if __name__ == '__main__':
    main()
