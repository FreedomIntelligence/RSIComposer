"""Compare sequence construction and top-1 selection with pinned SEAL source.

Only explicitly selected pure functions/constants are loaded; no training,
model API clients or top-level upstream scripts are executed.
"""
import argparse
import ast
import json
import random
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from rsisys.labs.seal import ANSWER_PROMPT, EDIT_PROMPT, select_edits, train_sequences
from rsisys.labs.seal_grader import GRADE_PROMPT

REVISION = '6d9c9f9ee392c6cc618e771f399d436d190f6ca4'


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('Upstream function set changed')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('upstream', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    revision = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError('Expected pinned SEAL revision')
    ns = {'re': re, 'List': List, 'Dict': Dict, 'Any': Any,
          'MAX_TRAIN_SEQS_PER_COMPLETION': 30, 'TRAINING_SEQUENCE_TEMPLATE': '{title}\n{completion_text}'}
    load_functions(args.upstream / 'general-knowledge/src/utils.py', ['_split_segments', 'build_train_sequences'], ns)
    load_functions(args.upstream / 'general-knowledge/src/EM/build_SFT_dataset.py', ['_top_k'], ns)
    rng, sequences, selections = random.Random(7309), 0, 0
    for n in range(40):
        for marker in ('', 'preamble\nImplications: '):
            completion = marker + ' --- '.join('fact ' + str(k) for k in range(n))
            for split in (False, True):
                for text in (completion, completion.replace(' --- ', '\n'),
                             'Question 1: why?\nAnswer: one\nQuestion 2: how?\nAnswer: two'):
                    expected = ns['build_train_sequences'](text, 'context', 'title', split_newlines=split)
                    if train_sequences(text, 'context', 'title', split_newlines=split) != expected:
                        raise AssertionError(f'Sequence construction mismatch: split_newlines={split}')
                    sequences += 1
        for _ in range(20):
            rows = [{'text': f'edit-{k}', 'stats': {'adapter_mean': rng.choice([0, .5, 1])}}
                    for k in range(max(1, n))]
            expected = ns['_top_k'](rows, 1, 'adapter_mean')
            actual = [r['completion'] for r in select_edits([
                {'doc_index': 0, 'adapter_mean': r['stats']['adapter_mean'], 'completion': r['text']} for r in rows])]
            if expected != actual:
                raise AssertionError('Stable top-1 mismatch')
            selections += 1
    source = (args.upstream / 'general-knowledge/src/data_generation/make_squad_data.py').read_text()
    tree = ast.parse(source)
    assignment = next(n for n in tree.body if isinstance(n, ast.AnnAssign) and
                      isinstance(n.target, ast.Name) and n.target.id == 'MAKE_SQUAD_DATA_TEMPLATES_BASE')
    templates = ast.literal_eval(assignment.value)
    if EDIT_PROMPT != templates['implications']:
        raise AssertionError('Self-edit prompt mismatch')
    tree = ast.parse((args.upstream / 'general-knowledge/src/utils.py').read_text())
    answer = next(n for n in tree.body if isinstance(n, ast.Assign) and
        any(isinstance(t, ast.Name) and t.id == 'SQUAD_ANSWER_TEMPLATE_BASE' for t in n.targets))
    if ANSWER_PROMPT.format(title='Title', question='Question') != ast.literal_eval(answer.value).format(question='Topic: Title\nQuestion'):
        raise AssertionError('Answer prompt mismatch')
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and
        any(isinstance(t, ast.Name) and t.id == 'SQUAD_GRADE_TEMPLATE' for t in n.targets))
    if GRADE_PROMPT != ast.literal_eval(assignment.value):
        raise AssertionError('Semantic grading prompt mismatch')
    args.output.write_text(json.dumps({'revision': revision, 'sequence_cases': sequences,
        'selection_cases': selections, 'prompt_exact': True, 'semantic_prompt_exact': True, 'answer_prompt_exact': True,
        'passed': True, 'scope': 'both sequence branches, prompts and stable selection; GPU Trainer validation is separate'}, indent=2))


if __name__ == '__main__':
    main()
