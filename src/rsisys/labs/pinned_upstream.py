"""Explicit, immutable upstream checkouts for optional research backends."""
import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

SEAL_REVISION = '6d9c9f9ee392c6cc618e771f399d436d190f6ca4'
TTRL_REVISION = '5806e119789132fcb43552beaf2532d7d0447213'


def checkout(path, revision):
    root = Path(path).resolve(strict=True)
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != revision:
        raise ValueError(f'Expected upstream revision {revision}, got {actual}')
    if subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Pinned upstream checkout has tracked modifications')
    return root


class TTRLReward:
    """Load the complete official reward package, without importing verl/Ray."""
    name = 'ttrl-official-' + TTRL_REVISION

    def __init__(self, root):
        root = checkout(root, TTRL_REVISION)
        package = root / 'verl/verl/utils/reward_score/ttrl_math'
        name = '_rsisys_ttrl_math_' + TTRL_REVISION
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, package / '__init__.py',
                submodule_search_locations=[str(package)])
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(name, None)
                raise
        self.module = sys.modules[name]
        # Load the exact pure vote function, not a second math normalizer.
        source = root / 'verl/verl/trainer/ppo/ttrl_utils.py'
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_majority_vote')
        from collections import Counter
        namespace = dict(Counter=Counter, List=list,
            extract_answer=self.module.extract_answer,
            simplify_expression_string=self.module.simplify_expression_string)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        self.vote = namespace['_majority_vote']

    def parse(self, completion):
        return self.module.extract_answer(completion)

    def is_correct(self, completion, gold):
        return bool(self.module.compute_score(completion, gold)['acc'])

    def rewards(self, completions):
        pseudo, ratio = self.vote(completions)
        return pseudo, ratio, [float(self.module.compute_score(text, pseudo)['score']) for text in completions]
