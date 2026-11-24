"""Pinned official TTRL reward and majority-vote adapter."""
import ast
import importlib.util
import sys
from threading import current_thread, main_thread

from rsicomposer.api import checkout

TTRL_REVISION = '5806e119789132fcb43552beaf2532d7d0447213'


class TTRLReward:
    """Load the complete official reward package, without importing verl/Ray."""
    name = 'ttrl-official-' + TTRL_REVISION

    def __init__(self, root):
        root = checkout(root, TTRL_REVISION)
        self.root = root
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

    def score_many(self, pairs):
        if current_thread() is not main_thread():
            from ._reward_worker import score_official_pairs
            return score_official_pairs(pairs, self.root)
        return [self.is_correct(completion, gold) for completion, gold in pairs]

    def rewards(self, completions):
        pseudo, ratio = self.vote(completions)
        return pseudo, ratio, [float(self.module.compute_score(text, pseudo)['score']) for text in completions]
