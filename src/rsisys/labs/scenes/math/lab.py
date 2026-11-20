"""Math scene: the evaluation Lab with math defaults and the official benchmark bundle."""
from rsisys.labs.evaluation import EvaluationLab

from .datasets import official_tasks


class MathLab(EvaluationLab):
    scene = 'math'
    title = 'Math'
    scorer = 'math-answer-exact-match/v1'
    group_field = 'benchmark'
    default_group = 'math'
    default_fields = {'group': 'benchmark'}
    default_verifier = 'competition_integer'
    default_max_output_tokens = 32768
    loaders = {'benchmarks_manifest': official_tasks}
