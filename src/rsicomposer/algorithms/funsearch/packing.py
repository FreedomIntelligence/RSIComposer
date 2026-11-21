"""Deterministic online bin packing with a bounded numeric priority language."""

import ast
import hashlib
import json
import math
import operator
import random
import re

import numpy as np

CAPACITY = 100
BASES = {
    'first_fit': 'def priority(item, bins):\n  return 0',
    'best_fit': 'def priority(item, bins):\n  return -(bins - item)',
}
MAX_SOURCE = 2000
MAX_NODES = 96
BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv}
CMP_OPS = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
           ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne}
FUNCS = {'where': (np.where, 3), 'minimum': (np.minimum, 2), 'maximum': (np.maximum, 2),
         'clip': (np.clip, 3), 'abs': (np.abs, 1), 'square': (np.square, 1)}


class InvalidPriority(ValueError):
    """A candidate exceeds the grammar or produces invalid scores."""


def _number(node):
    value = node.value
    if type(value) not in (int, float) or not -1e6 <= value <= 1e6:
        raise InvalidPriority('Only finite numeric constants in [-1e6, 1e6] are allowed')
    return lambda item, bins: float(value)


def _compile(node):
    if isinstance(node, ast.Constant):
        return _number(node)
    if isinstance(node, ast.Name) and node.id in ('item', 'bins'):
        return (lambda item, bins: item) if node.id == 'item' else (lambda item, bins: bins)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _compile(node.operand)
        return lambda item, bins: -value(item, bins)
    if isinstance(node, ast.BinOp) and type(node.op) in BIN_OPS:
        function = BIN_OPS[type(node.op)]
        left, right = _compile(node.left), _compile(node.right)
        return lambda item, bins: function(left(item, bins), right(item, bins))
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in CMP_OPS:
        function = CMP_OPS[type(node.ops[0])]
        left, right = _compile(node.left), _compile(node.comparators[0])
        return lambda item, bins: function(left(item, bins), right(item, bins))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS:
        function, arity = FUNCS[node.func.id]
        if len(node.args) != arity or node.keywords:
            raise InvalidPriority('Invalid numeric primitive arguments')
        arguments = [_compile(arg) for arg in node.args]
        return lambda item, bins: function(*(arg(item, bins) for arg in arguments))
    raise InvalidPriority(f'Unsupported priority syntax: {type(node).__name__}')


def normalize(source):
    """Parse an answer or continuation. Never compile or execute Python source."""
    if not isinstance(source, str) or not 0 < len(source.encode()) <= MAX_SOURCE:
        raise InvalidPriority('Invalid source size')
    source = source.strip()
    fenced = re.fullmatch(r'```(?:python)?\s*\n(.*?)\n```', source, re.DOTALL)
    if fenced:
        source = fenced[1].strip()
    if source.startswith('return '):
        source = 'def priority(item, bins):\n  ' + source
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise InvalidPriority('Malformed priority function') from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise InvalidPriority('Expected exactly one priority function')
    function = tree.body[0]
    args = function.args
    if (not re.fullmatch(r'priority(?:_v[0-9]+)?', function.name)
            or function.decorator_list or function.returns is not None or getattr(function, 'type_params', ())
            or [arg.arg for arg in args.args] != ['item', 'bins']
            or any(arg.annotation is not None for arg in args.args)
            or any((args.defaults, args.posonlyargs, args.vararg, args.kwonlyargs, args.kwarg))
            or len(function.body) != 1 or not isinstance(function.body[0], ast.Return)
            or function.body[0].value is None):
        raise InvalidPriority('Expected priority(item, bins) with exactly one return expression')
    expression = function.body[0].value
    if sum(1 for _ in ast.walk(expression)) > MAX_NODES:
        raise InvalidPriority('Priority expression is too complex')
    evaluator = _compile(expression)
    return 'def priority(item, bins):\n  return ' + ast.unparse(expression), evaluator


def pack(items, priority):
    """Give the policy feasible residual capacities, including one unused bin."""
    remaining = np.empty(len(items) + 1, dtype=np.float64)
    remaining[0] = CAPACITY
    used = 0
    for item in items:
        feasible = np.flatnonzero(remaining[:used + 1] >= item)
        capacities = remaining[feasible]
        try:
            with np.errstate(all='ignore'):
                scores = np.asarray(priority(float(item), capacities), dtype=np.float64)
        except (ArithmeticError, TypeError, ValueError) as exc:
            raise InvalidPriority('Priority evaluation failed') from exc
        if scores.shape == ():
            scores = np.broadcast_to(scores, capacities.shape)
        if scores.shape != capacities.shape or not np.isfinite(scores).all():
            raise InvalidPriority('Priority must give finite scores for all feasible bins')
        winner = int(feasible[np.argmax(scores)])
        remaining[winner] -= item
        if winner == used:
            used += 1
            remaining[used] = CAPACITY
    return used


def make_cases(split, count):
    """Deterministic experiment-wide cohorts, held constant across arms and seeds."""
    if split not in ('development', 'heldout'):
        raise ValueError(f'Unknown bin packing cohort: {split}')
    rng = random.Random(40953 if split == 'development' else 91731)
    families = ('uniform', 'weibull')
    cases = []
    for i in range(count):
        family = families[i % len(families)]
        size = 120 if split == 'development' or i % 4 < 2 else 250
        if family == 'uniform':
            items = [rng.randint(14, 67) for _ in range(size)]
        else:
            items = [max(1, min(CAPACITY, round(rng.weibullvariate(45, 3)))) for _ in range(size)]
        cases.append({'id': f'{split}-{i:03d}', 'family': family, 'items': items})
    return cases


def evaluate(source, cases):
    _, priority = normalize(source)
    rows = [{'id': case['id'], 'family': case['family'], 'n_items': len(case['items']),
             'bins': pack(case['items'], priority),
             'volume_bound': math.ceil(sum(case['items']) / CAPACITY)} for case in cases]
    total = sum(row['bins'] for row in rows)
    bound = sum(row['volume_bound'] for row in rows)
    return {'bins': total, 'volume_bound': bound, 'excess_fraction': (total - bound) / bound,
            'signature': [row['bins'] for row in rows], 'count': len(rows), 'rows': rows}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
