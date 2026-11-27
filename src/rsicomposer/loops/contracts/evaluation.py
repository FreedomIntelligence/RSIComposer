"""Which evaluation a receipt came from, in the terms a run-level goal compares.

A Lab reports measured values. A run-level goal additionally needs to know which
candidate produced them and under which scoring protocol, or a metric from one
configuration is indistinguishable from the same metric under another. Every Lab
reaches the Host through one dispatch method, so the identity is completed there
rather than restated by each Lab.

This lives in the contracts layer because both the decision loop and the default
decision harness dispatch Lab calls, and neither may import the other.
"""
import inspect

from rsicomposer.runtime.capabilities import optional_method

from .coordination import OperationResult

_INPUT_KEYS = ('split', 'task_ids', 'indices', 'dataset')


def _takes_argument(method):
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
               for p in parameters)


def input_refs(arguments):
    """The exact inputs an evaluation covered, as references stable across retries."""
    for key in _INPUT_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return (f'{key}:{value}',)
        if (isinstance(value, (list, tuple)) and value
                and all(isinstance(item, str) and item.strip() for item in value)):
            return tuple(f'{key}:{item}' for item in value)
    return None


def evaluation_identity(lab, name, arguments):
    """The candidate, scoring protocol and input set behind one Lab call."""
    alias = name.rpartition('.')[0]
    declare = optional_method(lab, 'primary_metric')
    if declare is None:
        return {}
    # A grouped Lab takes the alias; a single Lab declares one metric.
    spec = declare(alias) if _takes_argument(declare) else declare()
    return {'protocol_ref': getattr(spec, 'scorer', None),
            'candidate_id': arguments.get('instruction') or arguments.get('candidate_id'),
            'input_refs': input_refs(arguments)}


def tag_evaluation_identity(lab, name, call, result):
    """Complete an evaluation receipt with the identity a run goal matches on.

    A receipt without quality metrics is not a measurement, and a grouped Lab may
    say outright that the operation is not one. Fields a Lab already states keep
    their value: this fills gaps, it never overrides a Lab's own answer.
    """
    if not isinstance(result, OperationResult) or not isinstance(result.data, dict):
        return result
    if 'quality_metrics' not in result.data:
        return result
    classify = optional_method(lab, 'is_evaluation')
    if classify is not None and classify(name) is False:
        return result
    derived = evaluation_identity(lab, name, call.arguments)
    additions = {key: value for key, value in derived.items()
                 if value and key not in result.data}
    if not additions:
        return result
    from dataclasses import replace
    return replace(result, data={**result.data, **additions})
