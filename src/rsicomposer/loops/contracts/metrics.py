"""Shared measured-value validation; missing measurements remain absent."""
import math

from .lifecycle import ExecutionStatus


class InvalidMetrics(ValueError):
    """A provider supplied values that cannot be recorded as measurements."""


def has_scored_metrics(status, data):
    """Use only on validated operation data; zero scores are measurements too."""
    return (status == ExecutionStatus.COMPLETED and data.get('valid', True) is True
            and data.get('scored', True) is True and bool(data.get('quality_metrics')))


def validated_metrics(value, name='metrics'):
    if not isinstance(value, dict):
        raise InvalidMetrics(f'{name} must be a mapping of nonempty names to finite numbers')
    for key, number in value.items():
        if not isinstance(key, str) or not key.strip():
            raise InvalidMetrics(f'{name} has an invalid metric name: {key!r}')
        try:
            valid = type(number) in (int, float) and math.isfinite(number)
        except OverflowError:
            valid = False
        if not valid:
            raise InvalidMetrics(f'{name}[{key!r}] must be a finite number; got {number!r}')
    return dict(value)
