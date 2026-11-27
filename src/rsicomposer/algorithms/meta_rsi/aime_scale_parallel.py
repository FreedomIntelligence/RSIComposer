"""Score the wide-only AIME dev cohort on four independent hosted engines."""
from concurrent.futures import ThreadPoolExecutor, as_completed


_SHARDS = 4
_METRICS = ('count', 'correct', 'truncated', 'invalid_format')
_RESULT_KEYS = {*_METRICS, 'correctness'}


def _check_shard(report, size):
    if not isinstance(report, dict) or report.keys() != _RESULT_KEYS:
        raise RuntimeError('Unexpected AIME dev shard result shape')
    if any(type(report[metric]) is not int or report[metric] < 0 for metric in _METRICS):
        raise RuntimeError('AIME dev shard metrics must be nonnegative integers')
    positions = report['correctness']
    if (report['count'] != size or not isinstance(positions, list) or len(positions) != size
            or any(type(position) is not bool for position in positions)):
        raise RuntimeError('AIME dev shard did not return one boolean per input position')
    if (sum(positions) != report['correct']
            or report['correct'] + report['truncated'] + report['invalid_format'] > size):
        raise RuntimeError('Inconsistent AIME dev shard score accounting')


def evaluate_parallel(engines, problems, adapter):
    """Run unchanged greedy scoring on four contiguous, balanced dev shards.

    The caller supplies the validated dev cohort and one engine per separately
    bound single-GPU placement; this function neither allocates nor masks GPUs.
    Results match the serial scorer's shape and input order.
    """
    from .aime_scale_probe import evaluate

    engines, problems = tuple(engines), tuple(problems)
    if len(engines) != _SHARDS or len({id(engine) for engine in engines}) != _SHARDS:
        raise ValueError('Parallel AIME dev scoring requires four distinct engines')
    if adapter is None:
        raise ValueError('Parallel AIME dev scoring requires the wide adapter')
    if len(problems) < _SHARDS:
        raise ValueError('Parallel AIME dev scoring requires four nonempty shards')

    width, remainder = divmod(len(problems), _SHARDS)
    totals = dict.fromkeys(_METRICS, 0)
    correctness = [None] * len(problems)
    with ThreadPoolExecutor(max_workers=_SHARDS, thread_name_prefix='aime-dev') as pool:
        pending, start = {}, 0
        for index, engine in enumerate(engines):
            stop = start + width + (index < remainder)
            pending[pool.submit(evaluate, engine, problems[start:stop], adapter)] = (start, stop)
            start = stop
        for future in as_completed(pending):
            start, stop = pending[future]
            try:
                report = future.result()
                _check_shard(report, stop - start)
            except Exception as exc:
                exc.add_note(f'While scoring AIME dev shard covering positions [{start}, {stop})')
                raise
            correctness[start:stop] = report['correctness']
            for metric in _METRICS:
                totals[metric] += report[metric]

    if totals['count'] != len(problems) or any(position is None for position in correctness):
        raise RuntimeError('Incomplete parallel AIME dev result')
    return {**totals, 'correctness': correctness}
