"""Immutable candidate registration shared by artifact-search Labs."""
from copy import deepcopy
from functools import wraps
from threading import RLock

from rsicomposer.api import DEV, SEALED, Invariant, OperationCall, OperationSpec


def _registry_locked(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        if not hasattr(self, '_search_lock'):
            self._search_lock = RLock()
        with self._search_lock:
            return method(self, *args, **kwargs)
    return locked


def operation(name, **fields):
    return OperationSpec(name, f'Artifact search {name}', {'type': 'object',
        'properties': {key: {'type': kind} for key, kind in fields.items()},
        'required': list(fields)}, idempotent=True)


class SearchRegistry:
    """Candidate ownership and selection; split authorization belongs to the host."""
    search_splits = ('search', 'validation')
    sealed_split = 'test'

    def _search_init(self):
        self._search_lock = RLock()
        self.candidate_registry = {}
        self.baseline_ids = {}
        self.selected_id = None

    def profiles(self):
        return ('artifact-search/v1',)

    def invariants(self):
        # Proposals are the Controller's research step; the host still requires a measured
        # baseline before a candidate can be selected.
        return (Invariant.baseline_before('select'),)

    def split_roles(self):
        return {**dict.fromkeys(self.search_splits, DEV), self.sealed_split: SEALED}

    @_registry_locked
    def _search_state(self):
        return {'candidates': deepcopy(self.candidate_registry),
                'baselines': dict(self.baseline_ids), 'selected': self.selected_id}

    @_registry_locked
    def _restore_search(self, state):
        candidates = state['candidates']
        if any(self._candidate_id(candidate) != key for key, candidate in candidates.items()):
            raise ValueError('Candidate registry identity mismatch')
        if any(key not in candidates for key in state['baselines'].values()) or (
                state['selected'] is not None and state['selected'] not in candidates):
            raise ValueError('Selection references an unregistered candidate')
        self.candidate_registry = deepcopy(candidates)
        self.baseline_ids, self.selected_id = dict(state['baselines']), state['selected']

    def _search_specs(self, **evaluate_fields):
        return (operation('register', candidate='object'),
                operation('evaluate', candidate_id='string', split='string',
                          indices=['array', 'null'], **evaluate_fields),
                operation('select', candidate_id='string'))

    @_registry_locked
    def _register(self, request):
        candidate = request['candidate']
        key = self._candidate_id(candidate)
        # The validated content identity may have another equivalent artifact/path.
        self.candidate_registry.setdefault(key, deepcopy(candidate))
        return {'candidate_id': key}

    @_registry_locked
    def _candidate(self, request):
        key = request['candidate_id']
        if key not in self.candidate_registry:
            raise ValueError('Evaluate requires an immutable registered candidate')
        return deepcopy(self.candidate_registry[key])

    @_registry_locked
    def _select(self, request):
        self._candidate(request)
        self.selected_id = request['candidate_id']
        return {'candidate_id': self.selected_id}

    @_registry_locked
    def _record_baseline(self, request):
        if request.get('baseline') is True:
            self.baseline_ids.setdefault(request.get('label', 'initial'), request['candidate_id'])

    def _indices(self, request, size):
        indices = request.get('indices')
        if indices is None:
            return list(range(size))
        if (not isinstance(indices, list) or not indices or len(set(indices)) != len(indices)
                or any(type(i) is not int or not 0 <= i < size for i in indices)):
            raise ValueError('indices must be distinct valid example offsets')
        return indices

    def final_evaluations(self):
        if self.selected_id is None:
            return ()
        baseline = next(iter(self.baseline_ids.values()), None)
        if baseline is None:
            raise ValueError('Final measurement requires a measured baseline')
        return tuple(OperationCall('evaluate', {
            'candidate_id': key, 'split': self.sealed_split, 'indices': None,
            'request_id': f'host-final-{label}', 'round': -1, 'label': label})
            for label, key in (('test_initial', baseline), ('test_selected', self.selected_id)))
