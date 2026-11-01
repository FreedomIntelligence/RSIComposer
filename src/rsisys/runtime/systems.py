"""Versioned component application, independent of either loop or training engine."""
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import asdict
from typing import Protocol

from rsisys.loops.contracts.candidate import Candidate, SystemVersion
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.contracts.training import TrainingOrigin, TrainingResult
from rsisys.loops.decision_making.action_space import OperationCall, OperationResult, OperationSpec
from rsisys.loops.decision_making.action_space.capabilities import Capability


def _candidate_ids(candidate_ids: Iterable[str]) -> tuple[str, ...]:
    if isinstance(candidate_ids, (str, bytes)):
        raise ValueError('Expected a nonempty collection of unique candidate IDs')
    ids = tuple(candidate_ids)
    if (not ids or any(not isinstance(ident, str) or not ident.strip() for ident in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('Expected a nonempty collection of unique candidate IDs')
    return ids


def _application_key(idempotency_key: str) -> None:
    if not isinstance(idempotency_key, str) or not idempotency_key.strip():
        raise ValueError('Application requires an idempotency key')


class SystemActivationRejected(ValueError):
    """Activator confirms the requested composition was not applied.

    Use only for a known no-effect rejection or after verified rollback. An
    uncertain external outcome must keep the pending intent for reconciliation.
    """


class SystemActivator(Protocol):
    def activate(self, system: SystemVersion, *, idempotency_key: str) -> SystemVersion:
        """Prepare components and return the identical manifest only once ready.

        Retry the same key after an uncertain response. Implementations own external
        effects and atomic role binding; the registry cannot roll them back.
        Raise SystemActivationRejected only after confirming no composition was applied.
        """
        ...


class ReferenceSystemActivator:
    """Default for reference-only components; does not deploy or change weights."""
    def __init__(self, model: ModelRef | None = None):
        self.model = model

    def activate(self, system, *, idempotency_key):
        if system.model != self.model:
            raise SystemActivationRejected('Model deployment requires an explicit SystemActivator')
        return deepcopy(system)


class SystemRegistry:
    """Single-writer manifests with stale-parent checks and idempotent application.

    No implicit storage or resource acquisition. Custom recovery must persist this
    registry and the activator's receipts together; do not reconstruct on a retry.
    """
    def __init__(self, initial: SystemVersion, activator: SystemActivator):
        if not isinstance(initial, SystemVersion):
            raise TypeError('Expected SystemVersion')
        self._current = deepcopy(initial)
        self.activator = activator
        self._versions = {initial.version: deepcopy(initial)}
        self._candidates = {}
        self._training_origins = {}
        self._compositions = {}
        self._receipts = {}
        self._rejections = {}
        self._pending = None

    @property
    def current(self):
        return deepcopy(self._current)

    def get(self, version: str):
        return deepcopy(self._versions[version])

    def register(self, candidate: Candidate, *, model: ModelRef | None = None):
        """Register a component change relative to its parent's model revision."""
        return self._register(candidate, model=model, origin=None)

    def register_training(self, result: TrainingResult, *, origin: TrainingOrigin):
        """Validate training initialization independently of experiment ancestry.

        The caller supplies the submitted origin, not an inferred current model.
        Registration never authorizes applying an obsolete experimental parent.
        """
        if not isinstance(result, TrainingResult) or not isinstance(origin, TrainingOrigin):
            raise TypeError('Training registration requires TrainingResult and TrainingOrigin')
        result.__post_init__()
        origin.__post_init__()
        if result.base != origin.initial_model:
            raise ValueError('Training result base differs from origin.initial_model')
        if result.candidate.parent_system != origin.parent_system:
            raise ValueError('Training candidate parent differs from submitted origin')
        return self._register(result.candidate, model=result.model, origin=origin)

    def training_origin(self, candidate_id: str) -> TrainingOrigin | None:
        """Return retained submission provenance; ordinary candidates have none."""
        if candidate_id not in self._candidates:
            raise KeyError(candidate_id)
        return deepcopy(self._training_origins.get(candidate_id))

    def _register(self, candidate, *, model, origin):
        if not isinstance(candidate, Candidate) or not candidate.parent_system:
            raise ValueError('System candidate requires an explicit parent_system')
        parent = self._versions[candidate.parent_system]
        if origin is None and parent.model is not None and candidate.base_version != parent.model.revision:
            raise ValueError('Candidate model base revision differs from its parent model')
        target = SystemVersion(candidate.candidate_id,
            {**parent.components, **candidate.changes},
            model if model is not None else parent.model, parent.version)
        previous = self._versions.get(target.version)
        if previous is not None:
            if (previous != target or self._candidates.get(target.version) != candidate
                    or self._training_origins.get(target.version) != origin):
                raise ValueError('System version identity conflicts with existing content')
            return deepcopy(previous)
        self._versions[target.version] = deepcopy(target)
        self._candidates[target.version] = deepcopy(candidate)
        if origin is not None:
            self._training_origins[target.version] = deepcopy(origin)
        return deepcopy(target)

    def compose(self, candidate_ids: Iterable[str], *, version: str) -> SystemVersion:
        """Register a union of compatible changes without applying it.

        Source identity is an unordered set, independent of manifest equality.
        Candidates which retain the parent model do not override an explicit
        model change; distinct changed models cannot be combined.
        """
        ids = tuple(sorted(_candidate_ids(candidate_ids)))
        candidates = [self._candidates[ident] for ident in ids]
        parents = {candidate.parent_system for candidate in candidates}
        if len(parents) != 1:
            raise ValueError('Composition requires the same explicit parent system')
        component_bases = {candidate.base_version for candidate in candidates
                           if candidate.candidate_id not in self._training_origins}
        if len(component_bases) > 1:
            raise ValueError('Composition requires the same model base revision for component changes')
        parent = self._versions[candidates[0].parent_system]
        changes = {}
        model = parent.model
        for candidate in candidates:
            for name, ref in candidate.changes.items():
                if name in changes and changes[name] != ref:
                    raise ValueError(f'Conflicting component changes: {name}')
                changes[name] = ref
            proposed_model = self._versions[candidate.candidate_id].model
            if proposed_model != parent.model:
                if model != parent.model and model != proposed_model:
                    raise ValueError('Conflicting model changes in composition')
                model = proposed_model
        target = SystemVersion(version, {**parent.components, **changes}, model, parent.version)
        previous = self._versions.get(version)
        if previous is not None:
            if previous != target or self._compositions.get(version) != ids:
                raise ValueError('System version identity conflicts with existing content or sources')
            return deepcopy(previous)
        self._versions[version] = deepcopy(target)
        self._compositions[version] = ids
        return deepcopy(target)

    def adopt(self, candidate_ids: Iterable[str], *, version: str,
              idempotency_key: str) -> SystemVersion:
        """Compose and apply once, returning the activator-confirmed manifest.

        compose accepts known old parents, so activate can replay a receipt even
        after current advances, while still rejecting a fresh stale application.
        """
        _application_key(idempotency_key)
        self.compose(candidate_ids, version=version)
        return self.activate(version, idempotency_key=idempotency_key)

    def activate(self, candidate_id: str, *, idempotency_key: str):
        _application_key(idempotency_key)
        if idempotency_key in self._rejections:
            ident, message = self._rejections[idempotency_key]
            if ident != candidate_id:
                raise ValueError('Application key reused for a different candidate')
            raise SystemActivationRejected(message)
        if idempotency_key in self._receipts:
            ident, version = self._receipts[idempotency_key]
            if ident != candidate_id:
                raise ValueError('Application key reused for a different candidate')
            return deepcopy(version)
        target = self._versions[candidate_id]
        if candidate_id not in self._candidates and candidate_id not in self._compositions:
            raise ValueError('Only registered candidates or compositions can be applied')
        intent = (candidate_id, idempotency_key)
        if self._pending is not None and self._pending != intent:
            if self._pending[1] == idempotency_key:
                raise ValueError('Application key reused for a different candidate')
            raise RuntimeError('An unconfirmed system application must be reconciled first')
        if target.parent_version != self._current.version:
            raise ValueError('Candidate parent is stale; rebase explicitly before application')
        self._pending = intent
        # Preserve the intent on any exception: external work may already have run.
        try:
            actual = self.activator.activate(deepcopy(target), idempotency_key=idempotency_key)
        except SystemActivationRejected as exc:
            self._rejections[idempotency_key] = (candidate_id, str(exc))
            self._pending = None
            raise
        if actual != target:
            raise ValueError('Activator returned a different system manifest')
        self._current = deepcopy(target)
        self._receipts[idempotency_key] = (candidate_id, deepcopy(target))
        self._pending = None
        return deepcopy(target)


def system_adoption_capability(registry: SystemRegistry,
                               candidates: Callable[[], Iterable[Candidate]], *,
                               simulated: bool = False) -> Capability:
    """Explicitly bind system adoption; never install it or infer model weights.

    The live callback limits eligible candidates. Model changes must already be
    registered with an explicit ModelRef; automatic registration inherits the
    parent model. Validation is read-only and the handler rechecks the callback.
    """
    if not isinstance(registry, SystemRegistry) or not callable(candidates):
        raise TypeError('System adoption requires a registry and candidate callback')
    if type(simulated) is not bool:
        raise TypeError('simulated must be a bool')
    spec = OperationSpec('system_adopt', 'Adopt compatible candidates as one system manifest', {
        'type': 'object',
        'properties': {
            'candidate_ids': {'type': 'array', 'items': {'type': 'string', 'minLength': 1, 'pattern': r'\S'},
                              'minItems': 1, 'uniqueItems': True},
            'version': {'type': 'string', 'minLength': 1, 'pattern': r'\S'},
            'idempotency_key': {'type': 'string', 'minLength': 1, 'pattern': r'\S'},
        },
        'required': ['candidate_ids', 'version', 'idempotency_key'],
        'additionalProperties': False,
    }, read_only=False)

    def select(arguments):
        if (not isinstance(arguments, dict)
                or set(arguments) != {'candidate_ids', 'version', 'idempotency_key'}
                or not isinstance(arguments['candidate_ids'], (list, tuple))):
            raise ValueError('system_adopt requires candidate_ids, version and idempotency_key')
        ids = _candidate_ids(arguments['candidate_ids'])
        if not isinstance(arguments['version'], str) or not arguments['version'].strip():
            raise ValueError('System version must be nonempty')
        _application_key(arguments['idempotency_key'])
        known = {}
        for candidate in candidates():
            if not isinstance(candidate, Candidate):
                raise TypeError('Candidate callback must return Candidate instances')
            if candidate.candidate_id in known and known[candidate.candidate_id] != candidate:
                raise ValueError('Candidate callback contains conflicting identities')
            known[candidate.candidate_id] = deepcopy(candidate)
        if any(ident not in known for ident in ids):
            raise ValueError('system_adopt must select known candidates from its callback')
        selected = tuple(known[ident] for ident in ids)
        for candidate in selected:
            previous = registry._candidates.get(candidate.candidate_id)
            if previous is not None and previous != candidate:
                raise ValueError('Candidate identity conflicts with registered content')
        return ids, selected

    def validate(arguments):
        ids, _ = select(arguments)
        return OperationCall(spec.name, {'candidate_ids': list(ids),
            'version': arguments['version'], 'idempotency_key': arguments['idempotency_key']})

    def handle(arguments):
        ids, selected = select(arguments)
        for candidate in selected:
            if candidate.candidate_id not in registry._candidates:
                registry.register(candidate)
        actual = registry.adopt(ids, version=arguments['version'],
                                idempotency_key=arguments['idempotency_key'])
        components = ', '.join(sorted({name for candidate in selected for name in candidate.changes}))
        return OperationResult(f'Adopted {len(ids)} candidates; activated component changes: {components or "no component changes"}', (), simulated,
            data={'system': asdict(actual)}, adopted_candidate_ids=tuple(ids))

    return Capability.from_spec(spec, validate, handle)
