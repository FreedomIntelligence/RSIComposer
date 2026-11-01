"""Versioned component application, independent of either loop or training engine."""
from copy import deepcopy
from typing import Protocol

from rsisys.loops.contracts.candidate import Candidate, SystemVersion
from rsisys.loops.contracts.model_service import ModelRef


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
        self._receipts = {}
        self._rejections = {}
        self._pending = None

    @property
    def current(self):
        return deepcopy(self._current)

    def get(self, version: str):
        return deepcopy(self._versions[version])

    def register(self, candidate: Candidate, *, model: ModelRef | None = None):
        if not isinstance(candidate, Candidate) or not candidate.parent_system:
            raise ValueError('System candidate requires an explicit parent_system')
        parent = self._versions[candidate.parent_system]
        target = SystemVersion(candidate.candidate_id,
            {**parent.components, **candidate.changes},
            model if model is not None else parent.model, parent.version)
        previous = self._versions.get(target.version)
        if previous is not None:
            if previous != target or self._candidates.get(target.version) != candidate:
                raise ValueError('System version identity conflicts with existing content')
            return deepcopy(previous)
        self._versions[target.version] = deepcopy(target)
        self._candidates[target.version] = deepcopy(candidate)
        return deepcopy(target)

    def activate(self, candidate_id: str, *, idempotency_key: str):
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise ValueError('Application requires an idempotency key')
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
        if candidate_id not in self._candidates:
            raise ValueError('Only registered candidates can be applied')
        intent = (candidate_id, idempotency_key)
        if self._pending is not None and self._pending != intent:
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
