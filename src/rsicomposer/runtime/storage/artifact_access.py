"""Host-scoped artifact reads and write-time split labels for trusted plugins."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
import threading

from rsicomposer.loops.contracts.artifacts import ArtifactRef


class ArtifactAuthority:
    def __init__(self, store):
        self.store = store
        self._scope = ContextVar('rsisys-artifact-scope', default=None)
        self._lock, self._active = threading.RLock(), {}

    def component(self, name):
        return ComponentArtifacts(self, name)

    @contextmanager
    def scope(self, component, *, split, role, finishing):
        scope = {'owner': component, 'split': split, 'sealed': role is not None and not role.per_item_results,
                 'finishing': finishing}
        token, ident = self._scope.set(scope), object()
        with self._lock:
            self._active[ident] = scope
        try:
            yield
        finally:
            self._scope.reset(token)
            with self._lock:
                self._active.pop(ident, None)

    def current(self, component):
        current = self._scope.get()
        if current is not None and current['owner'] == component:
            return dict(current)
        with self._lock:
            candidates = [scope for scope in self._active.values() if scope['owner'] == component]
        if len(candidates) > 1:
            raise PermissionError('Ambiguous artifact operation scope')
        return dict(candidates[0]) if candidates else {'owner': component, 'split': None,
                                                      'sealed': False, 'finishing': False}

    def capture_scope(self, component):
        """Capture the host operation for a remote capability, never a client label."""
        current = self._scope.get()
        with self._lock:
            candidates = [(ident, scope) for ident, scope in self._active.items()
                          if scope['owner'] == component and (current is None or scope is current)]
            if len(candidates) > 1:
                raise PermissionError('Ambiguous remote artifact operation scope')
            if candidates:
                ident, scope = candidates[0]
                return component, ident, dict(scope)
        return component, None, {'owner': component, 'split': None, 'sealed': False, 'finishing': False}

    def captured_scope_active(self, capability):
        component, ident, scope = capability
        with self._lock:
            return ident is None or self._active.get(ident) == scope

    @contextmanager
    def use_captured_scope(self, capability):
        component, ident, scope = capability
        with self._lock:
            if ident is not None and self._active.get(ident) != scope:
                raise PermissionError('Remote operation scope has ended')
        token = self._scope.set(scope)
        try:
            yield
        finally:
            self._scope.reset(token)

    def restrict_result(self, value, *, owner, split):
        """Also mark pre-existing references returned by a sealed operation."""
        if isinstance(value, ArtifactRef):
            self.store.restrict(value, owner=owner, split=split)
        elif is_dataclass(value):
            self.restrict_result(asdict(value), owner=owner, split=split)
        elif isinstance(value, dict):
            fields = {'artifact_id', 'kind', 'uri', 'version', 'digest'}
            if set(value) == fields:
                self.store.restrict(ArtifactRef(**value), owner=owner, split=split)
            else:
                for item in value.values():
                    self.restrict_result(item, owner=owner, split=split)
        elif isinstance(value, (tuple, list)):
            for item in value:
                self.restrict_result(item, owner=owner, split=split)


class ComponentArtifacts:
    """No raw-store delegation: every read/resolve/enumeration checks the same ACL."""
    def __init__(self, authority, component):
        self._authority, self.component = authority, component

    def _check(self, ref):
        access = self._authority.store.access(ref)
        if access['sealed']:
            scope = self._authority.current(self.component)
            if (access['owner'] != self.component or not scope['finishing'] or not scope['sealed']
                    or access['split'] != scope['split']):
                raise PermissionError('Sealed per-item artifact is restricted to its host evaluation')

    def read(self, ref):
        self._check(ref)
        return self._authority.store.read(ref)

    def resolve(self, ref):
        self._check(ref)
        return self._authority.store.resolve(ref)

    def refs(self):
        visible = []
        for ref in self._authority.store.refs():
            try:
                self._check(ref)
            except PermissionError:
                continue
            visible.append(ref)
        return tuple(visible)

    def put(self, content, *, kind):
        scope = self._authority.current(self.component)
        return self._authority.store.put(content, kind=kind,
            access={key: scope[key] for key in ('owner', 'split', 'sealed')})

    def put_file(self, path, *, kind):
        scope = self._authority.current(self.component)
        return self._authority.store.put_file(path, kind=kind,
            access={key: scope[key] for key in ('owner', 'split', 'sealed')})
