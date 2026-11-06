"""External access configuration; operation semantics remain in definitions/handlers."""
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from inspect import getattr_static
from typing import Literal


@dataclass(frozen=True)
class SensitivePolicy:
    """Serializable explicit deny rules; everything else stays available.

    Each rule matches phase/kind/name and optionally an exact argument value or
    a canonical filesystem subtree. This is tool admission, not process isolation.
    Custom Python callbacks still need their own state integration.
    """
    policy_id: str
    version: str
    rules: tuple[dict, ...] = ()

    def __post_init__(self):
        import json
        if not all(isinstance(v, str) and v.strip() for v in (self.policy_id, self.version)):
            raise ValueError('Permission policy needs an ID and version')
        rules = json.loads(json.dumps(self.rules, allow_nan=False))
        for rule in rules:
            if (not isinstance(rule, dict) or not rule or set(rule) - {
                    'phase', 'kind', 'name', 'argument', 'equals', 'path_prefix'}
                    or ('equals' in rule and 'path_prefix' in rule)
                    or (('equals' in rule or 'path_prefix' in rule) != ('argument' in rule))):
                raise ValueError('Invalid sensitive permission rule')
            if any(not isinstance(rule[k], str) or not rule[k] for k in rule if k != 'equals'):
                raise ValueError('Permission selectors must be nonempty strings')
            if 'phase' in rule and rule['phase'] not in ('decision', 'review'):
                raise ValueError('Invalid permission phase')
            if 'kind' in rule and rule['kind'] not in ('operation', 'directive', 'control'):
                raise ValueError('Invalid permission kind')
            if 'path_prefix' in rule:
                from pathlib import Path
                if not Path(rule['path_prefix']).is_absolute():
                    raise ValueError('Sensitive path prefixes must be absolute')
        object.__setattr__(self, 'rules', tuple(rules))

    def __call__(self, request):
        from pathlib import Path
        for rule in self.rules:
            if any(getattr(request, k) != rule[k] for k in ('phase', 'kind', 'name') if k in rule):
                continue
            if 'argument' in rule:
                # Discovery cannot decide an argument-dependent rule.
                if request.action is None:
                    continue
                arguments = getattr(request.action, 'arguments', {})
                if rule['argument'] not in arguments:
                    continue
                value = arguments[rule['argument']]
                if 'equals' in rule and value != rule['equals']:
                    continue
                if 'path_prefix' in rule and (not isinstance(value, str) or not
                        Path(value).resolve().is_relative_to(Path(rule['path_prefix']).resolve())):
                    continue
            return False
        return True

    def to_dict(self):
        return {'protocol': 'sensitive-permission-policy/v1', 'policy_id': self.policy_id,
                'version': self.version, 'rules': deepcopy(list(self.rules))}

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict) or raw.get('protocol') != 'sensitive-permission-policy/v1' or set(raw) != {
                'protocol', 'policy_id', 'version', 'rules'}:
            raise ValueError('Unknown sensitive permission policy')
        return cls(raw['policy_id'], raw['version'], tuple(raw['rules']))


@dataclass(frozen=True)
class AccessRequest:
    phase: Literal['decision', 'review']
    kind: Literal['operation', 'directive', 'control']
    name: str
    read_only: bool = False
    # None during discovery; a normalized OperationCall/Directive/ReviewDecision at execution.
    action: object | None = None

    def __post_init__(self):
        if self.phase not in ('decision', 'review') or self.kind not in ('operation', 'directive', 'control'):
            raise ValueError('Unknown access phase or kind')
        if not isinstance(self.name, str) or not self.name or type(self.read_only) is not bool:
            raise ValueError('Invalid access operation metadata')


@dataclass(frozen=True)
class ActionPermissions:
    operations: frozenset[str] | None = None
    directives: frozenset[str] | None = None
    review_operations: frozenset[str] | Literal['read_only', 'all'] = 'read_only'
    allow_cancel: bool = True
    check: Callable[[AccessRequest], bool] | None = None

    def __post_init__(self):
        for name in ('operations', 'directives', 'review_operations'):
            value = getattr(self, name)
            if name == 'review_operations' and isinstance(value, str) and value in ('read_only', 'all'):
                continue
            if value is None and name != 'review_operations':
                continue
            if not isinstance(value, (set, frozenset, tuple, list)) or any(
                    not isinstance(v, str) or not v for v in value):
                raise ValueError(f'{name} must contain operation/mode names')
            object.__setattr__(self, name, frozenset(value))
        if type(self.allow_cancel) is not bool:
            raise TypeError('allow_cancel must be a bool')
        if self.check is not None and not callable(self.check):
            raise TypeError('check must be callable')

    @classmethod
    def unrestricted(cls):
        """Open all installed operations/modes; does not install tools or bypass lifecycle rules."""
        return cls(review_operations='all')

    def allows(self, request: AccessRequest) -> bool:
        if request.kind == 'operation':
            if self.operations is not None and request.name not in self.operations:
                return False
            if request.phase == 'review':
                selection = self.review_operations
                if selection == 'read_only' and not request.read_only:
                    return False
                if isinstance(selection, frozenset) and request.name not in selection:
                    return False
        elif request.kind == 'directive':
            if self.directives is not None and request.name not in self.directives:
                return False
        elif request.kind == 'control':
            if request.name == 'continue':
                return True  # Continuing makes no new mutation; it is always a valid fallback.
            if request.name != 'cancel' or not self.allow_cancel:
                return False
        if self.check is None:
            return True
        allowed = self.check(deepcopy(request))
        if type(allowed) is not bool:
            raise TypeError('Permission check must return bool')
        return allowed

    def to_dict(self):
        if self.check is not None and not isinstance(self.check, SensitivePolicy):
            raise TypeError('Custom permission callbacks need explicit snapshot integration')
        result = {'operations': sorted(self.operations) if self.operations is not None else None,
                'directives': sorted(self.directives) if self.directives is not None else None,
                'review_operations': sorted(self.review_operations) if isinstance(self.review_operations, frozenset)
                                     else self.review_operations,
                'allow_cancel': self.allow_cancel}
        if self.check is not None:
            result['sensitive_policy'] = self.check.to_dict()
        return result

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict) or set(raw) - {
                'operations', 'directives', 'review_operations', 'allow_cancel', 'sensitive_policy'}:
            raise ValueError('Unknown action permission fields')
        raw = deepcopy(raw)
        policy = raw.pop('sensitive_policy', None)
        return cls(**raw, check=SensitivePolicy.from_dict(policy) if policy is not None else None)


class AuthorizedActionSpace:
    """A phase view over any existing Action Space. Reads permissions anew on every call."""
    def __init__(self, source, phase='decision', *, permissions: ActionPermissions | None = None):
        if phase not in ('decision', 'review'):
            raise ValueError('Unknown action phase')
        self.source, self.phase, self._permissions = source, phase, permissions

    @property
    def permissions(self):
        # Optional capability must be explicitly declared; dynamic proxies may
        # fabricate arbitrary attributes (including an unrelated permissions object).
        value = self._permissions if self._permissions is not None else (
            getattr(self.source, 'permissions') if getattr_static(self.source, 'permissions', None) is not None
            else ActionPermissions())
        if not isinstance(value, ActionPermissions):
            raise TypeError('Expected ActionPermissions')
        return value

    def _request(self, kind, name, *, read_only=False, action=None):
        return AccessRequest(self.phase, kind, name, read_only, action)

    def describe(self):
        return tuple(spec for spec in self.source.describe() if self.permissions.allows(
            self._request('operation', spec.name, read_only=spec.read_only)))

    def validate(self, call):
        normalized = self.source.validate(call)
        spec = next((spec for spec in self.source.describe() if spec.name == normalized.name), None)
        if not self.permissions.allows(self._request(
                'operation', normalized.name, read_only=spec.read_only if spec is not None else False,
                action=normalized)):
            raise PermissionError(f'Operation is not permitted during {self.phase}: {normalized.name}')
        return normalized

    def directive_modes(self):
        return tuple(mode for mode in self.source.directive_modes() if self.permissions.allows(
            self._request('directive', mode)))

    def validate_directive(self, directive):
        self.source.validate_directive(directive)
        if not self.permissions.allows(self._request('directive', directive.mode, action=directive)):
            raise PermissionError(f'Directive is not permitted: {directive.mode}')

    def review_actions(self):
        return tuple(name for name in ('continue', 'cancel') if self.permissions.allows(
            self._request('control', name)))

    def validate_review(self, result):
        if not self.permissions.allows(self._request('control', result.action, action=result)):
            raise PermissionError(f'Review control is not permitted: {result.action}')


def authorized_space(source, phase='decision'):
    if isinstance(source, AuthorizedActionSpace):
        return AuthorizedActionSpace(source.source, phase, permissions=source._permissions)
    return AuthorizedActionSpace(source, phase)
