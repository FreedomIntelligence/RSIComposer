"""External access configuration; operation semantics remain in definitions/handlers."""
from collections.abc import Callable
from copy import copy, deepcopy
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
            if 'phase' in rule and rule['phase'] not in ('decision',):
                raise ValueError('Invalid permission phase')
            if 'kind' in rule and rule['kind'] not in ('operation',):
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
    phase: Literal['decision']
    kind: Literal['operation']
    name: str
    read_only: bool = False
    # None during discovery; a normalized OperationCall at execution.
    action: object | None = None

    def __post_init__(self):
        if self.phase not in ('decision',) or self.kind not in ('operation',):
            raise ValueError('Unknown access phase or kind')
        if not isinstance(self.name, str) or not self.name or type(self.read_only) is not bool:
            raise ValueError('Invalid access operation metadata')


@dataclass(frozen=True)
class ActionPermissions:
    operations: frozenset[str] | None = None
    check: Callable[[AccessRequest], bool] | None = None

    def __post_init__(self):
        if self.operations is not None:
            if not isinstance(self.operations, (set, frozenset, tuple, list)) or any(
                    not isinstance(v, str) or not v for v in self.operations):
                raise ValueError('operations must contain operation names')
            object.__setattr__(self, 'operations', frozenset(self.operations))
        if self.check is not None and not callable(self.check):
            raise TypeError('check must be callable')

    @classmethod
    def unrestricted(cls):
        """Open all installed operations/modes; does not install tools or bypass lifecycle rules."""
        return cls()

    def allows(self, request: AccessRequest) -> bool:
        if request.kind == 'operation':
            if self.operations is not None and request.name not in self.operations:
                return False
        if self.check is None:
            return True
        allowed = self.check(deepcopy(request))
        if type(allowed) is not bool:
            raise TypeError('Permission check must return bool')
        return allowed

    def to_dict(self):
        if type(self) is not ActionPermissions or (self.check is not None and type(self.check) is not SensitivePolicy):
            raise TypeError('Recovery, snapshots and nested workers require serializable ActionPermissions; '
                            'custom permission callbacks are unsupported')
        result = {'operations': sorted(self.operations) if self.operations is not None else None}
        if self.check is not None:
            result['sensitive_policy'] = self.check.to_dict()
        return result

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict) or set(raw) - {
                'operations', 'sensitive_policy'}:
            raise ValueError('Unknown action permission fields')
        raw = deepcopy(raw)
        policy = raw.pop('sensitive_policy', None)
        return cls(**raw, check=SensitivePolicy.from_dict(policy) if policy is not None else None)


def restore_action_permissions(state):
    """Restore Host admission before assembly. Older state did not record the policy."""
    if 'action_permissions' not in state:
        return ActionPermissions(operations=frozenset())
    raw = state['action_permissions']
    permissions = ActionPermissions.from_dict(raw)
    if permissions.to_dict() != raw:
        raise ValueError('Invalid saved action permissions')
    return permissions


def _same_permissions(left, right):
    # Do not invoke custom equality methods when deduplicating checks.
    return left is right or (type(left) is ActionPermissions and type(right) is ActionPermissions
        and left.operations == right.operations and (left.check is right.check or (
            type(left.check) is SensitivePolicy and type(right.check) is SensitivePolicy
            and left.check == right.check)))


class AuthorizedActionSpace:
    """A phase view over any existing Action Space. Reads permissions anew on every call."""
    def __init__(self, source, phase='decision', *, permissions: ActionPermissions | None = None,
                 required_permissions=()):
        if phase not in ('decision',):
            raise ValueError('Unknown action phase')
        self.source, self.phase, self._permissions = source, phase, permissions
        self._required_permissions = required_permissions

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

    def with_definitions(self, definitions):
        # Check collisions in the full catalog, then retain every permission layer.
        extended = copy(self)
        extended.source = self.source.with_definitions(definitions)
        return extended

    def _request(self, kind, name, *, read_only=False, action=None):
        return AccessRequest(self.phase, kind, name, read_only, action)

    def _policies(self):
        local = self.permissions
        if any(_same_permissions(local, policy) for policy in self._required_permissions):
            return self._required_permissions
        return (*self._required_permissions, local)

    def describe(self):
        policies = self._policies()
        return tuple(spec for spec in self.source.describe() if all(policy.allows(
            self._request('operation', spec.name, read_only=spec.read_only)) for policy in policies))

    def validate(self, call):
        normalized = self.source.validate(call)
        spec = next((spec for spec in self.source.describe() if spec.name == normalized.name), None)
        request = self._request('operation', normalized.name,
                                read_only=spec.read_only if spec is not None else False, action=normalized)
        if not all(policy.allows(request) for policy in self._policies()):
            raise PermissionError(f'Operation is not permitted during {self.phase}: {normalized.name}')
        return normalized


def authorized_space(source, phase='decision'):
    if isinstance(source, AuthorizedActionSpace):
        if source.phase == phase:
            return source
        return AuthorizedActionSpace(source.source, phase, permissions=source._permissions,
                                     required_permissions=source._required_permissions)
    return AuthorizedActionSpace(source, phase)


def enforce_host_permissions(action_space, policy):
    """Require the Host's policy in addition to any endpoint-local restrictions."""
    if not isinstance(policy, ActionPermissions):
        raise TypeError('Expected ActionPermissions')
    if type(action_space) is not AuthorizedActionSpace:
        # Preserve validation and discovery if an endpoint supplies its own subclass.
        return AuthorizedActionSpace(action_space, required_permissions=(policy,))
    space = action_space
    if any(_same_permissions(policy, existing) for existing in space._required_permissions):
        return space
    return AuthorizedActionSpace(space.source, space.phase, permissions=space._permissions,
                                 required_permissions=(*space._required_permissions, policy))
