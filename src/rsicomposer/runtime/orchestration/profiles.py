"""Host-owned validation and history for a single Lab instance."""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json

from rsicomposer.loops.contracts.coordination import OperationResult
from rsicomposer.loops.contracts.data import DatasetBinding
from rsicomposer.loops.contracts.lifecycle import ExecutionStatus
from rsicomposer.loops.contracts.model_service import ModelRef
from rsicomposer.loops.contracts.metrics import validated_metrics
from rsicomposer.loops.contracts.profiles import Invariant, MetricSpec, SPLIT_ROLES, SplitRole, get_profile
from rsicomposer.runtime.capabilities import optional_method


def _optional(value, name, default):
    method = optional_method(value, name)
    return default if method is None else method()


def _typed(value, kind):
    kinds = kind if isinstance(kind, list) else [kind]
    return any({'string': isinstance(value, str), 'integer': type(value) is int,
                'number': type(value) in (int, float), 'boolean': type(value) is bool,
                'object': isinstance(value, dict), 'array': isinstance(value, list),
                'null': value is None}.get(name, False) for name in kinds)


def evaluation_operations(lab, catalog, label):
    declarations = _optional(lab, 'evaluation_operations', ())
    operations = {spec.name for spec in catalog}
    if (not isinstance(declarations, tuple) or any(not isinstance(name, str) or name not in operations
            for name in declarations) or len(set(declarations)) != len(declarations)):
        raise ValueError(f'Lab {label} evaluation_operations() must name distinct declared operations')
    return frozenset(declarations)


class ProfileBinding:
    def __init__(self, alias, lab, catalog):
        self.alias = alias
        names = _optional(lab, 'profiles', ())
        if (not isinstance(names, tuple) or any(not isinstance(name, str) for name in names)
                or len(set(names)) != len(names)):
            raise TypeError(f'Lab {alias} profiles() must return distinct profile IDs as a tuple')
        self.profiles = tuple(get_profile(name) for name in names)
        for profile in self.profiles:
            try:
                profile.validate_catalog(catalog)
            except ValueError as error:
                raise ValueError(f'Lab {alias}: {error}') from error
        operations = {spec.name for spec in catalog}
        self.declared_evaluations = evaluation_operations(lab, catalog, alias)
        self.evaluations = self.declared_evaluations | frozenset(
            operation for profile in self.profiles for operation, fields in profile.receipt_fields.items()
            if operation in operations and fields.get('quality_metrics') == 'object')
        self.metric = _optional(lab, 'primary_metric', None)
        if self.metric is not None and not isinstance(self.metric, MetricSpec):
            raise TypeError(f'Lab {alias} primary_metric() must return MetricSpec')
        if any(profile.primary_metric for profile in self.profiles) and self.metric is None:
            raise ValueError(f'Lab {alias} must declare primary_metric()')
        self.model_roles = _optional(lab, 'model_roles', {})
        if (not isinstance(self.model_roles, Mapping) or any(not isinstance(name, str) or not name
                or not isinstance(model, ModelRef) for name, model in self.model_roles.items())):
            raise TypeError(f'Lab {alias} model_roles() must map names to ModelRef')
        self.roles = {}
        self.rules = []
        for profile in self.profiles:
            self.roles.update(profile.split_roles)
            self.rules.extend(profile.invariants)
        declared_roles = _optional(lab, 'split_roles', {})
        if not isinstance(declared_roles, Mapping):
            raise TypeError(f'Lab {alias} split_roles() must return a mapping')
        for name, role in declared_roles.items():
            if not isinstance(name, str) or not name or not isinstance(role, SplitRole):
                raise TypeError(f'Lab {alias} split_roles() must map split names to SplitRole')
            # Sealed profile roles cannot be weakened by a concrete Lab.
            if name in self.roles and self.roles[name].after_finish_only and role != self.roles[name]:
                raise ValueError(f'Lab {alias} cannot weaken sealed split {name}')
            self.roles[name] = role
        bindings = _optional(lab, 'data_bindings', ())
        if not isinstance(bindings, tuple) or any(not isinstance(item, DatasetBinding) for item in bindings):
            raise TypeError(f'Lab {alias} data_bindings() must return a DatasetBinding tuple')
        self.data_bindings = []
        seen = set()
        for item in sorted(bindings, key=lambda item: (item.name, item.split)):
            key = (item.name, item.split)
            if key in seen:
                raise ValueError(f'Lab {alias} data_bindings() must use distinct name/split pairs')
            seen.add(key)
            role = self.roles.get(item.split)
            if role is None or role != SPLIT_ROLES[item.role]:
                raise ValueError(f'Lab {alias} dataset role disagrees with split {item.split}')
            identity = item.identity
            if not role.inputs_visible:
                # Retain the recovery check without serializing private cohort metadata.
                fingerprint = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                identity = {'name': item.name, 'split': item.split, 'role': item.role,
                            'binding_sha256': fingerprint}
            self.data_bindings.append(identity)
        rules = _optional(lab, 'invariants', ())
        if not isinstance(rules, tuple) or any(not isinstance(rule, Invariant) for rule in rules):
            raise TypeError(f'Lab {alias} invariants() must return an Invariant tuple')
        self.rules = tuple(dict.fromkeys((*self.rules, *rules)))
        self.baseline = False
        self.candidates = set()
        self.once = set()
        self.measured = set()
        self.selected = None

    def identity(self):
        return {**({'model_roles': {name: asdict(model) for name, model in sorted(self.model_roles.items())}}
                   if self.model_roles else {}),
                **({'data_bindings': deepcopy(self.data_bindings)}
                   if self.data_bindings else {}),
                **({'evaluation_operations': sorted(self.declared_evaluations)} if self.declared_evaluations else {}),
                'profiles': [p.identity for p in self.profiles],
                'primary_metric': asdict(self.metric) if self.metric is not None else None,
                'split_roles': {name: asdict(role) for name, role in sorted(self.roles.items())},
                'invariants': [{**asdict(rule), 'key': list(rule.key)} for rule in self.rules]}

    def state(self):
        return {'baseline': self.baseline, 'candidates': sorted(self.candidates), 'once': sorted(self.once),
                'measured': sorted(self.measured), 'selected': self.selected}

    def restore(self, state):
        if (not isinstance(state, dict) or set(state) != {'baseline', 'candidates', 'once', 'measured', 'selected'}
                or type(state['baseline']) is not bool
                or any(not isinstance(keys, list) or any(not isinstance(key, str) for key in keys)
                       for keys in (state['candidates'], state['once'], state['measured']))
                or (state['selected'] is not None and state['selected'] not in state['measured'])):
            raise ValueError('Invalid Lab profile history')
        self.baseline = state['baseline']
        self.candidates, self.once = set(state['candidates']), set(state['once'])
        self.measured, self.selected = set(state['measured']), state['selected']

    def role(self, call):
        split = call.arguments.get('split')
        if split is None:
            return None
        role = self.roles.get(split) if isinstance(split, str) else None
        if self.profiles and role is None:
            raise ValueError(f'Lab {self.alias}: undeclared split {split}')
        return role

    def _once_key(self, rule, call):
        if any(key not in call.arguments for key in rule.key):
            raise ValueError(f'once_per requires request fields {rule.key}')
        return json.dumps([call.name, rule.key, [call.arguments[k] for k in rule.key]], sort_keys=True)

    @staticmethod
    def _candidate_key(call):
        value = call.arguments.get('candidate_id', call.arguments.get('model', call.arguments.get('artifact')))
        return json.dumps(value, sort_keys=True) if value is not None else None

    def check_finish(self, plan):
        for rule in sorted(self.rules, key=lambda rule: rule.kind != 'min_candidates'):
            if rule.kind == 'min_candidates' and len(self.candidates) < rule.count:
                raise ValueError(f'Final evaluation requires at least {rule.count} measured candidates')
            if rule.kind == 'measured_selection' and (self.selected is None or not plan):
                raise ValueError('Controller finish requires a measured selection and sealed evaluation plan')

    def check(self, call, *, finishing=False, history=True):
        role = self.role(call)
        if role is not None and role.after_finish_only and not finishing:
            raise ValueError(f'Lab {self.alias}: sealed split is available only to the host after Controller finish')
        if finishing and (role is None or not role.after_finish_only):
            raise ValueError('Host final evaluations must target a declared sealed split')
        if role is not None and not role.labels_visible and call.arguments.get('condition', 'none') != 'none':
            raise ValueError('Unlabeled split does not permit label-conditioned generation')
        if not history:
            return
        for rule in self.rules:
            if rule.kind == 'baseline_before' and rule.operation == call.name and not self.baseline:
                raise ValueError(f'{call.name} requires a successful baseline evaluation')
            if rule.kind == 'min_candidates' and finishing and len(self.candidates) < rule.count:
                raise ValueError(f'Final evaluation requires at least {rule.count} measured candidates')
            if rule.kind == 'measured_selection' and call.name == 'select':
                if self._candidate_key(call) not in self.measured:
                    raise ValueError('Selection requires a successfully measured candidate')
            if rule.kind == 'once_per' and (not rule.operation or rule.operation == call.name):
                if self._once_key(rule, call) in self.once:
                    raise ValueError(f'{call.name} violates once_per{rule.key}')

    def validate_result(self, call, result):
        if not isinstance(result, OperationResult) or result.status != ExecutionStatus.COMPLETED:
            return result
        try:
            for profile in self.profiles:
                for name, kind in profile.receipt_fields.get(call.name, {}).items():
                    if name not in result.data or not _typed(result.data[name], kind):
                        raise ValueError(f'{profile.identity}: {call.name} receipt requires {name} ({kind})')
            unscored = any(p.name == 'artifact-search' for p in self.profiles) and result.data.get('scored') is False
            if call.name == 'evaluate' and self.metric is not None and not unscored:
                self.metric.value(result.data)
            if 'metrics' in result.data:
                validated_metrics(result.data['metrics'], 'data.metrics')
            if call.name == 'evaluate' and result.data.get('valid') is False and result.data.get('scored'):
                raise ValueError('An invalid candidate cannot be reported as scored')
        except ValueError as error:
            return replace(result, status=ExecutionStatus.FAILED, error=str(error), adopted_candidate_ids=())
        return result

    def observe(self, call, result):
        if not isinstance(result, OperationResult) or result.status != ExecutionStatus.COMPLETED:
            return
        if call.name == 'evaluate' and result.data.get('scored', True):
            candidate = self._candidate_key(call)
            if candidate is not None:
                self.measured.add(candidate)
            if result.data.get('baseline', call.arguments.get('baseline')) is True:
                self.baseline = True
            elif candidate is not None:
                self.candidates.add(candidate)
        if call.name == 'select' and any(rule.kind == 'measured_selection' for rule in self.rules):
            self.selected = self._candidate_key(call)
        for rule in self.rules:
            if rule.kind == 'once_per' and (not rule.operation or rule.operation == call.name):
                self.once.add(self._once_key(rule, call))

    def aggregate(self, call, result, artifacts):
        """Keep sealed rows in a separately marked artifact, never in receipts."""
        role = self.role(call)
        if role is None or role.per_item_results:
            return result
        if not isinstance(result, OperationResult):
            raise TypeError('Sealed evaluations must return a terminal OperationResult')
        if artifacts is None:
            raise TypeError('Sealed evaluation requires the host artifact store')
        ref = artifacts.put(json.dumps(result.payload(), ensure_ascii=False, allow_nan=False).encode(),
                            kind='sealed-evaluation')
        data = {key: deepcopy(result.data[key]) for key in ('quality_metrics', 'training_metrics',
                'execution_metrics', 'metrics', 'valid', 'scored', 'count') if key in result.data}
        return replace(result, summary='Sealed evaluation receipt', artifacts=(ref,), data=data,
                       error='Sealed evaluation failed; see restricted evidence' if result.error else None,
                       adopted_candidate_ids=(), feedback_delivery=None)
