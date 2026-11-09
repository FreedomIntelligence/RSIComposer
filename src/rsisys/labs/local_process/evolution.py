"""Optional workspace-Lab tools for versioned candidate registration and activation."""
import hashlib
import json
from dataclasses import asdict

from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.decision import OperationSpec
from rsisys.loops.contracts.execution import LabToolError
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.runtime.storage.workspace_files import read_workspace_file


class EvolutionLab:
    """Optional lifecycle tools over an existing Lab, not another agent/loop."""
    def __init__(self, lab, registry, artifacts):
        self.lab, self.registry, self.artifacts = lab, registry, artifacts

    def __getattr__(self, name):
        return getattr(self.lab, name)

    def describe(self):
        return (*self.lab.describe(), *self.specifications())

    @staticmethod
    def specifications():
        text = {'type': 'string', 'minLength': 1}
        return (
            OperationSpec('system_status', 'Read both-side system identity and installed mutable targets.',
                {'type': 'object', 'properties': {}, 'additionalProperties': False}, True),
            OperationSpec('system_register', 'Register your own candidate.json from the workspace. '
                'Build/debug candidates with Lab tools. No candidate or reflect is required on each step. '
                'Manifest: {candidate_id,parent_system,base_version,affected_roles,components:{target:relative_json_file},'
                'models?:{role:{backend_id,model_id,revision}},model_bases?:{role:revision}}. '
                'Models require an installed ready host binding port. '
                'This freezes component bytes, validates installed ports and records ancestry; it does not activate.',
                {'type': 'object', 'properties': {'manifest_path': text},
                 'required': ['manifest_path'], 'additionalProperties': False}),
            OperationSpec('system_activate', 'Activate registered compatible candidates together. '
                'Meta targets are reflect within the same RSI optimizer; tasks, memory and accounts persist. '
                'A version receipt confirms binding activation, not training improvement.',
                {'type': 'object', 'properties': {'candidate_ids': {'type': 'array', 'items': text,
                    'minItems': 1, 'uniqueItems': True}, 'version': text, 'idempotency_key': text},
                 'required': ['candidate_ids', 'version', 'idempotency_key'], 'additionalProperties': False}))

    def execute(self, call):
        args = call.arguments
        if call.name == 'system_status':
            if args:
                raise LabToolError('system_status takes no arguments')
            return OperationResult('Current system bindings', (), False,
                data={'system': asdict(self.registry.current),
                      'installed_targets': self.registry.activator.configuration()})
        if call.name == 'system_register':
            self.lab.require_admission()
            if not isinstance(args, dict) or set(args) != {'manifest_path'}:
                raise LabToolError('Submit manifest_path only')
            def read(relative):
                return read_workspace_file(self.workspace, relative)
            try:
                spec = json.loads(read(args['manifest_path']))
                required = {'candidate_id', 'parent_system', 'base_version', 'affected_roles', 'components'}
                if (not isinstance(spec, dict) or not required <= set(spec)
                        or set(spec) - required - {'models', 'model_bases'}):
                    raise ValueError('Invalid system candidate manifest fields')
                def text(value):
                    return isinstance(value, str) and bool(value.strip()) and '\x00' not in value
                if any(not text(spec[k]) for k in ('candidate_id', 'parent_system', 'base_version')):
                    raise ValueError('Candidate, parent and base identities must be nonempty strings')
                roles = spec['affected_roles']
                if (not isinstance(roles, list) or any(not text(v) for v in roles)
                        or len(set(roles)) != len(roles)):
                    raise ValueError('affected_roles must be an array of unique nonempty role names')
                components = spec['components']
                if not isinstance(components, dict) or any(not text(k) or not text(v)
                        for k, v in components.items()):
                    raise ValueError('components must map target names to workspace JSON file paths')
                raw_models, bases = spec.get('models', {}), spec.get('model_bases', {})
                if (not isinstance(raw_models, dict) or not isinstance(bases, dict)
                        or any(not text(k) or k not in roles or not text(v) for k, v in bases.items())):
                    raise ValueError('Model bases must map affected roles to nonempty revisions')
                models = {}
                for name, raw in raw_models.items():
                    if not text(name) or name not in self.registry.activator.models or name not in roles:
                        raise ValueError('Model target needs an installed port and an affected role')
                    if (not isinstance(raw, dict) or set(raw) != {'backend_id', 'model_id', 'revision'}
                            or any(not text(v) for v in raw.values())):
                        raise ValueError('Models require nonempty backend_id, model_id and revision')
                    model = ModelRef(**raw)
                    self.registry.activator.models[name].prepare(asdict(model))
                    models[name] = model
                raw_components = {}
                for name, path in spec['components'].items():
                    if name not in self.registry.activator.ports:
                        raise ValueError('Target is not installed: ' + name)
                    role = self.registry.activator.ports[name].role
                    if role is not None and role not in spec['affected_roles']:
                        raise ValueError('Candidate affected_roles omit target role: ' + role)
                    raw = read(path)
                    self.registry.activator.ports[name].prepare(json.loads(raw))
                    raw_components[name] = raw
                self.lab.require_admission()
                refs = {}
                for name, raw in raw_components.items():
                    previous = self.registry._candidates.get(spec['candidate_id'])
                    prior_ref = previous.changes.get(name) if previous is not None else None
                    refs[name] = (prior_ref if prior_ref is not None and
                        prior_ref.digest == hashlib.sha256(raw).hexdigest() else
                        self.artifacts.put(raw, kind='system_component'))
                candidate = Candidate(spec['candidate_id'], spec['base_version'], refs,
                    tuple(roles), spec['parent_system'], bases)
                self.lab.require_admission()
                system = self.registry.register(candidate, models=models)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                raise LabToolError(str(exc)) from exc
            return OperationResult('Candidate frozen and registered; not activated', tuple(refs.values()),
                                   False, data={'system': asdict(system)})
        if call.name == 'system_activate':
            self.lab.require_admission()
            if (not isinstance(args, dict) or set(args) != {'candidate_ids', 'version', 'idempotency_key'}
                    or not isinstance(args['candidate_ids'], list)
                    or any(not isinstance(args[k], str) or not args[k].strip()
                           for k in ('version', 'idempotency_key'))):
                raise LabToolError('Submit candidate_ids, version and idempotency_key')
            try:
                system = self.registry.adopt(args['candidate_ids'], version=args['version'],
                                             idempotency_key=args['idempotency_key'])
            except (ValueError, KeyError) as exc:
                raise LabToolError(str(exc)) from exc
            return OperationResult('System bindings activated and read back', (), False,
                data={'system': asdict(system)}, adopted_candidate_ids=tuple(args['candidate_ids']))
        return self.lab.execute(call)
