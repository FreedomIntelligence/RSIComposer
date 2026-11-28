"""MetaRSI domain-agent operators over frozen math cohorts and verified rollouts."""
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from unicodedata import normalize

from rsicomposer.api import MathAttempt, MathLab, ModelRef, OperationCall, OperationResult, OperationSpec, json_bytes
from rsicomposer.labs.math.tasks import load_splits
from rsicomposer.loops.contracts.domain_agent import DomainAgentSpec, runtime_identity
from rsicomposer.loops.contracts.execution import LabToolError
from rsicomposer.loops.contracts.profiles import SPLIT_ROLES
from rsicomposer.runtime.storage.files import atomic_write

from . import aime, aime_synthesis
from .engine import (
    FixtureEngine, TransformersEngine, digest, fit_checkpoint, inherit_checkpoint, model_identity, model_stamp, read_checkpoint,
)
from .trial_inputs import mounted_inputs, validate_inputs, validate_seed_id


PROFILE = 'meta-rsi/v1'
OPERATORS = ('data', 'harness', 'model')
PROFILES = ('compact', 'detailed')
MATH_ROLES = {'train': 'train', 'search': 'dev', 'validation': 'dev', 'test': 'sealed'}
HARNESSES = {
    'seed': 'Solve the problem.',
    'compact': 'Solve the problem. Identify which operation is required before calculating.',
    'detailed': 'Solve the problem carefully. Identify each operation, work it out, and check your result.',
}
MAX_ROWS, MAX_COMPLETION = aime.MAX_ROWS, aime.MAX_COMPLETION
DEFAULT_OPTIONS = {'device': 'cpu', 'model_kind': 'causal_lm', 'seed': 73, 'max_new_tokens': 256, 'max_length': 2048,
                   'generation_batch_size': 1, 'train_batch_size': 1, 'gradient_accumulation_steps': 1,
                   'learning_rate': 2e-4, 'steps': 1, 'temperature': 0.7}


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _check_harness(value):
    if (not isinstance(value, dict) or set(value) != {'instruction', 'few_shot'}
            or value['instruction'] not in tuple(HARNESSES) or type(value['few_shot']) is not bool):
        raise ValueError('MetaRSI harness must select a bounded instruction and a boolean few_shot')


def _check_snapshot(value):
    if (not isinstance(value, dict) or set(value) != {'schema_version', 'harness', 'data'}
            or type(value['schema_version']) is not int or value['schema_version'] != 1
            or not isinstance(value['data'], list) or len(value['data']) > MAX_ROWS):
        raise ValueError('Only bounded, adapter-free MetaRSI snapshots can seed a run')
    _check_harness(value['harness'])
    for row in value['data']:
        if (not isinstance(row, dict) or set(row) != {'id', 'prompt', 'completion'}
                or not all(_text(text) for text in row.values()) or len(row['completion']) > MAX_COMPLETION):
            raise ValueError('Malformed MetaRSI verified trajectory')


def validate_config(config):
    required = {'manifest', 'split_roles'}
    if not isinstance(config, dict) or not required <= config.keys() <= required | {
            'engine', 'base_model', 'model_options', 'initial_snapshot', 'aime', 'wide_loop', 'model_driven', 'paper_kernel'}:
        raise ValueError('MetaRSI Lab requires manifest and split_roles')
    if not _text(config['manifest']) or config.get('engine', 'fixture') not in ('fixture', 'transformers'):
        raise ValueError('MetaRSI requires a math manifest and a fixture or transformers engine')
    roles = config['split_roles']
    if (not isinstance(roles, dict) or not {'train', 'search', 'validation'} <= roles.keys() <= MATH_ROLES.keys()
            or any(not _text(name) or (name in MATH_ROLES and name != role) for role, name in roles.items())
            or len(set(roles.values())) != len(roles)):
        raise ValueError('Assign distinct manifest splits to train, search, validation, and optionally test')
    if (type(config.get('model_driven', False)) is not bool
            or config.get('model_driven') and ('aime' not in config or 'test' in roles
                or 'initial_snapshot' in config or 'model_stage' in (config.get('wide_loop') or {}))):
        raise ValueError('Model-driven MetaRSI requires an AIME Lab with only train and development roles')
    if 'paper_kernel' in config:
        from .aime_release import validate_release_options
        if not config.get('model_driven'):
            raise ValueError('Paper kernel requires the model-driven AIME Lab')
        validate_release_options(config['paper_kernel'])
    real = config.get('engine', 'fixture') == 'transformers'
    if (real and not _text(config.get('base_model'))) or (not real and 'base_model' in config):
        raise ValueError('Only the transformers engine takes a pinned local base_model')
    options = config.get('model_options', {})
    wide = config.get('wide_loop')
    additions = {'enable_thinking', 'rollout_backend'}
    if wide is not None:
        from .aime_wide_loop import check_loop_config
        additions.update(check_loop_config(config, wide))
    if not isinstance(options, dict) or options.keys() - (DEFAULT_OPTIONS.keys() | additions):
        raise ValueError('Unknown MetaRSI model_options')
    for key, value in options.items():
        if key == 'device':
            valid = value == 'cpu' or (real and value == 'cuda')
        elif key == 'model_kind':
            valid = value in ('causal_lm', 'qwen3_5', 'qwen3_5_moe')
        elif key == 'enable_thinking':
            valid = type(value) is bool
        elif key == 'rollout_backend':
            valid = (value == 'vllm_graph' and 'aime' in config and real and options.get('device') == 'cuda'
                     and options.get('model_kind') == 'qwen3_5_moe')
        elif key in ('ordered_training_coverage', 'training_allow_uneven_tail'):
            valid = value is True
        elif key in ('learning_rate', 'temperature'):
            valid = type(value) in (int, float) and math.isfinite(value) and (value > 0 or key == 'temperature' and value == 0)
        else:
            valid = type(value) is int and value >= (0 if key == 'seed' else 1)
        if not valid:
            raise ValueError(f'Invalid MetaRSI model option: {key}')
    if 'aime' in config:
        options = aime.check_options(config['aime'])
        if 'initial_snapshot' in config:
            if wide is not None:
                from .aime_wide_loop import check_owned_ref
                check_owned_ref(config['initial_snapshot'])
            else:
                aime.check_snapshot(config['initial_snapshot'], options, HARNESSES)
    elif 'initial_snapshot' in config:
        _check_snapshot(config['initial_snapshot'])


def _manifest_subset(path, names, directory):
    """Mount selected cohorts only, preserving the original manifest's file resolution."""
    raw = json.loads(path.read_bytes())
    if (not isinstance(raw, dict) or raw.get('schema_version') != 1 or not isinstance(raw.get('splits'), dict)
            or not set(names) <= raw['splits'].keys()):
        raise ValueError('MetaRSI requires a version 1 math manifest with all assigned split names')
    if set(names) == raw['splits'].keys():
        return path

    def project(entry):
        if isinstance(entry, list):
            return [project(item) for item in entry]
        if isinstance(entry, dict):
            return {key: str(path.parent / value) if key in ('file', 'benchmarks_manifest') and isinstance(value, str)
                    else value for key, value in entry.items()}
        return entry

    subset = Path(directory) / 'manifest.json'
    subset.write_bytes(json_bytes({'schema_version': 1, 'splits': {name: project(raw['splits'][name]) for name in names}}))
    return subset


def _trial_settings(config):
    base = config.get('base_model')
    return (config.get('engine', 'fixture'), Path(base).expanduser().resolve() if base else None,
            {**DEFAULT_OPTIONS, **config.get('model_options', {})}, deepcopy(config.get('wide_loop')),
            aime.check_options(config['aime']) if 'aime' in config else None,
            config.get('model_driven', False), config.get('paper_kernel'))


def _cohort_identity(roles, rows, bindings):
    train = next(binding for binding in bindings if binding.split == roles['train'])
    train_rows = [(row.id, row.prompt, row.answer, row.verifier.spec) for row in rows[roles['train']]]
    measured = [row for role, name in roles.items() if role != 'train' for row in rows[name]]
    return ((train.content_digest, train.provenance, train_rows), {row.id for row in measured},
            {' '.join(normalize('NFKC', row.prompt).split()).casefold() for row in measured})


def _cohort_protocols(names, bindings, runtime_settings, aime_options, model_driven):
    return {role: digest({'profile': PROFILE, 'role': role, 'content': binding.content_digest,
        'benchmarks': binding.provenance['benchmarks'], 'runtime': runtime_settings,
        **({'synthesis': 'model-authored-v1' if model_driven else aime_options['data_policy']}
           if aime_options is not None and role == 'train' else {})})
        for role, name in names.items() for binding in bindings if binding.split == name}


def _load_trial_data(config, runtime):
    names = config['split_roles']
    path = Path(config['manifest']).expanduser().resolve()
    with TemporaryDirectory(prefix='meta-rsi-preflight-') as temporary:
        rows, bindings, _, _ = load_splits(_manifest_subset(path, names.values(), temporary), {
            name: SPLIT_ROLES[MATH_ROLES[role]] for role, name in names.items()}, budgets=False, runtime=runtime)
    return rows, bindings


def _check_trial(settings, parent, child_config, *, runtime=None):
    validate_config(child_config)
    if 'test' in child_config['split_roles']:
        raise ValueError('MetaRSI child trials may not mount a sealed split')
    if settings != _trial_settings(child_config):
        raise ValueError('MetaRSI trials must use the parent base model, engine and generation/training options')
    names = child_config['split_roles']
    rows, bindings = _load_trial_data(child_config, runtime)
    child = _cohort_identity(names, rows, bindings)
    if parent[0] != child[0]:
        raise ValueError('MetaRSI parent and child must share the same verified training cohort')
    if parent[1] & child[1] or parent[2] & child[2]:
        raise ValueError('MetaRSI child measurement cohorts must not overlap parent measurements')
    if runtime is not None:
        protocols = _cohort_protocols(names, bindings, runtime['settings'],
            aime.check_options(child_config['aime']) if 'aime' in child_config else None,
            child_config.get('model_driven', False))
        return mounted_inputs(names, rows, protocols)


def validate_trial_cohorts(parent_config, child_config):
    """Offline preflight. Live bindings use the parent's already mounted cohorts."""
    validate_config(parent_config)
    parent = _cohort_identity(parent_config['split_roles'], *_load_trial_data(parent_config, None))
    _check_trial(_trial_settings(parent_config), parent, child_config)


def _spec(name, description, properties, *, read_only=False):
    return OperationSpec(name, description, {'type': 'object', 'properties': properties,
        'required': list(properties), 'additionalProperties': False}, read_only=read_only, idempotent=True)


def _string(values=None):
    return {'type': 'string', **({'enum': list(values)} if values is not None else {'minLength': 1})}


@dataclass(frozen=True)
class _Runtime:
    lab: object
    name: str = 'meta-rsi-math'
    version: str = '1'
    model_access: str = 'engine'
    accepts: DomainAgentSpec = DomainAgentSpec('config', {
        key: {'type': kind, 'required': True} for key, kind in
        (('data', 'array'), ('harness', 'object'), ('model', 'object'))})

    @property
    def settings(self):
        return {'engine': self.lab.engine.name, 'base_model_sha256': self.lab.base_digest,
                **self.lab.options, 'decode': 'greedy', 'prompt': 'aime-system-prompt/v1' if self.lab.aime is not None else self.name + '/v' + self.version}

    def run(self, domain_agent, problems, run):
        return [answers[0] for answers in self.lab._generate(domain_agent, problems, count=1)]


class MetaRSILab(MathLab):
    profile = PROFILE
    source_budgets = False

    def __init__(self, context, config):
        validate_config(config)
        self._trial_settings = _trial_settings(config)
        self.aime = aime.check_options(config['aime']) if 'aime' in config else None
        self.wide = config.get('wide_loop')
        self._model_stage = None
        self.model_driven = config.get('model_driven', False)
        self.paper_kernel = config.get('paper_kernel')
        self._capacity = aime.data_capacity(self.aime)
        self.cohorts = dict(config['split_roles'])
        self.options = {**DEFAULT_OPTIONS, **config.get('model_options', {})}
        self.base_path = Path(config['base_model']).expanduser().resolve() if 'base_model' in config else None
        self._base_stamp = model_stamp(self.base_path) if self.aime is not None and self.base_path is not None else None
        self.base_digest = model_identity(self.base_path) if self.base_path is not None else None
        self._assert_base()
        engine_options = {key: value for key, value in self.options.items() if key not in ('steps', 'temperature')}
        if self.aime is not None:
            engine_options['report_truncated_attempts'] = True
        self.engine = (FixtureEngine() if self.base_path is None else
                       TransformersEngine(context, self.base_path, engine_options))
        self.root = Path(context.data_dir).resolve() / 'meta-rsi'
        self._parent_root = aime.owned_parent(self.root.parent) if self.aime is not None else None
        if self._parent_root is not None and 'test' in self.cohorts:
            raise ValueError('AIME nested trials may not mount a sealed split')
        self._lock = RLock()
        manifest = Path(config['manifest']).expanduser().resolve()
        with TemporaryDirectory(prefix='meta-rsi-manifest-') as temporary:
            super().__init__(context, {'manifest': str(_manifest_subset(manifest, self.cohorts.values(), temporary)),
                'split_roles': {name: MATH_ROLES[role] for role, name in self.cohorts.items()},
                'max_output_tokens': self.options['max_new_tokens'], 'primary_metric': {'aggregate': 'pooled-accuracy'}})
        self._train_rows = {row.id: row for row in self.datasets[self.cohorts['train']]}
        self._wide_bootstrap = None
        self._wide_bootstrap_verified = None
        self._wide_evidence = None
        if self.wide is not None:
            from .aime_wide_loop import load_bootstrap
            self._wide_bootstrap = load_bootstrap(self.wide)
            self._wide_evidence = {**self._wide_bootstrap['source'], **{key: self.wide[key]
                for key in ('checkpoint_sha256', 'aggregate_sha256')}}
            originals = [row['id'] for row in self._wide_bootstrap['rows']]
            missing = self._wide_bootstrap['failed_problem_ids']
            if len(set(originals + missing)) != 256 or set(originals + missing) != self._train_rows.keys():
                raise ValueError('Wide AIME bootstrap and non-admissions must exactly cover shared train')
        self._synthesis_sources = ({aime_synthesis.source_key(row): row for row in self._train_rows.values()}
                                   if self.aime and self.aime['data_policy'] == 'affine' else {})
        if self._synthesis_sources and len(self._synthesis_sources) != len(self._train_rows):
            raise ValueError('AIME synthesis source identity collision')
        self._candidates = {}
        self._measured = {}
        self._selected = None
        self._protocols = _cohort_protocols(self.cohorts, self.bindings, self.runtime.settings, self.aime, self.model_driven)
        self._trial_offers = {}  # Candidates are not active; only their own seed call can commit an offer.
        self._approved_trial_inputs = None  # The comparison protocol survives Controller replacement.
        # An observation about the actual mount, independent of any prospective Controller.
        self._mounted_inputs = (mounted_inputs(self.cohorts, self.datasets, self._protocols)
                                if 'test' not in self.cohorts and not self.model_driven else None)
        self._mounted_inputs_sha256 = digest(self._mounted_inputs) if self._mounted_inputs is not None else None

        seed = {'harness': {'instruction': 'seed', 'few_shot': False}, 'data': [], 'model': {'checkpoint': None}}
        inherited = deepcopy(config.get('initial_snapshot'))
        if self.aime is None:
            if inherited is not None:
                seed.update({key: inherited[key] for key in ('harness', 'data')})
            self._verify_data(seed['data'])
        else:
            seed['harness'].update(exemplars=[], revision=digest('aime-system-prompt/v1'))
            if inherited is not None:
                seed = self._import_snapshot(inherited)
            elif self.wide is not None:
                from .aime_loop_checkpoint import bootstrap_checkpoint
                bootstrap = self._wide_bootstrap
                seed['data'] = bootstrap['rows']
                aime.check_components(seed, self.aime, HARNESSES)
                self._verify_data(seed['data'])
                seed['model'] = bootstrap_checkpoint(Path(self.wide['checkpoint_run']), self.root / 'models',
                    self.base_path, self.base_digest, self._training_rows(seed['data']), self._wide_evidence)
                self._wide_bootstrap_verified = seed['model']['checkpoint_sha256']
        if self.wide is not None and 'model_stage' in self.wide:
            if self._parent_root is not None:
                raise ValueError('Isolated Model staging cannot run inside a nested trial')
            from .aime_model_stage import load_model_stage
            seed, proof, self._model_stage = load_model_stage(self, seed, self.wide['model_stage'])
            self._write_named(f'fresh-{self._id(seed)}.json', proof)
            self._write_named('model-stage.json', self._model_stage)
        self._seed = seed
        self._seed_id = self._id(seed)

    def validate_binding(self, requirement):
        mode = 'paper_kernel' if self.paper_kernel is not None else 'model' if self.model_driven else 'rule'
        if requirement.get('operation_mode') != mode:
            raise ValueError(f'MetaRSI Controller and Lab operation modes differ: Lab requires {mode}')
        fields = requirement.keys() - {'operation_mode'}
        if mode == 'rule' and not fields:
            return
        if mode == 'rule' and fields == {'trial_lab', 'approved_trial_inputs', 'seed_call_id', 'fresh_run'}:
            ticket = requirement['seed_call_id']
            validate_seed_id(ticket)
            if type(requirement['fresh_run']) is not bool:
                raise ValueError('Invalid MetaRSI trial admission mode')
            parent = _cohort_identity(self.cohorts, self.datasets, self.bindings)
            approved = _check_trial(self._trial_settings, parent, requirement['trial_lab'],
                                    runtime=runtime_identity(self.runtime))
            prior_state = requirement['approved_trial_inputs']
            if prior_state is not None and self._approved_trial_inputs is None and not requirement['fresh_run']:
                raise ValueError('Cannot restore past the trial seed without approval from this parent Lab')
            for prior in (self._trial_offers.get(ticket), self._approved_trial_inputs, prior_state):
                if prior is not None:
                    validate_inputs(prior)
                    if prior != approved:
                        raise ValueError('MetaRSI approved trial inputs changed since admission')
            if requirement['fresh_run']:
                self._approved_trial_inputs = approved
            elif self._approved_trial_inputs is None:
                self._trial_offers[ticket] = approved
        elif mode == 'rule' and fields == {'mounted_trial_inputs'}:
            expected = requirement['mounted_trial_inputs']
            validate_inputs(expected)
            if self._mounted_inputs != expected:
                raise ValueError('Mounted MetaRSI trial inputs differ from parent approval')
        elif mode != 'rule' and fields == {'min_delta'}:
            if type(requirement['min_delta']) not in (int, float) or not 0 <= requirement['min_delta'] <= 1:
                raise ValueError('Invalid model-driven MetaRSI release requirement')
            if self.paper_kernel is not None:
                from .aime_release import validate_release_options
                if validate_release_options(self.paper_kernel)['min_gain'] < requirement['min_delta']:
                    raise ValueError('Fixed paired release cannot be weaker than the controller margin')
        else:
            raise ValueError('Unknown MetaRSI Controller binding requirement')

    # MathLab's constructor calls self.validate_config on its *derived* configuration.
    def bind_actor(self, context, config):
        return ModelRef('local-weights', self.engine.name, self.base_digest or 'arithmetic-fixture')

    def make_runtime(self, config):
        return _Runtime(self)

    def resource_roles(self):
        return self.engine.resource_roles() if not self.engine.simulated else {}

    def split_roles(self):
        return {**super().split_roles(), **{role: self.roles[name] for role, name in self.cohorts.items()}}

    def evaluation_operations(self):
        return ('evaluate',)

    def candidate_identity_fields(self):
        return {'evaluate': ('candidate',)}

    def final_evaluations(self):
        with self._lock:
            if 'test' not in self.cohorts or self._selected is None:
                return ()
            if not self._measured.get(self._seed_id):
                raise ValueError('MetaRSI final evaluation requires a measured seed baseline')
            return tuple(OperationCall('evaluate', {'candidate': candidate, 'split': 'test', 'baseline': baseline})
                for candidate, baseline in ((self._seed_id, True), (self._selected, False)))

    def describe(self):
        return (
            _spec('seed', 'Publish the starting MetaRSI candidate.', {}),
            _spec('propose', 'Change only the requested data, harness or model component.', {
                'parent': _string(), 'operator': _string(OPERATORS), 'round': {'type': 'integer', 'minimum': 0},
                'profile': _string(PROFILES), **({'strategy': {'type': 'string', 'maxLength': 512}}
                                                if self.model_driven else {}),
                **({'program_step': {'type': 'object'}, 'transition_guidance': {'type': 'object'},
                    'capability_epoch': {'type': 'integer', 'minimum': 0}} if self.paper_kernel is not None else {})}),
            _spec('evaluate', 'Score one candidate on a frozen development cohort.', {
                'candidate': _string(), 'split': _string(('search', 'validation'))}),
            _spec('select', 'Release or retain the term system using the fixed evaluator.', {'candidate': _string(),
                **({'term_start': _string(), 'term': {'type': 'integer', 'minimum': 1}}
                   if self.paper_kernel is not None else {})}),
            _spec('snapshot', 'Export portable declarative state without model weights.', {'candidate': _string()},
                  read_only=True),
        ) + ((_spec('diagnose', 'Read the frozen aggregate signal from a completed development evaluation.', {
            'candidate': _string(), 'split': _string(aime.DEV_ROLES)}, read_only=True),) if self.aime is not None else ()) + (
            (_spec('policy', 'Propose one contract-bound action using the target model.', {
                'candidate': _string(), 'round': {'type': 'integer', 'minimum': 0, 'maximum': 4096},
                'request': {'type': 'object'}}),) if self.model_driven else ()) + (
            (_spec('compare', 'Apply the fixed paired development release gate.', {
                'incumbent': _string(), 'candidate': _string()}),) if self.paper_kernel is not None else ())

    def binding(self):
        return {**super().binding(), 'profile': PROFILE, 'cohorts': dict(self.cohorts), 'seed_candidate': self._seed_id,
                'cohort_protocols': dict(self._protocols),
                 **({'aime': dict(self.aime)} if self.aime is not None else {}),
                 **({'wide_bootstrap_sha256': self.wide['bootstrap_sha256']} if self.wide is not None else {}),
                 **({'model_driven': True} if self.model_driven else {}),
                 **({'paper_kernel_sha256': digest(self.paper_kernel)} if self.paper_kernel is not None else {})}

    def state(self):
        with self._lock:
            return {'binding': self.binding(), 'candidates': sorted(self._candidates),
                    'measured': deepcopy(self._measured), 'selected': self._selected,
                    'approved_trial_inputs': deepcopy(self._approved_trial_inputs), 'trial_offers': deepcopy(self._trial_offers)}

    def restore(self, state):
        with self._lock:
            if not isinstance(state, dict) or state.get('binding') != self.binding():
                raise ValueError('MetaRSI Lab input binding changed since the checkpoint')
            approved = state['approved_trial_inputs']
            if approved is not None:
                validate_inputs(approved)
            if self._approved_trial_inputs is not None and self._approved_trial_inputs != approved:
                raise ValueError('MetaRSI approved trial inputs differ from the checkpoint')
            offers = state['trial_offers']
            if not isinstance(offers, dict) or (approved is not None and offers):
                raise ValueError('Invalid MetaRSI trial admission checkpoint')
            for ticket, offered in offers.items():
                validate_seed_id(ticket)
                validate_inputs(offered)
            ids, measured, selected = (state.get(key) for key in ('candidates', 'measured', 'selected'))
            if (not isinstance(ids, list) or not all(_text(key) for key in ids) or len(set(ids)) != len(ids)
                    or (ids and self._seed_id not in ids) or not isinstance(measured, dict)
                    or any(key not in ids or not isinstance(splits, list) or not splits
                          or any(split not in ('search', 'validation') for split in splits)
                          for key, splits in measured.items())
                    or (selected is not None and (not _text(selected) or not measured.get(selected)))):
                raise ValueError('Invalid MetaRSI Lab state')
            candidates = {key: self._read_candidate(key) for key in ids}
            if self.aime is not None:
                for key, splits in measured.items():
                    for split in splits:
                        self._diagnostic(key, split)
            if selected is not None:
                release = self._selection(selected)
                path = self.root / f'selection-{selected}.json'
                if path.is_symlink() or path.read_bytes() != json_bytes(release):
                    raise ValueError('Published MetaRSI selection changed since the checkpoint')
            self._candidates, self._measured, self._selected = candidates, deepcopy(measured), selected
            if self.paper_kernel is not None and selected is not None and not self._release_qualified(selected):
                raise ValueError('Checkpointed selection no longer passes the paired release chain')
            self._approved_trial_inputs = deepcopy(approved)
            self._trial_offers = deepcopy(offers)

    def execute(self, call):
        args = call.arguments
        with self._lock:
            if call.name == 'compare' and self.paper_kernel is not None:
                if set(args) != {'incumbent', 'candidate'}:
                    raise LabToolError('Invalid paired selection request')
                return self._receipt(call, self._compare_candidates(args['incumbent'], args['candidate'], publish=True))
            if call.name == 'policy' and self.model_driven:
                return self._model_policy(call)
            if call.name == 'seed' and args == {}:
                key = self._write_candidate(self._seed)
                data = self._candidate_data(key, self._seed)
                approval = self._approved_trial_inputs or self._trial_offers.get(call.call_id)
                if approval is not None:
                    data['approved_trial_inputs'] = approval
                receipt = self._receipt(call, data, components=self._seed)
                if approval is not None:
                    self._approved_trial_inputs = approval
                    self._trial_offers.clear()
                self._candidates[key] = self._seed
                return receipt
            required = {'parent', 'operator', 'round', 'profile'} | ({'strategy'} if self.model_driven else set())
            if self.paper_kernel is not None:
                required |= {'program_step', 'capability_epoch', 'transition_guidance'}
            if call.name == 'propose' and set(args) == required:
                if (not _text(args['parent']) or args['operator'] not in OPERATORS or args['profile'] not in PROFILES
                        or type(args['round']) is not int or args['round'] < 0):
                    raise LabToolError('Invalid MetaRSI proposal arguments')
                if self.model_driven and (not _text(args['strategy']) or len(args['strategy'].encode()) > 512
                                          or not args['strategy'].isprintable()):
                    raise LabToolError('Invalid bounded operator strategy')
                if self._model_stage is not None and args != {'parent': self._seed_id, 'operator': 'model',
                                                              'round': 1, 'profile': 'detailed'}:
                    raise LabToolError('The isolated Model stage allows exactly one fixed Model proposal')
                if self.paper_kernel is not None:
                    from .model_policy import program_step_invalidator
                    if (type(args['capability_epoch']) is not int or args['capability_epoch'] < 0 or
                            program_step_invalidator(args['program_step'], [args['operator']],
                                resource_budget=args['program_step']['step_budget']) is not None):
                        raise LabToolError('The kernel action does not match its typed horizontal plan')
                return self._propose(call) if self.aime is None else self._propose_aime(call)
            if (self.aime is not None and call.name == 'diagnose' and set(args) == {'candidate', 'split'}
                    and _text(args['candidate']) and args['split'] in aime.DEV_ROLES):
                key, role = args['candidate'], args['split']
                self._candidate(key)
                if role not in self._measured.get(key, []):
                    raise LabToolError('AIME diagnose requires a completed frozen development evaluation')
                return self._receipt(call, {**self._candidate_data(key), **self._diagnostic(key, role)})
            if call.name == 'evaluate' and _text(args.get('candidate')):
                if set(args) == {'candidate', 'split'} and args['split'] in ('search', 'validation'):
                    return self._evaluate(call, baseline=args['candidate'] == self._seed_id)
                if (set(args) == {'candidate', 'split', 'baseline'} and args['split'] == 'test'
                        and 'test' in self.cohorts and self._selected is not None and type(args['baseline']) is bool
                        and args['candidate'] == (self._seed_id if args['baseline'] else self._selected)):
                    return self._evaluate(call, baseline=args['baseline'])  # Group authorizes host finish only.
            if call.name == 'select' and self.paper_kernel is not None:
                return self._release_term(call)
            if call.name in ('select', 'snapshot') and set(args) == {'candidate'} and _text(args['candidate']):
                key = args['candidate']
                candidate = self._candidate(key)
                if call.name == 'snapshot':
                    if self.aime is not None:
                        self._checkpoint(candidate['model'], candidate)
                        snapshot = ({'schema_version': 3, 'candidate_id': key, 'base_sha256': self.base_digest}
                                    if self.wide is not None else
                                    {'schema_version': 2, 'candidate_id': key, 'base_sha256': self.base_digest, **candidate})
                        return self._receipt(call, {**self._candidate_data(key), 'snapshot': snapshot})
                    if candidate['model']['checkpoint'] is not None:
                        raise LabToolError('Model checkpoints are not portable MetaRSI snapshots')
                    result = {**self._candidate_data(key), 'snapshot': {'schema_version': 1,
                        'harness': candidate['harness'], 'data': candidate['data']}}
                    return self._receipt(call, result)
                if self._model_stage is not None and key != self._seed_id:
                    raise LabToolError('The isolated Model stage does not promote its measured challenger')
                if not self._measured.get(key):
                    raise LabToolError('MetaRSI selection requires a measured candidate')
                if self.aime is not None and ('validation' not in self._measured[key]
                        or 'validation' not in self._measured.get(self._seed_id, [])):
                    raise LabToolError('AIME selection requires seed and candidate on frozen validation')
                if self.paper_kernel is not None and not self._release_qualified(key):
                    raise LabToolError('The paper kernel requires an independent development release decision')
                result = self._selection(key)
                self._write_named(f'selection-{key}.json', result)
                receipt = self._receipt(call, result, components=candidate)
                self._selected = key
                return receipt
            raise LabToolError('Use the documented MetaRSI Lab operation arguments')

    def _policy_completion(self, candidate, messages, round_number):
        checkpoint = self._checkpoint(candidate['model'], candidate)
        if self.engine.simulated:
            groups = self.engine.generate([{'prompt': messages, 'question': ''}], 1,
                model={'checkpoint': checkpoint}, harness=candidate['harness'], round_number=round_number,
                sample=False, temperature=0.)
        else:
            groups = self.engine.generate_policy(messages, checkpoint=checkpoint, round_number=round_number)
        if (not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], list)
                or len(groups[0]) != 1):
            raise RuntimeError('MetaRSI policy worker returned an incomplete response')
        # The hosted worker represents length-capped generations with a non-string status.
        return groups[0][0] if isinstance(groups[0][0], str) else ''

    def _model_policy(self, call):
        from .model_policy import PolicyRequest, run_policy
        args = call.arguments
        if (set(args) != {'candidate', 'round', 'request'} or type(args['round']) is not int
                or not 0 <= args['round'] <= 4096 or not aime.candidate_id(args['candidate'])):
            raise LabToolError('Invalid MetaRSI policy operation arguments')
        request = PolicyRequest.from_wire(args['request'])
        candidate = self._candidate(args['candidate'])
        name = f'policy-{digest(call.call_id)}.json'
        existing = self._read_owned(self.root, name)
        request_sha = digest(args)
        decode = 'same-pinned-target-greedy-no-thinking-768-v1'
        if existing is None:
            messages = request.messages()
            if self.paper_kernel is not None:
                from .kernel_evidence import operator_card
                messages[0]['content'] += '\nHost operator contracts: ' + json.dumps(
                    [operator_card(op).wire() for op in OPERATORS], sort_keys=True)
            response = self._policy_completion(candidate, messages, args['round'])
            output = run_policy(request, response)
            record = {'request_sha256': request_sha, 'response_sha256': digest(response), 'outcome': output,
                      'decode': decode}
            self._write_named(name, record)
        else:
            record = existing
        if (record.get('request_sha256') != request_sha or record.get('decode') != decode
                or not aime.sha256_text(record.get('response_sha256')) or not isinstance(record.get('outcome'), dict)):
            raise LabToolError('Published MetaRSI policy receipt does not match its request')
        return self._receipt(call, {**record, 'candidate_id': args['candidate']})

    def _release_term(self, call):
        args = call.arguments
        if (set(args) != {'candidate', 'term_start', 'term'} or type(args['term']) is not int or args['term'] < 1
                or not aime.candidate_id(args['term_start']) or not aime.candidate_id(args['candidate'])):
            raise LabToolError('Invalid original MetaRSI term release request')
        incumbent, candidate, term = args['term_start'], args['candidate'], args['term']
        name = f'term-release-{term}.json'
        saved = self._read_owned(self.root, name)
        if (saved is not None and saved.get('request') != args or
                saved is None and (self._selected or self._seed_id) != incumbent or
                not self._release_qualified(incumbent) or not self._release_qualified(candidate) or
                any('validation' not in self._measured.get(ident, ()) for ident in (incumbent, candidate))):
            raise LabToolError('Term release does not start at the last deployment or end at a measured trial')
        comparator = self._compare_candidates(incumbent, candidate) if candidate != incumbent else None
        selected = candidate if comparator and comparator['eligible'] else incumbent
        verdict = {'start': incumbent, 'working_end': candidate, 'term': term,
                   'decision': 'released' if selected == candidate and candidate != incumbent else 'retained',
                   'comparison': comparator}
        if saved is not None and saved != {'request': args, 'verdict': verdict}:
            raise LabToolError('Term release no longer replays its fixed verdict')
        self._write_named(name, {'request': args, 'verdict': verdict})
        data = self._selection(selected)
        self._write_named(f'selection-{selected}.json', data)
        result = self._receipt(call, {**data, 'term_release': verdict}, components=self._candidate(selected))
        self._selected = selected
        return result

    def _compare_candidates(self, incumbent, candidate, *, publish=False):
        from .aime_release import compare
        if (not aime.candidate_id(incumbent) or not aime.candidate_id(candidate)
                or any('validation' not in self._measured.get(k, ()) for k in (incumbent, candidate))):
            raise LabToolError('Paired release requires completed frozen development measurements')
        anchor, challenger = self._candidate(incumbent), self._candidate(candidate)
        if all(anchor[slot] == challenger[slot] for slot in ('harness', 'model')):
            raise LabToolError('A data-only change cannot enter the capability release gate')
        before, after = (self._read_owned(self.root, f'paired-validation-{k}.json') for k in (incumbent, candidate))
        exported = self._read_owned(self.root, f'operator-export-{candidate}.json')
        reconciliation = bool(exported and exported.get('reconciles') == incumbent and
                              self._strict_prompt_removal(anchor, challenger))
        result = compare(before, after, self.paper_kernel, equivalent_reconciliation=reconciliation)
        if (result['incumbent'] != incumbent or result['candidate'] != candidate
                or result['protocol_id'] != self._protocols['validation']):
            raise LabToolError('Paired selection protocol does not match the Lab')
        if publish and result['eligible']:
            if not self._release_qualified(incumbent):
                raise LabToolError('Paired release must descend from a previously eligible incumbent')
            self._write_named(f'approval-{candidate}-{incumbent}.json', result)
        return result

    def _release_qualified(self, key):
        def walk(ident, visiting):
            if ident == self._seed_id:
                return True
            if ident in visiting or not aime.candidate_id(ident) or len(visiting) >= 256:
                return False
            for entry in sorted(self.root.glob(f'approval-{ident}-metarsi-*.json')):
                approval = self._read_owned(self.root, entry.name)
                incumbent = approval['incumbent']
                if entry.name != f'approval-{ident}-{incumbent}.json' or approval.get('candidate') != ident:
                    raise LabToolError('Invalid paired selection proof identity')
                comparison = self._compare_candidates(incumbent, ident)
                if approval != comparison or approval['eligible'] is not True:
                    raise LabToolError('Paired selection proof changed after approval')
                if walk(incumbent, {*visiting, ident}):
                    return True
            return False
        return walk(key, set())

    def _selection(self, key):
        result = {'candidate_id': key, 'baseline_id': self._seed_id, 'profile': PROFILE,
                'training_performed': False, 'simulated': self.engine.simulated}
        if self.aime is not None:
            baseline, candidate = (self._diagnostic(item, 'validation') for item in (self._seed_id, key))
            if candidate['accuracy'] < baseline['accuracy']:
                raise LabToolError('AIME selection cannot regress relative to the seed on frozen validation')
            result['selection_evidence'] = {'baseline': baseline, 'candidate': candidate}
        return result

    def _candidate(self, key):
        if key not in self._candidates:
            raise LabToolError('Unknown MetaRSI candidate')
        return self._candidates[key]

    def _id(self, candidate):
        return 'metarsi-' + digest({'profile': PROFILE, 'base': self.base_digest, 'components': candidate,
                                    **({'schema_version': 2} if self.aime is not None else {})})

    def _write_named(self, name, data):
        path = self.root / name
        payload = json_bytes(data)
        if self.root.is_symlink() or path.is_symlink():
            raise ValueError('MetaRSI named artifacts cannot use symlinks')
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError('Published MetaRSI artifact does not match the replay')
        else:
            atomic_write(path, payload)

    def _write_candidate(self, value):
        key = self._id(value)
        self._write_named(f'{key}.json', value)
        return key

    def _read_candidate(self, key):
        if not key.startswith('metarsi-') or len(key) != 72 or any(c not in '0123456789abcdef' for c in key[8:]):
            raise ValueError('Invalid MetaRSI candidate identity')
        path = self.root / f'{key}.json'
        if self.root.is_symlink() or path.is_symlink():
            raise ValueError('MetaRSI candidate artifacts cannot use symlinks')
        value = json.loads(path.read_bytes())
        if (not isinstance(value, dict) or set(value) != {'harness', 'data', 'model'} or self._id(value) != key):
            raise ValueError('Published MetaRSI candidate changed since the checkpoint')
        if self.aime is None:
            _check_snapshot({'schema_version': 1, 'harness': value['harness'], 'data': value['data']})
        else:
            aime.check_components(value, self.aime, HARNESSES, allow_model_guidance=self.model_driven)
        self._verify_data(value['data'])
        self._checkpoint(value['model'], value)
        return value

    def _resolve_train_problem(self, ident):
        problem = self._train_rows.get(ident)
        if problem is None and self._synthesis_sources:
            derived = aime_synthesis.recover(ident, self._synthesis_sources)
            if derived is not None:
                problem = derived[0]
        if problem is None and self.model_driven:
            from .aime_domain_policy import recover_task
            recovered = recover_task(ident)
            if recovered is not None:
                problem = recovered[0]
        return problem

    def _verify_data(self, data):
        seen = set()
        for row in data:
            problem = self._resolve_train_problem(row['id'])
            key = row['id'] if self.wide is not None else (row['id'], row['completion'].strip())
            if (problem is None or row['prompt'] != problem.prompt or key in seen
                    or not self.verify([problem], [MathAttempt(row['completion'])])[0].correct):
                raise ValueError('MetaRSI snapshot/artifact data must be distinct, verified trajectories from this train split')
            seen.add(key)

    def _candidate_data(self, key, value=None, *, training=False):
        value = self._candidate(key) if value is None else value
        data = value['data']
        fresh = self._fresh_proof(key, value) if self.aime is not None else None
        model_available = bool(data) if self.aime is None else fresh is not None
        available = [*(['data'] if len(data) < self._capacity else []), 'harness', *(['model'] if model_available else [])]
        return {'candidate_id': key, 'available_operators': available, 'data_count': len(data),
                'training_performed': training,
                 **({'data_fresh': model_available} if self.aime is not None else {})}

    def _receipt(self, call, values, *, components=None):
        data = {'protocol': PROFILE, 'simulated': self.engine.simulated, **deepcopy(values)}
        if self._mounted_inputs_sha256 is not None:
            data['trial_inputs_sha256'] = self._mounted_inputs_sha256
        payload = {'operation': call.name, 'request': call.arguments, 'result': data}
        if components is not None:
            if self.wide is not None:
                payload['component_sha256'] = {name: digest(value) for name, value in components.items()}
            else:
                payload['components'] = components
        ref = self.artifacts.put(json_bytes(payload), kind=f'meta-rsi-{call.name}')
        return OperationResult(f'MetaRSI {call.name}', (ref,), self.engine.simulated, data=data, call_id=call.call_id)

    def _propose(self, call):
        args = call.arguments
        parent = self._candidate(args['parent'])
        operator, profile = args['operator'], args['profile']
        name = 'proposal-' + digest({'binding': self.binding(), 'request': args}) + '.json'
        path = self.root / name
        if self.root.is_symlink() or path.is_symlink():
            raise ValueError('MetaRSI proposal artifacts cannot use symlinks')
        if path.is_file():
            saved = json.loads(path.read_bytes())
            if saved['request'] != args:
                raise ValueError('Published MetaRSI proposal does not match its request')
            key, reason = saved['candidate_id'], saved['reason']
            candidate = self._read_candidate(key) if key is not None else parent
            if key is not None and [part for part in OPERATORS if parent[part] != candidate[part]] != [operator]:
                raise ValueError('Published MetaRSI proposal does not match its operator')
        else:
            candidate = deepcopy(parent)
            reason = 'no_component_change'
            if operator == 'data':
                candidate['data'] = self._collect(parent, profile, args['round'])
                reason = 'no_new_verified_trajectories'
            elif operator == 'harness':
                candidate['harness'] = {'instruction': profile, 'few_shot': profile == 'detailed'}
            elif parent['data']:
                candidate['model'] = self._fit(parent, args)
            else:
                reason = 'no_verified_data'
            key = self._write_candidate(candidate) if candidate != parent else None
            self._write_named(name, {'request': args, 'candidate_id': key, 'reason': reason if key is None else None})
        output = self._candidate_data(key or args['parent'], candidate, training=(
            key is not None and operator == 'model' and not self.engine.simulated))
        if key is None:
            output.update(candidate_id=None, reason=reason)
        receipt = self._receipt(call, {**args, **output}, components=candidate if key is not None else None)
        if key is not None:
            self._candidates[key] = candidate
        return receipt

    def _import_snapshot(self, snapshot):
        key = snapshot['candidate_id']
        if self.wide is None:
            candidate = {part: snapshot[part] for part in OPERATORS}
        else:
            if self._parent_root is None:
                raise ValueError('AIME wide snapshots can reference only the owned parent')
            candidate = self._read_owned(self._parent_root, f'{key}.json')
            aime.check_components(candidate, self.aime, HARNESSES)
        if snapshot['base_sha256'] != self.base_digest or key != self._id(candidate):
            raise ValueError('AIME snapshot does not match its exact candidate and pinned base')
        self._verify_data(candidate['data'])
        model = candidate['model']
        if self._parent_root is None:
            if model['checkpoint'] is not None:
                raise ValueError('AIME adapter snapshots require an owned parent run')
            return candidate
        path = self._parent_root / f'{key}.json'
        if (self._parent_root.is_symlink() or path.is_symlink() or not path.is_file()
                or path.read_bytes() != json_bytes(candidate)):
            raise ValueError('AIME child seed must match the owned parent candidate')
        if model['checkpoint'] is not None:
            inherit_checkpoint(self._parent_root / 'models', self.root / 'models', model,
                simulated=self.engine.simulated, validate=lambda payload: self._validate_fit(candidate, payload))
        proof = self._fresh_proof(key, candidate, root=self._parent_root)
        if proof is not None:
            self._write_named(f'refresh-{key}.json', proof)
        return candidate

    def _data_proof(self, key, candidate, previous_count):
        data = candidate['data']
        previous = {**candidate, 'data': data[:previous_count]}
        return {'candidate_id': key, 'source_candidate_id': self._id(previous),
                'train_protocol_id': self._protocols['train'], 'harness_sha256': digest(candidate['harness']),
                'model_sha256': digest(candidate['model']), 'previous_count': previous_count,
                'added_count': len(data) - previous_count, 'added_sha256': digest(data[previous_count:]),
                'cumulative_count': len(data), 'cumulative_sha256': digest(data)}

    def _fresh_proof(self, key, candidate, *, root=None):
        root = self.root if root is None else root
        name = f'refresh-{key}.json'
        value = self._read_owned(root, name)
        if value is None:
            return None
        count = value.get('previous_count') if isinstance(value, dict) else None
        if (type(count) is not int or not 0 <= count < len(candidate['data'])
                or key != self._id(candidate) or value != self._data_proof(key, candidate, count)):
            raise ValueError('AIME data refresh proof does not match the exact candidate and train protocol')
        return value

    @staticmethod
    def _read_owned(root, name):
        path = root / name
        if root.is_symlink() or path.is_symlink():
            raise ValueError('MetaRSI owned artifacts cannot use symlinks')
        if path.exists():
            if not path.is_file():
                raise ValueError('MetaRSI owned artifacts must be regular files')
            return json.loads(path.read_bytes())
        return None

    def _diagnostic_name(self, key, role):
        return f'diagnose-{key}-{self._protocols[role]}.json'

    def _diagnostic(self, key, role):
        value = self._read_owned(self.root, self._diagnostic_name(key, role))
        aime.check_metrics(value, key, role, self._protocols[role], len(self.datasets[self.cohorts[role]]))
        return value

    @staticmethod
    def _revise_prompt(parent_key, harness):
        # Avoid borrowing old D proof if a later H patch restores identical prompt text.
        harness['revision'] = digest({'parent': parent_key, 'instruction': harness['instruction'],
                                      'exemplars': harness['exemplars'],
                                      **({'guidance': harness['guidance']} if 'guidance' in harness else {})})

    @staticmethod
    def _strict_prompt_removal(parent, candidate):
        a, b = parent['harness'], candidate['harness']
        old, new = a['exemplars'], b['exemplars']
        return (parent['data'] == candidate['data'] and parent['model'] == candidate['model'] and
                a['instruction'] == b['instruction'] and new == [key for key in old if key in new] and
                ('guidance' not in b or b['guidance'] == a.get('guidance')) and
                (len(new) < len(old) or ('guidance' in a and 'guidance' not in b)))

    def _choose_harness(self, parent, proposals, profile):
        replies, candidates, replay_base = [], [], None
        for patch in proposals:
            value = deepcopy(parent)
            value['harness'].update(patch)
            if value['harness'].get('guidance') is None:
                value['harness'].pop('guidance', None)
            if value == parent:
                continue
            self._revise_prompt(self._id(parent), value['harness'])
            if replay_base is None:
                replay_base = self._harness_replay_base(parent, profile)
            replay = self._replay_harness_patch(value, replay_base)
            index = len(replies)
            replies.append({'patch_sha256': digest(patch), **replay})
            if replay['accepted']:
                candidates.append((replay['after']['correct'], -replay['after']['incomplete'], -index, value, patch))
        if not candidates:
            return parent, None, {'candidates': replies, 'selected_index': None}
        _, _, rank, candidate, patch = max(candidates, key=lambda row: row[:3])
        return candidate, patch, {'candidates': replies, 'selected_index': -rank}

    def _choose_fit(self, parent, args, proof, proposals):
        # Private fixed search uses development only; validation is a separate Host operation.
        trials, ranked = [], []
        problems = self.datasets[self.cohorts['search']]
        for patch in proposals:
            state = {**parent, 'model': self._fit(parent, args, proof=proof, proposal=patch)}
            index = len(trials)
            record = {'recipe_sha256': digest(patch), 'checkpoint': state['model']['checkpoint']}
            if len(proposals) > 1:
                attempts = [group[0] for group in self._generate(state, problems, count=1)]
                record.update(search_count=len(problems), search_correct=sum(check is not None and check.correct
                    for check in self.verify(problems, attempts)))
            trials.append(record)
            ranked.append((record.get('search_correct', 0), -index, state, patch))
        if not ranked:
            return parent, None, {'candidates': trials, 'selected_index': None}
        _, rank, candidate, patch = max(ranked, key=lambda row: row[:2])
        return candidate, patch, {'candidates': trials, 'selected_index': -rank}

    def _harness_replay_base(self, parent, profile):
        # Select the affected development-search shard using incumbent outcomes only.
        # Never expose individual questions, gold answers or completions to the proposal model.
        role = 'search'
        probes = self.datasets[self.cohorts[role]][:4 if profile == 'compact' else 8]
        initial_attempts = [batch[0] for batch in self._generate(parent, probes, count=1)]
        verified = self.verify(probes, initial_attempts)
        failures = [(p, a) for p, a, v in zip(probes, initial_attempts, verified, strict=True)
                    if v is None or not v.correct]
        scope = ('format_or_completion' if any(a.status in ('invalid_format', 'truncated') for _, a in failures)
                 else 'verified_failure')
        shard = ([p for p, a in failures if a.status in ('invalid_format', 'truncated')]
                 if scope == 'format_or_completion' else [p for p, _ in failures])
        if not shard:
            shard, scope = probes, 'no_observed_failure_replay'
        kept = {p.id for p in shard}
        before = [self.record(p, a, v) for p, a, v in zip(probes, initial_attempts, verified, strict=True) if p.id in kept]
        return shard, scope, before

    def _replay_harness_patch(self, candidate, replay_base):
        shard, scope, before = replay_base
        proposal = [batch[0] for batch in self._generate(candidate, shard, count=1)]
        after = [self.record(p, a, v) for p, a, v in zip(shard, proposal, self.verify(shard, proposal), strict=True)]
        def counts(rows):
            return {'correct': sum(row['correct'] for row in rows),
                    'incomplete': sum(row['status'] in ('truncated', 'invalid_format') for row in rows)}
        a, b = counts(before), counts(after)
        return {'scope': scope, 'count': len(shard), 'before': a, 'after': b,
                'measurement_sha256': digest({'protocol': self._protocols['search'], 'shard': [p.id for p in shard]}),
                'accepted': b['correct'] >= a['correct'] and b['incomplete'] <= a['incomplete']}

    def _prune_prompt(self, parent, candidate):
        role = 'search'
        problems = self.datasets[self.cohorts[role]]

        def replay(value):
            attempts = [group[0] for group in self._generate(value, problems, count=1)]
            records = [self.record(problem, attempt, check) for problem, attempt, check in
                       zip(problems, attempts, self.verify(problems, attempts), strict=True)]
            return aime.metrics(self._id(value), role, self._protocols[role], len(records),
                sum(row['correct'] for row in records), sum(row['status'] == 'invalid_format' for row in records),
                sum(row['status'] == 'truncated' for row in records),
                sum(row['finish_reason'] == 'prompt_budget' for row in records))

        harness = candidate['harness']
        removed = harness['exemplars'].pop()
        harness['few_shot'] = bool(harness['exemplars'])
        self._revise_prompt(self._id(parent), harness)
        baseline, proposed = replay(parent), replay(candidate)
        accepted = (proposed['accuracy'] + self.aime['replay_tolerance'] >= baseline['accuracy']
                    and proposed['invalid_format'] <= baseline['invalid_format']
                    and proposed['truncated'] <= baseline['truncated'])
        return {'removed_exemplar': removed, 'before': baseline, 'after': proposed, 'accepted': accepted}

    def _kernel_identity(self, key, components):
        from .kernel_evidence import build_state_identity
        return build_state_identity(key, components, protected={
            'base_model': self.base_digest, 'signal_compiler': 'meta-rsi/aime-diagnosis/v1',
            'verifier': self._protocols['train'], 'evaluator': {role: self._protocols[role] for role in aime.DEV_ROLES},
            'promotion_rule': self.paper_kernel, 'sealed_measurement': 'not-mounted',
            'release_rule': 'meta-rsi/aime-paired-release-v1', 'ledger': self._protocols, 'runtime_permissions': self.options,
        })

    def _paper_learning_signal(self, key, candidate, train_records=()):
        from .aime_diagnosis import project_evidence
        # Data-only states share behavior with their closest measured predecessor.
        # Verify the data lineage instead of assigning its own evaluation score.
        origin, origin_state = key, candidate
        while not self._measured.get(origin):
            proof = self._fresh_proof(origin, origin_state)
            if proof is None:
                break
            origin = proof['source_candidate_id']
            origin_state = self._candidate(origin)
        development = {role: self._diagnostic(origin, role) for role in self._measured.get(origin, ())
                       if role in aime.DEV_ROLES}
        return origin, project_evidence(development=development, train_records=train_records)

    def _paper_source(self, args, state):
        from .transitions import edge_contract, validate_artifact
        key = args['parent']
        source = self._read_owned(self.root, f'operator-export-{key}.json')
        if source is None:
            return None
        contract = edge_contract(source['operator'], args['operator'])
        if contract.adapter is not None:
            artifact = source['artifacts'][contract.input_kind]
            validate_artifact(artifact)
            if artifact['source_state'] != state or artifact['capability_epoch'] > args['capability_epoch']:
                raise LabToolError('Transition input was not exported from the active operator state')
        return source, contract

    def _paper_handoff(self, args, parent, transition, prepared):
        if prepared is None:
            return None
        from .transitions import build_artifact, build_transition
        source, contract = prepared
        before = transition['kernel_execution']['stages'][0]['state']
        epoch = args['capability_epoch']
        actor = transition.get('actor') or transition.get('synthesis', {}).get('actor') or {}
        _, signal = self._paper_learning_signal(args['parent'], parent)
        hashes = {'signal_sha256': digest(signal)}
        if contract.adapter is None:
            artifact = build_artifact('learning_signal', source_state=before, capability_epoch=epoch,
                                      evidence_sha256=hashes)
            output = None
        else:
            artifact = source['artifacts'][contract.input_kind]
            if contract.output_kind == 'harness_patch':
                if 'selected_harness_patch_sha256' not in transition:
                    return None
                hashes = {'harness_patch_sha256': transition['selected_harness_patch_sha256']}
            elif contract.output_kind == 'materialized_dataset':
                if not transition.get('training'):
                    return None
                checkpoint = transition['training']['checkpoint']
                recipe = read_checkpoint(self.root / 'models', checkpoint, simulated=self.engine.simulated)['recipe']
                hashes = {'masked_shards_sha256': recipe['training_rows_sha256'], 'recipe_sha256': digest(recipe)}
            elif contract.output_kind == 'model_learning_signal':
                hashes['learning_signatures_sha256'] = (actor.get('diagnosis') or {}).get('diagnosis_sha256') or digest(signal)
            output = build_artifact(contract.output_kind, source_state=before, capability_epoch=epoch,
                evidence_sha256=hashes, derived_from=artifact['artifact_sha256'])
        return build_transition(source['operator'], args['operator'], artifact=artifact, output=output,
                                target_state=before, capability_epoch=epoch)

    def _paper_export(self, key, candidate, args, transition):
        from .transitions import build_artifact
        op, epoch = args['operator'], args['capability_epoch'] + int(args['operator'] != 'data')
        state = self._kernel_identity(key, candidate)
        if op == 'data':
            proof = transition['refresh_proof']
            values = {'experience': {'experience_sha256': proof['added_sha256']},
                      'dataset': {'cumulative_sha256': proof['cumulative_sha256'], 'refresh_sha256': digest(proof)}}
        elif op == 'harness':
            values = {'genome': {'genome_sha256': state['component_sha256']['harness']}}
        else:
            checkpoint = candidate['model']['checkpoint']
            report = read_checkpoint(self.root / 'models', checkpoint, simulated=self.engine.simulated)
            values = {'checkpoint': {'checkpoint_sha256': checkpoint},
                      'training_report': {'training_report_sha256': digest(report)}}
        self._write_named(f'operator-export-{key}.json', {'operator': op,
            **({'reconciles': transition['reconciles']} if 'reconciles' in transition else {}),
            'artifacts': {kind: build_artifact(kind, source_state=state, capability_epoch=epoch,
                                               evidence_sha256=hashes) for kind, hashes in values.items()}})

    def _propose_aime(self, call):
        args = call.arguments
        parent_key, operator, profile = (args[key] for key in ('parent', 'operator', 'profile'))
        parent = self._candidate(parent_key)
        source = self._paper_source(args, self._kernel_identity(parent_key, parent)) if self.paper_kernel is not None else None
        name = 'proposal-' + digest({'binding': self.binding(), 'request': args}) + '.json'
        saved = self._read_owned(self.root, name)
        if saved is not None:
            if saved.get('request') != args:
                raise ValueError('Published AIME proposal does not match its request')
            key, reason, transition = (saved[field] for field in ('candidate_id', 'reason', 'transition'))
            candidate = self._read_candidate(key) if key is not None else parent
            if key is not None and [part for part in OPERATORS if parent[part] != candidate[part]] != [operator]:
                raise ValueError('Published AIME proposal does not match its operator')
            if key is not None and operator == 'data':
                proof = self._fresh_proof(key, candidate)
                if (proof is None or proof['source_candidate_id'] != parent_key or proof != transition['refresh_proof']):
                    raise ValueError('Published AIME data proposal is missing its exact original refresh proof')
        else:
            if self.paper_kernel is not None:
                from .kernel_evidence import operator_card, validate_action
                validate_action(operator, operator_card(operator).action)
            candidate = deepcopy(parent)
            transition = {'edges': [], 'requires_refresh': False}
            reason = 'no_component_change'
            if operator == 'data':
                old_count = len(parent['data'])
                if self.model_driven:
                    candidate['data'], transition['synthesis'] = self._collect_model_synthesis(call, parent)
                elif self.aime['data_policy'] == 'affine':
                    collect = self._collect_wide_synthesis if self.wide is not None else self._collect_synthesis
                    candidate['data'], transition['synthesis'] = collect(parent, profile, args['round'])
                else:
                    transition['rollout'] = {}
                    candidate['data'] = self._collect(parent, profile, args['round'], audit=transition['rollout'])
                new_count = len(candidate['data'])
                reason = 'data_capacity_reached' if old_count == self._capacity else 'no_new_verified_trajectories'
                transition['verified_added'] = new_count - old_count
                if new_count > old_count:
                    self._verify_data(candidate['data'])
                    key = self._id(candidate)
                    proof = self._fresh_proof(key, candidate) or self._data_proof(key, candidate, old_count)
                    if proof['source_candidate_id'] != parent_key:
                        raise ValueError('AIME refresh proof does not start from the requested parent')
                    self._write_named(f'refresh-{key}.json', proof)
                    transition['refresh_proof'] = proof
            elif operator == 'harness':
                if self.model_driven:
                    exemplars = list({row['id']: digest(row) for row in parent['data']
                                      if len(row['completion']) <= MAX_COMPLETION}.values())[-24:]
                    proposed, audit = self._domain_proposal(call, parent, eligible=exemplars)
                    transition['actor'] = audit
                    if proposed is None:
                        reason = 'domain_proposal_declined'
                    else:
                        if self.paper_kernel is not None:
                            candidate, proposed, replay = self._choose_harness(parent, proposed, profile)
                            transition['candidate_screen'] = replay
                            if proposed is None:
                                reason = 'no_harness_candidate_survived_replay'
                            else:
                                from .harness_genome import experience_patch, genome
                                before_prompt = genome(prompt=json.dumps(parent['harness'], sort_keys=True))
                                projected = experience_patch(before_prompt, {
                                    'prompt': json.dumps(candidate['harness'], sort_keys=True)}, writable_slots=('prompt',))
                                transition['selected_harness_patch_sha256'] = digest(projected)
                        else:
                            candidate['harness'].update(proposed)
                        if candidate != parent:
                            self._revise_prompt(parent_key, candidate['harness'])
                            if proposed['exemplars']:
                                transition['edges'].append('D→H')
                                transition['verified_exemplars'] = {'row_sha256': proposed['exemplars'],
                                    'train_protocol_id': self._protocols['train'], 'data_sha256': digest(parent['data'])}
                elif profile == 'compact' and parent['model']['checkpoint'] and parent['harness']['exemplars']:
                    replay = self._prune_prompt(parent, candidate)
                    transition['replay'] = replay
                    if replay['accepted']:
                        transition['edges'].append('M→H')
                    else:
                        candidate = deepcopy(parent)
                        reason = 'exemplar_replay_regressed'
                else:
                    exemplars, seen = [], set()
                    cap = 1 if profile == 'compact' else self.aime['max_exemplars']
                    for row in parent['data']:
                        if len(row['completion']) <= MAX_COMPLETION and row['id'] not in seen:
                            seen.add(row['id'])
                            exemplars.append(digest(row))
                            if len(exemplars) == cap:
                                break
                    candidate['harness'].update(instruction=profile, few_shot=bool(exemplars), exemplars=exemplars)
                    if candidate != parent:
                        self._revise_prompt(parent_key, candidate['harness'])
                    if exemplars and candidate != parent:
                        transition['edges'].append('D→H')
                        transition['verified_exemplars'] = {'row_sha256': exemplars,
                            'train_protocol_id': self._protocols['train'], 'data_sha256': digest(parent['data'])}
                if candidate != parent:
                    transition['edges'].append('H→D')
                    transition['requires_refresh'] = True
            elif not parent['data']:
                reason = 'no_verified_data'
            else:
                proof = self._fresh_proof(parent_key, parent)
                if proof is None:
                    reason = 'fresh_data_required'
                    transition['requires_refresh'] = True
                else:
                    proposed = None
                    if self.model_driven:
                        proposed, transition['actor'] = self._domain_proposal(call, parent)
                    if self.model_driven and proposed is None:
                        reason = 'domain_proposal_declined'
                    else:
                        if self.paper_kernel is not None:
                            candidate, chosen, screen = self._choose_fit(parent, args, proof, proposed)
                            transition['candidate_screen'] = screen
                            if chosen is None:
                                reason = 'no_model_candidate'
                            else:
                                transition['actor']['portfolio_sha256'] = transition['actor']['proposal_sha256']
                                transition['actor']['proposal_sha256'] = digest(chosen)
                        else:
                            candidate['model'] = self._fit(parent, args, proof=proof, proposal=proposed)
                if candidate != parent:
                    transition.update(edges=['D→M', 'M→D'], requires_refresh=True, refresh_proof=proof,
                        training={'base_sha256': self.base_digest, 'label_scope': 'assistant_only',
                                  'cumulative_count': len(parent['data']), 'checkpoint': candidate['model']['checkpoint']})
                    if self.wide is not None:
                        transition['training'].update(self._wide_fit_receipt(candidate['model']['checkpoint'], parent))
                    if self.paper_kernel is not None:
                        plan = self._curriculum(parent['data'], args['round'])
                        transition['training']['curriculum_sha256'] = digest(plan)
                        transition['training']['retention_monitor'] = self._replay_retention(parent, candidate, plan)
                    if self.wide is not None and (self._model_stage is not None or self.model_driven):
                        if transition['training'].get('training_seed') != 23:
                            raise ValueError('Pinned wide model-driven fit requires actual optimizer seed 23')
            if self.paper_kernel is not None:
                from .kernel_evidence import build_kernel_execution
                actor = transition.get('actor') or transition.get('synthesis', {}).get('actor') or {}
                before = self._kernel_identity(parent_key, parent)
                after = self._kernel_identity(self._id(candidate), candidate)
                rejected = before == after and (actor.get('accepted') is False or reason in (
                    'fresh_data_required', 'no_harness_candidate_survived_replay'))
                transition['kernel_execution'] = build_kernel_execution(operator, before=before, after=after,
                    signal=(actor.get('diagnosis') or {}).get('signal_sha256') or
                           digest(self._paper_learning_signal(parent_key, parent)[1]),
                    diagnosis=(actor.get('diagnosis') or {}).get('diagnosis_sha256'),
                    proposal=actor.get('portfolio_sha256', actor.get('proposal_sha256')),
                    validation='rejected' if rejected else 'accepted', allow_no_op=before == after)
                handoff = self._paper_handoff(args, parent, transition, source)
                transition['edges'] = [] if handoff is None else [
                    {'data': 'D', 'harness': 'H', 'model': 'M'}[handoff['source']] + '→' +
                    {'data': 'D', 'harness': 'H', 'model': 'M'}[handoff['target']]]
                if handoff is not None:
                    transition['handoff'] = handoff
                    if (handoff['adapter'] == 'redundancy_reconciliation'
                            and self._strict_prompt_removal(parent, candidate)):
                        transition['reconciles'] = parent_key
            if candidate != parent:
                aime.check_components(candidate, self.aime, HARNESSES, allow_model_guidance=self.model_driven)
                key = self._write_candidate(candidate)
                reason = None
            else:
                key = None
            self._write_named(name, {'request': args, 'candidate_id': key, 'reason': reason, 'transition': transition})
        if self.paper_kernel is not None:
            from .kernel_evidence import validate_kernel_execution
            execution = transition['kernel_execution']
            validate_kernel_execution(execution)
            if (execution['stages'][0]['state'] != self._kernel_identity(parent_key, parent) or
                    execution['stages'][4]['state'] != self._kernel_identity(key or parent_key, candidate)):
                raise ValueError('Persisted kernel execution no longer matches the proposal states')
            if 'handoff' in transition:
                from .transitions import validate_transition
                validate_transition(transition['handoff'], target_state=execution['stages'][0]['state'],
                                    capability_epoch=args['capability_epoch'])
            if key is not None:
                self._paper_export(key, candidate, args, transition)
        output = self._candidate_data(key or parent_key, candidate, training=(
            key is not None and operator == 'model' and not self.engine.simulated))
        if key is None:
            output.update(candidate_id=None, reason=reason)
        receipt = self._receipt(call, {**args, **output, 'transition': transition},
                                components=candidate if key is not None else None)
        if key is not None:
            self._candidates[key] = candidate
        return receipt

    def _assert_base(self):
        if self._base_stamp is not None and model_stamp(self.base_path) != self._base_stamp:
            raise ValueError('AIME pinned base model changed after admission')

    def _checkpoint(self, model, candidate=None):
        self._assert_base()
        key = model['checkpoint']
        if key is None:
            return None
        payload = read_checkpoint(self.root / 'models', key, simulated=self.engine.simulated)
        if digest(payload) != model['checkpoint_sha256']:
            raise ValueError('MetaRSI model component no longer matches its checkpoint')
        if self.aime is not None:
            self._validate_fit(candidate, payload)
        return key if self.engine.simulated else self.root / 'models' / key / 'adapter'

    def _training_rows(self, data):
        harness = {'instruction': 'seed', 'few_shot': False}
        if self.aime is not None:
            harness['exemplars'] = []
        return [{'prompt': self._prompt(self._resolve_train_problem(row['id']), harness, []),
                 'completion': row['completion'],
                 **({'id': digest([row['id'], row['completion']]), 'problem_id': row['id']}
                    if self.wide is not None else {})} for row in data]

    def _validate_fit(self, candidate, payload):
        recipe, count = payload['recipe'], payload['samples']
        if self.wide is not None and recipe.get('profile') == 'meta-rsi/aime-pool-scale-dev/v1':
            if count != 228 or candidate['data'][:count] != self._wide_bootstrap['rows']:
                raise ValueError('Wide seed adapter requires the exact inherited original trajectories')
            if self._wide_bootstrap_verified != digest(payload):
                from .aime_loop_checkpoint import verify_bootstrap_recipe
                verify_bootstrap_recipe(payload, Path(self.wide['checkpoint_run']), self.base_path, self.base_digest,
                                        self._training_rows(candidate['data'][:count]), self._wide_evidence)
                self._wide_bootstrap_verified = digest(payload)
            return
        if (not isinstance(recipe, dict) or not 1 <= count <= len(candidate['data'])
                or recipe.get('profile') != 'meta-rsi/aime/v2' or recipe.get('base_sha256') != self.base_digest
                or recipe.get('engine') != self.engine.name or recipe.get('options') != self.options
                or recipe.get('train_protocol_id') != self._protocols['train']
                or recipe.get('label_scope') != 'assistant_only' or not aime.sha256_text(recipe.get('refresh_sha256'))
                or recipe.get('source_rows_sha256' if self.paper_kernel is not None else
                              'unbucketed_rows_sha256' if self.wide is not None else 'training_rows_sha256')
                   != digest(self._training_rows(candidate['data'][:count]))):
            raise ValueError('AIME adapter is not bound to the pinned base and verified cumulative train rows')
        if self.paper_kernel is not None:
            source = candidate['data'][:count]
            plan = self._curriculum(source, recipe['round'])
            if (recipe.get('curriculum_sha256') != digest(plan) or
                    self._read_owned(self.root, f'curriculum-{digest(plan)}.json') != plan):
                raise ValueError('AIME adapter curriculum is missing or changed since fitting')
            rows = dict(zip((row['id'] for row in source), self._training_rows(source), strict=True))
            order = [rows[key] for key in plan['ordered_row_ids']]
            if recipe.get('unbucketed_rows_sha256' if self.wide is not None else 'training_rows_sha256') != digest(order):
                raise ValueError('AIME adapter row order does not match its verified curriculum')

    def _fit(self, candidate, args, *, proof=None, proposal=None):
        # Fresh LoRA over the pinned base; never warm-start from the previous adapter.
        self._assert_base()
        rows = self._training_rows(candidate['data'])
        original_digest = digest(rows)
        curriculum = self._curriculum(candidate['data'], args['round']) if self.paper_kernel is not None else None
        if curriculum is not None:
            by_problem = dict(zip((row['id'] for row in candidate['data']), rows, strict=True))
            rows = [by_problem[key] for key in curriculum['ordered_row_ids']]
        recipe = {'profile': PROFILE, 'engine': self.engine.name, 'base_sha256': self.base_digest,
                  'training_rows_sha256': digest(rows), 'round': args['round'], 'options': self.options,
                  'steps': self.options['steps'] * (2 if args['profile'] == 'detailed' else 1)}
        if curriculum is not None:
            recipe['source_rows_sha256'] = original_digest
            recipe['curriculum_sha256'] = digest(curriculum)
            recipe['curriculum_counts'] = {'samples': len(rows), 'batch_size': curriculum['batch_size'],
                                          'retention_monitors': len(curriculum['retention_monitor_ids'])}
            self._write_named(f'curriculum-{digest(curriculum)}.json', curriculum)
        if self.aime is not None:
            if not candidate['data'] or proof is None or proof != self._fresh_proof(self._id(candidate), candidate):
                raise ValueError('AIME fit requires durable verified fresh data on this exact candidate')
            self._verify_data(candidate['data'])
            recipe.update(profile='meta-rsi/aime/v2', label_scope='assistant_only',
                          train_protocol_id=self._protocols['train'], refresh_sha256=digest(proof))
        if self.wide is not None:
            from .aime_scale_probe import bucket_waves
            from rsicomposer.backends.training.lora.engine import LoRAEngine
            lengths = LoRAEngine(self.options).count_supervised_tokens(self.base_path, rows)
            recipe.update(unbucketed_rows_sha256=digest(rows), steps=(len(rows) + 15) // 16,
                          verified_supervised_tokens=sum(lengths),
                          fit_overrides={'learning_rate': (self.options['learning_rate'] * .5
                              if args['profile'] == 'detailed' else self.options['learning_rate'])})
            # Length bucketing preserves the planned update membership and uneven tail.
            rows = bucket_waves(rows, lengths, 16)
            recipe['training_rows_sha256'] = digest(rows)
            if self._model_stage is not None or self.model_driven:
                recipe['worker_round'] = 0  # Optimizer seed is the outer Seed 23, not seed + proposal ordinal.
            if self._model_stage is not None:
                recipe['model_stage_sha256'] = digest(self._model_stage)
        if proposal is not None:
            from .aime_domain_policy import compile_model_recipe
            if not self.engine.simulated:
                recipe['fit_overrides'] = compile_model_recipe(proposal, base_learning_rate=self.options['learning_rate'])
            recipe['operator_proposal_sha256'] = digest(proposal)
        return fit_checkpoint(self.root / 'models', self.engine, recipe, rows)

    def _wide_fit_receipt(self, checkpoint, candidate):
        from .aime_scale_receipt import verify_training_receipt
        root = self.root / 'models'
        payload = read_checkpoint(root, checkpoint, simulated=False)
        self._validate_fit(candidate, payload)
        recipe = payload['recipe']
        return verify_training_receipt(root / checkpoint / 'training.json', self._training_rows(candidate['data']),
            checkpoint, steps=recipe['steps'], full_token_count=recipe['verified_supervised_tokens'], world_size=4,
            global_batch_size=16, allow_uneven_tail=True,
            learning_rate=recipe['fit_overrides']['learning_rate'])

    def _prompt(self, problem, harness, data):
        if self.aime is not None:
            by_digest = {digest(row): row for row in data} if harness['exemplars'] else {}
            examples = [by_digest[key] for key in harness['exemplars'] if by_digest[key]['id'] != problem.id]
            demonstrations = ''.join(f"\n\nQuestion: {row['prompt']}\nAnswer: {row['completion']}" for row in examples)
            guidance = ('\nOptional solving guidance (does not change the answer format): ' + harness['guidance'] + '\n'
                        if 'guidance' in harness else '')
            return [{'role': 'system', 'content': HARNESSES[harness['instruction']] + guidance + ' '
                     + problem.verifier.answer_format + demonstrations}, {'role': 'user', 'content': problem.prompt}]
        demonstrations = ''
        if harness['few_shot']:
            examples = [row for row in data if row['id'] != problem.id][:2]
            demonstrations = ''.join(f"Question: {row['prompt']}\nAnswer: {row['completion']}\n\n" for row in examples)
        return (HARNESSES[harness['instruction']] + ' ' + problem.verifier.answer_format + '\n\n'
                + demonstrations + f'Question: {problem.prompt}\nAnswer:')

    def _generate(self, candidate, problems, *, count, round_number=0, sample=False, certificate=False):
        checkpoint = self._checkpoint(candidate['model'], candidate)
        harness = candidate['harness']
        rows = [{'question': problem.prompt, 'prompt': self._prompt(problem, harness, candidate['data'])}
                for problem in problems]
        if certificate:
            from .aime_curriculum import SUBSET_RESIDUE_CERTIFICATE_INSTRUCTION
            for row in rows:
                row['prompt'][-1]['content'] += '\n' + SUBSET_RESIDUE_CERTIFICATE_INSTRUCTION
        temperature = self.options['temperature'] if sample else 0.0
        answers = self.engine.generate(rows, count, model={'checkpoint': checkpoint}, harness=harness,
            round_number=round_number, sample=temperature > 0, temperature=temperature)
        if len(answers) != len(problems) or any(len(group) != count for group in answers):
            raise RuntimeError('MetaRSI engine returned an incomplete generation batch')
        def attempt(answer):
            if self.aime is not None and isinstance(answer, dict) and answer.get('status') == 'truncated':
                return MathAttempt(answer['completion'], 'truncated', answer['finish_reason'], simulated=self.engine.simulated)
            return MathAttempt(answer if _text(answer) else None, 'answered' if _text(answer) else 'invalid_format',
                               simulated=self.engine.simulated)
        return [[attempt(answer) for answer in group] for group in answers]

    def _domain_diagnosis(self, call, candidate, *, train_records=()):
        from .aime_diagnosis import parse, proposal_messages
        args, op = call.arguments, call.arguments['operator']
        origin, evidence = self._paper_learning_signal(args['parent'], candidate, train_records)
        state = digest({'kernel_state': self._kernel_identity(args['parent'], candidate)['state_sha256'],
                        'observed_behavior_source': origin,
                        'round': args['round'], 'profile': args['profile'], 'operator_policy': args['strategy']})
        messages = proposal_messages(op, state_sha256=state, evidence=evidence)
        binding = digest({'request': messages, 'round': args['round']})
        name = f'diagnosis-model-{binding}.json'
        record = self._read_owned(self.root, name)
        if record is None:
            raw = self._policy_completion(candidate, messages, args['round'])
            try:
                diagnosis = parse(op, raw, state_sha256=state, evidence=evidence)
                record = {'request_sha256': binding, 'response_sha256': digest(raw), 'accepted': True,
                          'diagnosis_sha256': diagnosis.sha256, 'diagnosis': diagnosis.to_dict()}
            except ValueError:
                record = {'request_sha256': binding, 'response_sha256': digest(raw), 'accepted': False}
            record['signal_sha256'] = digest(evidence)
            self._write_named(name, record)
        if record['request_sha256'] != binding or record['signal_sha256'] != digest(evidence) or (record['accepted'] and
                digest(record['diagnosis']) != record['diagnosis_sha256']):
            raise LabToolError('Immutable model diagnosis differs from its canonical state binding')
        return record

    def _domain_proposal(self, call, candidate, *, eligible=(), gap=None, train_records=()):
        from .aime_domain_policy import PROTOCOL, materialize_task, parse_portfolio, parse_proposal, proposal_messages
        args = call.arguments
        op = args['operator']
        context = {'round': args['round'], 'data_count': len(candidate['data']),
                   'remaining_capacity': self._capacity - len(candidate['data']),
                   'has_adapter': candidate['model']['checkpoint'] is not None,
                   'data_fresh': self._fresh_proof(args['parent'], candidate) is not None}
        for role in ('search', 'validation'):
            if role in self._measured.get(args['parent'], []):
                signal = self._diagnostic(args['parent'], role)
                context[role] = {k: signal[k] for k in ('count', 'correct', 'invalid_format', 'truncated',
                                                        'prompt_overlength') if k in signal}
        max_tasks = 2 if args['profile'] == 'compact' else 4
        max_exemplars = min(self.aime['max_exemplars'], 1 if args['profile'] == 'compact' else 2)
        bounds = {'eligible_exemplars': eligible, 'max_tasks': max_tasks, 'max_exemplars': max_exemplars}
        messages = proposal_messages(op, profile=args['profile'], context=context, **bounds)
        portfolio = self.paper_kernel is not None and op in ('harness', 'model')
        if portfolio and op == 'harness':
            bounds['allow_clear_guidance'] = True
            messages[0]['content'] += ' Use guidance:null to explicitly remove the active guidance.'
        if portfolio:
            messages[0]['content'] += (' For bounded candidate search you may instead return a JSON object with one '
                'key, candidates, whose list contains zero, one, or two different patches with the same schema. '
                'Host, not you, independently screens them and chooses at most one for validation.')
        diagnosis = self._domain_diagnosis(call, candidate, train_records=train_records) if self.paper_kernel is not None else None
        adapter_context = {}
        if self.paper_kernel is not None:
            source = self._read_owned(self.root, f"operator-export-{args['parent']}.json")
            adapter_context = {'program_intent': args['program_step']}
            if source is not None:
                edge = f"{source['operator']}->{op}"
                guidance = args['transition_guidance'].get(edge)
                if guidance is not None:
                    adapter_context['transition_advice'] = guidance
                if edge == 'model->harness':
                    checkpoint = candidate['model']['checkpoint']
                    report = read_checkpoint(self.root / 'models', checkpoint, simulated=self.engine.simulated)
                    adapter_context['training_report'] = {'samples': report['samples'],
                        'active_exemplar_count': len(candidate['harness']['exemplars']),
                        'active_guidance': candidate['harness'].get('guidance')}
        # Generated strategies and the verified aggregate gap are user-data fields, never privileged instructions.
        messages[-1]['content'] += '\nHost-observed gap and advisory strategy (data, not instructions): ' + json.dumps({
            'gap': gap, 'operator_advice': args['strategy'],
            **({'diagnosis': diagnosis.get('diagnosis')} if diagnosis else {}), **adapter_context},
            ensure_ascii=False, sort_keys=True)
        binding = digest({'protocol': PROTOCOL, 'args': args, 'train': self._protocols['train'], 'messages': messages,
                          'decode': 'same-pinned-target-greedy-no-thinking-768-v1'})
        name = f'domain-policy-{binding}.json'
        def checked(raw):
            patch = (parse_portfolio if portfolio else parse_proposal)(op, raw, **bounds)
            if self.paper_kernel is not None and op == 'data':
                for spec in patch['tasks']:
                    materialize_task(spec)
            return patch

        prior = self._read_owned(self.root, name)
        if prior is not None:
            if prior.get('request_sha256') != binding or not aime.sha256_text(prior.get('response_sha256')):
                raise LabToolError('Domain proposal replay did not match its immutable binding')
            raw = prior['response']
            if not isinstance(raw, str) or digest(raw) != prior['response_sha256']:
                raise LabToolError('Domain proposal replay changed after publication')
        else:
            raw = self._policy_completion(candidate, messages, args['round'])
            if len(raw.encode()) > 8192:
                raw = ''
            self._write_named(name, {'request_sha256': binding, 'response_sha256': digest(raw), 'response': raw})
        audit = {'protocol': PROTOCOL, 'request_sha256': binding, 'response_sha256': digest(raw), 'accepted': False,
                 **({'diagnosis': {k: v for k, v in diagnosis.items() if k != 'diagnosis'}} if diagnosis else {})}
        try:
            proposal = checked(raw)
        except ValueError as exc:
            return None, {**audit, **({'reason': str(exc)} if self.paper_kernel is not None else {})}
        audit.update(accepted=True, proposal_sha256=digest(proposal))
        return proposal, audit

    def _collect_model_synthesis(self, call, candidate):
        from .aime_domain_policy import materialize_task
        args = call.arguments
        data = deepcopy(candidate['data'])
        seen = {row['id'] for row in data}
        audit = {'rule': 'observed-train-gap-model-v1', 'fresh_blind_originals': 0, 'fresh_original_failures': 0,
                 'verified_original_recoveries': 0, 'nominated': 0, 'already_held': 0,
                 'rejected_by_anchor_or_verifier': 0, 'admitted': []}
        remaining = [ident for ident in (self._wide_bootstrap['failed_problem_ids'] if self.wide is not None
                                         else self._train_rows) if ident not in seen]
        if len(data) >= self._capacity or not remaining and self.paper_kernel is None:
            return data, audit
        start = (args['round'] * 7) % len(remaining) if remaining else 0
        width = (len(remaining) if self.paper_kernel is not None else
                 min(8 if args['profile'] == 'compact' else 16, len(remaining)))
        originals = [self._train_rows[remaining[(start + i) % len(remaining)]] for i in range(width)]
        blind = {**candidate, 'harness': {**candidate['harness'], 'few_shot': False, 'exemplars': []}}

        def admit(problem, attempt):
            completion = attempt.completion
            if (problem.id in seen or len(data) >= self._capacity or not completion
                    or len(completion) > aime.completion_char_limit(self.aime)):
                return None
            row = {'id': problem.id, 'prompt': problem.prompt, 'completion': completion.strip()}
            seen.add(problem.id)
            data.append(row)
            return row

        attempts = [rows[0] for rows in self._generate(blind, originals, count=1)] if originals else []
        checks = self.verify(originals, attempts)
        failed = [(p, a) for p, a, v in zip(originals, attempts, checks, strict=True) if v is None or not v.correct]
        audit.update(fresh_blind_originals=len(originals), fresh_original_failures=len(failed))
        for p, a, v in zip(originals, attempts, checks, strict=True):
            if v is not None and v.correct and (row := admit(p, a)) is not None:
                if self.paper_kernel is not None:
                    self._tag_curriculum(row, 'recovered')
                audit['verified_original_recoveries'] += 1
        if not failed or len(data) >= self._capacity:
            return data, audit
        # The model sees verified failure counts and broad train-only task types, never rows or answer keys.
        def classify(prompt):
            text = prompt.casefold()
            if any(w in text for w in ('ways', 'subsets', 'arrangements', 'choose', 'combinations', 'probability')):
                return 'counting_or_probability'
            if any(w in text for w in ('remainder', 'modulo', 'divisible', 'prime')):
                return 'number_theory'
            return 'other_math'
        public_gap = {label: sum(classify(p.prompt) == label for p, _ in failed) for label in (
            'counting_or_probability', 'number_theory', 'other_math')}
        public_gap['failed_attempts'] = len(failed)
        public_gap['truncated_attempts'] = sum(attempt.status == 'truncated' for _, attempt in failed)
        training_signal = [self.record(p, a, v) for p, a, v in zip(originals, attempts, checks, strict=True)]
        proposal, audit['actor'] = self._domain_proposal(call, candidate, gap=public_gap, train_records=training_signal)
        if proposal is None:
            return data, audit

        def norm(value):
            return ' '.join(normalize('NFKC', value).casefold().split())
        known = {norm(p.prompt) for name in self.cohorts.values() for p in self.datasets[name]}
        known.update(norm(row['prompt']) for row in data)
        pool = []
        for spec in proposal['tasks']:
            problem, provenance = materialize_task(spec)
            normalized = norm(problem.prompt)
            if problem.id not in seen and normalized not in known:
                known.add(normalized)
                pool.append((problem, provenance))
        audit['nominated'] = len(pool)
        if not pool:
            return data, audit
        greedy = [a[0] for a in self._generate(blind, [p for p, _ in pool], count=1)]
        unknown = [(p, provenance) for (p, provenance), check in zip(pool, self.verify(
            [p for p, _ in pool], greedy), strict=True) if check is None or not check.correct]
        audit['already_held'] = len(pool) - len(unknown)
        if self.paper_kernel is not None:
            unknown_ids = {p.id for p, _ in unknown}
            retained = [(p, provenance) for p, provenance in pool if p.id not in unknown_ids]
            if retained:
                problems = [p for p, _ in retained]
                monitor = [a[0] for a in self._generate(blind, problems, count=1, sample=True, certificate=True,
                                                         round_number=4096 + args['round'])]
                for (p, provenance), attempt, v in zip(retained, monitor, self.verify(problems, monitor), strict=True):
                    certificate = self._checked_certificate(p, attempt)
                    if v is not None and v.correct and certificate is not None and (row := admit(p, attempt)):
                        self._tag_curriculum(row, 'already_held')
                        audit.setdefault('consolidation', []).append({**provenance, 'item_sha256': digest(p.id),
                            'row_sha256': digest(row), 'certificate_sha256': digest(certificate)})
        if not unknown:
            return data, audit
        # Independent invocations with distinct round seeds; neither sees the other completion or the reference.
        problems = [p for p, _ in unknown]
        left = [a[0] for a in self._generate(blind, problems, count=1, sample=True,
                                               round_number=1024 + args['round'] * 2,
                                               certificate=self.paper_kernel is not None)]
        right = [a[0] for a in self._generate(blind, problems, count=1, sample=True,
                                                round_number=1025 + args['round'] * 2,
                                                certificate=self.paper_kernel is not None)]
        for (p, provenance), attempt, anchor, a, b in zip(unknown, left, right, self.verify(problems, left),
                                                        self.verify(problems, right), strict=True):
            proofs = []
            if self.paper_kernel is not None:
                proofs = [self._checked_certificate(p, atry) for atry in (attempt, anchor)]
            if a is None or not a.correct or b is None or not b.correct or any(proof is None for proof in proofs):
                audit['rejected_by_anchor_or_verifier'] += 1
            elif (row := admit(p, attempt)) is not None:
                if self.paper_kernel is not None:
                    self._tag_curriculum(row, 'frontier')
                audit['admitted'].append({**provenance, 'item_sha256': digest(p.id), 'row_sha256': digest(row),
                    'blind_completion_sha256': [digest(attempt.completion), digest(anchor.completion)],
                    **({'certificate_sha256': [digest(proof) for proof in proofs]} if proofs else {})})
        return data, audit

    @staticmethod
    def _checked_certificate(problem, attempt):
        from .aime_curriculum import check_subset_residue_completion, subset_residue_spec_from_id
        if not attempt.completion:
            return None
        tail = '\n'.join(attempt.completion.strip().splitlines()[-2:])
        try:
            return check_subset_residue_completion(subset_residue_spec_from_id(problem.id), tail)
        except ValueError:
            return None

    def _tag_curriculum(self, row, lane):
        self._write_named(f'curriculum-row-{digest(row)}.json', {'row_sha256': digest(row), 'lane': lane})

    def _curriculum(self, data, round_number):
        from .aime_curriculum import ALREADY_HELD, REPLAY, plan_curriculum
        labels = {}
        for row in data:
            key = digest(row)
            record = self._read_owned(self.root, f'curriculum-row-{key}.json')
            if record is not None and record.get('row_sha256') != key:
                raise ValueError('Curriculum provenance changed its verified row')
            labels[row['id']] = REPLAY if record is None else record['lane']
        plan = plan_curriculum(labels, seed=self.options['seed'], round_number=round_number,
                               min_retention=int(ALREADY_HELD in labels.values()))
        return plan

    def _replay_retention(self, parent, candidate, plan):
        ids = plan['retention_monitor_ids']
        if not ids:
            return {'count': 0, 'preserved': 0, 'regressed': 0, 'scope': 'verified_already_held_training_monitor'}
        problems = [self._resolve_train_problem(key) for key in ids]
        def vector(value):
            attempts = [group[0] for group in self._generate(value, problems, count=1)]
            return [check is not None and check.correct for check in self.verify(problems, attempts)]
        before, after = vector(parent), vector(candidate)
        return {'count': len(ids), 'preserved': sum(a and b for a, b in zip(before, after, strict=True)),
                'regressed': sum(a and not b for a, b in zip(before, after, strict=True)),
                'scope': 'verified_already_held_training_monitor'}

    def _collect_wide_synthesis(self, candidate, profile, round_number):
        data = deepcopy(candidate['data'])
        seen = {row['id'] for row in data}
        audit = {'rule': 'observed-train-gap-affine-v1', 'fresh_blind_originals': 0, 'fresh_original_failures': 0,
                 'verified_original_recoveries': 0, 'nominated': 0, 'already_held': 0,
                 'rejected_by_anchor_or_verifier': 0, 'admitted': []}
        if len(data) >= self._capacity:
            return data, audit
        width = 8 if profile == 'compact' else 16
        # Use only historical training rollout non-admissions, never evaluation questions.
        old_gaps = self._wide_bootstrap['failed_problem_ids']
        start = (round_number * 7) % len(old_gaps)
        originals = [self._train_rows[old_gaps[(start + i) % len(old_gaps)]]
                     for i in range(min(width, len(old_gaps)))]
        if not originals:
            return data, audit
        blind = {**candidate, 'harness': {**candidate['harness'], 'few_shot': False, 'exemplars': []}}
        def admit(problem, attempt):
            if problem.id in seen or len(data) >= self._capacity or not attempt.completion or (
                    len(attempt.completion) > aime.completion_char_limit(self.aime)):
                return None
            row = {'id': problem.id, 'prompt': problem.prompt, 'completion': attempt.completion.strip()}
            seen.add(problem.id)
            data.append(row)
            return row

        attempts = [item[0] for item in self._generate(blind, originals, count=1)]
        checks = self.verify(originals, attempts)
        audit['fresh_blind_originals'] = len(originals)
        failures = []
        for problem, attempt, check in zip(originals, attempts, checks, strict=True):
            if check is None or not check.correct:
                failures.append(problem)
            elif admit(problem, attempt) is not None:
                audit['verified_original_recoveries'] += 1
        audit['fresh_original_failures'] = len(failures)
        proposed = [aime_synthesis.propose(problem, self.options['seed'], round_number) for problem in failures]
        pool = [(problem, src) for pair in proposed if pair is not None
                for problem, src in (pair,) if problem.id not in seen]
        audit['nominated'] = len(pool)
        if not pool or len(data) >= self._capacity:
            return data, audit
        tasks = [p for p, _ in pool]
        held = self.verify(tasks, [item[0] for item in self._generate(blind, tasks, count=1)])
        unknown = [(p, src) for (p, src), check in zip(pool, held, strict=True) if check is None or not check.correct]
        audit['already_held'] = len(pool) - len(unknown)
        if not unknown:
            return data, audit
        tasks = [p for p, _ in unknown]
        draws = self._generate(blind, tasks, count=2, sample=True, round_number=1024 + round_number)
        authored = self.verify(tasks, [pair[0] for pair in draws])
        anchored = self.verify(tasks, [pair[1] for pair in draws])
        for (problem, source), pair, left, right in zip(unknown, draws, authored, anchored, strict=True):
            if left is None or not left.correct or right is None or not right.correct:
                audit['rejected_by_anchor_or_verifier'] += 1
                continue
            row = admit(problem, pair[0])
            if row is not None:
                audit['admitted'].append({'source_sha256': digest(source['source_id']), 'rule': source['rule'],
                    'item_sha256': digest(problem.id), 'row_sha256': digest(row)})
        return data, audit

    def _collect_synthesis(self, candidate, profile, round_number):
        data = deepcopy(candidate['data'])
        audit = {'rule': 'aime-affine-v1', 'nominated': 0, 'already_held': 0,
                 'rejected_by_anchor_or_verifier': 0, 'admitted': []}
        if len(data) >= MAX_ROWS:
            return data, audit
        train = tuple(self._train_rows.values())
        width = min(len(train), 2 if profile == 'compact' else 4, MAX_ROWS - len(data))
        start = (round_number * width) % len(train)
        prior = {row['id'] for row in data}
        nominations = [aime_synthesis.propose(train[(start + i) % len(train)], self.options['seed'], round_number)
                       for i in range(width)]
        pool = [(p, source) for pair in nominations if pair is not None for p, source in (pair,) if p.id not in prior]
        if not pool:
            return data, audit
        audit['nominated'] = len(pool)
        tasks = [p for p, _ in pool]
        # Held-knowledge check is a blind greedy solve by the current system.
        first = [group[0] for group in self._generate(candidate, tasks, count=1)]
        checks = self.verify(tasks, first)
        remaining = [(p, source) for (p, source), check in zip(pool, checks, strict=True)
                     if check is None or not check.correct]
        audit['already_held'] = len(pool) - len(remaining)
        if not remaining:
            return data, audit
        tasks = [p for p, _ in remaining]
        # The same target model independently draws a blind author answer and
        # an Anchor answer. Neither sees a reference, earlier reply, or dev row.
        left = [group[0] for group in self._generate(candidate, tasks, count=1, sample=True,
                                                    round_number=round_number * 2 + 1)]
        right = [group[0] for group in self._generate(candidate, tasks, count=1, sample=True,
                                                     round_number=round_number * 2 + 2)]
        author_checks, anchor_checks = self.verify(tasks, left), self.verify(tasks, right)
        for (problem, source), attempt, authored, anchored in zip(remaining, left, author_checks, anchor_checks,
                                                                  strict=True):
            if (authored is None or not authored.correct or anchored is None or not anchored.correct
                    or len(attempt.completion) > aime.completion_char_limit(self.aime)):
                audit['rejected_by_anchor_or_verifier'] += 1
                continue
            row = {'id': problem.id, 'prompt': problem.prompt, 'completion': attempt.completion.strip()}
            data.append(row)
            audit['admitted'].append({**source, 'item_id': problem.id, 'row_sha256': digest(row)})
        return data, audit

    def _collect(self, candidate, profile, round_number, *, audit=None):
        data = deepcopy(candidate['data'])
        if audit is not None:
            audit.update(attempted=0, truncated=0, verifier_correct=0, too_long_for_storage=0)
        if len(data) == MAX_ROWS:
            return data
        train = self.datasets[self.cohorts['train']]
        start = (round_number * MAX_ROWS) % len(train)
        cohort = tuple(train[(start + i) % len(train)] for i in range(min(MAX_ROWS, len(train))))
        count = 3 if profile == 'compact' else 4
        groups = self._generate(candidate, cohort, count=count, round_number=round_number, sample=True)
        problems = [problem for problem in cohort for _ in range(count)]
        attempts = [attempt for group in groups for attempt in group]
        seen = {(row['id'], row['completion'].strip()) for row in data}
        checks = self.verify(problems, attempts)
        completion_limit = aime.completion_char_limit(self.aime)
        if audit is not None:
            audit.update(attempted=len(attempts), truncated=sum(x.status == 'truncated' for x in attempts),
                verifier_correct=sum(x is not None and x.correct for x in checks),
                too_long_for_storage=sum(x is not None and x.correct and len(a.completion) > completion_limit
                                         for a, x in zip(attempts, checks, strict=True)))
        for problem, attempt, check in zip(problems, attempts, checks, strict=True):
            if check is None or not check.correct or len(attempt.completion) > completion_limit:
                continue
            if self.aime is not None and max(len(problem.id), len(problem.prompt)) > aime.MAX_TRAIN_TEXT:
                continue
            key = (problem.id, attempt.completion.strip())
            if key not in seen:
                data.append({'id': problem.id, 'prompt': problem.prompt, 'completion': attempt.completion.strip()})
                seen.add(key)
                if len(data) == MAX_ROWS:
                    break
        return data

    def _evaluate(self, call, *, baseline):
        role, key = call.arguments['split'], call.arguments['candidate']
        candidate = self._candidate(key)
        if (self.paper_kernel is not None and key != self._seed_id and
                self._fresh_proof(key, candidate) is not None):
            raise LabToolError('A verified Data stage is a training input, not a measured capability candidate')
        result = self.evaluate(call.call_id, {'split': self.cohorts[role], '_role': role, 'candidate_id': key,
            'baseline': baseline, '_candidate': candidate})
        if role != 'test':
            if self.aime is not None:
                name = self._diagnostic_name(key, role)
                if self._read_owned(self.root, name) is None:
                    data = result.data
                    self._write_named(name, aime.metrics(key, role, self._protocols[role], data['count'],
                        data['quality_metrics']['correct'], data['execution_metrics']['invalid_format'],
                        data['execution_metrics']['truncated'], data['prompt_overlength']))
            self._measured[key] = sorted(set(self._measured.get(key, [])) | {role})
        return result

    def solve(self, problems, arguments):
        return self.run_runtime(arguments['_candidate'], problems, arguments)

    def result_data(self, records, ref, arguments):
        role = arguments['_role']
        if self.paper_kernel is not None and role == 'validation':
            from .aime_release import make_measurement
            key = arguments['candidate_id']
            name = f'paired-validation-{key}.json'
            if self._read_owned(self.root, name) is None:
                self._write_named(name, make_measurement(key, self._protocols[role], records))
        return {'split': role, 'candidate_id': arguments['candidate_id'], 'baseline': arguments['baseline'],
                 'score': sum(record['correct'] for record in records) / len(records),
                 'protocol': PROFILE, 'protocol_id': self._protocols[role], 'training_performed': False,
                 **({'trial_inputs_sha256': self._mounted_inputs_sha256} if self._mounted_inputs_sha256 is not None else {}),
                 **({'prompt_overlength': sum(row['finish_reason'] == 'prompt_budget' for row in records)}
                    if self.aime is not None and role in aime.DEV_ROLES else {})}
