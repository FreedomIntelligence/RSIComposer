"""Budgeted MetaRSI policy: improve a task system, then evaluate edits to its improver."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from random import Random
from time import monotonic
from uuid import uuid4

from rsicomposer.api import Decision, ExecutionStatus, OperationCall

from .campaign import CROSS_TERM_TARGETS, VERTICAL_TARGETS, campaign_config, change_order, diagnose, route, vertical_patch
from .evidence import comparable, trial_score
from .trial_inputs import validate_inputs, validate_seed_id


OPERATORS = ('data', 'harness', 'model')
TARGETS = (*CROSS_TERM_TARGETS, 'proposal_policy', *VERTICAL_TARGETS)
PROFILES = ('compact', 'detailed')
NAME = 'meta_rsi'


def validate_config(config):
    allowed = {'order', 'profiles', 'evidence', 'history_seed', 'steps', 'warmup', 'min_delta',
               'trial_steps', 'trial_budget', 'trial_lab', 'meta_targets', 'allow_simulated_trials', 'trial',
               'campaign', 'round_offset', 'trial_inputs'}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError('Unknown MetaRSI Controller options')
    campaign = campaign_config(config.get('campaign'))
    offset = config.get('round_offset', 0)
    if type(offset) is not int or not 0 <= offset <= 256 or (offset and not config.get('trial')):
        raise ValueError('Only nested MetaRSI trials may inherit a bounded round offset')
    order = config.get('order', list(OPERATORS))
    if (not isinstance(order, list) or not order or any(op not in OPERATORS for op in order)
            or len(set(order)) != len(order)):
        raise ValueError('order must be a nonempty list of distinct MetaRSI operators')
    profiles = config.get('profiles', {})
    if not isinstance(profiles, dict) or set(profiles) - set(OPERATORS) or any(
            p not in PROFILES for p in profiles.values()):
        raise ValueError('profiles must map operators to compact or detailed')
    if config.get('evidence', 'recent') not in ('recent', 'ledger'):
        raise ValueError('evidence must be recent or ledger')
    steps, warmup, trial_steps = (config.get(k, v) for k, v in (('steps', 4), ('warmup', 1), ('trial_steps', 1)))
    if (type(steps) is not int or not 1 <= steps <= 24 or type(warmup) is not int or not 0 <= warmup < steps
            or type(trial_steps) is not int or not 1 <= trial_steps <= 6):
        raise ValueError('Invalid MetaRSI horizon, warmup, or trial steps')
    delta = config.get('min_delta', 0.)
    if type(delta) not in (int, float) or not math.isfinite(delta) or not 0 <= delta <= 1:
        raise ValueError('min_delta must be between 0 and 1')
    targets = config.get('meta_targets', [])
    if (not isinstance(targets, list) or any(t not in TARGETS for t in targets) or len(set(targets)) != len(targets)):
        raise ValueError('meta_targets must be distinct mutable Controller surfaces')
    for key in ('trial', 'allow_simulated_trials'):
        if type(config.get(key, False)) is not bool:
            raise ValueError(f'{key} must be boolean')
    history = config.get('history_seed', [])
    if (not isinstance(history, list) or len(history) > 24 or any(not isinstance(row, dict)
            or set(row) != {'operator', 'candidate', 'parent', 'data_count'} or row['operator'] not in OPERATORS
            or any(row[k] is not None and not isinstance(row[k], str) for k in ('candidate', 'parent'))
            or type(row['data_count']) is not int or row['data_count'] < 0 for row in history)):
        raise ValueError('Invalid development history seed')
    if config.get('trial') and targets:
        raise ValueError('A trial Controller must not start new trials')
    if config.get('trial'):
        validate_inputs(config.get('trial_inputs'))
    elif 'trial_inputs' in config:
        raise ValueError('Only a trial Controller takes a parent-approved input binding')
    if campaign is not None and (campaign['track'] == 'trial') != config.get('trial', False):
        raise ValueError('Campaign track and trial mode must agree')
    if campaign is None and set(targets) & VERTICAL_TARGETS.keys():
        raise ValueError('Per-operator policy trials require campaign mode')
    if campaign is not None and 'proposal_policy' in targets:
        raise ValueError('Use a named data_policy, harness_policy, or model_policy target in campaigns')
    if targets:
        lab, budget = config.get('trial_lab'), config.get('trial_budget')
        if not isinstance(lab, dict) or not isinstance(lab.get('manifest'), str) or not lab['manifest']:
            raise ValueError('Meta-optimization requires explicit dev-only trial Lab settings')
        roles = lab.get('split_roles')
        if not isinstance(roles, dict) or set(roles) != {'train', 'search', 'validation'}:
            raise ValueError('Child Labs must not mount a sealed split')
        if ('initial_snapshot' in lab or not isinstance(budget, dict) or 'seconds' not in budget
                or set(budget) - {'seconds', 'model_calls', 'gpu_seconds'}
                or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in budget.values())
                or ('model_calls' in budget and type(budget['model_calls']) is not int)):
            raise ValueError('Meta trials need finite positive explicit budgets and no pre-seeded snapshot')


def _hash(prefix, value):
    return prefix + hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                              separators=(',', ':')).encode()).hexdigest()[:24]


class MetaRSIController:
    def __init__(self, context, config):
        validate_config(config)
        self.context = context
        self.config = deepcopy(config)
        self.order = config.get('order', list(OPERATORS))
        self.profiles = config.get('profiles', {})
        self.campaign = campaign_config(config.get('campaign'))
        self.targets = config.get('meta_targets', [])
        if self.campaign and self.campaign['track'] != 'adaptive':
            self.targets = []
        self.round_offset = config.get('round_offset', 0)
        self.max_steps = config.get('steps', 4)
        self.warmup = config.get('warmup', 1)
        self.trial_steps = config.get('trial_steps', 1)
        self.margin = config.get('min_delta', 0.)
        self.s = {'phase': 'seed', 'pending': None, 'available': [], 'initial': None, 'current': None,
                  'scores': {}, 'candidate_data': {}, 'attempts': [], 'cycle': [], 'spent': 0,
                  'meta_index': 0, 'meta': [], 'trial_pair': None, 'snapshot': None,
                  'adopt_ref': None, 'active_version': None, 'failed': None, 'simulated': None}
        if self.campaign:
            self.s.update(term=None, terms=[], diagnoses={}, snapshot_start=None)
        if self.targets:
            self.s.update(trial_seed_call_id=uuid4().hex, trial_inputs=None)

    def required_operations(self):
        required = ['math.' + name for name in ('seed', 'propose', 'evaluate', 'select')]
        if self.campaign:
            required.append('math.diagnose')
        if self.targets:
            required += ['math.snapshot', 'reflect.build_candidate', 'reflect.start']
        return tuple(required)

    def required_lab_bindings(self):
        requirement = {'operation_mode': 'rule'}
        if self.targets:
            requirement.update(trial_lab=self.config['trial_lab'], seed_call_id=self.s['trial_seed_call_id'],
                               approved_trial_inputs=self.s['trial_inputs'], fresh_run=not self.context.restoring)
        elif self.config.get('trial'):
            requirement['mounted_trial_inputs'] = self.config['trial_inputs']
        return {'math': requirement}

    def state(self):
        return deepcopy(self.s)

    def restore(self, state):
        if not isinstance(state, dict) or set(state) != set(self.s):
            raise ValueError('Invalid MetaRSI Controller state')
        if self.targets:
            validate_seed_id(state['trial_seed_call_id'])
            if state['initial'] is not None or state['trial_inputs'] is not None:
                validate_inputs(state['trial_inputs'])
        self.s = deepcopy(state)

    def _send(self, phase, name, **args):
        call = (OperationCall(name, args, call_id=self.s['trial_seed_call_id'])
                if self.targets and phase == 'seed' else OperationCall(name, args))
        self.s['pending'] = {'id': call.call_id, 'phase': phase, 'name': name, 'args': args}
        if self.campaign and name.startswith('math.') and name != 'math.snapshot':
            self.s['pending']['issued_at'] = monotonic()
        return Decision.invoke(call)

    def _eval(self, phase, candidate, split):
        return self._send(phase, 'math.evaluate', candidate=candidate, split=split)

    def _context_rows(self, window):
        try:
            # Memory integration may append context after this JSON line.
            return json.JSONDecoder().raw_decode(window.context.lstrip())[0]['meta_rsi_evidence']
        except (ValueError, TypeError, KeyError):
            return []

    def _pick(self, rows):
        available = self.s['available']
        if not self.campaign and self.targets and self.s['snapshot'] is None:
            available = [operator for operator in available if operator != 'model']
        eligible = [op for op in self.order if op in available and op not in self.s['cycle']]
        if not eligible:
            self.s['cycle'] = []
            eligible = [op for op in self.order if op in available]
        if self.campaign:
            if self.campaign['routing'] == 'uniform' and eligible:
                seed = self.campaign['policy_seed'] * 257 + self.round_offset + self.s['spent']
                return Random(seed).choice(eligible)
            signal = self.s['diagnoses'].get(self.s['current'])
            basis = diagnose(signal, available)
            ranked = route(eligible, basis) if self.campaign['routing'] == 'diagnostic' else list(eligible)
            # Keeping more history can prevent redundant collection when fresh
            # verified rows are already usable or two recent attempts failed.
            data_rows = [row for row in rows if row['operator'] == 'data']
            if data_rows and (('model' in available and data_rows[-1]['data_count'] > 0)
                    or len(data_rows) >= 2 and all(row['candidate'] is None for row in data_rows[-2:])):
                ranked.sort(key=lambda op: op == 'data')
            return next(iter(ranked), None)
        # Same policy for both Harnesses. With evidence about recent Data-RSI,
        # move on once verified inputs exist or repeated collection failed.
        data_rows = [row for row in rows if row['operator'] == 'data']
        if data_rows and (data_rows[-1]['data_count'] > 0 or len(data_rows) >= 2 and all(
                row['candidate'] is None for row in data_rows[-2:])):
            eligible.sort(key=lambda op: op == 'data')
        return next(iter(eligible), None)

    def _record_candidate(self, payload):
        ident = payload['candidate_id']
        if not isinstance(ident, str) or not ident:
            raise ValueError('Lab did not return a candidate identity')
        available = payload.get('available_operators', list(OPERATORS))
        if not isinstance(available, list) or any(op not in OPERATORS for op in available):
            raise ValueError('Lab returned unknown available operators')
        self.s['candidate_data'][ident] = {'available': available, 'data_count': payload.get('data_count', 0)}
        return ident

    def _point(self, receipt, candidate):
        data = receipt.data
        score = data['score']
        if (data['candidate_id'] != candidate or type(score) not in (int, float) or not math.isfinite(score)
                or not 0 <= score <= 1 or type(data['count']) is not int or data['count'] < 1):
            raise ValueError('Invalid MetaRSI evaluation receipt')
        return score

    def _child_config(self, baseline, challenge, target):
        original = {k: deepcopy(self.config[k]) for k in ('order', 'profiles', 'evidence', 'min_delta')
                    if k in self.config}
        original.update(trial=True, steps=self.trial_steps, warmup=0, meta_targets=[],
                        trial_inputs=deepcopy(self.s['trial_inputs']),
                        history_seed=[{k: row[k] for k in ('operator', 'candidate', 'parent', 'data_count')}
                                      for row in self.s['attempts'][-24:]])
        if self.campaign:
            original['campaign'] = {**self.campaign, 'terms': 1, 'track': 'trial'}
            original['round_offset'] = self.s['spent']
        if target == 'proposal_policy' or target in VERTICAL_TARGETS:
            # Both arms target the same operator, changing only its proposal profile.
            original['order'] = [baseline]
            original['steps'] = 1
        return original if not challenge else {**original, **deepcopy(challenge)}

    def _pair(self, target):
        available = [op for op in self.order if op in self.s['available'] and (self.campaign or op != 'model')]
        if not available:
            return None
        override = None
        op = None
        contract = None
        if target == 'schedule':
            if len(available) < 2:
                return None
            # The challenger gives the least recently tried eligible operator priority.
            last = {row['operator']: i for i, row in enumerate(self.s['attempts'])}
            lead = min(available, key=lambda op: last.get(op, -1))
            if lead == available[0]:
                lead = available[1]
            revised = change_order(self.order, available, self.s['attempts']) if self.campaign else [
                lead, *(op for op in self.order if op != lead)]
            if revised is None:
                return None
            override = 'order', revised
        elif target in VERTICAL_TARGETS:
            op = VERTICAL_TARGETS[target]
            if op not in available:
                return None
            profiles, contract = vertical_patch(self.profiles, op)
            override = 'profiles', profiles
        elif target == 'proposal_policy':
            # Fix both trial arms to Harness-RSI if it is eligible; prefer it
            # because it has a directly measurable, bounded patch surface.
            op = 'harness' if 'harness' in available else available[0]
            mode = self.profiles.get(op, 'compact')
            override = 'profiles', {**self.profiles, op: 'detailed' if mode == 'compact' else 'compact'}
        else:
            override = 'evidence', 'ledger' if self.config.get('evidence', 'recent') == 'recent' else 'recent'
        key, value = override
        arms = [self._child_config(op, challenge, target) for challenge in ({}, {key: value})]
        if not self.campaign and target != 'proposal_policy':
            for arm in arms:
                arm['order'] = [name for name in arm.get('order', list(OPERATORS)) if name in available]
        return {'target': target, 'change': {key: value}, 'arms': arms, 'i': 0, 'results': [], 'ref': None,
                'version_id': None, 'snapshot': deepcopy(self.s['snapshot']), 'contract': contract}

    def _build_trial(self, pair):
        arm = pair['arms'][pair['i']]
        version_id = _hash('meta-trial-', [pair['target'], pair['i'], arm, pair['snapshot']])
        pair['version_id'] = version_id
        return self._send('build_trial', 'reflect.build_candidate',
                          version={'version_id': version_id, 'config': {'name': NAME, 'config': arm}})

    def _start_trial(self, pair):
        lab = deepcopy(self.config['trial_lab'])
        lab['manifest'] = str(Path(lab['manifest']).expanduser().resolve())
        if 'base_model' in lab and lab.get('engine') == 'transformers':
            lab['base_model'] = str(Path(lab['base_model']).expanduser().resolve())
        lab['initial_snapshot'] = pair['snapshot']
        pair['started'] = monotonic()
        return self._send('run_trial', 'reflect.start', candidate_ref=pair['ref'],
                          objective='Evaluate a fixed-budget MetaRSI Controller on independent development data',
                          budget=self.config['trial_budget'], labs={'math': {'name': NAME, 'config': lab}})

    def _complete_pair(self):
        s, pair = self.s, self.s['trial_pair']
        a, b = pair['results']
        entry = {'target': pair['target'], 'change': pair['change'], 'trials': [a, b], 'decision': 'kept'}
        if self.campaign:
            entry.update(term=s['term']['index'], stage=s['term']['stage'])
            if pair['contract'] is not None:
                entry['policy_diff'] = pair['contract']
        s['meta'].append(entry)
        s['meta_index'] += 1
        s['trial_pair'] = None
        if 'error' in a or 'error' in b or not comparable(a, b):
            entry['reason'] = 'missing or incomparable validation'
            return None
        if a['attempts'] == b['attempts']:
            entry['reason'] = 'candidate did not change the trial actions'
            return None
        if b['delta'] - a['delta'] <= self.margin:
            entry['reason'] = 'challenger did not exceed improvement margin'
            return None
        # An installed version is the already-tested configuration carried into
        # the parent mode, with explicit post-trial state, never a child job.
        history = [{k: row[k] for k in ('operator', 'candidate', 'parent', 'data_count')}
                   for row in s['attempts'][-24:]]
        config = {**self.config, **pair['change'], 'history_seed': history}
        ident = _hash('meta-adopt-', [s['meta_index'], config, a['evidence_ref'], b['evidence_ref']])
        entry.update(decision='adoption_requested', version_id=ident)
        s['phase'] = 'advance'
        handoff = self.state()
        handoff['active_version'] = ident
        handoff['meta'][-1]['decision'] = 'adopted'
        return self._send('build_adoption', 'reflect.build_candidate', version={
            'version_id': ident, 'config': {'name': NAME, 'config': config}, 'policy_state': handoff})

    def _consume(self, pending, receipt):
        phase, args, payload, s = pending['phase'], pending['args'], receipt.data, self.s
        success = receipt.status == ExecutionStatus.COMPLETED
        if phase in ('build_trial', 'run_trial') and not success:
            pair = s['trial_pair']
            failure = {'error': receipt.error or receipt.status.value}
            if 'started' in pair:
                failure.update(parent_observed_wall_seconds=max(0., monotonic() - pair.pop('started')),
                               requested_budget=deepcopy(self.config['trial_budget']))
            pair['results'].append(failure)
            s['phase'] = 'next_arm'
            return None
        if not success:
            s['failed'] = receipt.error or receipt.status.value
            s['phase'] = 'failed'
            return None
        if self.campaign and s['term'] is not None and 'issued_at' in pending:
            metrics = payload.get('execution_metrics', {})
            s['term']['task_operations'].append({
                'operation': pending['name'], 'elapsed_seconds': max(0., monotonic() - pending['issued_at']),
                'subsplit': args.get('split'), 'evaluated_items': payload.get('count', 0) if phase in
                ('baseline', 'search', 'validation') else 0,
                'input_tokens_reported_samples': metrics.get('input_tokens_reported_samples', 0),
                'output_tokens_reported_samples': metrics.get('output_tokens_reported_samples', 0),
                'input_tokens': metrics.get('input_tokens'), 'output_tokens': metrics.get('output_tokens'),
            })
        if pending['name'].startswith('math.'):
            if s['simulated'] is None:
                s['simulated'] = receipt.simulated
            elif s['simulated'] != receipt.simulated:
                raise ValueError('Unexpected real/simulated Math Lab transition')
        if phase == 'seed':
            if self.targets:
                validate_inputs(payload.get('approved_trial_inputs'))
                s['trial_inputs'] = deepcopy(payload['approved_trial_inputs'])
            ident = self._record_candidate(payload)
            s.update(initial=ident, current=ident, available=s['candidate_data'][ident]['available'])
            return self._eval('baseline', ident, 'validation')
        elif phase == 'baseline':
            s['scores'][s['current']] = self._point(receipt, s['current'])
            s['phase'] = 'advance'
            if self.campaign:
                self._open_term(1)
                return self._send('diagnose', 'math.diagnose', candidate=s['current'], split='validation')
        elif phase == 'diagnose':
            ident = args['candidate']
            if (payload.get('candidate_id') != ident or not payload.get('protocol_id')
                    or payload.get('accuracy') != s['scores'][ident]):
                raise ValueError('Diagnosis must reference the measured validation candidate')
            available = payload.get('available_operators', s['available'])
            if not isinstance(available, list) or any(op not in OPERATORS for op in available):
                raise ValueError('Diagnosis may not introduce unknown operators')
            signal = {k: payload[k] for k in ('protocol_id', 'count', 'accuracy', 'invalid_format')}
            signal['basis'] = diagnose(signal, available)
            if 'truncated' in payload:
                signal['truncated'] = payload['truncated']
            s['diagnoses'][ident] = signal
            s['available'] = available
            s['candidate_data'][ident]['available'] = available
            s['phase'] = 'advance'
        elif phase == 'propose':
            candidate = payload['candidate_id']
            row = {'operator': args['operator'], 'profile': args['profile'], 'parent': args['parent'],
                   'candidate': candidate, 'data_count': payload.get('data_count', 0),
                   'training_performed': payload.get('training_performed', False), 'selected': False}
            if self.campaign:
                row.update(term=s['term']['index'], round=args['round'],
                           diagnosis=s['diagnoses'].get(args['parent'], {}).get('basis', 'no_signal'))
                for key in ('reason', 'transition', 'data_fresh', 'candidate_kind'):
                    if key in payload:
                        row[key] = deepcopy(payload[key])
            s['attempts'].append(row)
            if candidate is not None and candidate != s['current']:
                self._record_candidate(payload)
                return self._eval('search', candidate, 'search')
            s['phase'] = 'advance'
        elif phase == 'search':
            row = s['attempts'][-1]
            row['search_score'] = self._point(receipt, row['candidate'])
            return self._eval('validation', row['candidate'], 'validation')
        elif phase == 'validation':
            row = s['attempts'][-1]
            ident = row['candidate']
            score = self._point(receipt, ident)
            s['scores'].setdefault(ident, score)
            score = s['scores'][ident]
            incumbent = s['scores'][s['current']]
            # Verified data can be valuable as an input to later operators even
            # when it alone leaves the task behavior unchanged; never regress.
            neutral_data = (row['operator'] == 'data' and score == incumbent and row['data_count']
                            > s['candidate_data'][s['current']]['data_count'])
            row.update(validation_score=score, selected=(score - incumbent > self.margin or neutral_data))
            if self.campaign:
                row['gain'] = score - incumbent
            if row['selected']:
                s.update(current=ident, available=s['candidate_data'][ident]['available'])
                if self.campaign:
                    return self._send('diagnose', 'math.diagnose', candidate=ident, split='validation')
            s['phase'] = 'advance'
        elif phase == 'snapshot':
            s['snapshot'] = payload['snapshot']
            if self.campaign:
                if payload.get('candidate_id') != s['current']:
                    raise ValueError('Terminal snapshot must match the released candidate')
                s['snapshot_start'] = s['current']
            s['phase'] = 'advance'
        elif phase == 'build_trial':
            pair = s['trial_pair']
            if payload['version_id'] != pair['version_id']:
                raise ValueError('Trial version does not match publication')
            pair['ref'] = payload['candidate_ref']
            return self._start_trial(pair)
        elif phase == 'run_trial':
            pair = s['trial_pair']
            try:
                result = trial_score(self.context.artifacts, receipt, version_id=pair['version_id'],
                                     candidate_ref=pair['ref'], allow_simulated=self.config.get(
                                         'allow_simulated_trials', False),
                                     expected_steps=pair['arms'][pair['i']]['steps'], start_id=s['current'],
                                     approved_inputs=s['trial_inputs'])
            except (KeyError, TypeError, ValueError, OSError) as exc:
                result = {'error': str(exc)}
            result['parent_observed_wall_seconds'] = max(0., monotonic() - pair.pop('started'))
            result['requested_budget'] = deepcopy(self.config['trial_budget'])
            pair['results'].append(result)
            s['phase'] = 'next_arm'
        elif phase == 'build_adoption':
            s['adopt_ref'] = payload['candidate_ref']
            return self._send('adopt', 'host.adopt_controller', version=s['adopt_ref'])
        elif phase == 'adopt':
            # The active successor receives the state immediately preceding the
            # request. This branch is fail-closed if the old policy keeps control.
            s['phase'] = 'await_adoption'
        elif phase in ('select', 'term_release'):
            if payload['candidate_id'] != s['current']:
                raise ValueError('Selection receipt differs from current system')
            if phase == 'term_release':
                self._record_term()
                s['phase'] = 'done' if s['term']['index'] == self.campaign['terms'] else 'advance'
            else:
                s['phase'] = 'done'
        return None

    def _open_term(self, index):
        s = self.s
        s['term'] = {'index': index, 'start': s['current'], 'spent_at_start': s['spent'],
                     'attempt_start': len(s['attempts']), 'stage': 'vertical', 'opened_at': monotonic(),
                     'controller_version': s['active_version'], 'order': list(self.order),
                     'profiles': deepcopy(self.profiles), 'task_operations': []}
        s.update(cycle=[], snapshot=None, snapshot_start=None)

    def _record_term(self):
        s, term = self.s, self.s['term']
        start, released = term['start'], s['current']
        rows = s['attempts'][term['attempt_start']:]
        operations = term['task_operations']
        meta_trials = [t for m in s['meta'] if m['term'] == term['index'] and m['stage'] == 'vertical'
                       for t in m.get('trials', [])]
        s['terms'].append({'term': term['index'], 'start_id': start, 'released_id': released,
                           'start_score': s['scores'][start], 'released_score': s['scores'][released],
                           'gain': s['scores'][released] - s['scores'][start],
                           'operator_budget': self.max_steps, 'operator_calls': len(rows),
                           'operators': [r['operator'] for r in rows],
                           'trained_candidates': sum(r['training_performed'] for r in rows),
                           'controller_version': term['controller_version'], 'order': term['order'],
                           'profiles': term['profiles'],
                           'term_elapsed_seconds': max(0., monotonic() - term['opened_at']),
                           'task_operation_seconds': sum(x['elapsed_seconds'] for x in operations),
                           'dev_items_scored': sum(x['evaluated_items'] for x in operations),
                           'vertical_trial_wall_seconds': sum(x.get('parent_observed_wall_seconds', 0.) for x in meta_trials),
                           'task_operation_costs': deepcopy(operations)})
        term['stage'] = 'cross_term'

    def _campaign_trial(self, targets):
        s, term = self.s, self.s['term']
        done = {x['target'] for x in s['meta'] if x['term'] == term['index'] and x['stage'] == term['stage']}
        for target in targets:
            if target in done:
                continue
            if s['snapshot_start'] != s['current']:
                return self._send('snapshot', 'math.snapshot', candidate=s['current'])
            pair = self._pair(target)
            if pair is not None:
                s['trial_pair'] = pair
                return self._build_trial(pair)
            s['meta'].append({'term': term['index'], 'stage': term['stage'], 'target': target,
                              'decision': 'kept', 'reason': 'no admissible matched challenger'})
        return None

    def _advance_campaign(self, window):
        s, term = self.s, self.s['term']
        if self.campaign['track'] == 'reference':
            return self._send('term_release', 'math.select', candidate=s['current'])
        if term['stage'] == 'cross_term':
            action = self._campaign_trial([x for x in self.targets if x in CROSS_TERM_TARGETS])
            if action is not None:
                return action
            self._open_term(term['index'] + 1)
            term = s['term']
        used = s['spent'] - term['spent_at_start']
        if self.warmup <= used < self.max_steps:
            action = self._campaign_trial([x for x in self.targets if x in VERTICAL_TARGETS])
            if action is not None:
                return action
        operator = self._pick(self._context_rows(window)) if used < self.max_steps else None
        if operator is None:
            return self._send('term_release', 'math.select', candidate=s['current'])
        s['cycle'].append(operator)
        s['spent'] += 1
        return self._send('propose', 'math.propose', parent=s['current'], operator=operator,
                          round=self.round_offset + s['spent'], profile=self.profiles.get(operator, 'compact'))

    def decide(self, window):
        s = self.s
        pending = s['pending']
        if pending:
            receipt = next((item for item in window.feedbacks if item.call_id == pending['id']), None)
            if receipt is None:
                return Decision('wait', reason='Awaiting MetaRSI operation')
            s['pending'] = None
            action = self._consume(pending, receipt)
            if action is not None:
                return action
        if s['phase'] == 'seed':
            return self._send('seed', 'math.seed')
        if s['phase'] == 'next_arm':
            pair = s['trial_pair']
            if pair['i'] == 0:
                pair['i'] = 1
                return self._build_trial(pair)
            s['phase'] = 'advance'
            action = self._complete_pair()
            if action is not None:
                return action
        if s['phase'] == 'await_adoption':
            return Decision('wait', reason='Awaiting Host Controller adoption boundary')
        if s['phase'] == 'failed':
            # Do not call select with an unvalidated candidate. Host also checks
            # that a final plan exists before any sealed measurement.
            return Decision('finish', reason='MetaRSI operation failed: ' + s['failed'])
        if s['phase'] == 'done':
            return Decision('finish', reason='MetaRSI selection complete')
        if self.campaign:
            return self._advance_campaign(window)
        while s['spent'] >= self.warmup and s['meta_index'] < len(self.targets):
            if s['snapshot'] is None:
                return self._send('snapshot', 'math.snapshot', candidate=s['current'])
            target = self.targets[s['meta_index']]
            pair = self._pair(target)
            if pair is not None:
                s['trial_pair'] = pair
                return self._build_trial(pair)
            s['meta'].append({'target': target, 'decision': 'kept', 'reason': 'no comparable alternative'})
            s['meta_index'] += 1
        operator = self._pick(self._context_rows(window)) if s['spent'] < self.max_steps else None
        if operator is None:
            return self._send('select', 'math.select', candidate=s['current'])
        s['cycle'].append(operator)
        s['spent'] += 1
        return self._send('propose', 'math.propose', parent=s['current'], operator=operator,
                          round=s['spent'], profile=self.profiles.get(operator, 'compact'))

    def report(self):
        s = self.s
        result = {'status': s['phase'], 'initial': s['initial'], 'selected': s['current'],
                  'scores': deepcopy(s['scores']), 'attempts': deepcopy(s['attempts']), 'meta': deepcopy(s['meta']),
                  'active_version': s['active_version'], 'simulated': s['simulated'], 'error': s['failed']}
        if self.campaign:
            result.update(campaign=deepcopy(self.campaign), terms=deepcopy(s['terms']),
                          diagnoses=deepcopy(s['diagnoses']), meta_trial_wall_seconds=sum(
                              t.get('parent_observed_wall_seconds', 0.) for m in s['meta'] for t in m.get('trials', [])))
        return result
