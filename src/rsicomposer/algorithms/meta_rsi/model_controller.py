"""Model-authored MetaRSI decisions; the Host owns scoring and activation."""
from copy import deepcopy
import json
from itertools import permutations

from rsicomposer.api import ExecutionStatus

from .controller import MetaRSIController, OPERATORS, _hash, validate_config
from .engine import digest

NAME = 'meta_rsi_model'


def validate_model_config(config):
    if not isinstance(config, dict):
        raise ValueError('MetaRSI model controller requires an object')
    if type(config.get('paper_kernel', False)) is not bool:
        raise ValueError('Invalid paper kernel selection')
    validate_config({k: v for k, v in config.items() if k not in ('scheduler', 'paper_kernel')})
    if (not config.get('campaign') or config['campaign'].get('track', 'adaptive') != 'adaptive'
            or config.get('meta_targets') or config.get('trial') or config.get('warmup', 0) != 0
            or config.get('steps', 4) < (1 if config.get('paper_kernel') else 2)):
        raise ValueError('Model-driven MetaRSI requires an adaptive campaign with a valid proposal budget')
    _scheduler(config)


def _scheduler(config):
    from .model_policy import compile_scheduler_request, seed_scheduler
    limit = config.get('steps', 4)
    original = config.get('paper_kernel', False)
    policy = deepcopy(config.get('scheduler', seed_scheduler(config.get('order', OPERATORS),
                                                              operator_budget=limit, original=original)))
    if original:
        compile_scheduler_request(policy, max_steps=limit, original=True, incumbent_id='configuration')
    else:
        compile_scheduler_request(policy, max_steps=limit)
    if set(policy['order']) != set(OPERATORS) or not config.get('paper_kernel') and policy['operator_budget'] < 2:
        raise ValueError('Invalid model scheduler operator set or horizon')
    return policy


class ModelMetaRSIController(MetaRSIController):
    def __init__(self, context, config):
        from .model_policy import seed_operators
        validate_model_config(config)
        super().__init__(context, {k: v for k, v in config.items() if k not in ('scheduler', 'paper_kernel')})
        self.paper_kernel = config.get('paper_kernel', False)
        self.config = deepcopy(config)
        self.scheduler = _scheduler(config)
        self.s.update(staged=None, operator_policies=seed_operators(self.profiles), policy_log=[],
                      last_released=None, declined_route=None, rewritten=[], rewrite_operator=None,
                      program=[], program_invalidations=[], program_stop=None, vertical_directive=None,
                      capability_epoch=0)

    def required_operations(self):
        return (*super().required_operations(), 'math.policy', 'reflect.build_candidate',
                *(('math.compare',) if self.paper_kernel else ()))

    def _open_term(self, index):
        super()._open_term(index)
        self.s['term'].update(order=list(self.scheduler['order']),
                              profiles={k: v['profile'] for k, v in self.s['operator_policies'].items()},
                              policy_sha256=digest(self.scheduler))
        self.s.update(staged=None, rewritten=[], rewrite_operator=None, declined_route=None, program=[],
                      program_stop=None, vertical_directive=None)

    def _evidence(self, *, operator=None):
        s, term = self.s, self.s['term']
        reviewing = term['stage'] == 'cross_term'
        count = 24 if self.scheduler['evidence'] == 'ledger' else 2
        if operator is not None:
            rows = [row for row in s['attempts'][term['attempt_start']:] if row['operator'] == operator][-count:]
        else:
            rows = s['attempts'][term['attempt_start']:] if reviewing else s['attempts'][-count:]
        actions = [{**{k: row[k] for k in ('operator', 'selected', 'data_count', 'training_performed', 'profile',
                 'gain', 'search_score', 'validation_score') if k in row}, 'candidate_created': bool(row['candidate'])}
                   for row in rows]
        evidence = {'term': term['index'], 'round': s['spent'], 'steps_remaining': max(0, self._remaining()),
                    'data_count': s['candidate_data'][s['current']]['data_count'], 'attempts': actions}
        if s['current'] in s['diagnoses']:
            evidence['diagnosis'] = deepcopy(s['diagnoses'][s['current']])
        if reviewing:
            released = s['terms'][-1]
            evidence.update({k: released[k] for k in ('start_score', 'released_score', 'gain', 'operator_calls',
                             'trained_candidates', 'dev_items_scored', 'task_operation_seconds')})
        if self.paper_kernel:
            for action, row in zip(actions, rows, strict=True):
                trans = row.get('transition') or {}
                screen = trans.get('candidate_screen', {}).get('candidates', [])
                action.update(typed_edges=trans.get('edges', []),
                              candidate_trials=len(screen),
                              training_trials=len(screen) if row['operator'] == 'model' else 0,
                              screening_items=sum(entry.get('count', entry.get('search_count', 0)) for entry in screen))
                if row['operator'] == 'data':
                    synthesis = trans.get('synthesis', {})
                    action.update(frontier=len(synthesis.get('admitted', ())),
                                  recovered=synthesis.get('verified_original_recoveries', 0),
                                  consolidation=len(synthesis.get('consolidation', ())),
                                  stage_disposition=row.get('stage_disposition'))
            if reviewing:
                evidence['training_trials'] = sum(action['training_trials'] for action in actions)
                evidence['screening_items'] = sum(action['screening_items'] for action in actions)
                evidence['vertical_rewrites'] = sum(row['phase'] == 'model_vertical' and
                    row['outcome'].get('proposal', {}).get('action') == 'rewrite_operator'
                    for row in s['policy_log'] if row['term'] == term['index'])
                evidence['resource_spent'] = s['spent'] - term['spent_at_start']
                release = released['term_release']
                evidence['term_release'] = release['decision']
                evidence['working_end_score'] = s['scores'][release['working_end']]
        return evidence

    def _schedule_request(self):
        from .model_policy import compile_scheduler_request
        options = {'order': tuple(permutations(OPERATORS)), 'diagnostics': None, 'evidence': ('recent', 'ledger'),
                   'operator_budget': tuple(range(1 if self.paper_kernel else 2, self.max_steps + 1)),
                   'prefer_axis': ('horizontal', 'vertical', 'balanced'),
                   'stop': ('never', 'on_solved'), 'routing': ('ordered', 'diagnostic', 'uniform')}
        if self.paper_kernel:
            options['transition_guidance'] = None
        return compile_scheduler_request(self.scheduler, max_steps=self.max_steps,
            options={k: v for k, v in options.items() if k in self.scheduler}, evidence=self._evidence(),
            **({'original': True, 'incumbent_id': self.s['last_released']} if self.paper_kernel else {}))

    def _remaining(self):
        s = self.s
        return self.scheduler['operator_budget'] - (s['spent'] - s['term']['spent_at_start'])

    def _request_policy(self, phase, request):
        return self._send(phase, 'math.policy', candidate=self.s['current'], round=self.s['spent'], request=request.wire())

    def _close_term(self):
        s = self.s
        if s['staged'] is not None:
            s['attempts'][s['staged']['attempt']]['stage_disposition'] = 'not_consumed_before_release'
        s['staged'] = None
        s['program'] = []
        s['program_stop'] = None
        return self._send('term_release', 'math.select', candidate=s['current'],
            **({'term_start': s['term']['start'], 'term': s['term']['index']} if self.paper_kernel else {}))

    def _advance_campaign(self, window):
        from .model_policy import compile_operator_request, compile_route_request, program_step_invalidator
        s, term = self.s, self.s['term']
        if term['stage'] == 'cross_term':
            if not term.get('meta_reviewed'):
                return self._request_policy('model_meta', self._schedule_request())
            if term['index'] == self.campaign['terms']:
                s['phase'] = 'done'
                return self.decide(window)
            self._open_term(term['index'] + 1)
        if (self._remaining() <= 0 or s['declined_route'] is not None or
                (self.paper_kernel and not s['program'] and s['program_stop'] == 'stop') or
                self.scheduler.get('stop') == 'on_solved' and s['scores'][s['current']] == 1.):
            return self._close_term()
        if s['rewrite_operator'] is not None:
            op = s['rewrite_operator']
            return self._request_policy('model_vertical', compile_operator_request(
                op, s['operator_policies'][op], evidence=self._evidence(operator=op if self.paper_kernel else None),
                **({'original': True, 'directive': s['vertical_directive'], 'incumbent_id': s['current']}
                   if self.paper_kernel else {})))
        parent = s['staged']['candidate'] if s['staged'] else s['current']
        permitted = s['candidate_data'][parent]['available']
        legal = [op for op in self.scheduler['order'] if op in permitted and
                 (op != 'model' or s['staged'] is not None) and
                 (self.paper_kernel or op != 'data' or self._remaining() >= 2)]
        if s['program']:
            step = s['program'][0]
            reason = program_step_invalidator(step, legal,
                **({'resource_budget': self._remaining()} if self.paper_kernel else {}))
            if reason is None:
                s['program'].pop(0)
                return self._dispatch(step['operator'], step=step if self.paper_kernel else None)
            s['program_invalidations'].append({'term': term['index'], 'round': s['spent'],
                'operator': step['operator'], 'reason': reason, 'cleared_steps': len(s['program'])})
            s['program'] = []
            s['program_stop'] = None
        if not legal:
            return self._close_term()
        rewrite = [op for op in self.scheduler['order'] if self.paper_kernel or op not in s['rewritten']]
        return self._request_policy('model_route', compile_route_request(legal, rewrite_operators=rewrite,
                                    allow_stop=True, evidence=self._evidence(),
                                    operator_policies=s['operator_policies'], scheduler=self.scheduler,
                                    program_operators=OPERATORS, max_program_steps=self._remaining(),
                                    **({'original': True, 'resource_budget': self._remaining()}
                                       if self.paper_kernel else {})))

    def _dispatch(self, operator, *, step=None):
        s = self.s
        parent = s['staged']['candidate'] if s['staged'] else s['current']
        s['spent'] += step['step_budget'] if step is not None else 1
        s['cycle'].append(operator)
        policy = s['operator_policies'][operator]
        return self._send('propose', 'math.propose', parent=parent, operator=operator, round=s['spent'],
                          profile=policy['profile'], strategy=policy['instruction'],
                          **({'program_step': step, 'capability_epoch': s['capability_epoch'],
                              'transition_guidance': self.scheduler['transition_guidance']}
                             if self.paper_kernel else {}))

    def _model_decision(self, pending, receipt):
        from .model_policy import PolicyRequest, run_policy
        phase, args, s = pending['phase'], pending['args'], self.s
        outcome = {'accepted': False, 'reason': 'operation_failed'}
        audit = {'term': s['term']['index'], 'phase': phase, 'candidate': args['candidate']}
        if receipt.status == ExecutionStatus.COMPLETED:
            report = receipt.data
            if report.get('candidate_id') != args['candidate'] or report.get('request_sha256') != digest(args):
                raise ValueError('MetaRSI actor receipt did not match the Host-issued request')
            outcome = report['outcome']
            audit.update(request_sha256=report['request_sha256'], response_sha256=report['response_sha256'])
            if outcome.get('accepted') is True:
                check = run_policy(PolicyRequest.from_wire(args['request']), json.dumps(outcome['proposal']))
                if check != outcome:
                    raise ValueError('MetaRSI receipt did not survive strict policy revalidation')
        audit['outcome'] = deepcopy(outcome)
        s['policy_log'].append(audit)
        proposal = outcome.get('proposal') if outcome.get('accepted') is True else None
        s['phase'] = 'advance'
        if phase == 'model_route':
            if proposal is None or proposal['action'] == 'stop':
                s['declined_route'] = outcome.get('reason', 'model_requested_stop')
                return None
            if proposal['action'] == 'program':
                s['program'] = deepcopy(proposal['steps'])
                s['program_stop'] = proposal.get('stop_intent')
                return None  # Recheck exact current candidate at the dispatch boundary, even for the first step.
            op = proposal['operator']
            if proposal['action'] == 'update_operator':
                s['rewrite_operator'] = op
                s['vertical_directive'] = deepcopy(proposal) if self.paper_kernel else None
                return None
            return self._dispatch(op)
        if phase == 'model_vertical':
            op = s['rewrite_operator']
            s['rewrite_operator'] = None
            s['vertical_directive'] = None
            s['rewritten'].append(op)
            if proposal is not None and proposal['action'] != 'keep':
                s['operator_policies'][op].update(proposal['patch'])
            return None
        term = s['term']
        entry = {'term': term['index'], 'stage': 'cross_term', 'target': 'schedule', 'decision': 'kept'}
        s['meta'].append(entry)
        term['meta_reviewed'] = True
        if proposal is None or proposal['action'] == 'keep':
            entry['reason'] = outcome.get('reason', 'model_kept_schedule')
            return None
        policy = {**self.scheduler, **proposal['patch']}
        config = {**self.config, 'scheduler': policy}
        validate_model_config(config)
        ident = _hash('meta-model-adopt-', [term['index'], policy, audit['response_sha256']])
        entry.update(decision='adoption_requested', version_id=ident, policy_diff=deepcopy(proposal['diff']))
        handoff = self.state()
        handoff['active_version'] = ident
        handoff['meta'][-1]['decision'] = 'adopted'
        return self._send('build_adoption', 'reflect.build_candidate', version={
            'version_id': ident, 'config': {'name': NAME, 'config': config}, 'policy_state': handoff})

    @staticmethod
    def _record_kernel_selection(row, *, accepted, decision):
        from .kernel_evidence import complete_kernel_evidence
        if 'kernel_execution' not in row.get('transition', {}):
            return
        execution = row['transition']['kernel_execution']
        state = execution['stages'][4 if accepted else 0]['state']
        row['kernel_evidence'] = complete_kernel_evidence(execution, selected=state, selection=decision)

    def _consume(self, pending, receipt):
        phase, args, s = pending['phase'], pending['args'], self.s
        if phase in ('model_route', 'model_vertical', 'model_meta'):
            return self._model_decision(pending, receipt)
        if phase == 'propose' and receipt.status != ExecutionStatus.COMPLETED:
            s['attempts'].append({'operator': args['operator'], 'profile': args['profile'], 'parent': args['parent'],
                'candidate': None, 'data_count': 0, 'training_performed': False, 'selected': False,
                'term': s['term']['index'], 'round': args['round'], 'reason': 'operator_failed'})
            s['declined_route'] = 'operator_failed'
            s['phase'] = 'advance'
            return None
        if phase == 'propose' and args['operator'] == 'data':
            data = receipt.data
            ident = data['candidate_id']
            transition = deepcopy(data['transition'])
            row = {'operator': 'data', 'profile': args['profile'], 'parent': args['parent'], 'candidate': ident,
                   'data_count': data['data_count'], 'training_performed': False, 'selected': False,
                   'term': s['term']['index'], 'round': args['round'], 'data_fresh': data['data_fresh'],
                   'reason': data.get('reason'), 'transition': transition}
            s['attempts'].append(row)
            if self.paper_kernel:
                self._record_kernel_selection(row, accepted=ident is not None,
                    decision={'scope': 'verified_data_export_not_deployed', 'staged': ident is not None})
            if ident is not None:
                proof = transition.get('refresh_proof', {})
                if (not row['data_fresh'] or proof.get('candidate_id') != ident or
                        proof.get('source_candidate_id') != args['parent'] or
                        proof.get('previous_count') != s['candidate_data'][args['parent']]['data_count'] or
                        row['data_count'] <= proof['previous_count']):
                    raise ValueError('Cannot stage a Data proposal without an exact verified refresh proof')
                self._record_candidate(data)
                if s['staged']:
                    s['attempts'][s['staged']['attempt']]['stage_disposition'] = 'continued_with_data'
                row['stage_disposition'] = 'staged'
                s['staged'] = {'candidate': ident, 'origin': s['current'], 'attempt': len(s['attempts']) - 1,
                               'proof_sha256': digest(proof)}
            s['phase'] = 'advance'
            return None
        if phase == 'propose' and s['staged'] is not None:
            if s['staged']['origin'] != s['current'] or args['parent'] != s['staged']['candidate']:
                raise ValueError('Staged handoff changed the scored incumbent or exact source')
            if args['operator'] == 'model' and receipt.data['candidate_id'] is not None and digest(
                    receipt.data['transition']['refresh_proof']) != s['staged']['proof_sha256']:
                raise ValueError('Model did not train from the staged dataset proof')
            s['attempts'][s['staged']['attempt']]['stage_disposition'] = (
                'consumed_by_' + args['operator'] if receipt.data['candidate_id'] else 'consumer_produced_no_candidate')
            s['staged'] = None
        if self.paper_kernel and phase == 'validation' and receipt.status == ExecutionStatus.COMPLETED:
            row = s['attempts'][-1]
            ident = row['candidate']
            s['scores'].setdefault(ident, self._point(receipt, ident))
            return self._send('paired_selection', 'math.compare', incumbent=s['current'], candidate=ident)
        if self.paper_kernel and phase == 'paired_selection' and receipt.status == ExecutionStatus.COMPLETED:
            from .aime_release import PROTOCOL
            row, evidence = s['attempts'][-1], receipt.data
            key = row['candidate']
            score, incumbent = s['scores'][key], s['scores'][s['current']]
            if (evidence['candidate'] != key or evidence['incumbent'] != s['current']
                    or evidence['protocol'] != PROTOCOL or abs(evidence['delta'] - (score - incumbent)) > 1e-12):
                raise ValueError('Paired release receipt did not match the measured candidate and incumbent')
            row.update(validation_score=score, gain=evidence['delta'], paired_selection=deepcopy(evidence),
                       selected=evidence['eligible'] is True)
            self._record_kernel_selection(row, accepted=row['selected'], decision=evidence)
            if row['selected']:
                s['capability_epoch'] += 1
                s.update(current=key, available=s['candidate_data'][key]['available'])
                return self._send('diagnose', 'math.diagnose', candidate=key, split='validation')
            s['phase'] = 'advance'
            return None
        if phase == 'term_release' and receipt.status == ExecutionStatus.COMPLETED:
            if self.paper_kernel:
                verdict = receipt.data['term_release']
                released = receipt.data['candidate_id']
                if (verdict['start'] != s['term']['start'] or verdict['working_end'] != s['current'] or
                        verdict['term'] != s['term']['index'] or released not in (verdict['start'], verdict['working_end'])):
                    raise ValueError('Host release must match the executed term, its origin and its working end')
                if released != s['current']:
                    s['capability_epoch'] += 1
                    s.update(current=released, available=s['candidate_data'][released]['available'])
            elif receipt.data['candidate_id'] != s['current']:
                raise ValueError('Host release must match the measured incumbent')
            self._record_term()
            if self.paper_kernel:
                s['terms'][-1].update(term_release=deepcopy(verdict), operator_budget=self.scheduler['operator_budget'],
                                      resource_spent=s['spent'] - s['term']['spent_at_start'])
            s['terms'][-1]['policy_sha256'] = s['term']['policy_sha256']
            s['last_released'] = s['current']
            s['phase'] = 'advance'
            return None
        action = super()._consume(pending, receipt)
        if (self.paper_kernel and phase == 'propose' and receipt.status == ExecutionStatus.COMPLETED and
                receipt.data['candidate_id'] is None):
            self._record_kernel_selection(s['attempts'][-1], accepted=False,
                                          decision={'scope': 'no_candidate'})
        return action

    def report(self):
        return {**super().report(), 'last_released': self.s['last_released'],
                'policy_decisions': deepcopy(self.s['policy_log']), 'scheduler': deepcopy(self.scheduler),
                'program_invalidations': deepcopy(self.s['program_invalidations']),
                'operator_policies': deepcopy(self.s['operator_policies'])}
