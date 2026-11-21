"""Controller candidates validated by independent, budgeted child Applications.

This Lab produces versions and evidence; adoption is an explicit Controller
operation through host.adopt_controller and never a side effect of validation.
"""
from dataclasses import asdict, replace
from hashlib import sha256
import json

from rsicomposer.loops.contracts.artifacts import ArtifactRef
from rsicomposer.loops.contracts.controller_version import ControllerVersion, EvidenceScope
from rsicomposer.loops.contracts.coordination import OperationResult
from rsicomposer.loops.contracts.decision import OperationSpec
from rsicomposer.loops.contracts.execution import LabToolError
from rsicomposer.loops.contracts.jobs import JobOutcome, JobRef, JobStatus
from rsicomposer.loops.contracts.lifecycle import ExecutionStatus
from rsicomposer.loops.contracts.model_service import ModelRef
from rsicomposer.runtime.execution.resources import ResourceRequest
from rsicomposer.runtime.state.controller_versions import load_controller_version, save_controller_version
from rsicomposer.runtime.storage.files import atomic_write, json_bytes


VERSION = '2'
_FIELDS = {
    'inspect': ({'scope': {'type': 'object'}}, ()),
    'build_candidate': ({'version': {'type': 'object'}}, ('version',)),
    'train_decision_model': ({'recipe_ref': {'type': 'string', 'minLength': 1},
        'experience_refs': {'type': 'array', 'items': {'type': 'string'}},
        'data_refs': {'type': 'array', 'items': {'type': 'object'}},
        'method': {'type': 'string', 'minLength': 1}}, ('recipe_ref',)),
    'start': ({'candidate_ref': {'type': 'object'}, 'objective': {'type': 'string', 'minLength': 1},
               'budget': {'type': 'object'}, 'labs': {'type': ['object', 'null']}},
              ('candidate_ref', 'objective', 'budget')),
    **{name: ({'job': {'type': 'object'}}, ('job',)) for name in ('poll', 'result', 'cancel', 'reconcile')},
}


def validate_config(config):
    if not isinstance(config, dict) or config:
        raise ValueError('Self-Reflection Lab currently accepts an empty configuration object')


class SelfReflectionLab:
    def __init__(self, context, config):
        validate_config(config)
        self.context = context
        self.nested = context.nested_run()
        self._candidates, self._jobs, self._training_jobs = {}, {}, set()
        self._closed = False
        self._journal = self.nested.root / 'candidate_jobs'
        self._journal.mkdir(exist_ok=True)
        for path in self._journal.glob('*.json'):
            ref = ArtifactRef(**json.loads(path.read_text()))
            self._jobs[path.stem] = asdict(ref)

    def resource_roles(self):
        return {'nested': ResourceRequest(gpus=0)}

    def describe(self):
        return tuple(OperationSpec(name, {
            'inspect': 'Read scoped parent evidence with sealed data and credentials excluded',
            'build_candidate': 'Store an immutable registered ControllerVersion without adopting it',
            'train_decision_model': 'Train a decision model and publish a candidate without adopting it',
            'start': 'Validate a candidate in an independent child run with a reserved parent budget',
            'poll': 'Observe candidate validation', 'result': 'Read terminal validation evidence',
            'cancel': 'Cancel validation and settle its actual budget usage',
            'reconcile': 'Find an uncertain nested submission without launching it again',
        }[name], {'type': 'object', 'properties': properties, 'required': list(required),
                 'additionalProperties': False}, read_only=name in {'poll', 'result', 'reconcile'},
            idempotent=name != 'inspect') for name, (properties, required) in _FIELDS.items())

    def execute(self, call):
        fields = _FIELDS.get(call.name)
        if (fields is None or not isinstance(call.arguments, dict)
                or set(call.arguments) - set(fields[0]) or set(fields[1]) - set(call.arguments)):
            raise LabToolError('Unknown self-reflection operation or invalid arguments')
        args = call.arguments
        if call.name == 'inspect':
            scope = EvidenceScope(**args.get('scope', {}))
            view = self.context.evidence_view(scope)
            ref = self.context.artifacts.put(json_bytes({'events': [asdict(event) for event in view.events()]}),
                                             kind='self-reflection-evidence')
            return OperationResult('Scoped parent evidence', (ref,), False, data={'evidence_ref': asdict(ref)})
        if call.name == 'build_candidate':
            raw = dict(args['version'])
            if raw.get('model') is not None:
                raw['model'] = ModelRef(**raw['model'])
            version = ControllerVersion(**raw)
            # Validate the registered selection before publishing a candidate.
            self.context.nested_manifest(version, None)
            ref = save_controller_version(self.context.artifacts, version)
            self._candidates[ref.artifact_id] = asdict(ref)
            return OperationResult('Controller candidate registered', (ref,), False,
                data={'candidate_ref': asdict(ref), 'version_id': version.version_id})
        if call.name == 'train_decision_model':
            if self._closed:
                raise RuntimeError('Self-Reflection Lab is closing')
            visible = {row.record_id for row in self.context.evidence_view().records()}
            if any(ref not in visible for ref in args.get('experience_refs', ())):
                raise ValueError('Training references must identify visible experience records')
            data_refs = tuple(ArtifactRef(**ref) for ref in args.get('data_refs', ()))
            for ref in data_refs:
                self.context.artifacts.read(ref)
            job = self.context.controller_learning().train(recipe_ref=args['recipe_ref'],
                experience_refs=args.get('experience_refs', ()), data_refs=data_refs,
                method=args.get('method', 'sft'), idempotency_key=call.call_id)
            self._training_jobs.add(job)
            return job
        if call.name == 'start':
            ref = ArtifactRef(**args['candidate_ref'])
            version = load_controller_version(self.context.artifacts, ref)
            ident = sha256(call.call_id.encode()).hexdigest()[:32]
            if ident in self._jobs and self._jobs[ident] != asdict(ref):
                raise ValueError('Nested validation call was reused for another candidate')
            atomic_write(self._journal / (ident + '.json'), json_bytes(asdict(ref)))
            self._jobs[ident] = asdict(ref)
            job = self.nested.start(version, args.get('labs'), objective=args['objective'],
                                    budget=args['budget'], call_id=call.call_id)
            return job
        job = JobRef(**args['job'])
        if call.name == 'reconcile':
            if job in self._training_jobs:
                raise LabToolError('Only nested submissions require reconcile; use poll for training jobs')
            outcome = self._outcome(self.nested.reconcile(job))
        else:
            outcome = (self.job_result(job) if call.name == 'result' else
                       self.cancel_job(job) if call.name == 'cancel' else self.poll_job(job))
        if outcome.terminal:
            if outcome.result is not None:
                return outcome.result
            return OperationResult('Self-reflection job ' + outcome.status.value, (), outcome.simulated,
                status=ExecutionStatus.CANCELLED if outcome.status == JobStatus.CANCELLED else ExecutionStatus.FAILED,
                error=outcome.error, data={'job': asdict(job), 'status': outcome.status.value})
        return OperationResult('Self-reflection job is running', (), outcome.simulated,
                               data={'job': asdict(job), 'status': outcome.status.value})

    def _outcome(self, outcome):
        if outcome.status != JobStatus.SUCCEEDED:
            return outcome
        result = dict(outcome.result)
        result['candidate_ref'] = self._jobs[outcome.job.job_id]
        evidence = ArtifactRef(**result['evidence_ref'])
        candidate = ArtifactRef(**result['candidate_ref'])
        receipt = OperationResult('Controller candidate validation completed', (candidate, evidence),
                                  False, data=result)
        return JobOutcome(outcome.job, outcome.status, result=receipt)

    def _training_outcome(self, outcome):
        if outcome.status != JobStatus.SUCCEEDED:
            return outcome
        ref = self.context.controller_learning().candidate(outcome.job)
        self._candidates[ref.artifact_id] = asdict(ref)
        result = outcome.result
        receipt = OperationResult('Decision-model candidate trained', (ref,),
            outcome.simulated or result.simulated, data={'candidate_ref': asdict(ref),
                'model': asdict(result.model), 'training_metrics': result.metrics})
        return replace(outcome, result=receipt)

    def poll_job(self, job):
        if job in self._training_jobs:
            return self._training_outcome(self.context.controller_learning().poll(job))
        return self._outcome(self.nested.poll(job))

    def job_result(self, job):
        if job in self._training_jobs:
            outcome = self.poll_job(job)
            if not outcome.terminal:
                raise RuntimeError('Decision-model training has not finished')
            return outcome
        return self._outcome(self.nested.job_result(job))

    def cancel_job(self, job):
        if job in self._training_jobs:
            return self._training_outcome(self.context.controller_learning().cancel(job))
        return self._outcome(self.nested.cancel(job))

    def reattach_job(self, job):
        if job in self._training_jobs:
            self.context.controller_learning().reattach(job)
            return
        if job.job_id not in self._jobs:
            raise ValueError('Unknown candidate validation reference')
        saved = self.nested._job(job)
        if saved['phase'] != 'terminal':
            self.nested._handle(saved)

    def shutdown(self, mode='cancel'):
        if mode not in {'drain', 'cancel'}:
            raise ValueError('Self-Reflection shutdown mode must be drain or cancel')
        self._closed = True
        complete = self.nested.shutdown(mode)
        for job in self._training_jobs:
            learning = self.context.controller_learning()
            outcome = learning.cancel(job) if mode == 'cancel' else learning.poll(job)
            complete = outcome.terminal and complete
        return complete

    def close(self):
        if not self.shutdown('cancel'):
            raise RuntimeError('Self-Reflection training cancellation is not yet confirmed')

    def state(self):
        return {'schema': 2, 'candidates': dict(self._candidates), 'jobs': dict(self._jobs),
                'training_jobs': [asdict(job) for job in sorted(self._training_jobs, key=lambda job: job.job_id)],
                'closed': self._closed, 'nested': self.nested.state()}

    def restore(self, state):
        if (not isinstance(state, dict) or set(state) != {'schema', 'candidates', 'jobs', 'nested', 'training_jobs', 'closed'}
                or type(state['schema']) is not int or state['schema'] != 2 or not isinstance(state['candidates'], dict)
                or not isinstance(state['jobs'], dict) or not isinstance(state['training_jobs'], list)
                or type(state['closed']) is not bool):
            raise ValueError('Unsupported Self-Reflection Lab state')
        self._training_jobs = {JobRef(**raw) for raw in state['training_jobs']}
        if len(self._training_jobs) != len(state['training_jobs']):
            raise ValueError('Duplicate Self-Reflection training jobs')
        self._closed = state['closed']
        self._candidates = dict(state['candidates'])
        for ident, ref in state['jobs'].items():
            if ident in self._jobs and self._jobs[ident] != ref:
                raise ValueError('Candidate validation journal conflicts with restored state')
            self._jobs[ident] = ref
        self.nested.restore(state['nested'])

    def validate_snapshot(self):
        self.nested.validate_snapshot()
        if self._training_jobs:
            self.context.controller_learning().validate_snapshot()
