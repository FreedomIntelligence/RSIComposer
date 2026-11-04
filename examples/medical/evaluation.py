"""Fixed development evaluator, raw generation evidence, and separate acceptance."""
import math
from copy import deepcopy
from dataclasses import asdict
from uuid import uuid4

from rsisys.loops.contracts.jobs import JobOutcome, JobRef, JobStatus
from rsisys.loops.contracts.model_service import GenerationRequest, GenerationResponse
from rsisys.loops.contracts.training import ModelActivationPending
from rsisys.loops.exploration.harness.evaluation import EvaluationResult
from .task import LABELS, encoded


def backend_inference(backend):
    """Adapt an explicitly injected ModelBackend; no provider discovery/imports."""
    return backend.generate


class MedicalEvaluationError(RuntimeError):
    """Terminal evaluation failure; inspect the attached raw partial evidence."""


def pareto_acceptance(task, candidate, incumbent):
    """Strict improvement on at least one accuracy, no drop on any accuracy."""
    deltas = {}
    for benchmark in task.development:
        key = benchmark.benchmark_id + '.accuracy'
        before, after = incumbent.metrics[key], candidate.metrics[key]
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
               for v in (before, after)):
            raise ValueError('Acceptance requires finite accuracies in [0, 1]')
        deltas[key] = after - before
    improved = [key for key, delta in deltas.items() if delta > 0]
    regressed = [key for key, delta in deltas.items() if delta < 0]
    return {'selection_rule': 'pareto', 'metric_deltas': deltas,
            'improved_metrics': improved, 'regressed_metrics': regressed,
            'accepted': bool(improved) and not regressed,
            'comparison': ('tradeoff' if improved and regressed else
                           'improvement' if improved else 'regression' if regressed else 'tie')}


class MedicalEvaluator:
    def __init__(self, task, artifacts, systems, candidates, inference, *, prepare_inference=None, acceptance_policy=None):
        if not callable(inference):
            raise TypeError('An explicit inference callable is required')
        if prepare_inference is not None and not callable(prepare_inference):
            raise TypeError('prepare_inference must be callable or None')
        if acceptance_policy is not None and not callable(acceptance_policy):
            raise TypeError('acceptance_policy must be callable or None')
        self.acceptance_policy = acceptance_policy
        self._task, self.artifacts, self.systems = task, artifacts, systems
        self.candidates, self.inference = candidates, inference
        self.prepare_inference = prepare_inference
        self._jobs, self._keys, self._outputs = {}, {}, {}
        self._failures, self._partial, self._interpretation_failures = {}, {}, {}
        self._states = {}
        self._measurement_systems = {}
        self._anchor = None
        self.evidence = {}
        self.task.check()

    @property
    def task(self):
        return self._task

    def measure(self, system):
        """Explicit synchronous baseline/local measurement; pending propagates."""
        while True:
            result = self.advance_measurement(system)
            if result is not None:
                return result

    def advance_measurement(self, system):
        """Prepare serving and infer at most one sample, retaining partial state."""
        previous = self._measurement_systems.get(system.version)
        if previous is not None and previous != system:
            # Reject a conflicting caller without poisoning a valid cached result
            # or attributing another manifest's partial evidence to this system.
            raise ValueError('Measurement system identity changed')
        self._measurement_systems[system.version] = deepcopy(system)
        if system.version in self._failures:
            raise MedicalEvaluationError(self._failures[system.version])
        try:
            return self._measure(system)
        except ModelActivationPending:
            raise  # A serving wait is not a failed measurement or a new intent.
        except Exception as exc:
            ref = self.artifacts.put(encoded({'kind': 'raw_development_failure',
                'protocol_ref': self.task.protocol_ref, 'system': asdict(system),
                'completed_predictions': self._partial.get(system.version, []),
                'error_type': type(exc).__name__, 'complete': False}), kind='medical_raw_evidence')
            self.evidence[ref.artifact_id] = ref
            message = f'Medical evaluation failed ({type(exc).__name__}); partial evidence: {ref.artifact_id}'
            self._failures[system.version] = message
            raise MedicalEvaluationError(message) from exc

    def _measure(self, system):
        if system.version in self._outputs:
            return deepcopy(self._outputs[system.version])
        state = self._states.get(system.version)
        if state is None:
            rows = self.task.check()
            if system.model is None:
                raise ValueError('Medical inference requires an explicit candidate model')
            state = {'system': deepcopy(system), 'rows': rows, 'predictions': [],
                     'instructions': self.artifacts.get(system.components['prompt']).decode('utf-8'),
                     'simulated': False, 'ready': False}
            self._states[system.version] = state
            self._partial[system.version] = state['predictions']
        if state['system'] != system:
            raise ValueError('Measurement system identity changed')
        if not state['ready']:
            if self.prepare_inference is not None:
                self.prepare_inference(deepcopy(system))
            state['ready'] = True
        rows, predictions = state['rows'], state['predictions']
        instructions, simulated = state['instructions'], state['simulated']
        if len(predictions) < len(rows):
            row = rows[len(predictions)]
            legal = LABELS[row.benchmark_id]
            request = GenerationRequest('actor', system.model, row.prompt,
                system=instructions + '\nReturn exactly one label: ' + ', '.join(legal) + '. No explanation.',
                max_output_tokens=self.task.max_output_tokens)
            response = self.inference(request)
            if not isinstance(response, GenerationResponse) or response.model != system.model:
                raise ValueError('Inference must return GenerationResponse with the requested model identity')
            if not isinstance(response.text, str) or type(response.simulated) is not bool:
                raise ValueError('Invalid inference response text or simulation marker')
            simulated |= response.simulated
            answer = response.text.strip()
            predictions.append({'id': row.id, 'benchmark_id': row.benchmark_id,
                'gold': row.gold, 'output': response.text, 'answer': answer,
                'legal': answer in legal, 'correct': answer in legal and answer == row.gold,
                'provider_model': response.provider_model, 'revision_verified': response.revision_verified,
                'usage': asdict(response.usage) if response.usage else None})
        state['simulated'] = simulated
        if len(predictions) < len(rows):
            return None
        metrics = {}
        for benchmark in self.task.development:
            subset = [r for r in predictions if r['benchmark_id'] == benchmark.benchmark_id]
            accuracy = sum(r['correct'] for r in subset) / len(subset)
            metrics[benchmark.benchmark_id + '.accuracy'] = accuracy
            metrics[benchmark.benchmark_id + '.invalid_format'] = float(sum(not r['legal'] for r in subset))
            f1s = []
            for label in LABELS[benchmark.benchmark_id]:
                tp = sum(r['gold'] == label and r['answer'] == label for r in subset)
                denominator = sum(r['gold'] == label for r in subset) + sum(r['answer'] == label for r in subset)
                f1s.append(2 * tp / denominator if denominator else 0.0)
            metrics[benchmark.benchmark_id + '.macro_f1'] = sum(f1s) / len(f1s)
            if benchmark.role == 'primary':
                metrics['score'] = accuracy
        raw = self.artifacts.put(encoded({'kind': 'raw_development_evaluation',
            'protocol_ref': self.task.protocol_ref, 'system': asdict(system),
            'predictions': predictions, 'metrics': metrics, 'simulated': simulated}), kind='medical_raw_evidence')
        self.evidence[raw.artifact_id] = raw
        result = EvaluationResult(system.version, self.task.protocol_ref, metrics, (raw.artifact_id,), simulated)
        self._outputs[system.version] = result
        return deepcopy(result)

    def establish_anchor(self):
        """Explicit baseline inference; never called by prepare/check/build."""
        if self._anchor is None:
            measured = self.measure(self.systems.current)
            self._anchor = deepcopy(measured)
        return deepcopy(self._anchor)

    def interpret(self, candidate_id):
        if candidate_id in self._interpretation_failures:
            raise MedicalEvaluationError(self._interpretation_failures[candidate_id])
        self.task.check()
        if self._anchor is None:
            raise RuntimeError('Measure the fixed initial anchor before candidate evaluation')
        result = self._outputs[candidate_id]
        incumbent = self._outputs[self.systems.current.version]
        primary = result.metrics['score'] > incumbent.metrics['score']
        guardrail = all(result.metrics[b.benchmark_id + '.accuracy'] >=
            self._anchor.metrics[b.benchmark_id + '.accuracy'] - self.task.max_guardrail_drop
            for b in self.task.development if b.role == 'guardrail')
        if result.simulated != incumbent.simulated or result.simulated != self._anchor.simulated:
            raise ValueError('Cannot compare simulated and provider measurements')
        decision = {'selection_rule': 'primary_guardrail',
                    'accepted': primary and (guardrail or not self.task.guardrails)}
        policy = self.acceptance_policy or (pareto_acceptance if self.task.selection_rule == 'pareto' else None)
        if policy is not None:
            decision = policy(self.task, deepcopy(result), deepcopy(incumbent))
            if not isinstance(decision, dict) or type(decision.get('accepted')) is not bool:
                raise ValueError('Acceptance policy must return a boolean accepted verdict')
        return {**decision, 'kind': 'acceptance_interpretation', 'candidate_id': candidate_id,
            'incumbent': incumbent.candidate_id, 'fixed_anchor': self._anchor.candidate_id,
            'protocol_ref': self.task.protocol_ref, 'primary_improved': primary,
            'guardrail_passed': guardrail, 'guardrails_enabled': self.task.guardrails,
            'source_refs': list(result.evidence_refs + incumbent.evidence_refs + self._anchor.evidence_refs)}

    def submit(self, request, *, idempotency_key):
        if not idempotency_key:
            raise ValueError('An evaluation idempotency key is required')
        if idempotency_key in self._keys:
            job = self._keys[idempotency_key]
            if self._jobs[job][0] != request:
                raise ValueError('Evaluation key reused with different content')
            return job
        if request.protocol_ref != self.task.protocol_ref or request.data_refs:
            raise ValueError('Only the frozen development protocol and inputs are permitted')
        if self.candidates.get(request.candidate.candidate_id) != request.candidate:
            raise ValueError('Evaluation requires a registered candidate with matching components')
        if self._anchor is None:
            raise RuntimeError('Initial anchor has not been measured')
        job = JobRef(uuid4().hex, 'medical-development')
        self._jobs[job] = (deepcopy(request), None)
        self._keys[idempotency_key] = job
        return job

    def poll(self, job):
        request, outcome = self._jobs[job]
        if outcome is None:
            try:
                result = self.advance_measurement(self.systems.get(request.candidate.candidate_id))
                if result is None:
                    return JobOutcome(job, JobStatus.RUNNING)
            except ModelActivationPending:
                return JobOutcome(job, JobStatus.RUNNING)
            except MedicalEvaluationError as exc:
                outcome = JobOutcome(job, JobStatus.FAILED, error=str(exc))
                self._jobs[job] = (request, outcome)
                return deepcopy(outcome)
            # A complete measurement is not a successful acceptance interpretation.
            # Fail closed while preserving its raw evidence, without re-querying it.
            try:
                interpretation = self.artifacts.put(encoded(self.interpret(result.candidate_id)),
                    kind='medical_interpretation')
            except Exception as exc:
                failure = self.artifacts.put(encoded({'kind': 'acceptance_interpretation_failure',
                    'candidate_id': result.candidate_id, 'protocol_ref': result.protocol_ref,
                    'source_refs': list(result.evidence_refs), 'error_type': type(exc).__name__,
                    'measurement_complete': True, 'acceptance_complete': False}),
                    kind='medical_interpretation')
                self.evidence[failure.artifact_id] = failure
                message = (f'Medical interpretation failed ({type(exc).__name__}); '
                           f'failure evidence: {failure.artifact_id}')
                self._interpretation_failures[result.candidate_id] = message
                outcome = JobOutcome(job, JobStatus.FAILED, error=message, simulated=result.simulated)
            else:
                self.evidence[interpretation.artifact_id] = interpretation
                result = EvaluationResult(result.candidate_id, result.protocol_ref, result.metrics,
                                          result.evidence_refs + (interpretation.artifact_id,), result.simulated)
                outcome = JobOutcome(job, JobStatus.SUCCEEDED, result, simulated=result.simulated)
            self._jobs[job] = (request, outcome)
        return deepcopy(outcome)

    def measurement(self, candidate_id):
        return deepcopy(self._outputs.get(candidate_id))

    def completed_evaluation(self, candidate_id):
        return any(request.candidate.candidate_id == candidate_id and outcome is not None
                   and outcome.status == JobStatus.SUCCEEDED
                   for request, outcome in self._jobs.values())

    def result(self, job):
        outcome = self._jobs[job][1]
        if outcome is None or outcome.status != JobStatus.SUCCEEDED:
            raise RuntimeError('Evaluation has no completed result')
        return deepcopy(outcome.result)

    def cancel(self, job):
        request, outcome = self._jobs[job]
        if outcome is None:
            predictions = self._partial.get(request.candidate.candidate_id, [])
            if predictions:
                ref = self.artifacts.put(encoded({'kind': 'raw_development_cancelled',
                    'candidate_id': request.candidate.candidate_id, 'protocol_ref': request.protocol_ref,
                    'completed_predictions': predictions, 'complete': False}), kind='medical_raw_evidence')
                self.evidence[ref.artifact_id] = ref
            self._jobs[job] = (request, JobOutcome(job, JobStatus.CANCELLED))
