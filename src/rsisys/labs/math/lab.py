"""Math Lab: frozen question-answer splits, answer verification and scored evidence.

The Lab owns its task data (tasks.py, benchmarks.py), answer verification
(verifiers.py) and the actor call; the Controller supplies the instruction being
measured. Any numeric or symbolic question-answer benchmark connects by
configuration: each split in the dataset manifest selects its verifier, and
fields renames row keys. Operations: inspect, evaluate (split, instruction,
optional development task_ids) and read_trace.

Methods on math tasks subclass MathLab and reuse its data, verifiers, scoring
and traces. Two hooks separate what differs between methods from what is shared:

  solve(problems, arguments) -> [Attempt]   How the candidate answers. The default
                                            asks the actor with arguments['instruction'];
                                            a subclass runs a harness, a checkpoint, etc.
  score(call_id, split, problems, attempts, arguments) -> OperationResult
                                            Verify, aggregate metrics, store the trace.

Smaller hooks: problems(arguments) picks the rows to answer, attempt(request,
generate) turns one model call into an Attempt (a subclass passes its own budgeted
generate), verify(problems, attempts) grades them, and result_data(records, ref,
arguments) adds fields to the result. bind_actor() names the model being measured
(the actor role by default). binding() is the input identity that traces carry; a
subclass that keeps more state extends state() and checks binding() on restore.

A subclass also extends describe()/execute() with its own operations, overrides
evaluate_schema() for different evaluate arguments, and adds answer formats to
the verifiers catalog that dataset manifests select from.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
import json
import multiprocessing
from time import monotonic

# Lab code imports its layers directly; rsisys.api is the composition layer.
from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.decision import OperationSpec
from rsisys.loops.contracts.execution import LabToolError
from rsisys.loops.contracts.model_service import GenerationRequest, ModelResponseError
from rsisys.loops.contracts.profiles import DEV, SEALED, TRAIN, MetricSpec
from rsisys.runtime.observation import report_samples

from .tasks import load_splits
from .verifiers import VERIFIERS


_TRACE_KIND = 'math-evaluation'
_TRACE_PAGE_CHARS = 4000
_SPLIT_ROLES = {'train': TRAIN, 'dev': DEV, 'sealed': SEALED}


@dataclass(frozen=True)
class Attempt:
    """One model answer to one problem, before verification.

    status is answered, truncated (the provider stopped at its output limit) or
    invalid_format (no usable text); only answered attempts are verified.
    extra holds a subclass's per-attempt evidence (trial, tool calls, errors),
    which the record keeps alongside the shared fields.
    """
    completion: str | None
    status: str = 'answered'
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)
    simulated: bool = False
    elapsed_seconds: float = 0.0
    extra: dict = field(default_factory=dict)


def _check_isolated(verifier, completion, answer):
    # Each spawned worker grades on its main thread, so signal-based time limits work.
    return verifier.check(completion, answer)


class MathLab:
    # Verifier names a dataset manifest may select; subclasses add their own formats.
    verifiers = VERIFIERS

    def __init__(self, context, config):
        self.validate_config(config)
        self.actor = self.bind_actor(context, config)
        self.models = context.models
        self.artifacts = context.artifacts
        self.role_names = dict(config['split_roles'])
        self.roles = {name: _SPLIT_ROLES[role] for name, role in self.role_names.items()}
        self.datasets, self.bindings, self.manifest_sha256 = load_splits(
            config['manifest'], self.roles, verifiers=self.verifiers, role_names=self.role_names)
        self.max_output_tokens = config.get('max_output_tokens', 32768)
        self.grading_workers = config.get('grading_workers', 1)

    def bind_actor(self, context, config):
        """The ModelRef of the model being measured."""
        actor = context.roles.get('actor')
        if actor is None:
            raise ValueError('Math Lab requires a configured actor model')
        return actor

    @staticmethod
    def validate_config(config):
        required = {'manifest', 'split_roles'}
        if not required <= config.keys() or config.keys() - required - {'max_output_tokens', 'grading_workers'}:
            raise ValueError('Math Lab requires manifest and split_roles')
        manifest, roles = config['manifest'], config['split_roles']
        if not isinstance(manifest, str) or not manifest.strip():
            raise ValueError('Math Lab manifest must be a nonempty file path')
        if (not isinstance(roles, dict) or not roles
                or any(not isinstance(name, str) or not name.strip() or role not in _SPLIT_ROLES
                       for name, role in roles.items())
                or list(roles.values()).count('sealed') > 1):
            raise ValueError('Assign math splits train, dev or sealed, with at most one combined sealed split')
        for key, default in (('max_output_tokens', 32768), ('grading_workers', 1)):
            value = config.get(key, default)
            if type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive integer')

    def split_roles(self):
        return dict(self.roles)

    def data_bindings(self):
        return self.bindings

    def primary_metric(self):
        return MetricSpec('accuracy', 'maximize', 'math-answer-exact-match/v1')

    def binding(self):
        """The data, actor and verifier identity every trace records."""
        return {'manifest_sha256': self.manifest_sha256, 'actor': asdict(self.actor),
                'max_output_tokens': self.max_output_tokens,
                'split_roles': dict(self.role_names),
                'verifiers': {name: sorted({row.verifier.name for row in rows})
                              for name, rows in self.datasets.items()}}

    def state(self):
        return self.binding()

    def restore(self, state):
        if state != self.binding():
            raise ValueError('Math Lab input binding changed since the checkpoint')

    def describe(self):
        trace_ref = {'type': 'object', 'properties': {
            'artifact_id': {'type': 'string'}, 'kind': {'type': 'string'},
            'uri': {'type': 'string'}, 'version': {'type': ['string', 'null']},
            'digest': {'type': ['string', 'null']}},
            'required': ['artifact_id', 'kind', 'uri', 'version', 'digest'],
            'additionalProperties': False}
        return (
            OperationSpec('inspect', 'Describe the visible evaluation splits and the actor budget.',
                          {'type': 'object', 'properties': {}, 'additionalProperties': False}, read_only=True),
            OperationSpec('evaluate', 'Score the actor using a fixed instruction and a frozen task split.',
                          self.evaluate_schema()),
            OperationSpec('read_trace', 'Read a page from one sample in a development evaluation.',
                          {'type': 'object', 'properties': {
                              'split': {'type': 'string'}, 'evaluation': trace_ref,
                              'index': {'type': 'integer', 'minimum': 0},
                              'offset': {'type': 'integer', 'minimum': 0}},
                           'required': ['split', 'evaluation', 'index'], 'additionalProperties': False},
                          read_only=True),
        )

    def evaluate_schema(self):
        """Input schema of evaluate: split and optional development task_ids, plus the candidate."""
        return {'type': 'object', 'properties': {
            'split': {'type': 'string', 'enum': list(self.datasets)},
            'instruction': {'type': 'string', 'minLength': 1},
            'task_ids': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'uniqueItems': True}},
            'required': ['split', 'instruction'], 'additionalProperties': False}

    def execute(self, call):
        args = call.arguments
        if call.name == 'inspect' and not args:
            return OperationResult('Math task environment', (), False, data={
                'protocol': 'math-environment/v1',
                'splits': {name: [{'id': row.id, 'benchmark': row.benchmark} for row in rows]
                           for name, rows in self.datasets.items() if self.roles[name].inputs_visible},
                'actor': asdict(self.actor), 'max_output_tokens': self.max_output_tokens,
            }, call_id=call.call_id)
        if call.name == 'evaluate':
            schema = self.evaluate_schema()
            if set(schema['required']) <= args.keys() <= set(schema['properties']):
                return self.evaluate(call.call_id, args)
        if call.name == 'read_trace' and {'split', 'evaluation', 'index'} <= args.keys() <= {
                'split', 'evaluation', 'index', 'offset'}:
            return OperationResult('Math trace page', (), False, data=self._read_trace(**args),
                                   call_id=call.call_id)
        raise LabToolError('Use the documented Math Lab operation arguments')

    def _select_rows(self, split, task_ids):
        if not isinstance(split, str) or split not in self.datasets:
            raise LabToolError('Unknown math evaluation split')
        rows = self.datasets[split]
        if task_ids is None:
            return rows
        if self.roles[split].after_finish_only:
            raise LabToolError('The sealed math evaluation must use the entire split')
        if (not isinstance(task_ids, list) or not task_ids
                or any(not isinstance(ident, str) for ident in task_ids)
                or len(set(task_ids)) != len(task_ids)):
            raise LabToolError('task_ids must be a nonempty array of distinct math problem IDs')
        by_id = {row.id: row for row in rows}
        if set(task_ids) - by_id.keys():
            raise LabToolError('Unknown math problem ID in task_ids')
        return tuple(by_id[ident] for ident in task_ids)

    def evaluate(self, call_id, arguments):
        """Answer the selected problems with solve() and score them."""
        split = arguments['split']
        problems = self.problems(arguments)
        started = monotonic()
        attempts = self.solve(problems, arguments)
        if len(attempts) != len(problems):
            raise RuntimeError('solve() must return one attempt per problem')
        return self.score(call_id, split, problems, attempts, arguments,
                          elapsed_seconds=monotonic() - started)

    def problems(self, arguments):
        """The rows evaluate() answers: the split, or the listed development task_ids."""
        return self._select_rows(arguments['split'], arguments.get('task_ids'))

    def solve(self, problems, arguments):
        """Ask the actor each problem after the instruction and the verifier's answer format."""
        instruction = arguments.get('instruction')
        if not isinstance(instruction, str) or not instruction.strip():
            raise LabToolError('instruction must be nonempty text')
        return [self.ask_actor(problem, instruction) for problem in problems]

    def ask_actor(self, problem, system):
        """One temperature-zero actor answer; the system text ends with the answer format."""
        request = GenerationRequest('actor', self.actor, problem.prompt,
            system=system + '\n\n' + problem.verifier.answer_format,
            temperature=0.0, max_output_tokens=self.max_output_tokens)
        return self.attempt(request)

    def attempt(self, request, generate=None):
        """One model call as an Attempt; truncated or empty replies are recorded, not raised."""
        started = monotonic()
        try:
            response = (generate or self.models.generate)(request)
        except ModelResponseError as error:
            if error.finish_reason == 'length':
                status = 'truncated'
            elif error.reason == 'empty_response':
                status = 'invalid_format'
            else:
                raise
            return Attempt(None, status, error.finish_reason, _usage(error.usage), False,
                           round(monotonic() - started, 6))
        return Attempt(response.text, 'truncated' if response.finish_reason == 'length' else 'answered',
                       response.finish_reason, _usage(response.usage), response.simulated,
                       round(monotonic() - started, 6))

    def verify(self, problems, attempts):
        """One AnswerCheck per answered attempt, None for the rest.

        Verifiers that must grade on a main thread (isolated) run in a pool of
        grading_workers processes when more than one is configured.
        """
        checks = [None] * len(attempts)
        answered = [i for i, attempt in enumerate(attempts)
                    if attempt.status == 'answered' and isinstance(attempt.completion, str)]
        pooled = [i for i in answered if self.grading_workers > 1 and problems[i].verifier.isolated]
        if pooled:
            with ProcessPoolExecutor(max_workers=self.grading_workers,
                                     mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = {i: pool.submit(_check_isolated, problems[i].verifier,
                                          attempts[i].completion, problems[i].answer) for i in pooled}
                for i, future in futures.items():
                    checks[i] = future.result()
        for i in answered:
            if checks[i] is None:
                checks[i] = problems[i].verifier.check(attempts[i].completion, problems[i].answer)
        return checks

    def record(self, problem, attempt, check):
        """The per-item evidence for one attempt and its check."""
        predicted, parsed, correct, status = None, False, False, attempt.status
        if check is not None:
            parsed, correct, predicted = check.parsed, check.correct, check.predicted
        if status == 'answered':
            status = 'correct' if correct else 'incorrect' if parsed else 'invalid_format'
        return {
            'id': problem.id, 'benchmark': problem.benchmark, 'question': problem.prompt,
            'expected': problem.expected, 'predicted': predicted,
            'completion': attempt.completion, 'correct': correct, 'parsed': parsed, 'status': status,
            'finish_reason': attempt.finish_reason, 'usage': dict(attempt.usage),
            'elapsed_seconds': attempt.elapsed_seconds, **attempt.extra,
        }

    def score(self, call_id, split, problems, attempts, arguments, *, elapsed_seconds):
        """Verify attempts, aggregate quality and execution metrics, and store the trace."""
        graded = monotonic()
        checks = self.verify(problems, attempts)
        grading_seconds = monotonic() - graded
        records = [self.record(problem, attempt, check)
                   for problem, attempt, check in zip(problems, attempts, checks, strict=True)]
        simulated = any(attempt.simulated for attempt in attempts)
        correct = sum(row['correct'] for row in records)
        quality = {'accuracy': correct / len(records), 'correct': correct}
        for name in sorted({row['benchmark'] for row in records}):
            group = [row for row in records if row['benchmark'] == name]
            count = sum(row['correct'] for row in group)
            quality.update({f'{name}_accuracy': count / len(group),
                            f'{name}_correct': count, f'{name}_count': len(group)})
        execution = {
            'elapsed_seconds': round(elapsed_seconds, 6),
            'grading_seconds': round(grading_seconds, 6),
            'num_samples': len(records),
            'invalid_format': sum(row['status'] == 'invalid_format' for row in records),
            'truncated': sum(row['status'] == 'truncated' for row in records),
        }
        for key in ('input_tokens', 'output_tokens'):
            counts = [row['usage'].get(key) for row in records]
            known = [count for count in counts if count is not None]
            execution[f'{key}_reported_samples'] = len(known)
            if len(known) == len(records):
                execution[key] = sum(known)

        sealed = self.roles[split].after_finish_only
        candidate = {key: value for key, value in arguments.items() if key not in ('split', 'task_ids')}
        # A distinct kind keeps standalone sealed artifacts restricted too.
        evidence = {'binding': self.binding(), 'split': split, **candidate, 'records': records}
        ref = self.artifacts.put(json.dumps(evidence, ensure_ascii=False, allow_nan=False).encode(),
                                 kind='sealed-evaluation' if sealed else _TRACE_KIND)
        if not sealed:
            report_samples([{key: row[key] for key in ('id', 'correct', 'parsed', 'expected', 'predicted')}
                            for row in records])
        return OperationResult('Math evaluation completed', (ref,), simulated, data={
            'split': split, 'model': self.actor.model_id, 'scored': True, 'count': len(records),
            'quality_metrics': quality, 'execution_metrics': execution, 'evidence_refs': [asdict(ref)],
            **self.result_data(records, ref, arguments),
        }, call_id=call_id)

    def result_data(self, records, ref, arguments):
        """Extra result fields a subclass's Controller reads; records and ref are this evaluation's."""
        return {}

    def _read_trace(self, split, evaluation, index, offset=0):
        if not isinstance(split, str) or self.roles.get(split) != DEV:
            raise LabToolError('Only development evaluation traces are available')
        if type(index) is not int or index < 0 or type(offset) is not int or offset < 0:
            raise LabToolError('index and offset must be nonnegative integers')
        try:
            ref = ArtifactRef(**evaluation)
        except (TypeError, ValueError) as error:
            raise LabToolError('evaluation must be a stored development artifact reference') from error
        if ref.kind != _TRACE_KIND:
            raise LabToolError('evaluation must reference a development trace')
        try:
            document = json.loads(self.artifacts.read(ref))
        except (KeyError, FileNotFoundError, ValueError) as error:
            raise LabToolError('Development trace could not be located with that reference') from error
        if document.get('split') != split or document.get('binding') != self.binding():
            raise LabToolError('Trace does not belong to this development binding')
        records = document['records']
        if index >= len(records):
            raise LabToolError('index is outside the recorded evaluation')
        sample = dict(records[index])
        sample.pop('question')
        text = sample.pop('completion') or ''
        end = offset + _TRACE_PAGE_CHARS
        return {'sample': sample, 'completion': text[offset:end],
                'next_offset': end if end < len(text) else None}


def _usage(usage):
    return asdict(usage) if usage is not None else {}
