"""Math Lab: frozen question-answer splits, actor inference and answer verification.

The Lab owns its task data (tasks.py, benchmarks.py), answer verification
(verifiers.py) and the actor call; the Controller supplies the instruction being
measured. Any numeric or symbolic question-answer benchmark connects by
configuration: each split in the scene manifest selects its verifier, and
fields renames row keys. Operations: inspect, evaluate (split, instruction,
optional development task_ids) and read_trace.
"""
from dataclasses import asdict
import json
from time import monotonic

# Lab code imports its layers directly; rsisys.api is the composition layer.
from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.decision import OperationSpec
from rsisys.loops.contracts.execution import LabToolError
from rsisys.loops.contracts.model_service import GenerationRequest, ModelResponseError
from rsisys.loops.contracts.profiles import DEV, SEALED, MetricSpec
from rsisys.runtime.observation import report_samples

from .tasks import load_splits


_TRACE_KIND = 'math-evaluation'
_TRACE_PAGE_CHARS = 4000
_SPLIT_ROLES = {'dev': DEV, 'sealed': SEALED}


class MathLab:
    def __init__(self, context, config):
        self.validate_config(config)
        self.actor = context.roles.get('actor')
        if self.actor is None:
            raise ValueError('Math Lab requires a configured actor model')
        self.models = context.models
        self.artifacts = context.artifacts
        self.roles = {name: _SPLIT_ROLES[role] for name, role in config['split_roles'].items()}
        self.datasets, self.bindings, self.manifest_sha256 = load_splits(config['manifest'], self.roles)
        self.max_output_tokens = config.get('max_output_tokens', 32768)

    @staticmethod
    def validate_config(config):
        required = {'manifest', 'split_roles'}
        if not required <= config.keys() or config.keys() - required - {'max_output_tokens'}:
            raise ValueError('Math Lab requires manifest and split_roles')
        manifest, roles = config['manifest'], config['split_roles']
        if not isinstance(manifest, str) or not manifest.strip():
            raise ValueError('Math Lab manifest must be a nonempty file path')
        if (not isinstance(roles, dict) or not roles
                or any(not isinstance(name, str) or not name.strip() or role not in _SPLIT_ROLES
                       for name, role in roles.items())
                or list(roles.values()).count('sealed') > 1):
            raise ValueError('Assign math splits dev or sealed, with at most one combined sealed split')
        tokens = config.get('max_output_tokens', 32768)
        if type(tokens) is not int or tokens < 1:
            raise ValueError('max_output_tokens must be a positive integer')

    def split_roles(self):
        return dict(self.roles)

    def data_bindings(self):
        return self.bindings

    def primary_metric(self):
        return MetricSpec('accuracy', 'maximize', 'math-answer-exact-match/v1')

    def state(self):
        return {'manifest_sha256': self.manifest_sha256, 'actor': asdict(self.actor),
                'max_output_tokens': self.max_output_tokens,
                'split_roles': {name: 'sealed' if role.after_finish_only else 'dev'
                                for name, role in self.roles.items()},
                'verifiers': {name: sorted({row.verifier.name for row in rows})
                              for name, rows in self.datasets.items()}}

    def restore(self, state):
        if state != self.state():
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
                          {'type': 'object', 'properties': {
                              'split': {'type': 'string', 'enum': list(self.datasets)},
                              'instruction': {'type': 'string', 'minLength': 1},
                              'task_ids': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1,
                                           'uniqueItems': True}},
                           'required': ['split', 'instruction'], 'additionalProperties': False}),
            OperationSpec('read_trace', 'Read a page from one sample in a development evaluation.',
                          {'type': 'object', 'properties': {
                              'split': {'type': 'string'}, 'evaluation': trace_ref,
                              'index': {'type': 'integer', 'minimum': 0},
                              'offset': {'type': 'integer', 'minimum': 0}},
                           'required': ['split', 'evaluation', 'index'], 'additionalProperties': False},
                          read_only=True),
        )

    def execute(self, call):
        args = call.arguments
        if call.name == 'inspect' and not args:
            return OperationResult('Math task environment', (), False, data={
                'protocol': 'math-environment/v1',
                'splits': {name: [{'id': row.id, 'benchmark': row.benchmark} for row in rows]
                           for name, rows in self.datasets.items() if self.roles[name].inputs_visible},
                'actor': asdict(self.actor), 'max_output_tokens': self.max_output_tokens,
            }, call_id=call.call_id)
        if call.name == 'evaluate' and {'split', 'instruction'} <= args.keys() <= {'split', 'instruction', 'task_ids'}:
            return self._evaluate(call.call_id, **args)
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

    def _sample(self, row, instruction):
        # The verifier names the answer format it reads after the Controller's instruction.
        request = GenerationRequest('actor', self.actor, row.prompt,
            system=instruction + '\n\n' + row.verifier.answer_format,
            temperature=0.0, max_output_tokens=self.max_output_tokens)
        started = monotonic()
        try:
            response = self.models.generate(request)
        except ModelResponseError as error:
            if error.finish_reason == 'length':
                status = 'truncated'
            elif error.reason == 'empty_response':
                status = 'invalid_format'
            else:
                raise
            completion, simulated, usage, finish = None, False, error.usage, error.finish_reason
        else:
            completion, simulated, usage, finish = (
                response.text, response.simulated, response.usage, response.finish_reason)
            status = 'truncated' if finish == 'length' else 'answered'
        elapsed = monotonic() - started
        predicted = None
        parsed = correct = False
        if status == 'answered' and isinstance(completion, str):
            check = row.verifier.check(completion, row.answer)
            parsed, correct, predicted = check.parsed, check.correct, check.predicted
        if status == 'answered':
            status = 'correct' if correct else 'incorrect' if parsed else 'invalid_format'
        return {
            'id': row.id, 'benchmark': row.benchmark, 'question': row.prompt, 'expected': row.expected,
            'predicted': predicted,
            'completion': completion, 'correct': correct, 'parsed': parsed, 'status': status,
            'finish_reason': finish, 'usage': asdict(usage) if usage is not None else {},
            'elapsed_seconds': round(elapsed, 6),
        }, simulated

    def _evaluate(self, call_id, split, instruction, task_ids=None):
        if not isinstance(instruction, str) or not instruction.strip():
            raise LabToolError('instruction must be nonempty text')
        rows = self._select_rows(split, task_ids)
        started = monotonic()
        records, simulated = [], False
        for row in rows:
            record, item_simulated = self._sample(row, instruction)
            records.append(record)
            simulated |= item_simulated

        correct = sum(row['correct'] for row in records)
        quality = {'accuracy': correct / len(records), 'correct': correct}
        for name in sorted({row['benchmark'] for row in records}):
            group = [row for row in records if row['benchmark'] == name]
            count = sum(row['correct'] for row in group)
            quality.update({f'{name}_accuracy': count / len(group),
                            f'{name}_correct': count, f'{name}_count': len(group)})
        execution = {
            'elapsed_seconds': round(monotonic() - started, 6),
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
        # A distinct kind keeps standalone sealed artifacts restricted too.
        evidence = {'binding': self.state(), 'split': split, 'instruction': instruction, 'records': records}
        ref = self.artifacts.put(json.dumps(evidence, ensure_ascii=False, allow_nan=False).encode(),
                                 kind='sealed-evaluation' if sealed else _TRACE_KIND)
        if not sealed:
            report_samples([{key: row[key] for key in ('id', 'correct', 'parsed', 'expected', 'predicted')}
                            for row in records])
        return OperationResult('Math evaluation completed', (ref,), simulated, data={
            'split': split, 'model': self.actor.model_id, 'scored': True, 'count': len(records),
            'quality_metrics': quality, 'execution_metrics': execution, 'evidence_refs': [asdict(ref)],
        }, call_id=call_id)

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
        if document.get('split') != split or document.get('binding') != self.state():
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
