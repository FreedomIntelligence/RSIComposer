"""One veRL GRPO training job bracketed by independent MathLab evaluations."""

from copy import deepcopy
import json
from pathlib import Path

from rsicomposer.api import (
    CheckpointRuntime, Decision, ExecutionStatus, HostedLoRAEngine, MathLab, MethodDefinition,
    ModelRef, ObservationProbe, OperationCall, OperationResult, OperationSpec, emit_observation, file_hash, report_version,
    report_version_evaluation, report_version_selection,
)
from ..shared.weight_observation import model_version


class Controller:
    def __init__(self):
        self.stage = 0
        self.pending = None

    def required_operations(self):
        return ('math.fit', 'math.evaluate')

    def state(self):
        return {'stage': self.stage, 'pending': deepcopy(self.pending)}

    def restore(self, state):
        self.stage, self.pending = state['stage'], deepcopy(state['pending'])

    def decide(self, window):
        if self.pending is not None:
            feedback = window.feedbacks[0] if window.feedbacks else None
            if feedback is None or feedback.call_id != self.pending.call_id:
                raise ValueError('Expected the receipt for the pending veRL experiment operation')
            if feedback.status != ExecutionStatus.COMPLETED:
                return Decision('finish', reason=f'{self.pending.name} ended with {feedback.status.value}')
            if self.pending.arguments.get('candidate') == 'trained':
                version = model_version(feedback.data['checkpoint'])
                report_version_selection(version, step=1, event_id=feedback.call_id, action='applied')
                emit_observation('verl_math_grpo.applied', feedback.call_id, subject=version,
                                 fields={'action': 'applied'})
        if self.stage == 3:
            return Decision('finish', reason='GRPO candidate evaluated against the baseline protocol')
        name, arguments = (('evaluate', {'split': 'dev', 'candidate': 'base'}), ('fit', {}),
                           ('evaluate', {'split': 'dev', 'candidate': 'trained'}))[self.stage]
        self.stage += 1
        self.pending = OperationCall(f'math.{name}', arguments)
        return Decision.invoke(deepcopy(self.pending), reason=f'Run {name} for the veRL GRPO experiment')


def validate_config(config):
    required = {'manifest', 'initial_model', 'backend'}
    if not required <= config.keys() or config.keys() - required - {'inference_python', 'evaluation_tokens'}:
        raise ValueError('veRL Math GRPO requires manifest, initial_model and backend')
    for key in ('manifest', 'initial_model', 'inference_python'):
        if key in config and (not isinstance(config[key], str) or not config[key].strip()):
            raise ValueError(f'{key} must be a nonempty path')
    backend = config['backend']
    if not isinstance(backend, dict) or set(backend) != {'name', 'config'} or not isinstance(backend['config'], dict):
        raise ValueError('Choose a registered GRPO backend with name and config')
    if type(config.get('evaluation_tokens', 64)) is not int or config.get('evaluation_tokens', 64) < 1:
        raise ValueError('evaluation_tokens must be a positive integer')


class GRPOMathLab(MathLab):
    source_budgets = False

    def __init__(self, context, config, trainer):
        validate_config(config)
        self.trainer = trainer
        self.trainer.bind_execution(context)
        self.initial_model = Path(config['initial_model']).resolve(strict=True)
        self.output = Path(context.data_dir).resolve() / 'verl-math-grpo' / 'checkpoint'
        self.checkpoint_files = None
        self.evaluation_tokens = config.get('evaluation_tokens', 64)
        evaluation = {'max_new_tokens': self.evaluation_tokens, 'max_length': 4096}
        if 'inference_python' in config:
            evaluation['inference_python'] = config['inference_python']
        self.inference = HostedLoRAEngine(evaluation)
        self.inference.bind_execution(context)
        manifest = Path(config['manifest'])
        roles = {'train': 'train', 'dev': 'dev'}
        if 'test' in json.loads(manifest.read_text())['splits']:
            roles['test'] = 'sealed'
        super().__init__(context, {'manifest': config['manifest'], 'split_roles': roles})

    def resource_roles(self):
        return {'rollout': self.inference.resource_roles()['rollout'], **self.trainer.resource_roles()}

    def bind_actor(self, context, config):
        return ModelRef('local-weights', self.inference.name, 'checkpoint')

    def make_runtime(self, config):
        return CheckpointRuntime(engine=self.inference.name, temperature=0.0, max_new_tokens=self.evaluation_tokens)

    def state(self):
        return {'binding': super().state(), 'checkpoint_files': self.checkpoint_files}

    def restore(self, state):
        super().restore(state['binding'])
        self.checkpoint_files = state['checkpoint_files']
        if self.checkpoint_files is not None:
            self._check_weights()

    def describe(self):
        return (*super().describe(), OperationSpec('fit', 'Train with online MathLab rewards.', {
            'type': 'object', 'properties': {}, 'additionalProperties': False}))

    def evaluate_schema(self):
        splits = [split for split in ('dev', 'test') if split in self.datasets]
        return {'type': 'object', 'properties': {'split': {'enum': splits},
            'candidate': {'enum': ['base', 'trained']}}, 'required': ['split', 'candidate'],
            'additionalProperties': False}

    def evaluation_operations(self):
        return ('evaluate',)

    def candidate_identity_fields(self):
        return {'evaluate': ('candidate',)}

    def final_evaluations(self):
        if self.checkpoint_files is None or 'test' not in self.datasets:
            return ()
        return tuple(OperationCall('evaluate', {'split': 'test', 'candidate': candidate})
                     for candidate in ('base', 'trained'))

    @staticmethod
    def _messages(problem):
        return [{'role': 'user', 'content': f'{problem.prompt}\n\n{problem.verifier.answer_format}'}]

    def _training_rows(self, split):
        import json
        return [{'messages': self._messages(problem), 'data_source': 'rsicomposer-math',
                 'ground_truth': json.dumps(problem.answer, allow_nan=False), 'extra_info': {
                  'task_id': problem.id, 'verifier_json': json.dumps(problem.verifier.spec, sort_keys=True)}}
                for problem in self.datasets[split]]

    def _check_weights(self):
        if not self.checkpoint_files or {p.name: file_hash(p) for p in self.output.glob('*.safetensors')} != self.checkpoint_files:
            raise ValueError('Published GRPO checkpoint is missing or has changed')

    def execute(self, call):
        if call.name != 'fit':
            result = super().execute(call)
            if call.name == 'evaluate' and result.status == ExecutionStatus.COMPLETED:
                candidate = call.arguments['candidate']
                version = model_version(str(self.initial_model if candidate == 'base' else self.output))
                if candidate == 'base':
                    report_version(version, label='Baseline', round_index=0, parent_ids=())
                report_version_evaluation(version, call.call_id)
            return result
        if call.arguments or self.checkpoint_files is not None or self.output.exists():
            raise ValueError('fit accepts no arguments and publishes one checkpoint per experiment')
        result = self.trainer.fit(self.initial_model, self._training_rows('train'), self._training_rows('dev'),
                                  self.output, reward_module=Path(__file__).with_name('reward.py'))
        self.checkpoint_files = result['checkpoint_files']
        self._check_weights()
        parent = model_version(str(self.initial_model))
        report_version(model_version(str(self.output)), label='GRPO checkpoint', round_index=1,
                       parent_ids=(parent,))
        return OperationResult('veRL GRPO checkpoint published', (), False, call_id=call.call_id,
                               data={'model': str(self.output), 'training_performed': True, 'training': result})

    def solve(self, problems, arguments):
        if set(arguments) != {'split', 'candidate'} or arguments['split'] not in ('dev', 'test'):
            raise ValueError('Evaluate only a configured development or sealed test split')
        if arguments['candidate'] == 'trained':
            self._check_weights()
            model = self.output
        elif arguments['candidate'] == 'base':
            model = self.initial_model
        else:
            raise ValueError('Choose the base or trained checkpoint')

        def generate(path, rows):
            prompts = [{'prompt': self._messages(problem)} for problem in rows]
            return [sample[0] for sample in self.inference.generate(path, prompts, 1, 0, False)]

        return self.run_runtime({'model': str(model)}, problems, arguments, context={'generate': generate})

    def result_data(self, records, ref, arguments):
        return {'checkpoint': str(self.initial_model if arguments['candidate'] == 'base' else self.output)}


def register(registry):
    def lab(context, config):
        validate_config(config)
        backend = config['backend']
        trainer = registry.resolve_component('backend', backend['name'], context, backend['config'])
        return GRPOMathLab(context, config, trainer)

    registry.register_method('verl_math_grpo', MethodDefinition.from_factories(
        controller=lambda context, config: Controller(**config), labs={'math': lab},
        observation_probes=(ObservationProbe('verl_math_grpo.applied', 'decision', (('action', 'id'),)),),
        version='1', validate_labs={'math': validate_config}))
