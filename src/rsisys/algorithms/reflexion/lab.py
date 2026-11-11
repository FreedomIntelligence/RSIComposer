"""Single HumanEval operations using the pinned author prompts and isolated tests."""
from dataclasses import asdict
import json
import random
import sys
from uuid import uuid4

from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.decision import OperationSpec
from rsisys.loops.contracts.model_service import GenerationRequest

from ..shared.experiment import ModelCalls
from .executor import SandboxedExecutor
from .protocol import BrokerModel, load_tasks


OPERATIONS = {
    'tests': '生成内部测试',
    'implement': '生成代码',
    'execute': '运行内部测试',
    'reflect': '根据失败反馈生成反思',
    'hidden': '运行隐藏测试',
}


class SessionClient:
    """Bind the author generator's model protocol to RSISys accounting."""
    def __init__(self, config, calls, model, seed, role):
        self.config, self.calls, self.model, self.seed, self.role = config, calls, model, seed, role

    def generate(self, _role, prompt='', *, messages=None, temperature, max_tokens, stop=None):
        return self.calls.text(GenerationRequest(
            self.role, self.model, prompt,
            messages=tuple(messages) if messages is not None else None,
            max_output_tokens=min(max_tokens, self.config['model']['max_output_tokens']),
            temperature=temperature, top_p=1, seed=self.seed, stop=stop,
        ))


class HumanEvalLab:
    def __init__(self, context, config):
        self.config, self.context = config, context
        self.source, self.tasks = load_tasks(config)
        self.calls = ModelCalls(context, config['max_model_calls'])
        sys.path.insert(0, str(self.source))
        try:
            import generators
            from generators import generator_factory
            from pathlib import Path

            if not Path(generators.__file__).resolve().is_relative_to(self.source / 'generators'):
                raise RuntimeError('Reflexion generator imported outside the pinned author checkout')
            self.generator = generator_factory('py')
        finally:
            sys.path.remove(str(self.source))
        self.executor = self.create_executor()

    def create_executor(self):
        return SandboxedExecutor(self.source)

    def describe(self):
        return tuple(OperationSpec(name, title, {'type': 'object'})
                     for name, title in OPERATIONS.items())

    def get(self, value, kind):
        ref = ArtifactRef(**value)
        if ref.kind != 'reflexion-' + kind:
            raise ValueError('Unexpected Reflexion artifact type')
        return json.loads(self.context.artifacts.get(ref))

    def put(self, value, kind):
        return self.context.artifacts.put(json.dumps(value, ensure_ascii=False, allow_nan=False).encode(),
                                          kind='reflexion-' + kind)

    def execute(self, call):
        request = call.arguments
        index, attempt = request['task'], request['attempt']
        if type(index) is not int or not 0 <= index < len(self.tasks):
            raise ValueError('Unknown HumanEval task index')
        task = self.tasks[index]
        name = call.name
        prompt = task['prompt']
        role = 'decision_maker' if name == 'reflect' else 'actor'
        # Match seeds across conditions even when one condition skips reflection.
        phase_seed = {'tests': 0, 'implement': 1, 'reflect': 2, 'execute': 3, 'hidden': 4}[name]
        seed = (self.config['seed'] + index * 100 + attempt * 5 + phase_seed) % 2**32
        model = BrokerModel(SessionClient(self.config, self.calls, self.context.roles[role], seed, role))
        artifacts, values = [], {}
        public = {'summary': '操作已完成。', 'changes': []}
        if name == 'tests':
            state = random.getstate()
            try:
                random.seed(seed)
                tests = self.generator.internal_tests(prompt, model, self.config['parameters']['max_internal_tests'])
            finally:
                random.setstate(state)
            artifacts.append(self.put(tests, 'tests'))
            values['tests'] = asdict(artifacts[-1])
            public = {'summary': f'生成了 {len(tests)} 条内部测试。',
                      'changes': [{'field': '内部测试', 'after': '\n'.join(tests)}]}
        elif name == 'implement':
            kwargs = {}
            if 'program' in request:
                kwargs.update(prev_func_impl=self.get(request['program'], 'program'),
                    feedback=self.get(request['feedback'], 'feedback'),
                    self_reflection=self.get(request['reflection'], 'reflection') if request['reflection'] else '')
            program = self.generator.func_impl(prompt, model, 'reflexion' if kwargs else 'simple', **kwargs)
            artifacts.append(self.put(program, 'program'))
            values['program'] = asdict(artifacts[-1])
            public = {'summary': '代码候选已生成。',
                      'changes': [{'field': '生成的代码', 'after': program}]}
        elif name == 'reflect':
            reflection = self.generator.self_reflection(
                self.get(request['program'], 'program'), self.get(request['feedback'], 'feedback'), model)
            artifacts.append(self.put(reflection, 'reflection'))
            values['reflection'] = asdict(artifacts[-1])
            public = {'summary': '反思已生成，将用于下一次代码生成。',
                      'changes': [{'field': '反思内容', 'after': reflection}]}
        elif name == 'execute':
            result = self.executor.execute(self.get(request['program'], 'program'), self.get(request['tests'], 'tests'))
            # Keep the failure trace bounded before using it as a model input.
            feedback = result.feedback[:4000]
            artifacts.append(self.put(feedback, 'feedback'))
            values.update(passed=result.is_passing, feedback=asdict(artifacts[-1]))
            public = {'summary': '内部测试通过。' if result.is_passing else '内部测试未全部通过。',
                      'changes': [{'field': '内部测试反馈', 'after': feedback}]}
        else:
            passed = self.executor.evaluate(task['entry_point'], self.get(request['program'], 'program'),
                                            task['test'], timeout=10)
            values.update(task_id=task['task_id'], passed=passed,
                          quality_metrics={'correct': int(passed), 'examples': 1})
            public['summary'] = '隐藏测试通过。' if passed else '隐藏测试未通过。'

        return OperationResult(f"Reflexion {name}: {request['request_id']}", tuple(artifacts), False,
            data={'request': request, 'receipt_id': uuid4().hex, 'public': public, **values})
