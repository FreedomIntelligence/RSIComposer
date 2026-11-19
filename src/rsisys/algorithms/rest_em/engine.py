"""ReST EM formatting and fresh-base model lineage over reusable LoRA backends."""
import json
from pathlib import Path
from typing import Protocol


class RestEMEngine(Protocol):
    name: str
    simulated: bool

    def generate(self, ref, rows, count, round_number, sample, *, temperature=0.7): ...
    def train(self, ref, rows, output, round_number, *, steps=None): ...


class FixtureRestEMEngine:
    """Uses supplied answers for protocol tests only; never claims learned weights."""
    name = 'fixture-v1'
    simulated = True

    def generate(self, ref, rows, count, round_number, sample, *, temperature=0.7):
        return [[row['answer'] if i % 2 == 0 else 'unparseable fixture response'
                 for i in range(count)] for row in rows]

    def train(self, ref, rows, output, round_number, *, steps=None):
        pass


class TransformersRestEMEngine:
    name = 'transformers-lora-v1'
    simulated = False
    backend_type = None

    def __init__(self, config=None, *, input_root=None):
        self.config = dict(config or {})
        self.input_root = Path(input_root or Path.cwd()).resolve()
        from rsisys.api import HostedLoRAEngine
        backend_config = {key: value for key, value in self.config.items() if key not in {
            'few_shot_data', 'answer_format', 'prompt_format', 'verifier', 'data_root',
            'output_subdir', 'engine', 'mode'}}
        backend_config['stop_strings'] = ['\nQuestion:'] if self.config.get('prompt_format', 'chat') == 'completion' else []
        self.backend = (self.backend_type or HostedLoRAEngine)(backend_config)
        self.demonstrations = []
        if self.config.get('few_shot_data'):
            from rsisys.api import final_boxed, numeric_answer
            parse_answer = final_boxed if self.config.get('answer_format') == 'latex' else numeric_answer
            data = self._path(self.config['few_shot_data'])
            self.demonstrations = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
            if not self.demonstrations or any(not isinstance(row, dict) or
                    not isinstance(row.get('prompt'), str) or not row['prompt'].strip() or
                    not isinstance(row.get('answer'), str) or parse_answer(row['answer']) is None
                    for row in self.demonstrations):
                raise ValueError('Few-shot data must contain prompt and an answer in the selected final-answer format')
            if len({r['prompt'].strip() for r in self.demonstrations}) != len(self.demonstrations):
                raise ValueError('Duplicate few-shot questions')

    def bind_execution(self, context):
        self.backend.bind_execution(context)

    def resource_roles(self):
        return self.backend.resource_roles()

    def _path(self, ref):
        path = Path(ref)
        return (path if path.is_absolute() else self.input_root / path).resolve(strict=True)

    def _prompt(self, prompt):
        ending = ('End each answer with the final answer in \\boxed{...}.' if self.config.get('answer_format') == 'latex'
                  else 'End each answer with #### followed by the numeric answer.')
        if self.config.get('prompt_format', 'chat') == 'completion':
            if any(row['prompt'].strip() == prompt.strip() for row in self.demonstrations):
                raise ValueError('Few-shot demonstration overlaps a target question')
            instruction = f'Solve each math problem step by step. {ending}\n\n'
            examples = ''.join(f"Question: {row['prompt']}\nAnswer: {row['answer']}\n\n"
                               for row in self.demonstrations)
            return instruction + examples + f'Question: {prompt}\nAnswer:'
        instruction = ending if self.config.get('answer_format') == 'latex' else 'End with #### followed by the numeric answer.'
        return [{'role': 'user', 'content': prompt + '\n' + instruction}]

    def _model(self, ref, *, train=False):
        selected = self._path(ref)
        manifest = selected / 'rest-em.json'
        if not manifest.is_file():
            return selected, None
        metadata = json.loads(manifest.read_text())
        if metadata['simulated']:
            raise ValueError('Cannot load fixture output as model weights')
        if train:
            raise ValueError('M-step must load initial weights, not a prior adapter')
        return self._path(metadata['parent']), selected / 'adapter'

    def generate(self, ref, rows, count, round_number, sample, *, temperature=0.7):
        base, adapter = self._model(ref)
        prompts = [{'prompt': self._prompt(row['prompt'])} for row in rows]
        return self.backend.generate(base, prompts, count, round_number, sample, adapter=adapter, temperature=temperature)

    def train(self, ref, rows, output, round_number, *, steps=None):
        base, _ = self._model(ref, train=True)
        prepared = [{'prompt': self._prompt(row['prompt']), 'completion': row['completion']} for row in rows]
        return self.backend.train(base, prepared, output, round_number, steps=steps)


class ParallelRestEMEngine(TransformersRestEMEngine):
    name = 'vllm-dp-ddp-lora-v1'

    def __init__(self, config=None, *, input_root=None):
        from rsisys.api import ParallelLoRAEngine
        self.backend_type = ParallelLoRAEngine
        super().__init__(config, input_root=input_root)
