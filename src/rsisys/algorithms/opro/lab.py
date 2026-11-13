"""GSM8K operations using the pinned OPRO prompt builder and evaluator."""
import hashlib
from rsisys.runtime.orchestration.monitor_snapshot import active_monitor
import json

from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.decision import OperationSpec
from rsisys.loops.contracts.model_service import GenerationRequest

from ..shared.experiment import ModelCalls
from ..shared.prompt_data import load_splits
from .protocol import OPTIMIZER_SYSTEM_V1, SCORER_SYSTEM_V1, upstream


class OPROGSM8KLab:
    def __init__(self, context, config):
        self.config = config
        self.session = ModelCalls(context, config['max_model_calls'])
        self.artifacts = context.artifacts
        self.models = context.roles
        self.groups = load_splits(config['files'], {k: config['data'][k + '_count']
                                                  for k in ('search', 'validation', 'test')})
        self.opt = upstream(config['source_dir'])

    def describe(self):
        return (
            OperationSpec('generate', 'Generate OPRO candidates from the supplied history', {'type': 'object'}),
            OperationSpec('score', 'Score one instruction on one GSM8K split', {'type': 'object'}),
        )

    def execute(self, call):
        if call.name == 'generate':
            values, evidence = self.generate(call.arguments)
        elif call.name == 'score':
            values, evidence = self.score(call.arguments)
        else:
            raise ValueError(f'Unknown OPRO Lab operation: {call.name}')
        payload = json.dumps({'operation': call.name, 'request': call.arguments,
                              'evidence': evidence}, ensure_ascii=False, allow_nan=False).encode()
        artifact = self.artifacts.put(payload, kind='opro-' + call.name)
        monitor = active_monitor.get()
        if monitor is not None and monitor.current_id and call.name == "score":
            monitor.add_samples(evidence.get("records", ()), monitor.current_id)
        return OperationResult(f"OPRO {call.name}: {call.arguments['request_id']}", (artifact,), False,
            data={'request': call.arguments, 'receipt_id': hashlib.sha256(payload).hexdigest(), **values})

    def complete(self, role, prompt, *, seed):
        model, params = self.config['model'], self.config['parameters']
        optimizer = role == 'decision_maker'
        adapted = params.get('local_adapter') == 'qwen-instructions-v1'
        system = OPTIMIZER_SYSTEM_V1 if optimizer else SCORER_SYSTEM_V1
        cap = model['max_output_tokens']
        if optimizer:
            cap = params.get('optimizer_max_output_tokens', cap)
        temperature = model['optimizer_temperature' if optimizer else 'scorer_temperature']
        return self.session.text(GenerationRequest(role, self.models[role], prompt,
            system=system if adapted else '', temperature=temperature, top_p=1,
            max_output_tokens=cap, seed=seed if adapted else None))

    def dataframe(self, split):
        import pandas as pd
        return pd.DataFrame([(row['question'], row['answer']) for row in self.groups[split]])

    def generate(self, request):
        params = self.config['parameters']
        history = [(row['instruction'], row['score'], row['round']) for row in request['history']]
        prompt = self.opt.gen_meta_prompt(
            old_instructions_and_scores=history, instruction_pos=params['instruction_pos'],
            optimizer_llm_name='gpt-4', old_instruction_score_threshold=.3,
            max_num_instructions=params['history_size'], meta_prompt_type='both_instructions_and_exemplars',
            few_shot_qa_pairs=True, include_qa=False, data=self.dataframe('search'),
            few_shot_index_list=request['exemplar_indices'], instructions_before_exemplars=True,
            num_score_buckets=100, dataset_name='gsm8k', task_name='gsm8k')
        count = params['candidates_per_round']
        responses, candidates = [], []
        for index in range(count):
            seed = (self.config['seed'] + request['round'] * count + index) % 2**32
            raw = self.complete('decision_maker', prompt, seed=seed)
            responses.append(raw)
            start = raw.find('<INS>')
            start = start + len('<INS>') if start >= 0 else 0
            end = raw.find('</INS>')
            instruction = self.opt.eval_utils.polish_sentence(raw[start:end if end >= 0 else len(raw)].strip())
            if (instruction and len(instruction) <= 500 and 'INS' not in instruction
                    and not any(char.isdigit() for char in instruction)):
                candidates.append(instruction)
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        return {'candidates': candidates, 'meta_prompt_sha256': digest}, {
            'meta_prompt_sha256': digest, 'responses': responses}

    def score(self, request):
        frame = self.dataframe(request['split'])

        def scorer(prompt, **_):
            prompts = [prompt] if isinstance(prompt, str) else prompt
            return [self.complete('actor', item, seed=self.config['seed']) for item in prompts]

        result = self.opt.eval_utils.evaluate_single_instruction(
            data=frame, instruction=request['instruction'], eval_index_all=list(range(len(frame))),
            batch_size=1, call_server_func=scorer, dataset_name='gsm8k', num_servers=1,
            extract_final_answer_by_prompting_again=False, instruction_pos=self.config['parameters']['instruction_pos'],
            is_multiple_choice=False, include_qa=False, evaluate_in_parallel=False,
            prediction_treat_as_number=True)
        score = float(result['accuracy'].mean())
        return {'score': score, 'count': len(result),
                'quality_metrics': {'accuracy': score, 'examples': len(result)}}, {
            'records': json.loads(result.reset_index().to_json(orient='records'))}
