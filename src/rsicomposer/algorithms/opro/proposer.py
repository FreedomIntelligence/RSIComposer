"""OPRO's optimizer step: the Controller's research model writes new instructions.

The proposer renders the pinned author meta-prompt from the scored instruction
history and the round's exemplar problems, which the Lab returns from its
search split, then samples candidates_per_round instructions from the
decision_maker model. It never scores candidates; the Lab does that.
"""
from dataclasses import dataclass
from hashlib import sha256
import json

from rsicomposer.api import ArtifactRef, GenerationRequest

from ..shared.experiment import ModelCalls
from .protocol import OPTIMIZER_SYSTEM_V1, upstream


@dataclass(frozen=True)
class Proposal:
    candidates: tuple[str, ...]   # Valid, polished instructions in sampling order.
    meta_prompt_sha256: str
    evidence: ArtifactRef         # Meta-prompt and raw replies, stored by the Controller.


class OPROProposer:
    def __init__(self, context, config):
        self.config, self.parameters = config, config['parameters']
        self.opt = upstream(config['source_dir'])
        self.model = context.roles['decision_maker']
        self.artifacts = context.artifacts
        # The Controller restores the count from its own state.
        self.calls = ModelCalls(context, self.parameters['rounds'] * self.parameters['candidates_per_round'])

    def complete(self, prompt, *, seed):
        adapted = self.parameters.get('local_adapter') == 'qwen-instructions-v1'
        model = self.config['model']
        return self.calls.text(GenerationRequest('decision_maker', self.model, prompt,
            system=OPTIMIZER_SYSTEM_V1 if adapted else '', temperature=model['optimizer_temperature'], top_p=1,
            max_output_tokens=self.parameters.get('optimizer_max_output_tokens', model['max_output_tokens']),
            seed=seed if adapted else None))

    def propose(self, history, exemplars, *, round_index):
        """history: [{instruction, score, round}]; exemplars: [{question, answer}] in exemplar order."""
        import pandas as pd

        params = self.parameters
        prompt = self.opt.gen_meta_prompt(
            old_instructions_and_scores=[(row['instruction'], row['score'], row['round']) for row in history],
            instruction_pos=params['instruction_pos'],
            optimizer_llm_name='gpt-4', old_instruction_score_threshold=.3,
            max_num_instructions=params['history_size'], meta_prompt_type='both_instructions_and_exemplars',
            few_shot_qa_pairs=True, include_qa=False,
            # The author reads few-shot rows by position; these are exactly the round's exemplars.
            data=pd.DataFrame([(row['question'], row['answer']) for row in exemplars]),
            few_shot_index_list=list(range(len(exemplars))), instructions_before_exemplars=True,
            num_score_buckets=100, dataset_name='gsm8k', task_name='gsm8k')
        count = params['candidates_per_round']
        responses, candidates = [], []
        for index in range(count):
            seed = (self.config['seed'] + round_index * count + index) % 2**32
            raw = self.complete(prompt, seed=seed)
            responses.append(raw)
            start = raw.find('<INS>')
            start = start + len('<INS>') if start >= 0 else 0
            end = raw.find('</INS>')
            instruction = self.opt.eval_utils.polish_sentence(raw[start:end if end >= 0 else len(raw)].strip())
            if (instruction and len(instruction) <= 500 and 'INS' not in instruction
                    and not any(char.isdigit() for char in instruction)):
                candidates.append(instruction)
        digest = sha256(prompt.encode()).hexdigest()
        evidence = self.artifacts.put(json.dumps({'round': round_index, 'meta_prompt': prompt,
            'responses': responses}, ensure_ascii=False, allow_nan=False).encode(), kind='opro-proposal')
        return Proposal(tuple(candidates), digest, evidence)
