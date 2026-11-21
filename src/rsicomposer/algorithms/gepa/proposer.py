"""GEPA's reflective mutation: the Controller's research step, using the decision model.

For the fixed-spend method, the proposer reads the parent's minibatch trace that
the Lab stored with its evaluation receipt, renders the pinned GEPA
instruction-proposal prompt, and asks the decision_maker model for one new
instruction. For the merge method, the pinned engine renders its own prompts and
reflect() answers them. Neither scores candidates; the Lab does that.
"""
from dataclasses import dataclass
from hashlib import sha256
import json

from rsicomposer.api import ArtifactRef, GenerationRequest

from ..shared.experiment import ModelCalls
from .protocol import upstream, valid_instruction


MUTATOR_SYSTEM = ('Write a single general-purpose instruction for solving unseen math word problems. '
                  'Keep the instruction concise and return it in a fenced text block.')
SCORE_ONLY_TEMPLATE = """I provided an assistant with the following instructions to perform a task for me:
```
<curr_instructions>
```

The following are task inputs and the assistant's correctness score for each one (1 correct, 0 incorrect):
```
<inputs_outputs_feedback>
```

Your task is to write a new instruction for the assistant. Read the inputs and scores carefully to infer the
task format and propose a generalizable strategy for solving the task more accurately.

Provide the new instructions within ``` blocks."""


@dataclass(frozen=True)
class Proposal:
    candidate: str | None      # None when the reply holds no valid instruction.
    meta_prompt_sha256: str
    evidence: ArtifactRef      # Meta-prompt and raw reply, stored by the Controller.


class GEPAProposer:
    def __init__(self, context, config, *, max_calls):
        self.parameters, self.seed = config['parameters'], config['seed']
        self.temperature = config['model']['optimizer_temperature']
        self.signature = upstream(config['source_dir'])[3]
        self.model = context.roles['decision_maker']
        self.artifacts = context.artifacts
        # The Controller restores the count from its own state.
        self.calls = ModelCalls(context, max_calls)

    def examples(self, records):
        """The parent's minibatch as GEPA's dataset_with_feedback."""
        reflective = self.parameters['condition'] == 'reflective'
        examples = []
        for row in records:
            example = {'Inputs': row['question']}
            if reflective:
                observed = row['completion'] or ''
                if len(observed) > 2400:
                    observed = observed[:1600] + '\n[middle excerpt omitted]\n' + observed[-800:]
                example['Generated Outputs'] = observed
                if row['correct']:
                    example['Feedback'] = 'Score 1. The final answer matches the reference answer.'
                else:
                    issue = 'The final answer is incorrect.' if row['parsed'] else 'No valid numeric final answer was found.'
                    example['Feedback'] = f"Score 0. {issue} Reference answer: {row['expected']}."
            else:
                example['Feedback'] = int(row['correct'])
            examples.append(example)
        return examples

    def parent_records(self, trace, receipt_id, request):
        """Records of the parent's minibatch evaluation, checked against its receipt."""
        raw = self.artifacts.read(trace)
        if trace.kind != 'math-evaluation' or sha256(raw).hexdigest() != receipt_id:
            raise ValueError('GEPA mutation requires a matching immutable parent score receipt')
        parent = json.loads(raw)
        if parent['request'] != request:
            raise ValueError('GEPA mutation receipt does not match the chosen parent minibatch')
        return parent['records']

    def complete(self, prompt, *, round_index):
        return self.calls.text(GenerationRequest('decision_maker', self.model, prompt, system=MUTATOR_SYSTEM,
            temperature=self.temperature, top_p=1,
            max_output_tokens=self.parameters['optimizer_max_output_tokens'],
            seed=(self.seed + round_index) % 2**32))

    def reflect(self, prompt, *, round_index):
        """Answer one reflection prompt rendered by the pinned engine, keeping the reply as evidence."""
        text = self.complete(prompt, round_index=round_index)
        self.artifacts.put(text.encode(), kind='gepa-reflect')
        return text

    def propose(self, instruction, records, *, round_index):
        reflective = self.parameters['condition'] == 'reflective'
        prompt = self.signature.prompt_renderer({'current_instruction_doc': instruction,
            'dataset_with_feedback': self.examples(records),
            'prompt_template': None if reflective else SCORE_ONLY_TEMPLATE})
        raw = self.complete(prompt, round_index=round_index)
        proposed = self.signature.output_extractor(raw)['new_instruction']
        digest = sha256(prompt.encode()).hexdigest()
        evidence = self.artifacts.put(json.dumps({'parent_instruction': instruction, 'meta_prompt': prompt,
            'response': raw, 'round': round_index}, ensure_ascii=False, allow_nan=False).encode(),
            kind='gepa-proposal')
        return Proposal(proposed if valid_instruction(proposed) else None, digest, evidence)
