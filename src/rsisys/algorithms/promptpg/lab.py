"""Run one PromptPG selector operation and route the frozen solver."""

from io import BytesIO
import json
from pathlib import Path
import random

from rsisys.api import LabFeedback, OperationSpec
from rsisys.loops.contracts.model_service import GenerationRequest

from ..shared.experiment import ModelCalls
from .protocol import PROMPT_OPTIONS, choice_pairs, load_author, load_cohort, score_answer


class PromptPGLab:
    def __init__(self, context, config):
        self.context, self.config = context, config
        self.calls = ModelCalls(context, config['max_model_calls'])
        self.cache = {}
        self.selector = None
        self.best = None
        self.restored = False
        self.rng = random.Random(config['seed'] + 10)

    def describe(self):
        return tuple(OperationSpec(name, label, {'type': 'object'}) for name, label in (
            ('prepare', 'Load the frozen encoder and benchmark'),
            ('train_batch', 'Perform one selector gradient update'),
            ('save', 'Persist the current selector'),
            ('restore', 'Restore the chosen selector'),
            ('evaluate', 'Measure one held-out batch under one condition'),
        ))

    def execute(self, call):
        handler = {'prepare': self.prepare, 'train_batch': self.train_batch, 'save': self.save,
                   'restore': self.restore, 'evaluate': self.evaluate}[call.name]
        data, artifacts = handler(call.arguments)
        return LabFeedback(f'PromptPG {call.name}', artifacts, False,
                           data={'request': call.arguments, **data})

    def prepare(self, request):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if request or self.selector is not None:
            raise ValueError('PromptPG can only be prepared once')
        source, self.questions, train, self.candidates, test = load_cohort(self.config)
        self.train_ids, self.test_ids = set(train), set(test)
        self.author, self.scorer = (load_author(source, name) for name in ('base_prompt', 'utilities'))
        encoder_path = Path(self.config['encoder_path']).resolve(strict=True)
        tokenizer = AutoTokenizer.from_pretrained(encoder_path, local_files_only=True)
        encoder = AutoModel.from_pretrained(encoder_path, local_files_only=True).eval().requires_grad_(False)
        ids = train + self.candidates + test
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(ids), 16):
                texts = [self.author.create_example_from_pid(key, self.questions, PROMPT_OPTIONS, test=True)
                         for key in ids[start:start + 16]]
                tokens = tokenizer(texts, max_length=512, truncation=True, padding=True, return_tensors='pt')
                chunks.append(encoder(**tokens).last_hidden_state[:, 0, :])
        self.encodings = dict(zip(ids, torch.cat(chunks), strict=True))
        self.candidate_vectors = torch.stack([self.encodings[key] for key in self.candidates])
        hidden = self.candidate_vectors.shape[1]
        self.generator = torch.Generator()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config['seed'])
            self.selector = torch.nn.Linear(hidden, hidden)
            self.generator.set_state(torch.get_rng_state())
        self.initial = {name: value.detach().clone() for name, value in self.selector.state_dict().items()}
        self.optimizer = torch.optim.Adam(self.selector.parameters(), lr=self.config['learning_rate'])
        ref = self.context.artifacts.put(json.dumps({'train': train, 'candidates': self.candidates, 'test': test}).encode(),
                                         kind='promptpg-cohorts')
        return {'train': train, 'test': test}, (ref,)

    def score(self, ids, pairs):
        for key, pair in zip(ids, pairs, strict=True):
            cache_key = key, tuple(pair)
            if cache_key in self.cache:
                continue
            prompt = self.author.build_prompt(self.questions, [self.candidates[i] for i in reversed(pair)],
                                              key, PROMPT_OPTIONS)
            prompt += '\nSolve the final question. End on its own line with The answer is followed by the answer.'
            model = self.context.roles['actor']
            text = self.calls.text(GenerationRequest('actor', model, prompt,
                                   temperature=0, top_p=1, max_output_tokens=self.config['max_output_tokens']))
            self.cache[cache_key] = score_answer(text, self.questions[key], self.scorer)
        return [self.cache[key, tuple(pair)] for key, pair in zip(ids, pairs, strict=True)]

    def train_batch(self, request):
        import torch

        ids = request['ids']
        if self.restored or not ids or len(set(ids)) != len(ids) or not set(ids) <= self.train_ids:
            raise ValueError('Invalid PromptPG training batch')
        x = torch.stack([self.encodings[key] for key in ids])
        scores = self.selector(x) @ self.selector(self.candidate_vectors).T
        log_probs = scores.log_softmax(dim=1)
        pairs = torch.multinomial(log_probs.detach().exp(), 2, replacement=False, generator=self.generator)
        passed = self.score(ids, pairs.tolist())
        rewards = torch.tensor([1 if item else -1 for item in passed])
        loss = -(log_probs.gather(1, pairs).sum(1) * rewards).sum()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite PromptPG policy loss')
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'correct': sum(passed), 'training_metrics': {'loss': float(loss.detach()), 'correct': sum(passed)}}, ()

    def save(self, request):
        import torch

        if self.restored:
            raise ValueError('Cannot select a checkpoint after entering held-out evaluation')
        state = {name: value.detach().clone() for name, value in self.selector.state_dict().items()}
        self.best = request['epoch'], state
        buffer = BytesIO()
        torch.save(state, buffer)
        return {}, (self.context.artifacts.put(buffer.getvalue(), kind='promptpg-selector'),)

    def restore(self, request):
        import torch

        if self.best is None or self.best[0] != request['epoch'] or self.restored:
            raise ValueError('No matching selected selector')
        if all(torch.equal(value, self.initial[name]) for name, value in self.best[1].items()):
            raise RuntimeError('The selected PromptPG weights did not change')
        self.selector.load_state_dict(self.best[1])
        self.restored = True
        return {}, ()

    def evaluate(self, request):
        import torch

        ids, arm = request['ids'], request['arm']
        if (not self.restored or not ids or len(set(ids)) != len(ids) or not set(ids) <= self.test_ids
                or arm not in ('trained', 'random', 'nearest')):
            raise ValueError('Invalid PromptPG held-out request')
        x = torch.stack([self.encodings[key] for key in ids])
        with torch.no_grad():
            if arm == 'trained':
                scores = self.selector(x) @ self.selector(self.candidate_vectors).T
            else:
                normalize = torch.nn.functional.normalize
                scores = normalize(x) @ normalize(self.candidate_vectors).T
        pairs = choice_pairs(scores.tolist(), arm, self.rng)
        passed = self.score(ids, pairs)
        evidence = [{'id': key, 'candidate_indices': pair, 'correct': correct}
                    for key, pair, correct in zip(ids, pairs, passed, strict=True)]
        artifact = self.context.artifacts.put(json.dumps(evidence).encode(), kind='promptpg-heldout')
        return {'correct': sum(passed), 'quality_metrics': {
            'accuracy': sum(passed) / len(ids), 'examples': len(ids)}}, (artifact,)
