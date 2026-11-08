"""SEAL general-knowledge adaptation (Zweiger et al., 2025).

Port of the open-source top-k self-edit / TTT / outer-SFT pipeline. The
local profile uses deterministic SQuAD token F1 instead of a paid GPT grader.
Prompts and sequence construction derive from upstream; see SEAL_NOTICE.txt.
"""
import math
import re
import string
from collections import Counter

from .weight_artifacts import WeightArtifacts
from .weight_engine import validate_engine

EDIT_PROMPT = ("Let's read the following passage and produce a list of implications "
               "derived directly or indirectly from the content.\n\nPassage:\n{title}\n{context}\n\nImplications:\n")
ANSWER_PROMPT = "Let's answer a question directly and concisely.\nQuestion: Topic: {title}\n{question}\nAnswer:\n"


def train_sequences(completion, context, title):
    marker = re.search(r'\nImplications:\s*', completion)
    if marker:
        completion = completion[marker.end():].lstrip()
    segments = [part.strip() for part in completion.split('---') if part.strip()] or [completion.strip()]
    return [title + '\n' + part for part in segments[:30] if part.strip()] + [title + '\n' + context.strip()]


def normalize_answer(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def squad_scores(prediction, gold):
    p, g = normalize_answer(prediction).split(), normalize_answer(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    f1 = 2 * common / (len(p) + len(g)) if p and g else float(p == g)
    return {'exact_match': float(p == g), 'f1': f1}


def select_edits(rows):
    selected = {}
    for row in rows:
        prior = selected.get(row['doc_index'])
        if prior is None or row['adapter_mean'] > prior['adapter_mean']:
            selected[row['doc_index']] = row
    return list(selected.values())


def validate_config(config):
    allowed = {'data_root', 'gpu_ids', 'inference_python', 'training_python', 'worker_timeout',
               'seed', 'max_length', 'max_tokens', 'lora_rank', 'inner_rank', 'inner_epochs',
               'outer_epochs', 'batch_size', 'inner_lr', 'outer_lr', 'gradient_checkpointing',
               'temperature', 'enforce_eager', 'inference_cuda_home', 'training_cuda_home'}
    if set(config) - allowed:
        raise ValueError('Unknown SEAL Lab option')
    validate_engine(config)


class SEALLab(WeightArtifacts):
    method = 'seal'
    engine_methods = ('generate', 'adapt', 'train')
    fields = {'evaluate': {'dataset': 'string'},
              'generate': {'dataset': 'string', 'candidates': 'integer'},
              'adapt': {'dataset': 'string'}, 'select': {'dataset': 'string'},
              'train': {'dataset': 'string'}}

    def __init__(self, context, config, *, scorer=None, scorer_name=None, **kwargs):
        validate_config(config)
        if scorer is not None and (not callable(scorer) or not isinstance(scorer_name, str) or not scorer_name.strip()):
            raise TypeError('Custom SEAL scorer must be callable with an explicit name/version')
        self.scorer = scorer or squad_scores
        self.scorer_name = scorer_name or 'squad-em-f1-v1'
        super().__init__(context, config, **kwargs)
        self.eval_contexts = self.eval_ref = None

    def _documents(self, ref):
        rows = self.read(ref)
        if not isinstance(rows, list) or not rows or any(
                not isinstance(r, dict) or any(not isinstance(r.get(k), str) or not r[k].strip()
                for k in ('title', 'context')) or not isinstance(r.get('questions'), list) or not r['questions']
                or any(not isinstance(q, dict) or any(not isinstance(q.get(k), str) or not q[k].strip()
                for k in ('question', 'answer')) for q in r['questions']) for r in rows):
            raise ValueError('Expected nonempty SQuAD document list with questions')
        if len({r['context'].strip() for r in rows}) != len(rows):
            raise ValueError('Duplicate documents')
        return rows

    def _generate(self, args):
        docs = self._documents(args['dataset'])
        if self.eval_contexts is None or self.eval_contexts & {r['context'].strip() for r in docs}:
            raise ValueError('Missing baseline or overlapping learning/evaluation documents')
        prompts = [EDIT_PROMPT.format(**doc) for doc in docs for _ in range(args['candidates'])]
        generated = self.engine.generate(str(self.path(args['model'])), prompts,
            self.root / f'round-{args["round"]}' / 'generation-workers', sample=True, seed_offset=args['round'])
        rows = [{**doc, 'doc_index': i, 'candidate': k, 'prompt': prompts[i * args['candidates'] + k],
                 'completion': generated[i * args['candidates'] + k]['completion']}
                for i, doc in enumerate(docs) for k in range(args['candidates'])]
        return self.write('generate', args['round'], rows)

    def _episodes(self, rows):
        return [{'index': i, 'sequences': train_sequences(r['completion'], r['context'], r['title']),
                 'questions': [{'prompt': ANSWER_PROMPT.format(title=r['title'], **q)} for q in r['questions']]}
                for i, r in enumerate(rows)]

    def _score_adaptations(self, rows, responses):
        output = []
        for row, response in zip(rows, responses, strict=True):
            if not response['weights_changed'] or response['steps'] < 1:
                raise ValueError('Inner SFT did not update parameters')
            questions = row['questions']
            scores = [self.scorer(p, q['answer']) for p, q in zip(response['predictions'], questions, strict=True)]
            baseline = [self.scorer(p, q['answer']) for p, q in zip(response['baseline'], questions, strict=True)]
            if any(not isinstance(s, dict) or set(s) != {'f1', 'exact_match'} or any(
                    type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in s.values())
                    for s in scores + baseline):
                raise ValueError('SEAL scorer must return finite f1/exact_match values between zero and one')
            output.append({**row, 'inner': response, 'scores': scores, 'baseline_scores': baseline,
                           'adapter_mean': sum(s['f1'] for s in scores) / len(scores)})
        return output

    def _adapt(self, args):
        rows = self.preceding(args['dataset'], 'generate', args['round'])
        responses = self.engine.adapt(str(self.path(args['model'])), self._episodes(rows),
            self.root / f'round-{args["round"]}' / 'adaptation-workers', seed_offset=args['round'])
        return {**self.write('adapt', args['round'], self._score_adaptations(rows, responses)), 'scorer': self.scorer_name}

    def _select(self, args):
        rows = self.preceding(args['dataset'], 'adapt', args['round'])
        # Stable top-1, including zero-reward ties, matches upstream top-k.
        accepted = [{'prompt': r['prompt'], 'completion': r['completion'],
                     'doc_index': r['doc_index'], 'candidate': r['candidate'], 'reward': r['adapter_mean']}
                    for r in select_edits(rows) if r['completion'].strip()]
        return self.write('select', args['round'], accepted)

    def _train(self, args):
        rows = self.preceding(args['dataset'], 'select', args['round'])
        return self.train_checkpoint(args, rows)

    def _evaluate(self, args):
        docs = self._documents(args['dataset'])
        ref = (str(self.path(args['dataset'])), self.inputs[str(self.path(args['dataset']))])
        if args['round'] == 0:
            self.eval_contexts = {r['context'].strip() for r in docs}
            self.eval_ref = ref
        elif ref != self.eval_ref:
            raise ValueError('Independent evaluation data changed')
        prompts = [EDIT_PROMPT.format(**doc) for doc in docs]
        outputs = self.engine.generate(str(self.path(args['model'])), prompts,
            self.root / f'round-{args["round"]}' / 'eval-generation-workers', sample=True, seed_offset=10000)
        rows = [{**doc, 'completion': output['completion']} for doc, output in zip(docs, outputs, strict=True)]
        responses = self.engine.adapt(str(self.path(args['model'])), self._episodes(rows),
            self.root / f'round-{args["round"]}' / 'eval-adaptation-workers', seed_offset=10000)
        measured = self._score_adaptations(rows, responses)
        scores = [s for r in measured for s in r['scores']]
        baseline = [s for r in measured for s in r['baseline_scores']]
        evidence = self.write('evaluate', args['round'], measured)
        return {'quality_metrics': {'adapted_f1': sum(s['f1'] for s in scores) / len(scores),
                'adapted_em': sum(s['exact_match'] for s in scores) / len(scores),
                'no_adapt_f1': sum(s['f1'] for s in baseline) / len(baseline),
                'questions': len(scores), 'documents': len(rows)}, 'evidence': evidence, 'scorer': self.scorer_name}
