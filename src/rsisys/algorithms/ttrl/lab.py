"""Unlabeled majority-vote rewards and on-policy GRPO for TTRL.

Target-set accuracy is transductive, as in upstream TTRL. Disjoint transfer
accuracy is reported separately. Gold answers never enter rollout/reward/train.
Independent reference implementation of https://arxiv.org/abs/2504.16084.
Upstream reference (Apache-2.0):
https://github.com/PRIME-RL/TTRL/tree/5806e119789132fcb43552beaf2532d7d0447213
"""
import math
from collections import Counter
from statistics import mean, stdev

from rsisys.labs.math_verifiers import MathVerifyVerifier, final_boxed
from rsisys.labs.weight_artifacts import WeightArtifacts
from rsisys.labs.weight_engine import validate_engine


def math_prompt(question):
    return question + '\nPlease reason step by step, and put your final answer within \\boxed{}.'


def majority_rewards(completions, verifier):
    """Stable plurality, invalid-answer zero reward; no reference-answer input."""
    if len(completions) < 2:
        raise ValueError('A GRPO group requires multiple responses')
    official = getattr(verifier, 'rewards', None)
    if callable(official):
        pseudo, ratio, rewards = official(completions)
        avg, std = mean(rewards), stdev(rewards)
        return {'pseudo_answer': pseudo, 'majority_ratio': ratio, 'rewards': rewards,
                'advantages': [(r - avg) / (std + 1e-6) for r in rewards]}
    canonical, representatives = [], {}
    for completion in completions:
        parsed = verifier.parse(completion)
        if parsed is None:
            canonical.append(None)
            continue
        # Math-Verify already parses LaTeX without Python eval. Canonical SymPy
        # objects give equivalent rational/LaTeX spellings the same vote key.
        key = str(parsed[0] if isinstance(parsed, (list, tuple)) else parsed)
        canonical.append(key)
        representatives.setdefault(key, final_boxed(completion))
    counts = Counter(value for value in canonical if value is not None)
    if not counts:
        return {'pseudo_answer': None, 'majority_ratio': 0.0,
                'rewards': [0.0] * len(completions), 'advantages': [0.0] * len(completions)}
    key, votes = counts.most_common(1)[0]
    pseudo = representatives[key]
    rewards = [float(verifier.is_correct(completion, pseudo)) for completion in completions]
    avg, std = mean(rewards), stdev(rewards)
    return {'pseudo_answer': pseudo, 'majority_ratio': votes / len(completions),
            'rewards': rewards, 'advantages': [(r - avg) / (std + 1e-6) for r in rewards]}


def validate_config(config):
    allowed = {'data_root', 'output_subdir', 'gpu_ids', 'inference_python', 'training_python', 'worker_timeout',
               'seed', 'max_length', 'max_tokens', 'lora_rank', 'batch_size', 'rl_lr', 'kl_beta',
               'gradient_checkpointing', 'temperature', 'enforce_eager', 'stop',
               'inference_cuda_home', 'training_cuda_home', 'eval_samples', 'evaluation_temperature',
               'evaluation_top_p', 'top_p', 'apply_chat_template'}
    if set(config) - allowed:
        raise ValueError('Unknown TTRL Lab option')
    validate_engine(config)
    if config.get('batch_size', 1) != 1:
        raise ValueError('GRPO currently requires one sequence per GPU microbatch')
    beta = config.get('kl_beta', 0.001)
    if type(beta) not in (int, float) or not math.isfinite(beta) or beta < 0:
        raise ValueError('kl_beta must be finite and nonnegative')
    temperature = config.get('temperature', 0.6)
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('RL sampling temperature must be positive')
    if type(config.get('eval_samples', 1)) is not int or config.get('eval_samples', 1) < 1:
        raise ValueError('eval_samples must be a positive integer')
    for key in ('top_p', 'evaluation_top_p'):
        value = config.get(key, 0.95)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f'{key} must be in (0, 1]')
    value = config.get('evaluation_temperature', 0.6)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('evaluation_temperature must be positive')
    if type(config.get('apply_chat_template', False)) is not bool:
        raise ValueError('apply_chat_template must be boolean')


class TTRLLab(WeightArtifacts):
    method = 'ttrl'
    fields = {'evaluate': {'dataset': 'string'},
              'rollout': {'dataset': 'string', 'samples_per_prompt': 'integer'},
              'reward': {'dataset': 'string'}, 'train': {'dataset': 'string'}}

    def __init__(self, context, config, *, verifier=None, **kwargs):
        self.validate_config(config)
        self.verifier = MathVerifyVerifier() if verifier is None else verifier
        if (not isinstance(getattr(self.verifier, 'name', None), str) or not self.verifier.name.strip()
                or any(not callable(getattr(self.verifier, k, None)) for k in ('parse', 'is_correct'))):
            raise TypeError('TTRL verifier must declare a name/version and parse/is_correct')
        super().__init__(context, config, **kwargs)
        self.target_prompts = self.transfer_prompts = self.eval_ref = None

    validate_config = staticmethod(validate_config)

    def _rollout(self, args):
        rows = self.read(args['dataset'])
        if not isinstance(rows, list) or not rows or any(
                not isinstance(r, dict) or set(r) != {'id', 'prompt'} or
                any(not isinstance(r[k], str) or not r[k].strip() for k in ('id', 'prompt')) for r in rows):
            raise ValueError('TTRL learning file must contain only id and prompt; gold labels are forbidden')
        prompts = {r['prompt'].strip() for r in rows}
        if len(prompts) != len(rows) or len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate adaptation questions')
        if self.target_prompts is None or prompts != self.target_prompts or prompts & self.transfer_prompts:
            raise ValueError('Adaptation must exactly match target questions and exclude transfer questions')
        texts = [math_prompt(r['prompt']) for r in rows for _ in range(args['samples_per_prompt'])]
        replies = self.engine.generate(str(self.path(args['model'])), texts,
            self.root / f'round-{args["round"]}' / 'rollout-workers', sample=True, seed_offset=args['round'])
        output = [{'id': r['id'], 'prompt': math_prompt(r['prompt']),
                   'responses': replies[i * args['samples_per_prompt']:(i + 1) * args['samples_per_prompt']]}
                  for i, r in enumerate(rows)]
        return self.write('rollout', args['round'], output)

    def _reward(self, args):
        rows = self.preceding(args['dataset'], 'rollout', args['round'])
        rewarded = []
        group_stats = []
        for group in rows:
            result = majority_rewards([r['completion'] for r in group['responses']], self.verifier)
            group_stats.append({'id': group['id'], **result})
            for response, reward, advantage in zip(group['responses'], result['rewards'], result['advantages'], strict=True):
                rewarded.append({'prompt': group['prompt'], 'completion': response['completion'],
                    'response_ids': response['response_ids'], 'reward': reward, 'advantage': advantage,
                    'group_id': group['id']})
        evidence = self.write('reward', args['round'], rewarded)
        groups = self.write('reward-groups', args['round'], group_stats)
        return {**evidence, 'groups': groups, 'reward_source': 'model_majority_without_gold',
                'verifier': self.verifier.name,
                'nonzero_advantages': sum(abs(r['advantage']) > 1e-12 for r in rewarded)}

    def _train(self, args):
        rows = self.preceding(args['dataset'], 'reward', args['round'])
        if not any(abs(r['advantage']) > 1e-12 for r in rows):
            raise ValueError('No learning signal: every group has constant rewards')
        return self.train_checkpoint(args, rows, algorithm='grpo')

    def _evaluate(self, args):
        data = self.read(args['dataset'])
        if not isinstance(data, dict) or set(data) != {'target', 'transfer'}:
            raise ValueError('Evaluation file must separately declare target and disjoint transfer questions')
        for rows in data.values():
            if not isinstance(rows, list) or (not rows and (rows is data['target'] or not getattr(self, 'allow_empty_transfer', False))) or any(not isinstance(r, dict) or
                    any(not isinstance(r.get(k), str) or not r[k].strip() for k in ('id', 'prompt', 'answer'))
                    for r in rows):
                raise ValueError('Malformed evaluation data')
        target = {r['prompt'].strip() for r in data['target']}
        transfer = {r['prompt'].strip() for r in data['transfer']}
        if target & transfer or len(target) != len(data['target']) or len(transfer) != len(data['transfer']):
            raise ValueError('Duplicate or overlapping target/transfer questions')
        ref = (str(self.path(args['dataset'])), self.inputs[str(self.path(args['dataset']))])
        if args['round'] == 0:
            self.target_prompts, self.transfer_prompts, self.eval_ref = target, transfer, ref
        elif ref != self.eval_ref:
            raise ValueError('Evaluation data changed')
        rows = [r for split in ('target', 'transfer') for r in data[split]]
        count = self.config.get('eval_samples', 1)
        sampled = count > 1
        outputs = self.engine.generate(str(self.path(args['model'])),
            [math_prompt(r['prompt']) for r in rows for _ in range(count)],
            self.root / f'round-{args["round"]}' / 'evaluation-workers', sample=sampled, seed_offset=10000,
            **({'temperature': self.config.get('evaluation_temperature', 0.6),
                'top_p': self.config.get('evaluation_top_p', 0.95)} if sampled else {}))
        if len(outputs) != len(rows) * count:
            raise ValueError('Missing evaluation samples')
        scored = []
        for i, row in enumerate(rows):
            for sample, output in enumerate(outputs[i * count:(i + 1) * count]):
                scored.append({'id': row['id'], 'sample': sample,
                    'split': 'target' if i < len(data['target']) else 'transfer',
                    'completion': output['completion'], 'correct': self.verifier.is_correct(output['completion'], row['answer'])})
        metrics = {}
        for split in ('target', 'transfer'):
            subset = [r for r in scored if r['split'] == split]
            if subset:
                metrics[split + '_accuracy'] = sum(r['correct'] for r in subset) / len(subset)
            metrics[split + '_questions'] = len(subset) // count
        metrics['samples_per_question'] = count
        return {'quality_metrics': metrics, 'evidence': self.write('evaluate', args['round'], scored),
                'verifier': self.verifier.name}
