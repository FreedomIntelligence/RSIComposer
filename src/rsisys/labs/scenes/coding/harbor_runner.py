"""Run one real Harbor trial. The parent Lab owns scheduling and recovery."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

from .agents import HarborCodingAgent, agent_configuration
from rsisys.runtime.storage.files import atomic_write, json_bytes


def atomic_json(path, data):
    atomic_write(Path(path), json_bytes(data))


def is_measured(rewards, error):
    # Harbor still grades an exhausted total agent time budget. It is a real
    # model failure, not an infrastructure retry. Other exceptions stay
    # incomplete even if a fallback verifier happened to produce zero.
    budget_exhausted = (isinstance(error, dict)
                        and error.get('exception_type') == 'AgentTimeoutError'
                        and (error.get('policy') or {}).get('has_valid_reward') is True)
    return bool(rewards) and (error is None or budget_exhausted)


async def run_one(args):
    spec = (json.loads(Path(args.agent_spec).read_text()) if args.agent_spec
            else HarborCodingAgent().to_json())
    spec = HarborCodingAgent(**spec).to_json()
    artifact = str(Path(args.artifact).resolve(strict=True))
    agent, paths = agent_configuration(spec, {
        'artifact': artifact, 'artifact_dir': str(Path(artifact) if Path(artifact).is_dir() else Path(artifact).parent),
        'endpoint': args.endpoint, 'model': args.model, 'max_steps': args.max_steps,
        'max_output_tokens': args.max_tokens, 'total_output_tokens': args.total_tokens,
        'seed': args.seed, 'thinking': args.thinking, 'agent_timeout': args.agent_timeout})
    sys.path[:0] = paths
    # Optional dependencies load only in this deployed task worker.
    from harbor.models.trial.config import TrialConfig
    from harbor.trial.trial import Trial
    config = TrialConfig(
        task={'path': str(Path(args.task).resolve())},
        trial_name=args.trial_name, trials_dir=Path(args.output).resolve(),
        agent=agent, environment={'type': 'docker', 'delete': True},
        verifier={'max_timeout_sec': args.verifier_timeout})
    trial = await Trial.create(config)
    result = await trial.run()
    raw = result.model_dump(mode='json')
    atomic_json(Path(args.output) / args.trial_name / 'official-result.json', raw)
    verifier = raw.get('verifier_result') or {}
    rewards = verifier.get('rewards')
    error = raw.get('exception_info')
    # A real task failure is zero; missing grades or infrastructure errors are
    # not silently converted into a measured zero or excluded from denominators.
    measured = is_measured(rewards, error)
    score = None
    if measured:
        values = list(rewards.values())
        if not all(type(x) in (int, float) and 0 <= x <= 1 for x in values):
            raise ValueError('Expected finite binary/bounded benchmark rewards')
        score = sum(values) / len(values)
    receipt = {'task': args.task, 'trial_name': args.trial_name,
               'measured': measured, 'score': score, 'rewards': rewards,
               'exception': error, 'official_result': str(Path(args.output).resolve() /
                   args.trial_name / 'official-result.json'),
               'agent_result': raw.get('agent_result'),
               'task_checksum': raw.get('task_checksum')}
    atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    for name in ('task', 'endpoint', 'output', 'trial-name', 'receipt'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--artifact', '--candidate', dest='artifact', required=True)
    parser.add_argument('--agent-spec')
    parser.add_argument('--model', required=True)
    parser.add_argument('--agent-timeout', type=int, default=1800)
    parser.add_argument('--verifier-timeout', type=int, default=1800)
    parser.add_argument('--max-steps', type=int, default=48)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--total-tokens', type=int, default=98304)
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--thinking', action='store_true')
    asyncio.run(run_one(parser.parse_args()))


if __name__ == '__main__':
    main()
