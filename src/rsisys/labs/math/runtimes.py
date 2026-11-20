"""Runtimes for math benchmarks: how a candidate answers a batch of problems.

A runtime is named in the Lab's wiring (never in candidate data) and run by
MathLab.solve(). Its identity (name, version, model access, accepted candidate
shape) is part of each benchmark's protocol, and the candidate's content hash is
recorded on every receipt.

    instruction   One actor call per problem; the candidate is the instruction text.
    step-program  A sandboxed step(problem, history) program asks the Lab to call a
                  fixed model pool until it returns an answer; the candidate is the
                  program's source file.
    checkpoint    A trained checkpoint generates through an engine; the candidate is the
                  checkpoint reference.

run(candidate, problems, host) takes a batch because engines and trial pools work in
batches. `host` is what the Lab lends the runtime: its model access and budgets. A
runtime never reaches the host's artifacts, graders or other splits.
"""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
import subprocess
import time

from rsisys.loops.contracts.agent_runtime import CandidateRejected, CandidateSpec
from rsisys.loops.contracts.execution import LabToolError

from .lab import Attempt


@dataclass(frozen=True)
class InstructionRuntime:
    name: str = 'instruction'
    version: str = '1'
    model_access: str = 'lab-mediated'
    accepts: CandidateSpec = CandidateSpec('instruction', {'instruction': {'type': 'string', 'required': True}}, 'data')

    def run(self, candidate, problems, host):
        instruction = candidate['instruction']
        if not isinstance(instruction, str) or not instruction.strip():
            raise LabToolError('instruction must be nonempty text')
        return [host.ask_actor(problem, instruction) for problem in problems]


@dataclass(frozen=True)
class StepProgramRuntime:
    """A harness program stepped in a sandbox, with model calls made by the Lab on its behalf."""
    max_calls: int = 3
    name: str = 'step-program'
    version: str = '1'
    model_access: str = 'lab-mediated'
    accepts: CandidateSpec = CandidateSpec('program', {'harness': {'type': 'string', 'required': True}}, 'sandboxed-code')

    def run(self, candidate, problems, host):
        """host provides pool, executor, sandbox_slots, candidate_errors, seed, trials_per_problem and workers (FixedModelHarnessLab)."""
        trials = host.trials_per_problem
        with ThreadPoolExecutor(max_workers=host.workers) as pool:
            return list(pool.map(lambda item: copy_context().run(self.trial, host, candidate['harness'], *item),
                                 [(k // trials, k % trials, problem) for k, problem in enumerate(problems)]))

    def trial(self, host, harness, index, trial, problem):
        history, actions, covered = [], [], set()
        error, answer = None, ''
        started = time.monotonic()
        try:
            for step in range(self.max_calls + 1):
                with host.sandbox_slots:
                    result = host.executor(harness, problem.prompt, history)
                covered.update(result['covered_lines'])
                action = result['action']
                actions.append(action)
                if 'answer' in action:
                    answer = action['answer']
                    break
                if step == self.max_calls:
                    answer = history[-1]['content'] if history else ''
                    break
                history.append(host.pool.complete(action['messages'], seed=host.seed + index * 100 + trial * 10 + step))
        except (*host.candidate_errors, subprocess.TimeoutExpired) as exc:
            # A candidate crash is an observed failed trial; transport and verifier failures propagate.
            error, answer = str(exc), ''
        usage = {name: sum(reply[key] for reply in history)
                 for name, key in (('input_tokens', 'prompt_tokens'), ('output_tokens', 'completion_tokens'))}
        return Attempt(answer, usage=usage, elapsed_seconds=round(time.monotonic() - started, 6), extra={
            'trial': trial, 'tokens': sum(reply['tokens'] for reply in history), 'calls': len(history),
            'actions': actions, 'history': history, 'covered_lines': sorted(covered), 'error': error})


@dataclass(frozen=True)
class CheckpointRuntime:
    """A trained checkpoint, generated from by an engine the Lab owns."""
    name: str = 'checkpoint'
    version: str = '1'
    model_access: str = 'engine'
    accepts: CandidateSpec = CandidateSpec('checkpoint', {'model': {'type': 'string', 'required': True}}, 'data')

    def run(self, candidate, problems, host):
        """host.generate(model, problems) -> one completion per problem."""
        return [Attempt(text, simulated=host.simulated) for text in host.generate(candidate['model'], problems)]


RUNTIMES = {
    'instruction': lambda options: _no_options(InstructionRuntime, options),
    'step-program': lambda options: StepProgramRuntime(**_only(options, {'max_calls'}, 'step-program')),
    'checkpoint': lambda options: _no_options(CheckpointRuntime, options),
}


def _only(options, allowed, name):
    if set(options) - allowed:
        raise ValueError(f'{name} accepts only {sorted(allowed)}')
    return options


def _no_options(factory, options):
    if options:
        raise ValueError(f'{factory().name} takes no options')
    return factory()


__all__ = ['RUNTIMES', 'InstructionRuntime', 'StepProgramRuntime', 'CheckpointRuntime', 'CandidateRejected']
