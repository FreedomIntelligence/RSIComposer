"""Runtimes for math benchmarks: how a domain agent answers a batch of problems.

The domain agent is what the Lab measures; the RSI policy in the Controller submits
and revises it. A runtime is named in the Lab's wiring (never in the domain agent's
own data) and run by MathLab.solve(). Everything that changes what a score means
(temperature, token limits, seed policy, trials, timeouts, call budgets) is a field of
the runtime, and the runtime's `settings` is exactly what is recorded in each
benchmark's protocol: the recorded protocol and the executed one are the same object.
Name, version, model access and accepted domain agent shape are recorded with it, and the
domain agent's content hash goes on every receipt.

    instruction   One actor call per problem; the domain agent is the instruction text.
    step-program  A sandboxed step(problem, history) program (the domain agent's own
                  harness, not the Controller's) asks the Lab to call a fixed model pool
                  until it returns an answer; the domain agent is the program's source file.
    checkpoint    A trained checkpoint generates through an engine; the domain agent is the
                  checkpoint reference.
    prompt-template  A method's own prompt template, rendered around one instruction and
                  answered by the actor under a call budget (GEPA, OPRO).

run(domain_agent, problems, run) takes a batch because engines and trial pools work in
batches. `run` is a Run: this call's own context and the resources the Lab lends it, so
nothing about one evaluation is stored on the Lab and two evaluations cannot see each
other. A runtime never reaches the host's artifacts, graders or other splits.
"""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, field
from hashlib import sha256
import subprocess
import time

from rsisys.loops.contracts.benchmark import canonical_json
from rsisys.loops.contracts.domain_agent import DomainAgentRejected, DomainAgentSpec
from rsisys.loops.contracts.execution import LabToolError

from .attempt import Attempt


@dataclass(frozen=True)
class Run:
    """One call to a runtime: its own context, plus what the Lab lends.

    Built per evaluation and passed down, never stored on the Lab, so two evaluations in
    flight cannot see each other's request, round, directory or trial count.
    """
    ask: object                  # ask(problem, system, *, seed, temperature, max_output_tokens) -> Attempt
    request: dict                # The evaluate arguments of this call.
    trials: int = 1              # Independent trials per problem, for runtimes that repeat.
    context: dict = field(default_factory=dict)   # Lab-specific resources: pool, executor, engine hooks.


def _settings(**values):
    return values


@dataclass(frozen=True)
class InstructionRuntime:
    """One actor call per problem. temperature and max_output_tokens belong to the protocol."""
    temperature: float = 0.0
    max_output_tokens: int | None = None   # None: each benchmark's own budget (its source entry).
    name: str = 'instruction'
    version: str = '2'
    model_access: str = 'lab-mediated'
    accepts: DomainAgentSpec = DomainAgentSpec('instruction', {'instruction': {'type': 'string', 'required': True}}, 'data')

    @property
    def settings(self):
        return _settings(temperature=self.temperature, max_output_tokens=self.max_output_tokens)

    def run(self, domain_agent, problems, run):
        instruction = domain_agent['instruction']
        if not isinstance(instruction, str) or not instruction.strip():
            raise LabToolError('instruction must be nonempty text')
        return [run.ask(problem, instruction, seed=None, temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens) for problem in problems]


@dataclass(frozen=True)
class PromptTemplateRuntime:
    """A method's own prompt template around one instruction, answered by the actor under a call budget.

    GEPA and OPRO use this so their prompts stay identical to the pinned upstream while the
    parameters that decide scores are still part of the recorded protocol.
    """
    template: str                     # Names the upstream prompt format, e.g. 'gepa-instruction/v1'.
    system: str = ''
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = 0
    max_output_tokens: int | None = None
    max_model_calls: int | None = None
    name: str = 'prompt-template'
    version: str = '1'
    model_access: str = 'lab-mediated'
    accepts: DomainAgentSpec = DomainAgentSpec('instruction', {'instruction': {'type': 'string', 'required': True}}, 'data')
    render: object = None             # render(problem, instruction, request) -> prompt; named by template

    @property
    def settings(self):
        return _settings(template=self.template, system=self.system, temperature=self.temperature, top_p=self.top_p,
                         seed=self.seed, max_output_tokens=self.max_output_tokens, max_model_calls=self.max_model_calls)

    def run(self, domain_agent, problems, run):
        instruction = domain_agent['instruction']
        attempts = []
        for problem in problems:
            prompt = self.render(problem, instruction, run.request)
            attempts.append(run.ask(problem, prompt, system=self.system, seed=self.seed, temperature=self.temperature,
                                    top_p=self.top_p, max_output_tokens=self.max_output_tokens, raw_prompt=True))
        return attempts


@dataclass(frozen=True)
class StepProgramRuntime:
    """A domain agent harness program stepped in a sandbox, with model calls made by the Lab on its behalf."""
    max_calls: int = 3
    seed: int = 73
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 180
    trials: int = 2                  # Default independent trials per problem; a call may ask for another count.
    sandbox_timeout: int = 5
    name: str = 'step-program'
    version: str = '3'
    model_access: str = 'lab-mediated'
    accepts: DomainAgentSpec = DomainAgentSpec('program', {'harness': {'type': 'string', 'required': True}}, 'sandboxed-code')

    @property
    def settings(self):
        return _settings(max_calls=self.max_calls, seed=self.seed, temperature=self.temperature,
                         max_tokens=self.max_tokens, timeout=self.timeout, trials=self.trials,
                         sandbox_timeout=self.sandbox_timeout)

    def run(self, domain_agent, problems, run):
        """run.context provides pool, executor, sandbox_slots, domain_agent_errors and workers."""
        trials = run.trials
        with ThreadPoolExecutor(max_workers=run.context['workers']) as pool:
            # Capture a separate copy in the caller for each worker's model calls.
            futures = [pool.submit(copy_context().run, self.trial, run, domain_agent['harness'],
                                   k % trials, problem)
                       for k, problem in enumerate(problems)]
            return [future.result() for future in futures]

    def trial(self, run, harness, trial, problem):
        context = run.context
        history, actions, covered = [], [], set()
        error, answer = None, ''
        started = time.monotonic()
        try:
            for step in range(self.max_calls + 1):
                with context['sandbox_slots']:
                    result = context['executor'](harness, problem.prompt, history, timeout=self.sandbox_timeout)
                covered.update(result['covered_lines'])
                action = result['action']
                actions.append(action)
                if 'answer' in action:
                    answer = action['answer']
                    break
                if step == self.max_calls:
                    answer = history[-1]['content'] if history else ''
                    break
                key = canonical_json([self.seed, problem.benchmark, problem.id, trial, step])
                seed = int.from_bytes(sha256(key.encode()).digest()[:4], 'big')
                history.append(context['pool'].complete(action['messages'], seed=seed))
        except (*context['domain_agent_errors'], subprocess.TimeoutExpired) as exc:
            # A domain agent crash is an observed failed trial; transport and verifier failures propagate.
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
    version: str = '2'
    model_access: str = 'engine'
    engine: str = 'unspecified'       # The engine's own identity (name/version), recorded in the protocol.
    temperature: float = 0.0
    max_new_tokens: int | None = None
    accepts: DomainAgentSpec = DomainAgentSpec('checkpoint', {'model': {'type': 'string', 'required': True}}, 'data')

    @property
    def settings(self):
        return _settings(engine=self.engine, temperature=self.temperature, max_new_tokens=self.max_new_tokens)

    def run(self, domain_agent, problems, run):
        """run.context['generate'](model, problems) -> one completion per problem; run.context['simulated']."""
        return [Attempt(text, simulated=run.context.get('simulated', False))
                for text in run.context['generate'](domain_agent['model'], problems)]


RUNTIMES = {
    'instruction': lambda options: InstructionRuntime(**_only(options, {'temperature', 'max_output_tokens'}, 'instruction')),
    'step-program': lambda options: StepProgramRuntime(**_only(
        options, {'max_calls', 'seed', 'temperature', 'max_tokens', 'timeout', 'trials', 'sandbox_timeout'},
        'step-program')),
    'checkpoint': lambda options: CheckpointRuntime(**_only(options, {'engine', 'temperature', 'max_new_tokens'}, 'checkpoint')),
}


def _only(options, allowed, name):
    if set(options) - allowed:
        raise ValueError(f'{name} accepts only {sorted(allowed)}')
    return options


__all__ = ['RUNTIMES', 'Run', 'InstructionRuntime', 'PromptTemplateRuntime', 'StepProgramRuntime',
           'CheckpointRuntime', 'DomainAgentRejected']
