"""Simulated parallel decision DAG, using CPU artifact work and no model service.

Run from RSISys with Python 3.12+::

    PYTHONPATH=src python3.12 examples/decision_graph.py --run-dir runs/decision-graph
    rsimonitor --runs-dir runs

The existing monitor UI can then preview this run. The policy is a scripted
fixture, not an autonomous researcher. Two actual worker threads overlap; all
candidate quality and execution semantics are deliberately toy/simulated.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from threading import Event, Lock
from time import monotonic_ns
from uuid import uuid4

if __package__:
    from .qwen_toy.monitor import Monitor
else:
    from qwen_toy.monitor import Monitor
from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.candidate import Candidate, SystemVersion
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.jobs import JobRef
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.decision_making.action_space import ModularDecisionActionSpace, OperationCall
from rsisys.loops.decision_making.action_space.capabilities import Capabilities
from rsisys.loops.decision_making.action_space.directives import (
    DirectiveDefinition, DirectiveDefinitions, DirectiveSpec,
)
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.tools import BoundDecisionTools
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.loops.exploration.action_space import EchoActionSpace
from rsisys.loops.exploration.harness import ConcurrentExplorationHarness
from rsisys.loops.exploration.harness.commands import CommandHandler
from rsisys.loops.exploration.harness.engine import DefaultExplorationHarness
from rsisys.loops.exploration.loop import ExplorationLoop
from rsisys.runtime.artifacts import FileArtifactStore
from rsisys.runtime.files import atomic_write, json_bytes
from rsisys.runtime.provenance import RunSource
from rsisys.runtime.runner import Runner
from rsisys.runtime.systems import ReferenceSystemActivator, SystemRegistry, system_adoption_capability


BASE = ModelRef('mock', 'artifact-actor', 'base')
OBJECTIVE = 'Simulated: parallel prompt/data candidates, confirmed adoption, then consume both components'
CANDIDATE_IDS = ('prompt-candidate', 'data-candidate')
VERSION = 'prompt-and-data-v1'
PROMPT = 'SIMULATED adopted prompt: total={total}; count={count}'
DATA = [3, 5, 8]


@dataclass(frozen=True)
class ComponentRequest:
    component: str
    parent_system: str


class ComponentWorkers:
    """Bounded real CPU threads; events make overlap deterministic without sleeps."""
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='simulated-component')
        self.started = {name: Event() for name in ('prompt', 'data')}
        self.release = Event()
        self.data_done = Event()
        self.data_received = Event()
        self.lock = Lock()
        self.futures, self.intervals = {}, {}

    def submit(self, request):
        component = request.component
        if component not in self.futures:
            self.futures[component] = self.pool.submit(self._produce, request)
        if not self.started[component].wait(5):
            raise RuntimeError(f'Simulated {component} worker did not start')
        return self.futures[component]

    def _produce(self, request):
        component = request.component
        with self.lock:
            self.intervals[component] = {'started_ns': monotonic_ns(), 'parent_system': request.parent_system}
        self.started[component].set()
        if not self.release.wait(5):
            raise RuntimeError('Both simulated workers must be running before release')
        # Complete data first to exercise feedback arriving out of dispatch order.
        if component == 'prompt' and not self.data_done.wait(5):
            raise RuntimeError('Simulated data worker did not finish')
        if component == 'prompt' and not self.data_received.wait(5):
            raise RuntimeError('Data feedback must be delivered before prompt completion')
        content = PROMPT.encode() if component == 'prompt' else json_bytes(DATA)
        artifact = self.artifacts.put(content, kind=component)
        candidate = Candidate(f'{component}-candidate', BASE.revision,
                              {component: artifact}, ('actor',), parent_system=request.parent_system)
        with self.lock:
            self.intervals[component]['finished_ns'] = monotonic_ns()
        if component == 'data':
            self.data_done.set()
        return candidate

    def close(self):
        self.release.set()
        self.data_received.set()
        self.pool.shutdown(wait=True, cancel_futures=True)


class ComponentCommand(CommandHandler):
    def __init__(self, workers):
        self.workers = workers

    def prepare(self, harness, state):
        request = state.directive.payload
        if not isinstance(request, ComponentRequest) or request.component not in ('prompt', 'data'):
            raise ValueError('Expected an explicit prompt/data component request')
        state.request = request

    def advance(self, harness, state, action_space):
        if state.cancel_requested:
            return Feedback(state.directive.directive_id, ExecutionStatus.CANCELLED,
                            'Simulated component proposal cancelled', True)
        future = self.workers.submit(state.request)
        state.job = JobRef(state.directive.directive_id, 'simulated-cpu-components')
        state.phase = 'running'
        if not future.done():
            return None
        candidate = future.result()
        return Feedback(state.directive.directive_id, ExecutionStatus.COMPLETED,
                        f'Simulated {state.request.component} candidate produced; not yet adopted', True,
                        artifact_refs=tuple(candidate.changes.values()), candidates=(candidate,), jobs=(state.job,))


class ConsumeCurrentSystem(CommandHandler):
    """Read the live registry here, rather than pass candidate artifacts in a directive."""
    def __init__(self, registry, artifacts, consumed):
        self.registry, self.artifacts, self.consumed = registry, artifacts, consumed

    def prepare(self, harness, state):
        if state.directive.payload is not None or state.directive.input_refs:
            raise ValueError('Consumption must resolve both components from registry.current')

    def advance(self, harness, state, action_space):
        if state.cancel_requested:
            return Feedback(state.directive.directive_id, ExecutionStatus.CANCELLED,
                            'Simulated consumption cancelled', True)
        current = self.registry.current
        prompt = self.artifacts.get(current.components['prompt']).decode()
        values = json.loads(self.artifacts.get(current.components['data']))
        rendered = prompt.format(total=sum(values), count=len(values))
        self.consumed.append({'version': current.version, 'components': asdict(current)['components'],
                              'prompt': prompt, 'data': values, 'output': rendered, 'simulated': True})
        output = self.artifacts.put(rendered.encode(), kind='simulated-output')
        return Feedback(state.directive.directive_id, ExecutionStatus.COMPLETED,
                        f'Simulated execution consumed registry.current {current.version}: {rendered}', True,
                        execution_metrics={'total': sum(values), 'count': len(values)}, artifact_refs=(output,))


class RecordingActivator(ReferenceSystemActivator):
    def __init__(self):
        super().__init__(BASE)
        self.calls = []

    def activate(self, system, *, idempotency_key):
        actual = super().activate(system, idempotency_key=idempotency_key)
        self.calls.append((deepcopy(actual), idempotency_key))
        return actual


class ScriptedGraphPolicy:
    """Use only explicit executor lineage from Context; never reconstruct ancestry."""
    model = BASE

    def __init__(self, registry, monitor):
        self.registry, self.monitor = registry, monitor
        self.producers = {}
        self.wait_for_producers = None
        self.adoption = self.consumption = None

    def decide(self, context):
        decision = self._choose(context)
        self.monitor.decision(decision, context, 1)
        return decision

    def _choose(self, context):
        for component in ('prompt', 'data'):
            if component not in self.producers:
                decision = Decision('explore', directive=Directive(
                    f'produce-{component}', context.objective, f'Propose {component} change',
                    mode='propose_component', payload=ComponentRequest(component, self.registry.current.version),
                    simulated=True), simulated=True,
                    research_attempt_title=f'Improve the {component} component independently',
                    reason=f'Improve {component} independently from the same parent; inspect its artifact feedback.')
                self.producers[component] = decision.decision_id
                return decision
        candidates = {c.candidate_id for c in context.candidates}
        graph = {row['decision_id']: row for row in context.decision_graph}
        if not set(CANDIDATE_IDS) <= candidates:
            if self.wait_for_producers is None:
                self.wait_for_producers = Decision('wait', simulated=True,
                    source_decision_ids=tuple(self.producers.values()),
                    research_attempt_title='Combine and verify both improvements',
                    reason='Both proposals must finish before joint adoption.')
            return self.wait_for_producers
        if self.adoption is None:
            # Missing integration is an error, never an excuse to infer producers
            # from dispatch order or fall back to manually manufactured edges.
            origins = tuple(context.candidate_origins[ident] for ident in CANDIDATE_IDS)
            if any(graph[ident]['status'] != 'completed' for ident in origins):
                raise RuntimeError('Both producer decisions must be completed')
            decision = Decision('invoke', operation=OperationCall('system_adopt', {
                'candidate_ids': list(CANDIDATE_IDS), 'version': VERSION,
                'idempotency_key': 'simulated-joint-adoption-v1'}), simulated=True,
                depends_on_decision_ids=origins, source_decision_ids=origins,
                **({'research_attempt_id': graph[self.wait_for_producers.decision_id]['research_attempt_id']}
                   if self.wait_for_producers else {'research_attempt_title': 'Combine and verify both improvements'}),
                reason='Both independent proposals completed; adopt prompt and data together and verify the manifest.')
            self.adoption = decision.decision_id
            return decision
        if self.consumption is None:
            adoption = graph[self.adoption]
            if (adoption['status'] != 'completed'
                    or set(adoption['adopted_candidate_ids']) != set(CANDIDATE_IDS)):
                raise RuntimeError('Consumption requires executor-confirmed adoption of both candidates')
            decision = Decision('explore', directive=Directive('consume-current', context.objective,
                'Verify adopted changes', mode='consume_current', simulated=True),
                simulated=True, depends_on_decision_ids=(self.adoption,), source_decision_ids=(self.adoption,),
                research_attempt_id=adoption['research_attempt_id'],
                reason='Joint adoption is confirmed; execute with both live components and check total and count.')
            self.consumption = decision.decision_id
            return decision
        if graph[self.consumption]['status'] == 'completed':
            return Decision('finish', simulated=True, depends_on_decision_ids=(self.consumption,),
                research_attempt_id=graph[self.consumption]['research_attempt_id'],
                reason='Simulated execution verified both adopted components; the fork/join chain is complete.')
        return Decision('wait', simulated=True, reason='Wait for the execution receipt before finishing.')


@dataclass
class GraphDemo:
    runner: Runner
    registry: SystemRegistry
    activator: RecordingActivator
    workers: ComponentWorkers
    policy: ScriptedGraphPolicy
    monitor: Monitor
    artifacts: FileArtifactStore
    children: dict
    consumed: list

    def close(self):
        self.workers.close()


def build_demo(run_dir):
    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'monitor.json').exists():
        raise FileExistsError(f'Choose a fresh run directory: {root}')
    source = RunSource(root.name)
    atomic_write(root / 'run.json', json_bytes(source.to_dict()))
    artifacts = FileArtifactStore(root / 'artifacts')
    initial = SystemVersion('initial', {
        'prompt': artifacts.put(b'BASE prompt: count={count}', kind='prompt'),
        'data': artifacts.put(json_bytes([1]), kind='data'),
    }, BASE)
    activator = RecordingActivator()
    registry = SystemRegistry(initial, activator)
    monitor = Monitor(root, BASE, OBJECTIVE, source=source)
    monitor.data.update(simulated=True, experiment_name='Simulated parallel prompt/data DAG',
                        project={'project_id': 'decision-graph', 'name': 'Decision graph examples'})
    workers = ComponentWorkers(artifacts)
    policy = ScriptedGraphPolicy(registry, monitor)
    experience = BasicExperienceSystem(InMemoryExperienceStore())
    children, consumed = {}, []

    def factory(directive):
        actor = type('SimulatedActor', (), {'model': BASE})()
        child = DefaultExplorationHarness(actor, object(), experience, command_handlers={
            'propose_component': ComponentCommand(workers),
            'consume_current': ConsumeCurrentSystem(registry, artifacts, consumed),
        })
        children[directive.directive_id] = child
        return child

    harness = DefaultDecisionHarness(policy, BoundDecisionTools({}), experience, max_outstanding=2)
    capabilities = Capabilities((system_adoption_capability(registry,
        candidates=lambda: tuple(harness.state.candidates.values()), simulated=True),))
    # Commands are explicit demo extensions; no core directive modes are changed.
    catalog = DirectiveDefinitions((
        DirectiveDefinition(DirectiveSpec('propose_component', 'Simulated component proposal', {
            'type': 'object', 'properties': {'component': {'enum': ['prompt', 'data']},
                'parent_system': {'type': 'string'}}, 'required': ['component', 'parent_system'],
            'additionalProperties': False}, accepts_input_refs=False), lambda args, candidates: ComponentRequest(**args)),
        DirectiveDefinition(DirectiveSpec('consume_current', 'Execute live registry components', {
            'type': 'object', 'properties': {}, 'additionalProperties': False}, accepts_input_refs=False),
            lambda args, candidates: None),
    ))
    harness.tools = BoundDecisionTools(capabilities.bindings)
    decision_loop = DecisionMakingLoop(harness, ModularDecisionActionSpace(capabilities.definitions, directives=catalog))
    exploration = ExplorationLoop(ConcurrentExplorationHarness(factory, max_in_flight=2), EchoActionSpace())
    return GraphDemo(Runner(decision_loop, exploration, parallel=True), registry, activator,
                     workers, policy, monitor, artifacts, children, consumed)


def publish_candidates(demo):
    """Candidate cards only; Monitor.decision/advance own all decision graph fields."""
    state = demo.runner.decision_making.harness.state
    records = state.decision_records
    adopted = {ident for record in records.values() for ident in record.get('adopted_candidate_ids', ())}
    for candidate in state.candidates.values():
        demo.monitor.candidates[candidate.candidate_id] = {
            'candidate_id': candidate.candidate_id,
            'status': 'adopted' if candidate.candidate_id in adopted else 'proposed',
            'base_revision': candidate.base_version,
        }
    demo.monitor.write()


def run_demo(run_dir, *, max_steps=100):
    demo = build_demo(run_dir)
    steps, completion_order, max_in_flight = [], [], 0
    try:
        for _ in range(max_steps):
            result = demo.runner.step(OBJECTIVE)
            steps.append(asdict(result))
            pending = demo.runner.exploration.pending_all()
            max_in_flight = max(max_in_flight, len(pending))
            if len(demo.workers.futures) == 2 and not demo.workers.release.is_set():
                if not all(event.is_set() for event in demo.workers.started.values()):
                    raise RuntimeError('Both workers must actually start before releasing their work')
                demo.workers.release.set()
            demo.monitor.sync_research_attempts(demo.runner.decision_making.harness.state.decision_records)
            demo.monitor.advance(result)
            publish_candidates(demo)
            if result.kind == 'completed':
                completion_order.append(result.directive_id)
                if result.directive_id == 'produce-data':
                    demo.workers.data_received.set()
            if result.kind == 'finished':
                break
            # Let actual CPU futures progress without timing-sensitive busy loops.
            running = [future for future in demo.workers.futures.values() if not future.done()]
            if demo.workers.release.is_set() and running:
                wait(running, timeout=0.01, return_when=FIRST_COMPLETED)
        else:
            raise RuntimeError('Simulated decision DAG did not finish within its step budget')
        records = deepcopy(demo.runner.decision_making.harness.state.decision_records)
        report = {'simulated': True, 'policy': 'scripted fixture', 'status': 'completed',
                  'run_dir': str(demo.monitor.root), 'max_in_flight': max_in_flight,
                  'completion_order': completion_order, 'worker_intervals': demo.workers.intervals,
                  'activations': len(demo.activator.calls), 'current': asdict(demo.registry.current),
                  'consumed': demo.consumed, 'decision_graph': list(records.values()),
                  'candidate_origins': dict(demo.runner.decision_making.harness.state.candidate_origins)}
        atomic_write(demo.monitor.root / 'decision_memory.json', json_bytes({
            'history': [asdict(row) for row in demo.runner.decision_making.harness.memory.history()]}))
        atomic_write(demo.monitor.root / 'steps.jsonl', b''.join(json_bytes(step) + b'\n' for step in steps))
        atomic_write(demo.monitor.root / 'result.json', json_bytes(report))
        demo.monitor.stage('simulated_dag_completed', status='completed')
        return report
    except BaseException:
        demo.monitor.stage('simulated_dag_failed', status='failed')
        raise
    finally:
        demo.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path,
                        default=Path('runs') / f'decision-graph-{uuid4().hex[:8]}')
    args = parser.parse_args()
    print(json.dumps(run_demo(args.run_dir), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
