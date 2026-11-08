"""Offline current-window probes; retain original review JSON as history."""
import json
from pathlib import Path

from rsisys.config import build_demo
from rsisys.research import ResearchAssembly, assemble_research
from rsisys.loops.decision_making.feedback.learning import (
    checked_processor_result, FeedbackLearning, SerializedLearningBackend)
from rsisys.loops.decision_making.feedback.pipeline import FeedbackPipeline
from rsisys.loops.decision_making.harness.learning_harness import enable_feedback_learning
from tests.feedback.test_feedback_components import ExternalContext
from tests.feedback.test_feedback_learning_integration import ImmediateExecutor
from tests.feedback.test_feedback_required_outputs import CONFIG, MODEL, EmptyBackend, observation

results = {}

class ContextOnly:
    def context(self, feedback):
        return 'A factual context fragment'

try:
    ResearchAssembly(context_memory=ContextOnly())
except TypeError as exc:
    results['context_only_provider'] = {'accepted': False, 'error': str(exc)}
else:
    results['context_only_provider'] = {'accepted': True, 'protocol': 'context(query) -> str'}

result = checked_processor_result({
    'status': 'completed', 'charged_calls': 0, 'context': 'A factual context fragment'
}, 0)
assert result['status'] == 'failed'
results['legacy_processor_field_guard'] = result

runner, _ = build_demo(CONFIG)
with runner:
    service = assemble_research(runner.decision_making,
        assembly=ResearchAssembly(context_memory=ContextOnly(),
            memory_processor=lambda event, model: {'context': 'native context', 'charged_calls': 999}),
        backend=EmptyBackend(), executor=ImmediateExecutor(), owner=runner.assembly)
    ident = service.observe(observation())
    service.refresh()
    results['native_product_window'] = {'status': service.learning.state['jobs'][ident]['status'],
        'output': service.outputs()[0]['output'], 'host_charged_calls': service.learning.state['attempted']}

class Stage:
    max_calls_per_job = 1
    def __init__(self, status):
        self.status = status
        self.calls = 0
    def configuration(self):
        return {'status': self.status}
    def recover(self, job):
        return None
    def process(self, job, backend, limits):
        self.calls += 1
        return {'status': self.status, 'error': 'fixture memory failure' if self.status == 'failed' else None,
                'charged_calls': 0, 'proposals': []}

memory, proposer = Stage('failed'), Stage('completed')
pipeline = FeedbackPipeline(memory_processor=memory, improvement_proposals=proposer)
learning = FeedbackLearning(SerializedLearningBackend(EmptyBackend()), processor=pipeline,
                            executor=ImmediateExecutor())
ident = learning.capture_receipt(observation(), MODEL)
result = pipeline.process(learning.state['jobs'][ident], learning.backend, {**learning.limits, 'max_calls': 2})
learning.close()
assert memory.calls == 1 and proposer.calls == 1
assert result['products']['improvement_proposals']['status'] == 'completed'
results['proposal_dependency'] = {
    'memory_calls': memory.calls, 'proposer_calls': proposer.calls, 'products': result['products']}

class FlakyContext(ExternalContext):
    fail = True
    def memory(self, job):
        if self.fail:
            self.fail = False
            raise RuntimeError('fixture context store unavailable')
        return super().memory(job)

runner, _ = build_demo(CONFIG)
with runner:
    loop = runner.decision_making
    backend = EmptyBackend()
    enable_feedback_learning(loop, backend=backend, executor=ImmediateExecutor(),
                             context_memory=FlakyContext(), trajectory_archive=None)
    service = loop.feedback_service
    ident = service.observe(observation())
    context = loop.harness._context('objective', loop.action_space)
    results['publication_isolated_from_decision_context'] = {
        'delivery': service.learning.state['jobs'][ident]['publication_delivery'],
        'processing_status': service.learning.state['jobs'][ident]['status'],
        'has_context': bool(context.memory_context)}
    loop.harness._context('objective', loop.action_space)
    assert not service.learning.state['jobs'][ident].get('published', False)
    service.retry_publication(ident)
    service.refresh()
    assert service.learning.state['jobs'][ident]['published']
    assert len(backend.requests) == 1
    results['publication_retry'] = {'published': True, 'fixture_generation_requests': len(backend.requests)}

output = {'scope': 'offline fixtures; public windows and explicit compatibility boundaries',
          'real_model_inference': False, 'real_training': False, 'probes': results}
path = Path(__file__).with_name('feedback-contract-probe-current.json')
path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
print(path.read_text())
