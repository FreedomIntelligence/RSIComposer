"""Memory prototype: copy this file to give a Controller memory across steps.

Memory turns evidence from the run into text context for the Policy. The host
feeds it every EvidenceEvent and asks it for context before each decision.
Bind it with ControllerDefinition(policy, memory=factory), where factory(config)
returns MemoryAssembly(ResearchAssembly(context_memory=YourMemory())).

  Required
    context(query)      Return text for the next decision; '' is valid.
                        query.objective and query.latest_feedback are available.

  Optional, recognized by the host when present
    observe(event)      Receive each EvidenceEvent. event_id is stable across
                        deliveries, so deduplicate by it.
    export(), restore(state)
                        Save and reload Memory state for durable runs.
    seed_version(state) Load a parent Controller's Memory into a new version.
    rebuild(events)     Rebuild an empty Memory from recorded events.

The memory text reaches the Policy in window.context. This Memory keeps the best
score seen for each operation and reports it as one line per operation.
"""


class BestScoreMemory:
    def __init__(self):
        self.seen = set()
        self.best = {}

    def observe(self, event):
        if event.event_id in self.seen or event.kind != 'receipt':
            return
        self.seen.add(event.event_id)
        # A receipt event carries the whole OperationResult in output and the
        # OperationCall in request.
        data = event.output.get('data', {}) if isinstance(event.output, dict) else {}
        metrics = data.get('quality_metrics')
        if not metrics:
            return
        name = event.request.get('name', 'operation') if isinstance(event.request, dict) else 'operation'
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                key = f'{name}/{metric}'
                self.best[key] = max(value, self.best.get(key, value))

    def context(self, query):
        return '\n'.join(f'best {key}: {value}' for key, value in sorted(self.best.items()))

    def export(self):
        return {'seen': sorted(self.seen), 'best': dict(self.best)}

    def restore(self, state):
        self.seen, self.best = set(state['seen']), dict(state['best'])

    def seed_version(self, state):
        self.restore(state)
