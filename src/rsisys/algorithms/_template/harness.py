"""Harness prototype: copy this file to change what a Controller's Policy sees.

A Harness sits between the host and the Policy. It builds the decision window
the Policy reads and can give the Policy local tools that run without a Lab.
Bind it with ControllerDefinition(policy, harness=factory). Without a Harness,
the Controller uses the default one, which also adds Controller Memory.

  Required
    window(objective, capabilities, feedback)
                        Return the DecisionInput for the next decide() call.
                        capabilities describes every callable operation;
                        feedback holds the receipts delivered at this step.
    observe(event)      Receive each EvidenceEvent the run records.
    tools()             OperationSpecs for Controller-local tools.
    execute_tool(call)  Run one local tool; return an OperationResult.

  Optional
    state(), restore()  Save and reload Harness state for durable runs.
    close()             Release resources when the Controller is replaced.

A Harness replaces the default one. Memory is installed by the default Harness,
so a replacement does not receive memory context. If you need Memory, either
keep the default Harness or add the memory text to window.context yourself.

This Harness adds a short summary of recent receipts to window.context, so a
Policy can compare all candidates instead of only the newest receipt. It also
offers one local tool, `note`, that stores a line of text in that summary.
"""
from rsisys.api import DecisionInput, OperationResult, OperationSpec


class ScoreboardHarness:
    def __init__(self, context, config):
        # Called as factory(context, config), like Policy and Lab factories.
        self.limit = config.get('scoreboard_rows', 20)
        self.rows = []

    def window(self, objective, capabilities, feedback):
        for receipt in feedback:
            metrics = receipt.data.get('quality_metrics', {})
            self.rows.append(f'{receipt.summary}: {metrics}' if metrics else receipt.summary)
        context = 'Scoreboard (most recent last):\n' + '\n'.join(self.rows[-self.limit:])
        return DecisionInput(objective, context=context, capabilities=capabilities, feedbacks=feedback)

    def observe(self, event):
        # Every recorded receipt and Controller event arrives here. This Harness
        # builds its summary from window() feedback, so it ignores events.
        pass

    def tools(self):
        return (OperationSpec('note', 'Add one line to the scoreboard shown in context', {
            'type': 'object', 'properties': {'text': {'type': 'string', 'minLength': 1}},
            'required': ['text'], 'additionalProperties': False,
        }, read_only=False),)

    def execute_tool(self, call):
        if call.name != 'note':
            raise ValueError(f'Unknown Harness tool: {call.name}')
        self.rows.append('note: ' + call.arguments['text'])
        return OperationResult('Note added', (), False)

    def state(self):
        return {'rows': list(self.rows)}

    def restore(self, state):
        self.rows = list(state['rows'])
