"""Method completion checks; all experiment choices remain model-generated."""
from dataclasses import replace

from rsicomposer.api import ExecutionStatus, ModelDecisionPolicy


class HarnessResearchPolicy(ModelDecisionPolicy):
    method = 'meta_agent'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selected = False
        self.selection = None
        self.max_output_tokens = 4096
        self.lab_aliases = ()

    def required_profiles(self):
        return dict.fromkeys(self.lab_aliases, 'artifact-search/v1')

    def _decision_request(self, context):
        return replace(super()._decision_request(context), max_output_tokens=self.max_output_tokens)

    def decide(self, context):
        for result in context.operation_results:
            if (result.status == ExecutionStatus.COMPLETED and not result.simulated
                    and result.data.get('selection_recorded') is True
                    and isinstance(result.data.get('selected_candidate'), str)):
                self.selected = True
                self.selection = result.data
        return super().decide(context)

    def state(self):
        return {**super().state(), 'selected': self.selected, 'selection': self.selection}

    def restore(self, state):
        super().restore({'model': state['model']})
        self.selected, self.selection = state['selected'], state['selection']

    def report(self):
        data = self.selection or {}
        return {'method': self.method, 'status': 'selected' if self.selected else 'incomplete',
                'selected_candidate': data.get('selected_candidate'),
                'sealed_evaluation': 'host_after_finish',
                'training_performed': False, 'self_reflection_implemented': False}

    def _control_instructions(self, context):
        if self.selected:
            return super()._control_instructions(context)
        operations = ', '.join(spec.name for spec in context.operations if spec.name.endswith('.select'))
        return ('Choose a measured candidate through ' + operations +
                ', wait for its selection receipt, then finish. The host enforces the baseline and '
                'minimum measured candidate requirements and runs sealed evaluation after finish.')

    def decision_schema(self, context):
        schema = super().decision_schema(context)
        if not self.selected:
            schema['oneOf'] = [branch for branch in schema['oneOf']
                               if branch['properties']['kind'].get('const') != 'finish']
        return schema

    def _parse_decision(self, response, context):
        decision = super()._parse_decision(response, context)
        if decision.kind == 'finish' and not self.selected:
            raise ValueError('Call the installed select operation with a measured candidate before finish.')
        return decision
