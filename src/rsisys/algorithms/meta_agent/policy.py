"""Method completion checks; all experiment choices remain model-generated."""
from dataclasses import replace

from rsisys.harnesses.decision.policy import ModelDecisionPolicy
from rsisys.loops.contracts.lifecycle import ExecutionStatus


class HarnessResearchPolicy(ModelDecisionPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finalized = False
        self.finalization = None
        self.max_output_tokens = 4096

    def _decision_request(self, context):
        return replace(super()._decision_request(context), max_output_tokens=self.max_output_tokens)

    def decide(self, context):
        for result in context.operation_results:
            if (result.status == ExecutionStatus.COMPLETED and not result.simulated
                    and result.data.get('selection_sealed') is True
                    and 'baseline' in result.data and 'selected' in result.data):
                self.finalized = True
                self.finalization = result.data
        return super().decide(context)

    def report(self):
        data = self.finalization or {}
        return {'method': 'meta_agent', 'status': 'completed' if self.finalized else 'incomplete',
                'selected_candidate': data.get('selected_candidate'),
                'evaluations': {split: {key: data[split][key] for key in
                    ('score', 'cost', 'samples', 'harness_sha256', 'dataset_sha256') if key in data[split]}
                    for split in ('baseline', 'selected') if split in data},
                'score_delta': data.get('score_delta'), 'selection_sealed': self.finalized,
                'training_performed': False, 'self_reflection_implemented': False}

    def _control_instructions(self, context):
        if self.finalized:
            return super()._control_instructions(context)
        operations = ', '.join(spec.name for spec in context.operations if spec.name.endswith('.finalize'))
        return ('Completion is unavailable: no finalization receipt exists. Continue using installed operations. '
                'When you choose to stop research, invoke ' + operations +
                ' with your own selected candidate, wait for its result, then finish. '
                'Do not return kind=finish before that receipt.')

    def decision_schema(self, context):
        schema = super().decision_schema(context)
        if not self.finalized:
            schema['oneOf'] = [branch for branch in schema['oneOf']
                               if branch['properties']['kind'].get('const') != 'finish']
        return schema

    def _parse_decision(self, response, context):
        decision = super()._parse_decision(response, context)
        if decision.kind == 'finish' and not self.finalized:
            raise ValueError('No completed finalization receipt. Call the installed finalize operation '
                             'with your own selected candidate before finish. A finish reason does not execute a Lab operation.')
        return decision
