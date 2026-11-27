"""Versioned projection of development receipts for the MetaRSI scheduler."""
from copy import deepcopy
import json

from rsicomposer.api import DecisionInput


class SchedulerHarness:
    def __init__(self, context, config):
        self.mode = config.get('evidence', 'recent')
        self.rows = deepcopy(config.get('history_seed', []))

    def observe(self, event):
        # Only completed dev evaluations and proposal outcomes, never sealed rows.
        request = event.request or {}
        if event.source != 'lab' or event.status != 'completed' or request.get('name') not in (
                'math.propose', 'math.evaluate'):
            return
        args, data = request.get('arguments', {}), (event.output or {}).get('data', {})
        if request['name'] == 'math.propose':
            self.rows.append({'operator': args['operator'], 'candidate': data.get('candidate_id'),
                              'parent': args['parent'], 'data_count': data.get('data_count', 0)})
        elif args.get('split') == 'validation':
            self.rows.append({'candidate': args['candidate'], 'score': data['score']})
        self.rows = self.rows[-64:]

    def window(self, objective, capabilities, feedback):
        rows = self.rows[-1:] if self.mode == 'recent' else self.rows
        proposals = [row for row in rows if 'operator' in row]
        return DecisionInput(objective, capabilities=capabilities, feedbacks=feedback,
                             context=json.dumps({'meta_rsi_evidence': deepcopy(proposals)}, allow_nan=False))

    def tools(self):
        return ()

    def execute_tool(self, call):
        raise ValueError('SchedulerHarness has no local tools')

    def state(self):
        return {'rows': deepcopy(self.rows)}

    def restore(self, state):
        self.rows = deepcopy(state['rows'])
