"""Observe each receipt without conflating data provenance and weight ancestry."""
from rsicomposer.api import (ObservationProbe, emit_observation, observation_id, report_version,
                              report_version_evaluation, report_version_selection)


def weight_probes(method):
    return (ObservationProbe(method + '.data', 'decision', (('round', 'number'), ('count', 'number'),
                ('stage', 'id'), ('model', 'id')), 'table'),
            ObservationProbe(method + '.checkpoint', 'decision', (('round', 'number'), ('model', 'id')), 'lineage'))


def model_version(value):
    return observation_id('weights', value)


def observe_weight_receipt(method, call, data):
    args = call.arguments
    stage = call.name.rsplit('.', 1)[-1]
    number = args['round']
    scope = f'round-{number}'
    model = data.get('model', args.get('model'))
    if stage in ('generate', 'filter'):
        dataset = data.get('dataset')
        if not isinstance(dataset, str) or not dataset:
            return
        links = ({'generated_by': [model_version(args['model'])]} if stage == 'generate'
                 else {'derived_from': [observation_id('dataset', args['samples'])]})
        emit_observation(method + '.data', call.call_id, scope=scope,
                         subject=observation_id('dataset', dataset), links=links,
                         fields={'round': number, 'stage': stage, 'count': data.get('count'), 'model': model})
    elif stage == 'train':
        if not isinstance(model, str) or not model:
            return
        parent = model_version(args['parent'])
        subject = model_version(model)
        report_version(subject, label=f'Checkpoint {number}', round_index=number, parent_ids=(parent,))
        emit_observation(method + '.checkpoint', call.call_id, scope=scope, subject=subject,
                         links={'derived_from': [parent], 'trained_on': [observation_id('dataset', args['dataset'])]},
                         fields={'round': number, 'model': model})
    elif stage == 'evaluate' and isinstance(model, str):
        version = model_version(model)
        if args.get('baseline'):
            report_version(version, label='Baseline', round_index=0, parent_ids=())
        report_version_evaluation(version, call.call_id)
    elif stage == 'select' and isinstance(model, str):
        report_version_selection(model_version(model), step=number, event_id=call.call_id)
