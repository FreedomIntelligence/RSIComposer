"""Optional context-or-memory routing for the example local Lab assembly."""
from copy import deepcopy
from dataclasses import replace

from rsisys.loops.contracts.decision import OperationCall
from rsisys.loops.contracts.feedback_modules import FeedbackAdmission
from rsisys.loops.decision_making.action_space import OperationResult
from rsisys.loops.decision_making.harness.research_state import (
    archive_available, archived_operation_result_ids, full_history_ids,
)
from rsisys.runtime.storage.json_tools import result_id


class RoutedLab:
    """Add a routing hint to result collection without constraining Lab internals.

    The wrapped Lab receives its original arguments. Its exact native result is
    recorded before the working-context projection chooses a presentation.
    """
    def __init__(self, lab):
        self.lab = lab

    @property
    def workspace(self):
        return self.lab.workspace

    def describe(self):
        specs = []
        for spec in self.lab.describe():
            if spec.name == 'lab_result':
                schema = deepcopy(spec.input_schema)
                schema['properties']['feedback_route'] = {'enum': ['context', 'memory'],
                    'description': 'context (default): immediate transient feedback only; memory: '
                                   'organize reusable evidence with the memory provider.'}
                spec = replace(spec, input_schema=schema, description=spec.description +
                    ' Choose feedback_route=context for quick tests/debugging, or memory for reusable findings. '
                    'The default context route does not trigger memory organization.')
            specs.append(spec)
        return tuple(specs)

    def execute(self, call):
        if call.name == 'lab_result':
            args = dict(call.arguments)
            route = args.pop('feedback_route', 'context')
            if route not in ('context', 'memory'):
                from rsisys.loops.contracts.execution import LabToolError
                raise LabToolError('feedback_route must be context or memory')
            call = OperationCall(call.name, args)
        return self.lab.execute(call)

    def runtime_context(self):
        return self.lab.runtime_context()

    def shutdown(self, mode):
        return self.lab.shutdown(mode)

    def require_admission(self):
        return self.lab.require_admission()

    def close(self):
        return self.lab.close()


class ContextOrMemoryAdmission:
    """One organization route per observation; retrieval is always available."""
    def admit(self, event, pending):
        request = event.request if isinstance(event.request, dict) else {}
        arguments = request.get('arguments') or {}
        memory = request.get('name') == 'lab_result' and arguments.get('feedback_route') == 'memory'
        return FeedbackAdmission(organize=memory,
            reason='explicit_memory_route' if memory else 'immediate_context_route')


class ImmediateFeedbackWindow:
    """Expire archived instant observations; memory-routed bodies stay in archives.

    Only a derived working view changes. The latest instant observation and live
    dependencies survive; no nonarchivable result can be removed. A memory-route
    receipt acknowledges the request, never claims organization succeeded.
    """
    def __init__(self, recent=8):
        if type(recent) is not int or recent < 1:
            raise ValueError('recent must be positive')
        self.recent = recent

    def __call__(self, context):
        if not (archive_available(context, 'history') and archive_available(context, 'operation_results')):
            return context
        rows = [h for h in context.history if h.kind == 'operation']
        keep = {h.record_id for h in rows[-self.recent:]} | full_history_ids(context)
        from rsisys.loops.decision_making.harness.observations import failed_observation
        keep.update(row.record_id for row in [h for h in rows if failed_observation(h)][-2:])
        archived = archived_operation_result_ids(context)
        retained, memory_ids, history = set(), set(), []
        for row in context.history:
            if row.kind != 'operation':
                history.append(row)
                continue
            card = row.details.get('evidence_card', {})
            output = row.details.get('result')
            digest = result_id(output) if isinstance(output, dict) else card.get('operation_result_sha256')
            operation = row.details.get('operation') or {}
            route = (operation.get('arguments') or {}).get('feedback_route', card.get('feedback_route'))
            if route == 'memory' and digest in archived:
                memory_ids.add(digest)
                history.append(replace(row, summary='Memory organization requested; consult feedback status and retrieved memory.',
                    details={'feedback_route': 'memory', 'operation_result_sha256': digest,
                        'read_path': {'operation': 'research_archive', 'collection': 'history', 'record_id': row.record_id}}))
            elif row.record_id in keep or digest not in archived:
                if digest:
                    retained.add(digest)
                history.append(row)
        results = []
        def source_id(row):
            card = row.data.get('evidence_card', {})
            return card.get('sha256') if isinstance(card, dict) and card.get('schema_version') == 1 else result_id(row.payload())
        newest = {source_id(row): i for i, row in enumerate(context.operation_results)}
        for index, row in enumerate(context.operation_results):
            digest = source_id(row)
            if digest in archived and index != newest[digest]:
                continue
            if digest in memory_ids:
                results.append(OperationResult('Memory organization requested; inspect feedback status for completion.',
                    row.artifacts, row.simulated, data={'feedback_route': 'memory', 'source_sha256': digest,
                        'read_path': {'operation': 'research_archive', 'collection': 'operation_results', 'record_id': digest}}))
            elif digest in retained or digest not in archived or index == len(context.operation_results) - 1:
                results.append(row)
        agenda = {**context.research_agenda, 'instant_feedback_window': {
                'recent_operation_limit': self.recent, 'raw_history_unchanged': True,
                'older_observations': {'operation': 'research_archive', 'collection': 'history'},
                'memory_route': 'organization_request_only; consult feedback status and retrieved memory'}}
        recent_reads = rows[-3:]
        pages = [((row.details.get('result') or {}).get('data') or {}) for row in recent_reads]
        if (len(pages) == 3 and all((row.details.get('operation') or {}).get('name') == 'lab_read_file'
                                   for row in recent_reads)
                and pages[0].get('page_sha256') and all(
                    tuple(page.get(k) for k in ('path', 'offset', 'file_version', 'page_sha256')) ==
                    tuple(pages[0].get(k) for k in ('path', 'offset', 'file_version', 'page_sha256'))
                    for page in pages[1:])):
            page = pages[-1]
            agenda['instant_feedback_window']['repeated_read'] = {
                'observation': 'The same unchanged file page was read three times; no new content was obtained.',
                'path': page['path'], 'offset': page['offset'], 'eof': page['eof'],
                'next_page': ({'operation': 'lab_read_file', 'arguments': {'path': page['path'],
                    'offset': page['next_offset'], 'expected_version': page['file_version']}}
                    if page['next_offset'] is not None else None)}
        return replace(context, history=tuple(history), operation_results=tuple(results), research_agenda=agenda)
