"""Optional memory admission for LocalProcessLab receipts; not a Lab protocol."""
from rsisys.loops.contracts.feedback_modules import FeedbackAdmission
from rsisys.runtime.json_tools import result_id


class LocalLabMemoryAdmission:
    """Archive all observations; organize terminal evidence and actionable errors.

    The decision already receives every receipt. Code writes, submission/cancel
    acknowledgements and busy backpressure do not require an additional LLM call.
    Equivalent queued errors share one organization job; raw events remain exact.
    """
    def __init__(self, *, submission_operations=('lab_start',)):
        self.submission_operations = frozenset(submission_operations)

    @staticmethod
    def _signature(event):
        request = event.request if isinstance(event.request, dict) else {}
        output = event.output if isinstance(event.output, dict) else {}
        return result_id({'request': request, 'status': event.status,
                          'error': output.get('error'), 'summary': output.get('summary'),
                          'data': output.get('data')})

    def admit(self, event, pending):
        request = event.request if isinstance(event.request, dict) else {}
        result = event.output if isinstance(event.output, dict) else {}
        data = result.get('data', {})
        data = data if isinstance(data, dict) else {}
        name = request.get('name')
        if name in self.submission_operations and data.get('reason') == 'execution_busy':
            return FeedbackAdmission(False, reason='unchanged_execution_backpressure')
        if event.status == 'completed' and name in self.submission_operations | {'lab_write_file', 'lab_cancel', 'lab_export'}:
            return FeedbackAdmission(False, reason='nonterminal_receipt_available_in_context_and_archive')
        # Every terminal producer result gets its own source-backed organization.
        # Only equivalent errors are coalesced; scientific results are not merged.
        coalesce = ()
        if event.status == 'failed' and name != 'lab_result':
            signature = self._signature(event)
            coalesce = tuple(item.event.event_id for item in pending
                if item.event.status == 'failed' and self._signature(item.event) == signature)
        return FeedbackAdmission(coalesce=coalesce, reason='terminal_evidence_or_actionable_feedback')
