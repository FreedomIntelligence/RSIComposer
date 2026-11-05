"""Code pages survive the actual evidence projection without losing the middle."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from rsisys.adapters.local_lab import LocalProcessLab
from rsisys.adapters.local_lab_feedback import ImmediateFeedbackWindow
from rsisys.loops._memory import HistoryEntry
from rsisys.loops.contracts.decision import OperationCall, OperationSpec
from rsisys.loops.contracts.execution import LabToolError
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.harness.research_state import bounded_evidence_view
from rsisys.loops.decision_making.harness.state import DecisionContext
from rsisys.runtime.artifacts import InMemoryArtifactStore
from rsisys.runtime.json_tools import encoded
from rsisys.runtime.local_process import LocalProcessBackend


class FilePagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.backend = LocalProcessBackend(Path(self.temp.name) / 'executor')
        self.addCleanup(self.backend.close_all)
        self.lab = LocalProcessLab(self.backend, InMemoryArtifactStore())
        self.addCleanup(self.lab.close)
        self.archive = OperationSpec('research_archive', 'fixture archive capability', {
            'type': 'object', 'properties': {'collection': {'enum': ['history', 'operation_results']}}}, True)

    def call(self, name, **args):
        return self.lab.execute(OperationCall('lab_' + name, args))

    def context(self, results, arguments):
        history = tuple(HistoryEntry(str(i), 'operation', None, result.summary, False,
            details={'operation': asdict(OperationCall('lab_read_file', args)), 'result': result.payload()})
            for i, (result, args) in enumerate(zip(results, arguments)))
        return DecisionContext('Read code', (), (self.archive,), tuple(results), history=history)

    def test_large_code_unicode_and_json_escapes_round_trip_through_projection(self):
        source = '# prefix\n' * 650 + 'def extract_prompt_text(item):\n    return item["messages"][0]\n' + (
            '# 中文 "quoted" \\ slash\tcontrol\n' * 650)
        self.call('write_file', path='candidate.py', content=source)
        chunks, offset, version = [], 0, None
        while True:
            args = {'path': 'candidate.py', 'offset': offset, 'max_bytes': 1024 * 1024}
            if version is not None:
                args['expected_version'] = version
            result = self.call('read_file', **args)
            self.assertLessEqual(len(encoded(result.payload()).encode()), 5500)
            view = bounded_evidence_view(self.context([result], [args]))
            self.assertEqual(view.operation_results[-1].data['content'], result.data['content'])
            self.assertEqual(view.history[-1].details['result']['data']['content'], result.data['content'])
            policy = ModelDecisionPolicy(Mock(spec=['generate']), ModelRef('fixture', 'meta', 'v1'))
            wire_context = json.loads(policy._decision_request(view).prompt)
            self.assertEqual(wire_context['operation_results'][-1]['data']['content'], result.data['content'])
            self.assertEqual(wire_context['history'][-1]['details']['result']['data']['content'], result.data['content'])
            chunks.append(result.data['content'])
            self.assertEqual(result.data['file_sha256'], hashlib.sha256(source.encode()).hexdigest())
            version = result.data['file_version']
            if result.data['eof']:
                self.assertIsNone(result.data['next_offset'])
                break
            self.assertGreater(result.data['next_offset'], offset)
            offset = result.data['next_offset']
        self.assertGreater(len(chunks), 3)
        self.assertEqual(''.join(chunks), source)
        self.assertTrue(any('def extract_prompt_text' in c for c in chunks))

    def test_edit_between_pages_is_explicit_and_repeat_hint_is_transient(self):
        self.call('write_file', path='code.py', content='x' * 10000)
        args = {'path': 'code.py', 'max_bytes': 100000}
        results = [self.call('read_file', **args) for _ in range(3)]
        original = self.context(results, [args] * 3)
        projected = ImmediateFeedbackWindow()(original)
        hint = projected.research_agenda['instant_feedback_window']['repeated_read']
        self.assertEqual(hint['next_page']['arguments']['offset'], results[-1].data['next_offset'])
        self.assertEqual(original.research_agenda, {})
        version = results[0].data['file_version']
        self.call('write_file', path='code.py', content='y' * 10000)
        with self.assertRaisesRegex(LabToolError, 'changed between pages'):
            self.call('read_file', path='code.py', offset=results[0].data['next_offset'], expected_version=version)

    def test_small_utf8_windows_make_exact_progress_or_report_boundary_error(self):
        self.call('write_file', path='unicode.py', content='中abc')
        first = self.call('read_file', path='unicode.py', max_bytes=4)
        self.assertEqual(first.data['content'], '中a')
        self.assertEqual(self.call('read_file', path='unicode.py', offset=4).data['content'], 'bc')
        with self.assertRaisesRegex(LabToolError, 'UTF-8'):
            self.call('read_file', path='unicode.py', max_bytes=1)


if __name__ == '__main__':
    unittest.main()
