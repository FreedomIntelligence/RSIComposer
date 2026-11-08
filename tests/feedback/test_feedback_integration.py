import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from rsisys.application import Application, ApplicationConfig
from rsisys.config import DemoConfig, build_demo
from rsisys.experience.records import ExperienceRecord
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.snapshot.local import LocalSnapshotCapture


class FeedbackIntegrationTests(unittest.TestCase):
    def test_injected_feedback_reaches_policy_without_mutating_evidence(self):
        runner, _ = build_demo(DemoConfig.load(Path('configs/demo.json')))
        try:
            harness = runner.decision_making.harness
            raw = ExperienceRecord('evidence', 'runtime_feedback', 'actor', None, 'observed failure', False)
            harness.experience.record(raw)
            selected = ExperienceRecord('derived', 'insight', 'shared', None, 'investigate efficiency', False,
                                        source_refs=('evidence',), details={'cost': 7})
            selector = Mock()
            selector.select.return_value = (selected,)
            harness.feedback_selector = selector
            contexts = []
            harness.policy = Mock(decide=lambda context: (contexts.append(context) or Decision('finish')))
            harness.step('research objective', runner.decision_making.action_space)
            selector.select.assert_called_once_with(harness.experience, 'research objective', limit=harness.experience_limit)
            self.assertEqual(contexts[0].experience, (selected,))
            contexts[0].experience[0].details['cost'] = 99
            self.assertEqual(selected.details['cost'], 7)
            self.assertEqual(harness.experience.retrieve(), (raw,))
        finally:
            runner.close()

    def test_snapshot_rejects_custom_selector_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            from dataclasses import replace
            config = replace(ApplicationConfig.load(Path('configs/local.json')), data_dir=Path(tmp))
            with Application(config) as app:
                app.runner.decision_making.harness.feedback_selector = Mock()
                with self.assertRaisesRegex(TypeError, 'feedback selectors'):
                    LocalSnapshotCapture(app).capture('custom-selection')
                self.assertFalse((Path(tmp) / 'snapshots').exists())
