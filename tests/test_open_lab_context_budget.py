"""The host reserves output before deriving any input admission allowance."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from examples.medical.open_lab_experiment import host_context_compactor
from examples.medical import open_lab_experiment as launcher


class OpenLabContextBudgetTests(unittest.TestCase):
    def test_production_cli_passes_capacity_to_the_same_host_budget(self):
        with TemporaryDirectory() as directory, patch.object(launcher, 'run',
                side_effect=lambda args: host_context_compactor(vars(args))) as run:
            compactor = launcher.main(['--root', str(Path(directory) / 'run'),
                '--task', '/tmp/public-task', '--model-path', '/tmp/weights',
                '--python', '/usr/bin/python3', '--endpoint', 'http://127.0.0.1:18000/v1',
                '--run-id', 'fixture', '--context-window-tokens', '200000',
                '--context-budget', '{"trigger_tokens":160000,"target_tokens":48000,'
                '"max_target_tokens":144000,"target_increment_tokens":24000}'])
            run.assert_called_once()
            self.assertEqual(compactor.input_budget_tokens, 190784)
            self.assertEqual(compactor.trigger_tokens, 160000)

    def test_native_200k_context_keeps_the_full_reply_and_safety_reservation(self):
        compactor = host_context_compactor({'context_window_tokens': 200000,
            'context_budget': {'trigger_tokens': 160000, 'target_tokens': 48000,
                'max_target_tokens': 144000, 'target_increment_tokens': 24000}})
        self.assertEqual(compactor.input_budget_tokens, 190784)
        self.assertEqual(compactor.trigger_tokens, 160000)
        self.assertEqual(compactor.max_target_tokens, 144000)

    def test_existing_64k_service_gets_a_hard_budget_without_changing_soft_goals(self):
        compactor = host_context_compactor({})
        self.assertEqual(compactor.input_budget_tokens, 56320)
        self.assertEqual(compactor.trigger_tokens, 48000)
        self.assertEqual(compactor.max_target_tokens, 40000)

    def test_capacity_and_compression_settings_cannot_spend_output_reserve(self):
        for config in ({'context_window_tokens': True}, {'context_safety_tokens': -1},
                       {'context_window_tokens': 48000}, {'context_safety_tokens': 15000},
                       {'context_budget': {'input_budget_tokens': 200000}},
                       {'context_budget': {'trigger_tokens': 200000}}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                host_context_compactor(config)
