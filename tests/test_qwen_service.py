"""Pure service regressions: no torch, model loading, sockets or GPU calls."""
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.qwen_toy import serve


class FinishGenerationTests(unittest.TestCase):
    def test_eos_at_token_limit_is_stop(self):
        for eos in (3, [3], [2, 3]):
            with self.subTest(eos=eos):
                self.assertEqual(
                    serve.finish_generation('answer', [7, 3], 2, eos),
                    ('answer', 'stop'),
                )

    def test_eos_before_limit_is_stop(self):
        self.assertEqual(serve.finish_generation('answer', [7, 3], 8, 3),
                         ('answer', 'stop'))

    def test_token_exhaustion_without_final_eos_is_length(self):
        for tokens, eos in (([7, 8], 3), ([7, 8], None), ([7, 8], []), ([3, 8], 3)):
            with self.subTest(tokens=tokens, eos=eos):
                self.assertEqual(serve.finish_generation('answer', tokens, 2, eos),
                                 ('answer', 'length'))

    def test_short_or_empty_generation_is_stop(self):
        for tokens in ([], [7]):
            with self.subTest(tokens=tokens):
                self.assertEqual(serve.finish_generation('', tokens, 2, None), ('', 'stop'))

    def test_user_marker_at_token_limit_is_stop(self):
        for stop in ('<END>', ['missing', '<END>']):
            with self.subTest(stop=stop):
                self.assertEqual(
                    serve.finish_generation('answer<END>extra', [7, 8], 2, 3, stop),
                    ('answer', 'stop'),
                )

    def test_earliest_marker_wins_independently_of_list_order(self):
        for stop in (['<LATER>', '<END>'], ['<END>', '<LATER>']):
            with self.subTest(stop=stop):
                self.assertEqual(
                    serve.finish_generation('answer<END>extra<LATER>', [7, 8], 2, 3, stop),
                    ('answer', 'stop'),
                )

    def test_marker_at_start_returns_empty_stopped_content(self):
        self.assertEqual(serve.finish_generation('<END>extra', [7, 8], 2, 3, '<END>'),
                         ('', 'stop'))

    def test_unmatched_or_empty_markers_do_not_hide_exhaustion(self):
        for stop in (None, [], '', [''], 'missing', ['missing', '']):
            with self.subTest(stop=stop):
                self.assertEqual(serve.finish_generation('answer', [7, 8], 2, 3, stop),
                                 ('answer', 'length'))


class ServiceArgumentTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.model_dir = stack.enter_context(tempfile.TemporaryDirectory())
        stack.enter_context(patch.dict(os.environ, {}, clear=True))

    def parse(self, *args):
        return serve.parse_args(['--model', self.model_dir, *args])

    def test_cpu_defaults_to_loopback_and_four_threads(self):
        args = self.parse()
        self.assertEqual((args.device, args.host, args.threads), ('cpu', '127.0.0.1', 4))

    def test_cuda_default_host_without_initializing_cuda(self):
        self.assertEqual(self.parse('--device', 'cuda').host, '0.0.0.0')

    def test_explicit_host_overrides_device_default(self):
        for device, host in (('cpu', '0.0.0.0'), ('cuda', '127.0.0.1')):
            with self.subTest(device=device):
                self.assertEqual(self.parse('--device', device, '--host', host).host, host)

    def test_cli_device_overrides_environment_for_default_host(self):
        os.environ['QWEN_DEVICE'] = 'cuda'
        self.assertEqual(self.parse().host, '0.0.0.0')
        self.assertEqual(self.parse('--device', 'cpu').host, '127.0.0.1')


if __name__ == '__main__':
    unittest.main()
