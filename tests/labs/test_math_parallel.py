"""Parallel execution boundaries: ordering, supervision, and process cleanup."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

from rsisys.labs.math import validate_config
from rsisys.labs.math_parallel import shard_requests, gather_results, run_workers
from rsisys.labs.math_worker import training_groups


class MathParallelTests(unittest.TestCase):
    def test_shards_do_not_expose_gold_and_gather_preserves_samples(self):
        rows = [{'prompt': str(i), 'answer': 'secret'} for i in range(11)]
        shards = shard_requests(rows, 3, 8)
        self.assertEqual(sum(map(len, shards)), 33)
        self.assertLessEqual(max(map(len, shards)) - min(map(len, shards)), 1)
        for shard in shards:
            for item in shard:
                self.assertEqual(set(item), {'index', 'sample', 'prompt'})
        replies = [[{**item, 'completion': f"{item['index']}:{item['sample']}"}
                    for item in reversed(shard)] for shard in reversed(shards)]
        self.assertEqual(gather_results(replies, rows, 3), [[f'{i}:{s}' for s in range(3)] for i in range(11)])
        with self.assertRaisesRegex(ValueError, 'Missing'):
            gather_results(replies[:-1], rows, 3)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            gather_results(replies + replies[:1], rows, 3)

    def test_training_tail_never_duplicates_or_drops_examples(self):
        groups = training_groups(list(range(19)), 1, 2, 8)
        self.assertEqual([len(g) for g in groups], [16, 3])
        self.assertEqual([x for g in groups for x in g], list(range(19)))

    def test_config_rejects_bad_resources(self):
        validate_config({'engine': 'vllm', 'gpu_ids': list(range(8)), 'gpu_memory_utilization': .85})
        for overrides in ({'gpu_ids': [0, 0]}, {'gpu_ids': []}, {'gpu_ids': [True]},
                          {'gpu_memory_utilization': 1}, {'gpu_memory_utilization': float('nan')},
                          {'max_num_seqs': 0}, {'device': 'cpu'}, {'train_gradient_checkpointing': 1}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                validate_config({'engine': 'vllm', **overrides})

    @unittest.skipUnless(os.name == 'posix', 'POSIX process groups')
    def test_failure_terminates_other_worker_and_reports_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            script = "import pathlib,os,time; pathlib.Path('" + str(directory / 'pid') + "').write_text(str(os.getpid())); time.sleep(30)"
            bad = "import time; time.sleep(.2); raise ValueError('deliberate worker failure')"
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, 'deliberate worker failure'):
                run_workers([[sys.executable, '-c', script], [sys.executable, '-c', bad]],
                            [dict(os.environ)] * 2, directory, 10)
            self.assertLess(time.monotonic() - started, 5)
            pid = int((directory / 'pid').read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    @unittest.skipUnless(os.name == 'posix', 'POSIX symlink')
    def test_worker_path_keeps_virtualenv_bin_even_for_symlink_python(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            virtual_bin = directory / 'bin'
            virtual_bin.mkdir()
            executable = virtual_bin / 'python'
            executable.symlink_to(sys.executable)
            script = "import os; print(os.environ['PATH'].split(os.pathsep)[0])"
            run_workers([[str(executable), '-c', script]], [dict(os.environ)], directory, 3)
            self.assertEqual((directory / 'worker-0.log').read_text().strip(), str(virtual_bin))

    def test_timeout_and_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaises(TimeoutError):
                run_workers([[sys.executable, '-c', 'import time; time.sleep(30)']], [dict(os.environ)], directory, .1)
            run_workers([[sys.executable, '-c', 'pass']], [dict(os.environ)], directory, 3)
