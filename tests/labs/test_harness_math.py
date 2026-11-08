"""Measurement isolation, honest costs and immutable data in harness search."""

import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from rsisys.api import OperationCall
from rsisys.labs.harness_math import HarnessMathLab
from rsisys.labs.harness_runtime import HarnessExecutionError
from tests.support.math import make_math_selection


class Pool:
    def complete(self, messages, **options):
        return {
            "content": "#### 8",
            "tokens": 30,
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "endpoint": "local",
            "finish_reason": "stop",
        }


class HarnessMathTests(unittest.TestCase):
    def setup_lab(self, executor, **options):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        make_math_selection(root / "data")
        harness = root / "harness.py"
        harness.write_text(
            'def step(problem, history):\n    return {"answer": "#### 8"}\n'
        )
        config = {
            "endpoints": ["http://127.0.0.1:1/v1"],
            "model": "fixed",
            "search_data": str(root / "data/train.jsonl"),
            "heldout_data": str(root / "data/eval.jsonl"),
            "verifier": "strict_numeric",
            "workers": 2,
        }
        config.update(options)
        lab = HarnessMathLab(
            SimpleNamespace(data_dir=root / "run"),
            config,
            pool=Pool(),
            executor=executor,
        )
        call = OperationCall(
            "evaluate",
            {
                "harness": str(harness),
                "dataset": config["search_data"],
                "split": "search",
                "round": 0,
                "trials": 2,
            },
        )
        return lab, call

    def test_trial_history_and_gold_are_isolated_and_tokens_are_measured(self):
        seen = []

        def step(path, problem, history):
            seen.append((problem, len(history)))
            return {
                "action": {"answer": history[-1]["content"]}
                if history
                else {"messages": [{"role": "user", "content": problem}]},
                "covered_lines": [1, 2],
            }

        lab, call = self.setup_lab(step, sandbox_workers=1)
        result = lab.execute(call)
        self.assertEqual(result.data["score"], 0.5)
        self.assertEqual(result.data["cost"], 30)
        self.assertEqual(result.data["samples"], 4)
        self.assertFalse(result.simulated)
        self.assertFalse(result.data["training_performed"])
        self.assertTrue(all(size in (0, 1) for _, size in seen))
        self.assertEqual(result.data["endpoints_used"], ["local"])

    def test_sandbox_admission_has_its_own_limit(self):
        active = peak = 0
        lock = threading.Lock()

        def step(*args):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            threading.Event().wait(0.01)
            with lock:
                active -= 1
            return {"action": {"answer": "#### 8"}, "covered_lines": [1]}

        lab, call = self.setup_lab(step, workers=4, sandbox_workers=1)
        result = lab.execute(call)
        self.assertEqual(peak, 1)
        self.assertEqual(result.data["samples"], 4)
        self.assertEqual(result.data["harness_errors"], 0)

    def test_candidate_crashes_are_failures_but_transport_errors_propagate(self):
        def step(*args):
            raise HarnessExecutionError("bad candidate")

        lab, call = self.setup_lab(step)
        result = lab.execute(call)
        self.assertEqual(result.data["score"], 0)
        self.assertEqual(result.data["harness_errors"], 4)
        lab.executor = lambda p, q, h: {
            "action": {"messages": [{"role": "user", "content": q}]},
            "covered_lines": [1],
        }

        class Offline:
            def complete(self, *args, **kwargs):
                raise ConnectionError("offline")

        lab.pool = Offline()
        with self.assertRaises(ConnectionError):
            lab.execute(call)

    def test_cross_split_substitution_and_changed_inputs_are_rejected(self):
        lab, call = self.setup_lab(lambda *args: None)
        bad = OperationCall("evaluate", {**call.arguments, "split": "heldout"})
        with self.assertRaisesRegex(ValueError, "frozen split"):
            lab.execute(bad)
        Path(call.arguments["dataset"]).write_text("")
        with self.assertRaisesRegex(ValueError, "changed after launch"):
            lab.execute(call)

    def test_completed_trials_are_scored_while_other_generations_are_pending(self):
        scored = threading.Event()
        lock = threading.Lock()
        calls = 0

        class DelayedPool(Pool):
            def complete(self, *args, **kwargs):
                nonlocal calls
                with lock:
                    first = calls == 0
                    calls += 1
                if not first and not scored.wait(timeout=5):
                    raise RuntimeError("Grading waited for all generations")
                return super().complete(*args, **kwargs)

        def step(path, problem, history):
            return {
                "action": {"answer": history[-1]["content"]}
                if history
                else {"messages": [{"role": "user", "content": problem}]},
                "covered_lines": [1],
            }

        lab, call = self.setup_lab(step)
        original = lab.verifier

        class Verifier:
            name = original.name

            def is_correct(inner, completion, answer):
                self.assertIs(threading.current_thread(), threading.main_thread())
                scored.set()
                return original.is_correct(completion, answer)

        lab.pool, lab.verifier = DelayedPool(), Verifier()
        result = lab.execute(call)
        self.assertEqual(result.data["score"], 0.5)
        self.assertEqual(result.data["rewards"], [[1, 1], [0, 0]])
        self.assertEqual(result.data["samples"], 4)
