"""Math measurement Lab for fixed-weight, executable harness evolution."""

import json
import multiprocessing
import subprocess
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path

from rsisys.backends.serving.chat import ChatPool
from rsisys.harnesses.python_step import (
    HarnessExecutionError,
    sandbox_step,
    validate_source,
)
from rsisys.labs.scenes.math.verifiers import MathVerifyVerifier, StrictNumericVerifier
from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.decision import OperationSpec
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.runtime.storage.files import file_hash as digest


@lru_cache(maxsize=1)
def _worker_verifier():
    return MathVerifyVerifier()


def _score_math(pair):
    # Each spawned worker scores on its main thread, preserving signal timeouts.
    return _worker_verifier().is_correct(*pair)


def validate_config(config):
    allowed = {
        "endpoints",
        "model",
        "verifier",
        "max_calls",
        "max_tokens",
        "workers",
        "grading_workers",
        "sandbox_workers",
        "seed",
        "temperature",
        "timeout",
        "output_subdir",
        "search_data",
        "heldout_data",
    }
    if set(config) - allowed:
        raise ValueError("Unknown harness_math option")
    for key in ("endpoints", "model", "search_data", "heldout_data"):
        if key not in config:
            raise ValueError("Missing harness_math " + key)
    for key in (
        "max_calls",
        "max_tokens",
        "workers",
        "grading_workers",
        "sandbox_workers",
        "timeout",
    ):
        if key in config and (type(config[key]) is not int or config[key] < 1):
            raise ValueError("Invalid " + key)
    if config.get("verifier", "math_verify") not in ("math_verify", "strict_numeric"):
        raise ValueError("Invalid verifier")
    output = config.get("output_subdir", "harness-math")
    if (
        not isinstance(output, str)
        or Path(output).is_absolute()
        or ".." in Path(output).parts
    ):
        raise ValueError("Invalid output_subdir")


class RRSIMathLab:
    def __init__(
        self, context, config, *, pool=None, executor=sandbox_step, verifier=None
    ):
        validate_config(config)
        self.config = config
        self.root = Path(context.data_dir) / config.get("output_subdir", "harness-math")
        self.root.mkdir(parents=True, exist_ok=False)
        self.pool = pool or ChatPool(
            config["endpoints"],
            config["model"],
            max_tokens=config.get("max_tokens", 2048),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 180),
        )
        self.executor = executor
        self._sandbox_slots = threading.BoundedSemaphore(
            config.get("sandbox_workers", 32)
        )
        self.verifier = verifier or (
            MathVerifyVerifier()
            if config.get("verifier", "math_verify") == "math_verify"
            else StrictNumericVerifier()
        )
        self._counter = 0
        self._harnesses = {}
        self.datasets = {
            k: self._read(config[k + "_data"]) for k in ("search", "heldout")
        }
        for key in ("id", "prompt"):
            if {row[key] for row in self.datasets["search"]} & {
                row[key] for row in self.datasets["heldout"]
            }:
                raise ValueError("Search and heldout datasets overlap by " + key)
        self.dataset_hashes = {k: digest(config[k + "_data"]) for k in self.datasets}

    @staticmethod
    def _read(path):
        rows = [
            json.loads(line)
            for line in Path(path).read_text().splitlines()
            if line.strip()
        ]
        if not rows or any(
            not all(
                isinstance(r.get(k), str) and r[k].strip()
                for k in ("id", "prompt", "answer")
            )
            for r in rows
        ):
            raise ValueError("Invalid math data")
        if len({r["id"] for r in rows}) != len(rows) or len(
            {r["prompt"] for r in rows}
        ) != len(rows):
            raise ValueError("Duplicate math tasks")
        return rows

    def describe(self):
        return (
            OperationSpec(
                "evaluate",
                "Measure a fixed model with one immutable Python harness.",
                {
                    "type": "object",
                    "properties": {
                        k: {"type": t}
                        for k, t in [
                            ("harness", "string"),
                            ("dataset", "string"),
                            ("split", "string"),
                            ("trials", "integer"),
                            ("round", "integer"),
                        ]
                    },
                    "required": ["harness", "dataset", "split", "trials", "round"],
                    "additionalProperties": False,
                },
            ),
        )

    def _trial(self, item, harness):
        index, row, trial = item
        history, actions, covered = [], [], set()
        error = None
        answer = ""
        try:
            for step in range(self.config.get("max_calls", 3) + 1):
                with self._sandbox_slots:
                    result = self.executor(harness, row["prompt"], history)
                covered.update(result["covered_lines"])
                action = result["action"]
                actions.append(action)
                if "answer" in action:
                    answer = action["answer"]
                    break
                if step == self.config.get("max_calls", 3):
                    answer = history[-1]["content"] if history else ""
                    break
                reply = self.pool.complete(
                    action["messages"],
                    seed=self.config.get("seed", 73) + index * 100 + trial * 10 + step,
                )
                history.append(reply)
        except (HarnessExecutionError, subprocess.TimeoutExpired) as exc:
            error = str(exc)
            answer = ""
        # Math-Verify uses signal-based time limits: scoring runs on the main thread.
        return {
            "id": row["id"],
            "prompt": row["prompt"],
            "trial": trial,
            "completion": answer,
            "tokens": sum(r["tokens"] for r in history),
            "calls": len(history),
            "actions": actions,
            "history": history,
            "covered_lines": sorted(covered),
            "error": error,
        }

    def execute(self, call):
        request = dict(call.arguments)
        if call.name != "evaluate":
            raise ValueError("Unknown harness_math operation")
        split = request["split"]
        if (
            split not in self.datasets
            or Path(request["dataset"]).resolve()
            != Path(self.config[split + "_data"]).resolve()
        ):
            raise ValueError("Dataset reference does not match frozen split")
        if digest(request["dataset"]) != self.dataset_hashes[split]:
            raise ValueError("Dataset changed after launch")
        if type(request["trials"]) is not int or not 1 <= request["trials"] <= 16:
            raise ValueError("Invalid trial count")
        source = Path(request["harness"]).read_text()
        validate_source(source)
        harness_hash = digest(request["harness"])
        key = str(Path(request["harness"]).resolve())
        if key in self._harnesses and self._harnesses[key] != harness_hash:
            raise ValueError("Previously measured harness changed")
        self._harnesses[key] = harness_hash
        rows = self.datasets[split]
        for row in rows:
            if len(row["prompt"]) > 20 and row["prompt"] in source:
                raise ValueError("Task-specific problem in harness source")
        items = [
            (i, row, j) for i, row in enumerate(rows) for j in range(request["trials"])
        ]
        started = time.monotonic()
        traces = [None] * len(items)
        grading_workers = (
            self.config.get("grading_workers", 1)
            if type(self.verifier) is MathVerifyVerifier
            else 1
        )
        with ExitStack() as stack:
            grader = (
                stack.enter_context(
                    ProcessPoolExecutor(
                        max_workers=grading_workers,
                        mp_context=multiprocessing.get_context("spawn"),
                    )
                )
                if grading_workers > 1
                else None
            )
            pool = stack.enter_context(
                ThreadPoolExecutor(max_workers=self.config.get("workers", 32))
            )
            pending = {
                pool.submit(self._trial, item, request["harness"]): index
                for index, item in enumerate(items)
            }
            scoring = {}
            for future in as_completed(pending):
                index = pending[future]
                trace = traces[index] = future.result()
                pair = (trace["completion"], rows[items[index][0]]["answer"])
                if grader is None:
                    trace["correct"] = self.verifier.is_correct(*pair)
                else:
                    scoring[grader.submit(_score_math, pair)] = index
            dispatched = time.monotonic()
            for future in as_completed(scoring):
                traces[scoring[future]]["correct"] = future.result()
        finished = time.monotonic()
        if digest(request["harness"]) != harness_hash:
            raise ValueError("Harness changed during evaluation")
        rewards = []
        for i, row in enumerate(rows):
            group = []
            for trial in range(request["trials"]):
                trace = traces[i * request["trials"] + trial]
                group.append(int(trace["correct"]))
            rewards.append(group)
        output = self.root / f"evaluation-{self._counter:03d}.jsonl"
        self._counter += 1
        output.write_text(
            "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in traces)
        )
        data = {
            "request": request,
            "harness": request["harness"],
            "harness_sha256": harness_hash,
            "dataset_sha256": self.dataset_hashes[split],
            "traces": str(output),
            "traces_sha256": digest(output),
            "score": sum(map(sum, rewards)) / len(traces),
            "cost": sum(t["tokens"] for t in traces) / len(traces),
            "rewards": rewards,
            "samples": len(traces),
            "model": self.config["model"],
            "training_performed": False,
            "verifier": self.verifier.name,
            "harness_errors": sum(t["error"] is not None for t in traces),
            "timing": {
                "elapsed_seconds": finished - started,
                "generation_and_dispatch_seconds": dispatched - started,
                "grading_tail_seconds": finished - dispatched,
                "grading_workers": grading_workers,
                "trials_per_second": len(traces) / (finished - started),
                "completion_tokens_per_second": sum(
                    reply["completion_tokens"]
                    for trace in traces
                    for reply in trace["history"]
                )
                / (finished - started),
            },
            "endpoints_used": sorted(
                {reply["endpoint"] for t in traces for reply in t["history"]}
            ),
        }
        # Candidate crashes are observed failed trials; transport/verifier failures propagate.
        (output.with_suffix(".json")).write_text(json.dumps(data, indent=2))
        return OperationResult(
            "Harness measured with fixed model weights",
            (),
            False,
            ExecutionStatus.COMPLETED,
            data=data,
        )
