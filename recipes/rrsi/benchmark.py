"""Measure one fixed harness without invoking an RSI decision-maker."""

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from rsisys.api import OperationCall
from rsisys.labs.harness_math import HarnessMathLab


def monitor_gpus(stop, samples):
    while not stop.is_set():
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
                stderr=subprocess.DEVNULL,
            )
            samples.append(
                {
                    "time": time.time(),
                    "gpus": [
                        [float(value.strip()) for value in line.split(",")]
                        for line in output.splitlines()
                    ],
                }
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        stop.wait(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--grading-workers", type=int)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--split", choices=("search", "heldout"), default="heldout")
    parser.add_argument("--harness")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    options = dict(config["framework"]["labs"]["harness"]["config"])
    for name in ("workers", "grading_workers"):
        value = getattr(args, name)
        if value is not None:
            options[name] = value
    harness = (
        args.harness or config["framework"]["controller"]["config"]["initial_harness"]
    )
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "benchmark-config.json").write_text(
        json.dumps(
            {"lab": options, "harness": harness, "trials": args.trials}, indent=2
        )
    )
    lab = HarnessMathLab(SimpleNamespace(data_dir=root), options)
    call = OperationCall(
        "evaluate",
        {
            "harness": harness,
            "dataset": options[args.split + "_data"],
            "split": args.split,
            "trials": args.trials,
            "round": 0,
        },
    )
    samples = []
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_gpus, args=(stop, samples), daemon=True)
    monitor.start()
    started = time.monotonic()
    (root / "status.json").write_text(
        json.dumps({"stage": "running", "time": time.time()})
    )
    try:
        result = lab.execute(call)
        elapsed = time.monotonic() - started
        traces = [
            json.loads(line)
            for line in Path(result.data["traces"]).read_text().splitlines()
        ]
        completion_tokens = sum(
            reply["completion_tokens"] for trace in traces for reply in trace["history"]
        )
        summary = {
            "elapsed_seconds": elapsed,
            "samples": len(traces),
            "trials_per_second": len(traces) / elapsed,
            "completion_tokens": completion_tokens,
            "completion_tokens_per_second": completion_tokens / elapsed,
            "gpu_mean_utilization": {
                str(int(gpu)): sum(
                    row[1]
                    for sample in samples
                    for row in sample["gpus"]
                    if row[0] == gpu
                )
                / sum(row[0] == gpu for sample in samples for row in sample["gpus"])
                for gpu in {row[0] for sample in samples for row in sample["gpus"]}
            },
            "measurement": result.data,
            "controller_invoked": False,
            "training_performed": False,
        }
        (root / "result.json").write_text(json.dumps(summary, indent=2))
        (root / "status.json").write_text(
            json.dumps({"stage": "completed", "time": time.time()})
        )
        print(
            json.dumps(
                {key: value for key, value in summary.items() if key != "measurement"}
            )
        )
    except BaseException as exc:
        (root / "status.json").write_text(
            json.dumps({"stage": "failed", "error": repr(exc)})
        )
        raise
    finally:
        stop.set()
        monitor.join(timeout=6)
        (root / "gpu-samples.json").write_text(json.dumps(samples))


if __name__ == "__main__":
    main()
