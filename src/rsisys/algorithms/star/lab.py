"""Execute one STaR collection, fit or evaluation operation."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import monotonic, sleep

from rsisys.api import LabFeedback, OperationSpec
from rsisys.runtime.storage.files import atomic_write, json_bytes

from ..shared.numeric_answers import extract_answer, score_math
from rsisys.runtime.orchestration.monitor_snapshot import active_monitor
from ..shared.prompt_data import load_splits
from .gpu import admit_gpu
from .protocol import MAX_LENGTH, curriculum, keep_reasoning, render_prompt, write_records
from .records import GLOBAL_BATCH, training_example


def atomic_json(path, value):
    atomic_write(path, json_bytes(value))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def filter_overlength(source, destination, base_model):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    kept, dropped = [], 0
    for line in source.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        try:
            training_example(tokenizer, render_prompt(tokenizer, row["prompt"]),
                             row["completion"], max_tokens=MAX_LENGTH)
        except ValueError as exc:
            if str(exc) != f"Prompt plus completion exceeds {MAX_LENGTH} tokens":
                raise
            dropped += 1
        else:
            kept.append(row)
    if len(kept) < GLOBAL_BATCH:
        raise RuntimeError(f"STaR has only {len(kept)} eligible rationales after length filtering")
    write_records(destination, kept)
    return dropped


class StarLab:
    def __init__(self, context, config):
        self.root = Path(context.data_dir)
        self.groups = groups = load_splits(config['files'], config['counts'], splits=('train', 'dev', 'test'))
        self.devices = admit_gpu()
        self.seed = config['seed']
        self.schedule = curriculum(config['rounds'], config['plain_rounds'])
        self.warmup_steps = config['warmup_steps']
        self.base = Path(config['base_model']).resolve(strict=True)
        self.deadline = monotonic() + config['max_seconds']
        self.models, self.datasets = {'base': self.base}, {}
        self.dev_file = self.root / 'development.jsonl'
        atomic_json(self.root / 'cohorts.json', {split: [row['id'] for row in rows] for split, rows in groups.items()})
        atomic_json(self.root / 'allocated_devices.json', self.devices)
        development = []
        for row in groups["dev"]:
            answer = extract_answer(row["answer"], require_marker=False)
            if answer is None:
                raise ValueError(f"Invalid GSM8K reference for {row['id']}")
            development.append({"id": row["id"], "prompt": row["question"],
                                "answer": "\\boxed{" + str(answer) + "}"})
        write_records(self.dev_file, development)

    def describe(self):
        return (
            OperationSpec("evaluate", "Score a committed model on a fixed GSM8K split", {"type": "object"}),
            OperationSpec("collect", "Generate and filter one round of supervised rationales", {"type": "object"}),
            OperationSpec("fit", "Train the original base on this round's filtered rationales", {"type": "object"}),
        )

    def execute(self, call):
        request = call.arguments
        actions = {"evaluate": self.evaluate, "collect": self.collect, "fit": self.fit}
        summary, data = actions[call.name](request)
        return LabFeedback(summary, (), False, data={"request": request, **data})

    def workers(self, commands):
        monitor = active_monitor.get()
        if monitor is not None:
            monitor.observe_devices(self.devices)
        processes = []
        try:
            for command, env, log in commands:
                with log.open("x") as stream:
                    process = subprocess.Popen(command, cwd=self.root, env=env, stdout=stream,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                processes.append((process, log))
            while processes:
                if monotonic() > self.deadline:
                    raise TimeoutError("STaR reached its configured run budget")
                for process, log in processes[:]:
                    code = process.poll()
                    if monitor is not None and log.name == "worker.log":
                        monitor.read_training_log(log)
                    if code is None:
                        continue
                    if code:
                        raise RuntimeError(f"STaR worker exited with {code}; see {log}")
                    processes.remove((process, log))
                if processes:
                    sleep(1)
        finally:
            for process, _ in processes:
                stop_process(process)

    def generate(self, items, model, directory, *, hinted=False):
        directory.mkdir()
        commands, receipts = [], []
        for rank, device in enumerate(self.devices):
            shard = items[rank::len(self.devices)]
            if not shard:
                continue
            prefix = directory / f"gpu-{rank}"
            input_path, output = prefix.with_suffix(".input.json"), prefix.with_suffix(".result.json")
            payload = [{"id": item["id"], "question": item["question"]} for item in shard]
            if hinted:
                for request, item in zip(payload, shard, strict=True):
                    request["hint"] = item["answer"]
            env = dict(os.environ, NVIDIA_VISIBLE_DEVICES=device['uuid'], CUDA_VISIBLE_DEVICES=device['index'])
            commands.append(([sys.executable, '-B', '-m', 'rsisys.algorithms.star.sample',
                              str(self.models[model]), str(input_path), str(output)], env, prefix.with_suffix('.log')))
            atomic_json(input_path, payload)
            receipts.append((output, [item["id"] for item in shard]))
        self.workers(commands)
        generated = {}
        for path, ids in receipts:
            responses = read_json(path)
            if [row["id"] for row in responses] != ids:
                raise RuntimeError(f"Unexpected STaR generation identities in {path}")
            generated.update((row["id"], row["text"]) for row in responses)
        return generated

    def evaluate(self, request):
        split, model, label = request["split"], request["model"], request["label"]
        if split not in ("dev", "test"):
            raise ValueError("Evaluation must use a fixed held-out split")
        items = self.groups[split]
        directory = self.root / f"evaluation-{label}"
        outputs = self.generate(items, model, directory)
        rows = [{"id": row["id"], "text": outputs[row["id"]],
                 **score_math(outputs[row["id"]], row["answer"])} for row in items]
        write_records(directory / "scores.jsonl", rows)
        metrics = {"correct": sum(row["correct"] for row in rows), "total": len(rows),
                   "parsed": sum(row["parsed"] for row in rows)}
        metrics["accuracy"] = metrics["correct"] / metrics["total"]
        atomic_json(directory / "metrics.json", metrics)
        return f"GSM8K {label}: {metrics['correct']}/{metrics['total']}", {
            "protocol": f"GSM8K STaR {label}", "quality_metrics": metrics, "model": model,
            "evidence": str(directory / "scores.jsonl")}

    def collect(self, request):
        number, rationalize, _ = self.schedule[request["round"] - 1]
        expected_model = "base" if number == 1 else f"round-{number - 1}"
        if request["model"] != expected_model or request["rationalize"] is not rationalize:
            raise ValueError("STaR collection differs from the registered curriculum")
        items = self.groups["train"]
        directory = self.root / f"round-{number}"
        directory.mkdir()
        outputs = self.generate(items, expected_model, directory / "plain")
        accepted, misses = [], []
        for item in items:
            row = keep_reasoning(item, outputs[item["id"]])
            if row is None:
                misses.append(item)
            else:
                accepted.append({**row, "origin": "plain"})
        plain_count = len(accepted)
        if rationalize and misses:
            hinted = self.generate(misses, expected_model, directory / "rationalized", hinted=True)
            for item in misses:
                row = keep_reasoning(item, hinted[item["id"]])
                if row is not None:
                    accepted.append({**row, "origin": "rationalized"})
        path = directory / "accepted.jsonl"
        write_records(path, accepted)
        metrics = {"generated_plain": len(items), "accepted_plain": plain_count,
                   "generated_rationalized": len(misses) if rationalize else 0,
                   "accepted_rationalized": len(accepted) - plain_count,
                   "accepted_total": len(accepted)}
        atomic_json(directory / "collection.json", {"model": expected_model, **metrics})
        self.datasets[number] = path
        return f"第 {number} 轮保留 {len(accepted)}/{len(items)} 条推理", {
            "dataset": str(path), "collection": metrics}

    def fit(self, request):
        number, _, steps = self.schedule[request["round"] - 1]
        dataset = self.datasets[number]
        if (request["dataset"] != str(dataset) or request["steps"] != steps):
            raise ValueError("STaR fit request differs from the committed data and schedule")
        directory = dataset.parent / "fit"
        directory.mkdir()
        filtered = directory / "supervised.jsonl"
        dropped = filter_overlength(dataset, filtered, self.base)
        command = [sys.executable, "-B", "-m", "torch.distributed.run", "--standalone",
                   "--nproc-per-node", "4", "--module", "rsisys.algorithms.star.fit",
                   "--base-model", str(self.base), "--train-jsonl", str(filtered),
                   "--validation-jsonl", str(self.dev_file), "--output-dir", str(directory),
                   "--seed", str(self.seed + number), "--max-steps", str(steps),
                   "--dropped-overlength", str(dropped), "--warmup-steps", str(self.warmup_steps)]
        self.workers([(command, os.environ.copy(), directory / "worker.log")])
        receipt = read_json(directory / "training_result.json")
        checkpoint = directory / "checkpoint"
        if (receipt["optimizer_steps"] != steps or receipt["text_tensors_changed"] < 1
                or receipt["base_model"] != str(self.base)
                or receipt["checkpoint"] != str(checkpoint)
                or not (checkpoint / "model.safetensors.index.json").is_file()):
            raise RuntimeError("STaR did not produce the expected updated checkpoint")
        model = f"round-{number}"
        self.models[model] = checkpoint
        metrics = {key: receipt[key] for key in (
            "optimizer_steps", "text_tensors_changed", "completion_loss_first",
            "completion_loss_last", "train_rows", "dropped_overlength")}
        return f"第 {number} 轮完成 {steps} 次更新", {
            "model": model, "checkpoint": str(checkpoint), "training_metrics": metrics}
