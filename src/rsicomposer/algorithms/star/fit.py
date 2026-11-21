"""Fixed-step STaR update on the shared Qwen3.5 FSDP2 training backend."""
import argparse
from itertools import islice
import json
import math
from pathlib import Path

from . import training as full_text
from .protocol import LEARNING_RATE, MAX_LENGTH, MAX_NEW_TOKENS
from .records import GLOBAL_BATCH, optimizer_batches


def fit_fixed(args, base, config, shards, weight_map, tokenizer, verifier, train, dev, *,
              rank, device, output, staging):
    import torch
    import torch.distributed as dist

    checkpoint = full_text.checkpoint
    model = checkpoint.initialize(base, config, shards, rank=rank, device=device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=LEARNING_RATE, foreach=False)
    epochs = math.ceil(args.max_steps / (len(train) // GLOBAL_BATCH))
    batches = islice(optimizer_batches(len(train), epochs=epochs, seed=args.seed), args.max_steps)
    losses = []
    model.train()
    for step, indices in enumerate(batches, 1):
        rate = LEARNING_RATE * min(step / args.warmup_steps, 1.0)
        for group in optimizer.param_groups:
            group["lr"] = rate
        loss = full_text.train_batch(model, optimizer, [train[index] for index in indices],
                                     rank=rank, device=device)
        losses.append(loss)
        if rank == 0:
            print(json.dumps({"step": step, "learning_rate": rate, "completion_loss": loss}), flush=True)
    if len(losses) != args.max_steps:
        raise RuntimeError("STaR optimizer did not reach its scheduled step count")
    snapshot = staging / "trained" if rank == 0 else None
    changed = checkpoint.export(model, base, snapshot, shards, weight_map, rank=rank)
    if rank == 0:
        if not changed:
            raise RuntimeError("STaR update did not change any exported text tensors")
        full_text._publish(output, snapshot, {
            "status": "completed", "optimizer_steps": len(losses), "base_model": str(base),
            "checkpoint": str(output / "checkpoint"), "text_tensors_changed": changed,
            "text_tensors_total": sum(name.startswith(checkpoint.TEXT_PREFIXES) for name in weight_map),
            "train_rows": len(train), "dropped_overlength": args.dropped_overlength,
            "global_batch": GLOBAL_BATCH, "learning_rate": LEARNING_RATE,
            "warmup_steps": args.warmup_steps, "max_length": MAX_LENGTH, "packing": False,
            "completion_loss_first": losses[0], "completion_loss_last": losses[-1],
            "seed": args.seed,
        })
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("base-model", "train-jsonl", "validation-jsonl", "output-dir"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, choices=range(1, 7913), required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--dropped-overlength", type=int, required=True)
    parser.set_defaults(max_length=MAX_LENGTH, validation_max_new_tokens=MAX_NEW_TOKENS,
                        verifier="strict_numeric", qwen_math_chat=True)
    args = parser.parse_args()
    if args.warmup_steps < 1:
        parser.error('warmup steps must be positive')
    full_text.run(args, fit=fit_fixed)


if __name__ == "__main__":
    main()
