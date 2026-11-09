"""Freeze disjoint RRSI search/heldout tasks from a prepared MATH release."""

import argparse
import hashlib
import json
import random
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare(source_dir, output_dir, search_size=64, heldout_size=128, seed=7308):
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    train_path, test_path = source_dir / "train.jsonl", source_dir / "test.jsonl"
    train, heldout = read_rows(train_path), read_rows(test_path)
    if not 1 <= search_size <= len(train) or not 1 <= heldout_size <= len(heldout):
        raise ValueError("Requested split size is invalid")
    search = train[:search_size]
    random.Random(seed).shuffle(heldout)
    heldout = heldout[:heldout_size]
    for key in ("id", "prompt"):
        for rows in (search, heldout):
            if len({r[key] for r in rows}) != len(rows):
                raise ValueError("Duplicate " + key)
        if {r[key] for r in search} & {r[key] for r in heldout}:
            raise ValueError("Search/heldout overlap by " + key)
    output_dir.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, rows in (("search", search), ("heldout", heldout)):
        path = output_dir / (name + ".jsonl")
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "search_size": search_size,
        "heldout_size": heldout_size,
        "seed": seed,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (train_path, test_path)
        },
        "output_sha256": hashes,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--search-size", type=int, default=64)
    parser.add_argument("--heldout-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7308)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.source_dir,
                args.output_dir,
                args.search_size,
                args.heldout_size,
                args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
