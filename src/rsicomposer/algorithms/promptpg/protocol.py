"""Pinned PromptPG TabMWP rendering, cohorts and answer scoring."""

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import random
import re
from string import ascii_uppercase
from types import SimpleNamespace

PROMPT_OPTIONS = SimpleNamespace(option_inds=list(ascii_uppercase), prompt_format="TQ-SA")


def load_author(source, name):
    path = source / "run_gpt3_rl" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"promptpg_author_{name}", path)
    module = importlib.util.module_from_spec(spec)
    state = random.getstate()
    try:
        spec.loader.exec_module(module)
    finally:
        random.setstate(state)
    return module


def split_data(train, test, seed, train_count=160, candidate_count=20, test_count=1000):
    rng = random.Random(seed)
    sample = rng.sample(sorted(train), train_count + candidate_count)
    heldout = random.Random(11010).sample(sorted(test), test_count)
    if set(sample) & set(heldout):
        raise ValueError("TabMWP question IDs overlap across splits")
    return sample[:train_count], sample[train_count:], heldout


def choice_pairs(scores, mode, rng):
    if mode == "random":
        return [rng.sample(range(len(row)), 2) for row in scores]
    return [sorted(range(len(row)), key=lambda i: (-row[i], i))[:2] for row in scores]


def score_answer(completion, item, utils):
    matches = re.findall(r"(?im)^[ \t]*(?:the answer is|final answer[ \t]*:)[ \t]*(.+)$", completion)
    if not matches:
        return False
    final = matches[-1].strip()
    choices = item["choices"]
    if choices:
        label = re.fullmatch(r"\(?([A-Z])\)?[.。]?", final, re.I)
        if label and ascii_uppercase.index(label[1].upper()) >= len(choices):
            return False
    prediction = utils.extract_prediction(final, choices, list(ascii_uppercase))
    normalize = utils.normalize_answer
    return normalize(prediction.strip(), item["unit"]).casefold() == normalize(
        item["answer"], item["unit"]).casefold()


def load_cohort(config):
    source = Path(config['source_dir']).resolve(strict=True)
    for name, expected in SOURCE_HASHES.items():
        if sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'PromptPG author source differs: {name}')
    data = source / 'data/tabmwp'
    train = json.loads((data / 'problems_train.json').read_text())
    test = json.loads((data / 'problems_test.json').read_text())
    if len(train) != 23059 or len(test) != 7686:
        raise ValueError('Unexpected TabMWP release size')
    train_ids, candidates, test_ids = split_data(
        train, test, config['seed'], config['train_count'], config['candidate_count'], config['test_count'])
    questions = {key: train[key] for key in train_ids + candidates} | {key: test[key] for key in test_ids}
    return source, questions, train_ids, candidates, test_ids


SOURCE_HASHES = {
    'data/tabmwp/problems_train.json': '63e57d9a212740f080d4e08657d7aeda3d9f95b26c74a1bdfd9c576846d9b828',
    'data/tabmwp/problems_test.json': 'c47607dad36477725087cb4e5317fe1e89a69dfed404eca430673bda36fda45c',
    'run_gpt3_rl/base_prompt.py': '10f8fd7c74ab561fc46d60b4ca65e7d342cdcfae4fe714cc351c5a73e290ca8b',
    'run_gpt3_rl/utilities.py': '4b0e1040581f603a298ee9df8eba0aff37b2650323e17b4c844549563343fac9',
}
