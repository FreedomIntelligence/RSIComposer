"""STaR curriculum, numeric rationale filtering and Qwen prompt rendering."""

import json
import re

from rsisys.api import MarkedNumericVerifier


MAX_LENGTH, MAX_NEW_TOKENS = 1024, 768
LEARNING_RATE = 1e-6
VERIFIER = MarkedNumericVerifier()
ANSWER_MARKER = re.compile(r'####|\\boxed\{|(?:final answer|the answer is|答案是)', re.I)


def curriculum(rounds, plain_rounds):
    return [(number, number > plain_rounds, min(int(40 * 1.2 ** (number - 1)), 7912))
            for number in range(1, rounds + 1)]


def keep_reasoning(item, answer):
    if not VERIFIER.is_correct(answer, item["answer"]):
        return None
    marker = list(ANSWER_MARKER.finditer(answer))[-1]
    rationale = re.sub(r'(?:final answer|the answer is|答案是)\s*[:：]?\s*$', '',
                       answer[:marker.start()].strip(), flags=re.I).strip(" \t\n:：*$")
    if not any(ch.isalpha() for ch in rationale):
        return None
    return {"id": item["id"], "prompt": item["question"],
            "completion": rationale + "\n\\boxed{" + str(VERIFIER.parse_reference(item["answer"])) + "}"}


def write_records(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def render_prompt(tokenizer, problem):
    text = problem.rstrip() + '\n\nSolve step by step. End with the final answer in a single \\boxed{...}.'
    return tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
