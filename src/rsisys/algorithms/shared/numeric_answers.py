"""Deterministic GSM8K scoring; no eval, model judge, or execution."""
from fractions import Fraction
import re

NUMBER = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:\s*/\s*[-+]?\d+)?"


def extract_answer(text, *, require_marker=True):
    if not isinstance(text, str) or len(text) > 200000:
        return None
    markers = []
    for pattern in (r"####\s*([^\n]+)", r"\\boxed\{([^{}]+)\}",
                    r"(?:final answer|the answer is|答案是)\s*[:：]?\s*([^\n]+)"):
        markers.extend((match.start(), match.group(1).strip())
                       for match in re.finditer(pattern, text, re.I))
    if markers:
        # A later explicit correction supersedes an earlier answer, whatever
        # marker style it uses. Never fall back to an older parsable answer.
        value = max(markers, key=lambda item: item[0])[1]
    elif not require_marker:
        value = text.strip()
    else:
        return None
    value = value.strip(" $￥¥€£").rstrip("。")
    # Permit a final sentence period, never extract arbitrary intermediate numbers.
    if value.endswith("."):
        value = value[:-1]
    if not re.fullmatch(NUMBER, value) or len(value) > 256:
        return None
    exponent = re.search(r"[eE]([-+]?\d+)", value)
    if exponent and (len(exponent[1].lstrip('+-')) > 4 or abs(int(exponent[1])) > 1024):
        return None
    try:
        return Fraction(value.replace(",", "").replace(" ", ""))
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def score_math(prediction, reference):
    expected = extract_answer(reference, require_marker=False)
    if expected is None:
        raise ValueError("Reference answer is not a finite supported number")
    observed = extract_answer(prediction)
    return {"correct": observed is not None and observed == expected,
            "parsed": observed is not None, "predicted": str(observed) if observed is not None else None,
            "expected": str(expected), "protocol": "gsm8k-explicit-final-exact-fraction/v2"}
