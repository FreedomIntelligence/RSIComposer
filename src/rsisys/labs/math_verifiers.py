"""Answer verification components; no engine or framework runtime dependency."""
from fractions import Fraction
import re
from typing import Protocol
from functools import lru_cache
from importlib.metadata import version


class MathVerifier(Protocol):
    name: str

    def parse(self, text: str) -> object | None: ...
    def is_correct(self, completion: str, answer: str) -> bool: ...


def numeric_answer(text):
    """Require a final #### number or numeric \\boxed{...}; never eval text."""
    boxed = re.findall(r'\\boxed\{([^{}]+)\}', text)
    value = text.rsplit('####', 1)[1].strip() if '####' in text else (boxed[-1].strip() if boxed else '')
    if not re.fullmatch(r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/[+-]?\d+)?', value):
        return None
    try:
        return Fraction(value.replace(',', ''))
    except (ValueError, ZeroDivisionError):
        return None


class StrictNumericVerifier:
    """Strict final numeric answers; not a general symbolic MATH verifier."""
    name = "strict-numeric-v1"
    parse = staticmethod(numeric_answer)

    def is_correct(self, completion, answer):
        expected = self.parse(answer)
        return expected is not None and self.parse(completion) == expected


def final_boxed(text):
    """Extract the last balanced boxed/fbox answer, including nested LaTeX braces."""
    matches = list(re.finditer(r'\\(?:boxed|fbox)\s*\{', text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for index in range(start, len(text)):
        if text[index] == '{':
            depth += 1
        elif text[index] == '}':
            depth -= 1
            if depth == 0:
                return '\\boxed{' + text[start:index] + '}' if text[start:index].strip() else None
    return None


class MathVerifyVerifier:
    """Optional Math-Verify adapter; compare final boxed answers, never reasoning."""
    def __init__(self):
        from math_verify import LatexExtractionConfig, parse, verify
        self.name = f'math-verify-{version("math-verify")}-boxed-v1'
        self._parse, self._verify = parse, verify
        self._extraction = [LatexExtractionConfig(boxed_match_priority=0)]

    @lru_cache(maxsize=4096)
    def parse(self, text):
        boxed = final_boxed(text)
        if boxed is None:
            return None
        parsed = self._parse(boxed, extraction_config=self._extraction,
                             fallback_mode='first_match', extraction_mode='first_match',
                             parsing_timeout=2)
        return parsed or None

    def is_correct(self, completion, answer):
        expected, actual = self.parse(answer), self.parse(completion)
        return expected is not None and actual is not None and bool(
            self._verify(expected, actual, strict=True, timeout_seconds=2))
