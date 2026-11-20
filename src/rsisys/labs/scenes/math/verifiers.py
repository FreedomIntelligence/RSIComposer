"""Answer verification components; no engine or framework runtime dependency."""
from fractions import Fraction
from functools import lru_cache
from importlib.metadata import version
import json
import re
import subprocess
import sys
from threading import current_thread, main_thread
from typing import Protocol


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


class CompetitionIntegerVerifier:
    """Compare a terminal boxed integer with a separately parsed gold label."""

    def __init__(self, maximum):
        if type(maximum) is not int or maximum not in (999, 99999):
            raise ValueError('maximum must be 999 or 99999')
        self.maximum = maximum
        self.name = f'competition-integer-0-{maximum}-v1'

    def parse_gold(self, value):
        if type(value) is int:
            return value if 0 <= value <= self.maximum else None
        if not isinstance(value, str) or not re.fullmatch(r'[0-9]+', value.strip()):
            return None
        digits = value.strip().lstrip('0') or '0'
        if len(digits) > len(str(self.maximum)):
            return None
        return int(digits)

    def parse(self, text):
        if not isinstance(text, str):
            return None
        # Match only the last marker so an unfinished final box cannot fall
        # back to an earlier value in the reasoning.
        markers = list(re.finditer(r'\\boxed\s*\{', text))
        if not markers:
            return None
        answer = re.fullmatch(r'\s*([0-9]+)\s*\}\s*(?:\$\$?|\\[\]\)])?\s*\.?\s*',
                              text[markers[-1].end():])
        return self.parse_gold(answer.group(1)) if answer else None

    def is_correct(self, completion, answer):
        expected = self.parse_gold(answer)
        return expected is not None and self.parse(completion) == expected


class MathVerifyVerifier:
    """Optional Math-Verify adapter; compare final boxed answers, never reasoning."""
    def __init__(self):
        from math_verify import LatexExtractionConfig, parse, verify
        self.name = f'math-verify-{version("math-verify")}-boxed-v1'
        self._parse, self._verify = parse, verify
        self._extraction = [LatexExtractionConfig(boxed_match_priority=0)]

    def parse(self, text):
        """Return the native parsed expression for callers on the main thread."""
        boxed = final_boxed(text)
        return self._parse_boxed(boxed) if boxed is not None else None

    @lru_cache(maxsize=4096)
    def _parse_boxed(self, boxed):
        return self._parse(boxed, extraction_config=self._extraction,
                           fallback_mode='first_match', extraction_mode='first_match',
                           parsing_timeout=2) or None

    def _score_boxed(self, actual, expected):
        parsed = self._parse_boxed(actual)
        if parsed is None or expected is None:
            return parsed is not None, False
        gold = self._parse_boxed(expected)
        return True, gold is not None and bool(self._verify(gold, parsed, strict=True, timeout_seconds=2))

    def _grade(self, actual, expected):
        if actual is None:
            return False, False
        if current_thread() is main_thread():
            return self._score_boxed(actual, expected)
        # Math-Verify uses SIGALRM on the main thread. Keep its scoring limits
        # in the child and bound its total lifetime, including imports.
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'rsisys.labs.scenes.math._symbolic_worker'],
                input=json.dumps([actual, expected]), capture_output=True, text=True, check=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError('Math-Verify worker exceeded its 10-second deadline') from error
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f'Math-Verify worker exited with code {error.returncode}') from error
        parsed, correct = json.loads(result.stdout)
        return parsed, correct

    def accepts_reference(self, answer):
        return self._grade(final_boxed(answer), None)[0]

    def score(self, completion, answer):
        """Return (parsed, correct) from either the main thread or a Lab worker."""
        return self._grade(final_boxed(completion), final_boxed(answer))

    def is_correct(self, completion, answer):
        return self.score(completion, answer)[1]
