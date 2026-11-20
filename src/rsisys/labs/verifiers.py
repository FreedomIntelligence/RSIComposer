"""Answer verifiers: one interface, selected by name in a Lab's config.

A verifier compares a model completion with a reference answer. Labs depend on
the AnswerVerifier interface, never on a concrete class, so the same Lab can
score numeric, competition-integer or symbolic answers by configuration:

    "verifier": "math_verify"
    "verifier": {"name": "competition_integer", "maximum": 999}

Required by every verifier: name (a versioned scorer identity used in
MetricSpec.scorer), parse(text) and is_correct(completion, answer).
Subclassing AnswerVerifier adds parse_reference(), check() and is_correct()
defaults, the answer_format instruction text, and the isolated flag for
verifiers that must grade on a process main thread.
"""
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from importlib.metadata import version
import json
import re
import subprocess
import sys
from threading import current_thread, main_thread


@dataclass(frozen=True)
class AnswerCheck:
    """One graded completion. predicted is the parsed answer as text, when the verifier exposes one."""
    parsed: bool
    correct: bool
    predicted: str | None = None


class AnswerVerifier:
    """Base class. Subclasses set name and implement parse(); override the rest when needed."""
    name: str
    answer_format = 'Put the final answer in one \\boxed{...}.'
    # True when grading must run on a Python main thread (for example SIGALRM
    # timeouts). Labs grading from worker threads then use a process per call.
    isolated = False

    @property
    def spec(self):
        """JSON configuration that rebuilds this verifier with build_verifier()."""
        raise NotImplementedError

    def parse(self, text):
        raise NotImplementedError

    def parse_reference(self, answer):
        """Normalize a gold answer, or return None when this verifier cannot grade it."""
        return self.parse(answer) if isinstance(answer, str) else None

    def check(self, completion, answer):
        expected = self.parse_reference(answer)
        if expected is None:
            raise ValueError(f'Reference answer is invalid for {self.name}')
        predicted = self.parse(completion) if isinstance(completion, str) else None
        return AnswerCheck(predicted is not None, predicted is not None and predicted == expected,
                           None if predicted is None else str(predicted))

    def is_correct(self, completion, answer):
        return self.check(completion, answer).correct


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


class StrictNumericVerifier(AnswerVerifier):
    """A final #### number or numeric box; not a general symbolic MATH verifier."""
    name = 'strict-numeric-v1'
    answer_format = 'Put the final number in one \\boxed{...}.'
    spec = {'name': 'strict_numeric'}
    parse = staticmethod(numeric_answer)


_MARKED_NUMBER = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:\s*/\s*[-+]?\d+)?"


def marked_number(text, *, require_marker=True):
    """Read the last explicitly marked final number (####, \\boxed or 'the answer is')."""
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
    if not re.fullmatch(_MARKED_NUMBER, value) or len(value) > 256:
        return None
    exponent = re.search(r"[eE]([-+]?\d+)", value)
    if exponent and (len(exponent[1].lstrip('+-')) > 4 or abs(int(exponent[1])) > 1024):
        return None
    try:
        return Fraction(value.replace(",", "").replace(" ", ""))
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


class MarkedNumericVerifier(AnswerVerifier):
    """GSM8K scoring: the last marked final number; a bare reference number is accepted."""
    name = 'gsm8k-explicit-final-exact-fraction/v2'
    answer_format = 'End with one line containing #### followed by the numeric answer only.'
    spec = {'name': 'marked_numeric'}
    parse = staticmethod(marked_number)

    def parse_reference(self, answer):
        return marked_number(answer, require_marker=False)


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


class CompetitionIntegerVerifier(AnswerVerifier):
    """Compare a terminal boxed integer with a separately parsed gold label."""

    def __init__(self, maximum=99999):
        if type(maximum) is not int or maximum not in (999, 99999):
            raise ValueError('maximum must be 999 or 99999')
        self.maximum = maximum
        self.name = f'competition-integer-0-{maximum}-v1'
        self.answer_format = f'Put the final integer from 0 through {maximum} in one terminal \\boxed{{...}}.'

    @property
    def spec(self):
        return {'name': 'competition_integer', 'maximum': self.maximum}

    def parse_reference(self, answer):
        if type(answer) is int:
            return answer if 0 <= answer <= self.maximum else None
        if not isinstance(answer, str) or not re.fullmatch(r'[0-9]+', answer.strip()):
            return None
        digits = answer.strip().lstrip('0') or '0'
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
        return self.parse_reference(answer.group(1)) if answer else None


def boxed_reference(answer):
    """A gold answer as one final box: its own last box, or the whole answer boxed."""
    if type(answer) is int:
        answer = str(answer)
    if not isinstance(answer, str) or not answer.strip():
        return None
    return final_boxed(answer) or '\\boxed{' + answer.strip() + '}'


class MathVerifyVerifier(AnswerVerifier):
    """Optional Math-Verify adapter; compare final boxed answers, never reasoning."""
    answer_format = 'Put the final mathematical answer in one \\boxed{...}.'
    spec = {'name': 'math_verify'}
    # Math-Verify uses SIGALRM, which only works on the main thread.
    isolated = True

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

    def grade_boxed(self, actual, expected):
        """(parsed, correct) for two final boxes; must run on the main thread."""
        parsed = self._parse_boxed(actual)
        if parsed is None or expected is None:
            return parsed is not None, False
        gold = self._parse_boxed(expected)
        return True, gold is not None and bool(self._verify(gold, parsed, strict=True, timeout_seconds=2))

    def _grade(self, actual, expected):
        if actual is None:
            return False, False
        if current_thread() is main_thread():
            return self.grade_boxed(actual, expected)
        # Keep Math-Verify's scoring limits in a child main thread and bound
        # its total lifetime, including imports.
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'rsisys.labs._verify_worker'],
                input=json.dumps([actual, expected]), capture_output=True, text=True, check=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError('Math-Verify worker exceeded its 10-second deadline') from error
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f'Math-Verify worker exited with code {error.returncode}') from error
        parsed, correct = json.loads(result.stdout)
        return parsed, correct

    def parse_reference(self, answer):
        """The reference as one final box, when Math-Verify can parse it."""
        boxed = boxed_reference(answer)
        return boxed if boxed is not None and self._grade(boxed, None)[0] else None

    def check(self, completion, answer):
        if not isinstance(completion, str):
            return AnswerCheck(False, False)
        parsed, correct = self._grade(final_boxed(completion), boxed_reference(answer))
        return AnswerCheck(parsed, correct)


def parse_reference(verifier, answer):
    """A verifier's normalized gold answer; parse() for verifiers without parse_reference()."""
    method = getattr(verifier, 'parse_reference', None)
    return method(answer) if callable(method) else verifier.parse(answer)


def score_record(verifier, completion, answer):
    """Per-item evidence fields for one graded completion."""
    check = verifier.check(completion, answer)
    expected = parse_reference(verifier, answer)
    return {'correct': check.correct, 'parsed': check.parsed, 'predicted': check.predicted,
            'expected': None if expected is None else str(expected), 'verifier': verifier.name}


VERIFIERS = {
    'strict_numeric': StrictNumericVerifier,
    'marked_numeric': MarkedNumericVerifier,
    'competition_integer': CompetitionIntegerVerifier,
    'math_verify': MathVerifyVerifier,
}


def _options(spec):
    if isinstance(spec, str):
        spec = {'name': spec}
    if not isinstance(spec, dict) or spec.get('name') not in VERIFIERS:
        raise ValueError('verifier must be one of ' + ', '.join(sorted(VERIFIERS))
                         + ', as a name or an object with name and options')
    options = dict(spec)
    return options.pop('name'), options


def validate_verifier(spec, *, allowed=None):
    """Check a verifier selection without importing optional dependencies."""
    name, options = _options(spec)
    if allowed is not None and name not in allowed:
        raise ValueError('verifier must be one of ' + ', '.join(sorted(allowed)))
    if name == 'competition_integer':
        if set(options) - {'maximum'}:
            raise ValueError('competition_integer accepts only maximum')
        CompetitionIntegerVerifier(**options)
    elif options:
        raise ValueError(f'{name} takes no options')
    return name


def build_verifier(spec):
    """Construct a verifier from a name or {"name": ..., options}."""
    validate_verifier(spec)
    name, options = _options(spec)
    return VERIFIERS[name](**options)
