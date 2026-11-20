"""Run a bounded Math-Verify grade on an isolated Python main thread."""
import json
import sys

from .verifiers import MathVerifyVerifier


def main():
    actual, expected = json.load(sys.stdin)
    json.dump(MathVerifyVerifier().grade_boxed(actual, expected), sys.stdout)


if __name__ == '__main__':
    main()
