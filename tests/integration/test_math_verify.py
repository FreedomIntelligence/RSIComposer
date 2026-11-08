import os
import unittest
from rsisys.labs.math_verifiers import MathVerifyVerifier

@unittest.skipUnless(os.environ.get('RSISYS_MATH_VERIFY_INTEGRATION') == '1',
                     'opt-in pinned Math-Verify integration')
class SymbolicVerifierTests(unittest.TestCase):
    def test_symbolic_equivalence_and_no_reasoning_fallback(self):
        verifier = MathVerifyVerifier()
        for gold, predicted in ((r'\boxed{\frac{1}{2}}', r'\boxed{0.5}'),
                                (r'\boxed{\sqrt{2}}', r'\boxed{\frac{2}{\sqrt{2}}}'),
                                (r'\boxed{(x+1)^2}', r'\boxed{x^2+2x+1}'),
                                (r'\boxed{\{1,2\}}', r'\boxed{\{2,1\}}')):
            with self.subTest(gold=gold):
                self.assertTrue(verifier.is_correct(predicted, gold))
        for predicted, gold in ((r'\boxed{2}', r'\boxed{-2}'),
                                (r'\boxed{(2,1)}', r'\boxed{(1,2)}'),
                                ('reasoning contains 42 but has no final answer', r'\boxed{42}'),
                                (r'work \boxed{42}; final \boxed{41}', r'\boxed{42}')):
            self.assertFalse(verifier.is_correct(predicted, gold))
