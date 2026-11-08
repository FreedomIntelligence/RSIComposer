"""Final-answer isolation and opt-in real symbolic equivalence checks."""
import unittest
from rsisys.labs.math_verifiers import final_boxed, numeric_answer


class BoxedAnswerTests(unittest.TestCase):
    def test_last_nested_box_and_malformed_answers(self):
        self.assertEqual(final_boxed(r'work \boxed{7}; final \boxed{\frac{1}{\sqrt{2}}}.'),
                         r'\boxed{\frac{1}{\sqrt{2}}}')
        self.assertEqual(final_boxed(r'\fbox {x+1}'), r'\boxed{x+1}')
        for text in ('no final answer 42', r'\boxed{}', r'\boxed{\frac{1}{2}', r'\boxed{1} then \boxed{'):
            self.assertIsNone(final_boxed(text))


class RestEMTests(unittest.TestCase):
    def test_numeric_verifier_rejects_nonfinal_or_ambiguous_answers(self):
        for value in ('42', 'work: 42; answer unclear', '#### 42 or 43', '#### nan', '#### 1/0', r'\boxed{x+1}'):
            self.assertIsNone(numeric_answer(value))
        self.assertEqual(numeric_answer('#### 1,200'), 1200)
        self.assertEqual(numeric_answer(r'\boxed{0.5}'), numeric_answer('#### 1/2'))
