"""The pinned OPRO GSM8K grader as a MathLab answer verifier.

It reproduces the author evaluator (evaluate_single_instruction) for one decode,
no second extraction prompt and numeric predictions: the reply is normalized by
metrics.get_normalized_prediction, cut at its first line and any "Q:", then
compared with the reference by get_accuracy_of_list. OPRO's dataset manifest
selects it as "opro_official"; any other verifier in the catalog can replace it.
"""
from rsicomposer.api import AnswerCheck, AnswerVerifier


class OPROVerifier(AnswerVerifier):
    name = 'opro-official-gsm8k'
    answer_format = 'End with: The answer is <number>.'

    def __init__(self, eval_utils):
        self.eval_utils = eval_utils

    @property
    def spec(self):
        return {'name': 'opro_official'}

    def parse(self, text):
        choice = self.eval_utils.metrics.get_normalized_prediction(
            text, treat_as_number=True, num_decimals=0, treat_as_bool=False)
        # The author's _extract_second_round_answer_for_parsing, applied to every choice.
        return choice.strip(':').strip().split('\n')[0].split('Q:')[0]

    def parse_reference(self, answer):
        # The author compares against the raw dataset answer.
        return answer if isinstance(answer, str) and answer.strip() else None

    def check(self, completion, answer):
        if self.parse_reference(answer) is None:
            raise ValueError(f'Reference answer is invalid for {self.name}')
        choice = self.parse(completion) if isinstance(completion, str) else ''
        accuracy = self.eval_utils.get_accuracy_of_list(
            true_answer=answer, pred_answer_list=[choice], input_text='', treat_include_as_correct=False)
        return AnswerCheck(bool(choice), bool(accuracy == 1), choice or None)
