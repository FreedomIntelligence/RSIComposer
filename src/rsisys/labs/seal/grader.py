"""SEAL's semantic grade prompt; failures never become negative rewards.

Prompt derived from pinned SEAL utils.py; see rsisys.labs.seal.lab for attribution and MIT license. Credentials
are read by the optional OpenAI SDK from the environment, never from run JSON.
"""
import os
import re

GRADE_PROMPT = (
    "You are a grading assistant. Your job is to determine whether a student's answer correctly answers the question based solely on the provided gold answer. "
    "Do not use any outside knowledge. The student answer can include additional information, but it must at least fully convey the gold answer and must not contradict it. "
    "Ignore style, phrasing, or extra details that do not affect correctness. Respond ONLY with 'yes' or 'no'.\n\n"
    "Question: {question}\nGold answer: {gold}\nStudent answer: {pred}\n"
    "Is the student answer correct based solely on the gold answer? Respond 'yes' or 'no'."
)


class SEALSemanticGrader:
    name = 'seal-gpt-4.1-semantic-v1'

    def __init__(self, *, client=None):
        if client is None:
            if not os.environ.get('OPENAI_API_KEY'):
                raise ValueError('SEAL paper profile requires OPENAI_API_KEY for GPT-4.1 semantic grading')
            from openai import OpenAI
            client = OpenAI(timeout=120, max_retries=2)
        self.client = client
        self.cache = {}

    def __call__(self, prediction, gold):
        raise TypeError('Semantic grading requires the question; use score_question')

    def score_question(self, prediction, question, title):
        if not prediction.strip():
            return {'f1': 0.0, 'exact_match': 0.0}
        prompt = GRADE_PROMPT.format(question=f"Topic: {title}\n{question['question']}",
            gold=question['answer'], pred=prediction.strip())
        if prompt not in self.cache:
            response = self.client.responses.create(model='gpt-4.1', input=prompt)
            text = response.output_text.strip()
            yes, no = bool(re.search(r'\byes\b', text, re.I)), bool(re.search(r'\bno\b', text, re.I))
            if yes == no:
                raise ValueError('Ambiguous semantic grader response; refusing to fabricate a reward')
            self.cache[prompt] = float(yes)
        score = self.cache[prompt]
        # These historical internal field names carry a binary semantic score.
        # Public paper-profile metrics use accuracy, never lexical F1/EM.
        return {'f1': score, 'exact_match': score}
