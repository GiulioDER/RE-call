from __future__ import annotations

from benchmarks.llm import Completer

#: The exact token the generator must emit when the memories don't answer the question.
NO_ANSWER = "NO_ANSWER"


def is_abstention(answer: str) -> bool:
    """True iff the generated answer is exactly the abstention token (case/space-insensitive).

    Requires the WHOLE answer to be the token — an answer that merely mentions ``NO_ANSWER`` in a
    sentence is a real (if odd) answer, not an abstention.
    """
    return answer.strip().casefold() == NO_ANSWER.casefold()


GEN_SYSTEM_PROMPT = (
    "You answer questions about a conversation using ONLY the provided memories. "
    f"If the answer is not present in the memories, respond with exactly {NO_ANSWER} and nothing "
    "else. Do not use outside knowledge. Keep answers short."
)

JUDGE_SYSTEM_PROMPT = (
    "You are grading whether a predicted answer matches the gold answer to a question. "
    "Reply with exactly YES if the prediction is correct (same meaning as gold), otherwise NO."
)


def generate_answer(completer: Completer, context: str, question: str) -> str:
    user = f"Memories:\n{context}\n\nQuestion: {question}\nAnswer:"
    return completer(GEN_SYSTEM_PROMPT, user).strip()


def judge_correct(completer: Completer, question: str, gold: str, answer: str) -> bool:
    user = f"Question: {question}\nGold answer: {gold}\nPredicted answer: {answer}\nCorrect?"
    verdict = completer(JUDGE_SYSTEM_PROMPT, user).strip().casefold()
    return verdict.startswith("yes")
