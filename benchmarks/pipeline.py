from __future__ import annotations

#: The exact token the generator must emit when the memories don't answer the question.
NO_ANSWER = "NO_ANSWER"


def is_abstention(answer: str) -> bool:
    """True iff the generated answer is exactly the abstention token (case/space-insensitive).

    Requires the WHOLE answer to be the token — an answer that merely mentions ``NO_ANSWER`` in a
    sentence is a real (if odd) answer, not an abstention.
    """
    return answer.strip().casefold() == NO_ANSWER.casefold()
