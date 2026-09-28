"""W5: pin the user's own stated preferences at the top of an advice-shaped Search.

Round two, 2026-09-28. E1 (preferences) scored 41.46 on the first Textual Full; on LongMemEval-S the
preference questions score 0.45 with every piece of evidence already in the top 10, and PrefEval
finds preferences ignored in long contexts. Zep moved LongMemEval-S preference from 30.0 to 53.3 with
gpt-4o-mini by showing preferences as facts. This selects W1 facts whose relation is ``preference``
and whose words overlap the question, and renders each as the user's own dated, quoted statement,
for the first slots of a Search that asks for advice. The statement is verbatim (W1 keeps a fact
only when its value is in the cited words), so the item is evidence, never an answer.

Pure functions; serving waits for W1's facts to be stored and returned (W1 stage B).
"""

from __future__ import annotations

from collections.abc import Sequence
import re

from recall_aml.conversation_records import ConversationFact

_ADVICE = re.compile(
    r"\b(?:recommend\w*|suggest\w*|advice|advise|ideas?|tips?|what should i|which (?:\w+ ){0,3}should i|"
    r"can you (?:help me )?(?:choose|pick|plan|find)|help me (?:choose|pick|plan|find|decide)|"
    r"any (?:good )?(?:options|places|books|recipes|movies|ways))\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are for how i in is it me my of on or should some the to what which with you your can any".split()
)
MAX_PINNED = 3


def is_advice_query(query: str) -> bool:
    return bool(_ADVICE.search(query))


def _terms(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.casefold()) if word not in _STOP and len(word) > 2}


def _when(fact: ConversationFact) -> float:
    return fact.mention_time.timestamp() if fact.mention_time else 0.0


def select_preferences(facts: Sequence[ConversationFact], query: str, *, limit: int = MAX_PINNED) -> list[ConversationFact]:
    """Preference facts sharing words with the question, most overlap first, newest first on ties;
    one per key, keeping the key's newest statement."""
    wanted = _terms(query)
    newest: dict[str, ConversationFact] = {}
    for fact in facts:
        if fact.relation != "preference":
            continue
        current = newest.get(fact.key)
        if current is None or _when(fact) >= _when(current):
            newest[fact.key] = fact
    scored = [
        (len(wanted & (_terms(f.value) | _terms(f.subject) | _terms(f.attribute))), f)
        for f in newest.values()
    ]
    scored = [item for item in scored if item[0] > 0]
    scored.sort(key=lambda item: (item[0], _when(item[1])), reverse=True)
    return [fact for _, fact in scored[:limit]]


def rendered_preference(fact: ConversationFact) -> str:
    day = fact.mention_time.date().isoformat() if fact.mention_time else "undated"
    who = fact.speaker or "User"
    return f'{who} stated ({day}): "{fact.value}"'
