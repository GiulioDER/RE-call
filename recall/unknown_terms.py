"""Abstain when a question names something the memory has never contained.

A calibrated threshold on top cosine answers "is the memory ABOUT this question?", not "does it
STATE the answer?". A question that differs from an answerable one by a single name it has never
seen ("In QQ01, what happens when I press ACTION1?", against lessons about AR25) scores as high as
the real thing: measured 2026-10-08 on lessons about seventeen ARC-AGI-3 public games, 20 of 20
such questions about invented games got a trusted hit, every one about another game, at
confidence 0.77 to 1.00. The score cannot see the difference; the vocabulary can. If the question's own identifier for its subject never occurs in the memory,
no hit can be about that subject, however similar the rest of the sentence is.

Only ENTITY-LIKE tokens are checked: a token of three or more characters that contains both a
letter and a digit (``QQ01``, ``A320``), is written in capitals (``SKU``, ``CERBERUS``), or has a
capital after its first letter (``PgVector``). Plain words are never checked, because a question
is phrased in words its answer need not contain ("happens", "worth", a misspelling), and refusing
on those would refuse real answers. This covers the near-miss whose subject is a name the memory
lacks. It does not cover a near-miss built entirely from the memory's own words ("the URL of the
portal named in Amendment A-1"); that needs a reader, not a vocabulary.

The check needs a store that can say whether a term occurs anywhere in it: ``unknown_terms``.
The lite store answers from its FTS5 vocabulary. A store without the method is skipped and the
result says so (``unknown_term_check="unavailable"``); nothing about its verdicts changes.
``RECALL_UNKNOWN_TERM_GATE=0`` switches the gate off and the result says that too.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Final

from recall.trust_verdicts import decision_state_for
from recall.types import TrustedHit, TrustedResult, Verdict

#: A run of letters and digits, the way a person writes an identifier; split on everything else so
#: "A-320" yields "A" and "320", which the length floor then drops or keeps on their own.
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_MIN_LENGTH = 3

GATE_ENV = "RECALL_UNKNOWN_TERM_GATE"
VERDICT: Final[Verdict] = "unknown_term"


def entity_like_tokens(text: str) -> list[str]:
    """The query's tokens that name something, in first-seen order, without repeats."""
    seen: dict[str, None] = {}
    for token in _TOKEN.findall(text):
        if len(token) < _MIN_LENGTH:
            continue
        has_letter = any(c.isalpha() for c in token)
        has_digit = any(c.isdigit() for c in token)
        letters = [c for c in token if c.isalpha()]
        capitals = len(letters) >= 2 and all(c.isupper() for c in letters)
        inner_capital = any(c.isupper() for c in token[1:])
        if (has_letter and has_digit) or capitals or inner_capital:
            seen.setdefault(token, None)
    return list(seen)


@dataclass(frozen=True)
class UnknownTermCheck:
    """What the gate found. ``status`` is ``checked``, ``unavailable`` or ``disabled``."""

    status: str
    unknown: tuple[str, ...] = ()


def gate_enabled(env: Mapping[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    return source.get(GATE_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def check_unknown_terms(store: Any, query: str, env: Mapping[str, str] | None = None) -> UnknownTermCheck:
    if not gate_enabled(env):
        return UnknownTermCheck("disabled")
    lookup = getattr(store, "unknown_terms", None)
    if not callable(lookup):
        return UnknownTermCheck("unavailable")
    tokens = entity_like_tokens(query)
    if not tokens:
        return UnknownTermCheck("checked")
    return UnknownTermCheck("checked", tuple(lookup(tokens)))


def unknown_term_reason(unknown: tuple[str, ...]) -> str:
    names = ", ".join(repr(t) for t in unknown)
    return (
        f"the question names {names}, which this memory has never contained, so no retrieved "
        f"memory can be about it (unknown term)"
    )


def apply_unknown_term_gate(trusted: TrustedResult, check: UnknownTermCheck) -> TrustedResult:
    """Demote every ``ok`` hit to ``unknown_term`` when the question names an absent term.

    Only ``ok`` hits move: a superseded or expired hit keeps the verdict that explains it. Order
    is kept, which is still valid-first because nothing is valid any more.
    """
    diagnostics = replace(trusted.diagnostics, unknown_term_check=check.status,
                          unknown_terms=check.unknown)
    if not check.unknown:
        return replace(trusted, diagnostics=diagnostics)
    hits: list[TrustedHit] = [
        replace(h, verdict=VERDICT) if h.verdict == "ok" else h for h in trusted.hits
    ]
    return replace(
        trusted,
        hits=hits,
        abstained=True,
        reason=unknown_term_reason(check.unknown),
        decision_state=decision_state_for(hits, gap_warning=trusted.gap_warning),
        diagnostics=diagnostics,
    )
