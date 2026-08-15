"""Mechanical, judge-free scoring of an answer against hand-extracted fact anchors.

The supersession probe compares two arms on 11 rows. An LLM judge would cost money, would need
pinning, and at n=11 would add a second source of variance to a measurement that already cannot
support an effect size. `answer_facts` in EnterpriseRAG-Bench carry hard anchors instead:
`POST /v1/capacity/migrations/start`, `+$0.085`, `sha256-only`, `70-84`. Those are checkable with
`in`.

⚠️ **This measures TOKEN PRESENCE, not correctness**, and the distinction is the whole reason the
pre-registration reports only the DELTA between arms. A correct answer phrased differently scores
as a miss, so the absolute rate is a lower bound. An answer that lists every anchor while asserting
the opposite scores 1.0. Both arms are scored by the same instrument, so a difference between them
is interpretable in a way that neither absolute number is.

Three anchor kinds, and the third is what keeps the denominator honest:

- **positive**: at least one spelling in `any`, or every string in `all`, must appear.
- **negative**: none of the strings may appear. These are the sharpest, because "must not invent
  `/v1/capacity/migration/start`" is exactly checkable.
- **unanchorable**: a fact deliberately NOT scored, recorded with the reason. "Must not present the
  legacy endpoint as primary" is about emphasis, and no substring decides it. Counting those facts
  in the denominator would understate both arms; dropping them silently would overstate the
  instrument's reach. They are listed so the human read knows where to look.

⚠️ **The anchor file is NOT committed**, for the same reason the evidence fixture is not: 46 literal
strings drawn from the answer key of a live benchmark are answer-key material. Its SHA-256 is
published in the pre-registration, and the schema is here so the file can be rebuilt.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Fixed BEFORE the run, so it cannot be tuned afterwards to flatter an arm. Normalisation is
#: deliberately shallow: casefold, collapse whitespace, and unify the dash and comparison glyphs a
#: model may render differently from the source. Nothing stems, nothing strips punctuation, because
#: a scorer that normalises aggressively starts matching things that are not there.
_WS = re.compile(r"\s+")
_DASHES = str.maketrans({"–": "-", "—": "-", "−": "-"})
_COMPARATORS = {"≥": ">=", "≤": "<="}


def normalize(text: str) -> str:
    """The one normalisation both arms and every anchor pass through."""
    text = unicodedata.normalize("NFKC", text)
    for glyph, plain in _COMPARATORS.items():
        text = text.replace(glyph, plain)
    text = text.translate(_DASHES)
    return _WS.sub(" ", text).casefold().strip()


@dataclass(frozen=True)
class RowScore:
    question_id: str
    hits: int
    total: int
    missed: tuple[str, ...]
    violated: tuple[str, ...]

    @property
    def rate(self) -> float:
        """Undefined rather than 1.0 when a row has no scorable anchor.

        A row with an empty denominator has not scored perfectly, it has not been measured, and
        returning 1.0 would let an unanchorable row inflate an arm.
        """
        if not self.total:
            raise ValueError(f"{self.question_id} has no scorable anchors; rate is undefined")
        return self.hits / self.total


def _present(haystack: str, anchor: Mapping[str, Any]) -> bool:
    if "any" in anchor:
        return any(normalize(str(s)) in haystack for s in anchor["any"])
    if "all" in anchor:
        return all(normalize(str(s)) in haystack for s in anchor["all"])
    raise ValueError(f"anchor {anchor.get('id')!r} has neither 'any' nor 'all'")


def score_row(answer: str, row: Mapping[str, Any]) -> RowScore:
    """Score one answer. A negative anchor scores a HIT when its strings are absent."""
    haystack = normalize(answer)
    hits = 0
    missed: list[str] = []
    violated: list[str] = []

    for anchor in row.get("positive", ()):
        if _present(haystack, anchor):
            hits += 1
        else:
            missed.append(str(anchor["id"]))
    for anchor in row.get("negative", ()):
        if _present(haystack, anchor):
            violated.append(str(anchor["id"]))
        else:
            hits += 1

    total = len(row.get("positive", ())) + len(row.get("negative", ()))
    return RowScore(
        question_id=str(row.get("question_id", "?")),
        hits=hits,
        total=total,
        missed=tuple(missed),
        violated=tuple(violated),
    )


def anchors_digest(anchors: Mapping[str, Any]) -> str:
    """Same canonicalisation as the evidence fixture, for the same auditability reason."""
    payload = json.dumps(anchors, indent=1, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_anchors(path: Path) -> dict[str, dict[str, Any]]:
    """Load and structurally validate, refusing a file that cannot score what it claims to."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, Any]] = payload["rows"]
    for question_id, row in rows.items():
        row["question_id"] = question_id
        positive: Sequence[Mapping[str, Any]] = row.get("positive", ())
        negative: Sequence[Mapping[str, Any]] = row.get("negative", ())
        if not positive and not negative:
            raise ValueError(f"{question_id} has no scorable anchors")
        seen: set[str] = set()
        for anchor in [*positive, *negative]:
            anchor_id = str(anchor.get("id", ""))
            if not anchor_id:
                raise ValueError(f"{question_id} has an anchor with no id")
            if anchor_id in seen:
                raise ValueError(f"{question_id} repeats anchor id {anchor_id!r}")
            seen.add(anchor_id)
            if ("any" in anchor) == ("all" in anchor):
                raise ValueError(f"{question_id}:{anchor_id} needs exactly one of any/all")
            strings = anchor.get("any") or anchor.get("all") or ()
            if not strings or any(not str(s).strip() for s in strings):
                raise ValueError(f"{question_id}:{anchor_id} has an empty match string")
    return rows


__all__ = [
    "RowScore",
    "anchors_digest",
    "load_anchors",
    "normalize",
    "score_row",
]
