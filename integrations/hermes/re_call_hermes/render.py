"""Turn RE-call's compact `recall_search` answer into the block Hermes injects before a turn.

Three outcomes, and the model must be able to tell them apart:

- citable hits: the ones RE-call says a present-state answer may cite (`evidence.citable`);
- retired hits: superseded or expired memories the server returned, flagged so the model does not
  state them as current;
- abstention: nothing cleared RE-call's calibrated threshold for this message.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

HEADER = "## RE-call memory"
RETIRED_VERDICTS = frozenset({"superseded", "expired"})
MAX_HIT_CHARS = 600

ABSTAIN_LINE = (
    "RE-call abstained: nothing in memory passed its calibrated trust check for this message. "
    "Do not answer questions about the present from memory."
)
CITABLE_LEAD = (
    "These memories passed RE-call's calibrated trust check for this message. "
    "Name the memo id when you rely on one."
)
RETIRED_LEAD = "Retired memories (superseded or expired): do not state these as current."


@dataclass(frozen=True)
class Rendered:
    """The injected text and how many citable memories it holds (for Hermes's indicator)."""

    text: str
    citable: int


def _clip(text: str) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= MAX_HIT_CHARS else flat[: MAX_HIT_CHARS - 3] + "..."


def _validity(hit: Mapping[str, Any]) -> str:
    parts = []
    if hit.get("valid_from"):
        parts.append(f"valid from {hit['valid_from']}")
    if hit.get("valid_until"):
        parts.append(f"valid until {hit['valid_until']}")
    return f" ({', '.join(parts)})" if parts else ""


def _citable_ids(answer: Mapping[str, Any]) -> set[str] | None:
    evidence = answer.get("evidence")
    if not isinstance(evidence, Mapping):
        return None  # a server run with RECALL_SEARCH_EVIDENCE=off sends none
    return {str(c) for c in evidence.get("citable") or []}


def render_search(answer: Mapping[str, Any]) -> Rendered:
    """The prefetch block for one compact `recall_search` answer."""
    hits = [h for h in answer.get("hits") or [] if isinstance(h, Mapping)]
    citable_ids = _citable_ids(answer)
    abstained = bool(answer.get("abstained"))

    citable: list[Mapping[str, Any]] = []
    retired: list[Mapping[str, Any]] = []
    for hit in hits:
        verdict = str(hit.get("verdict") or "")
        if verdict in RETIRED_VERDICTS:
            retired.append(hit)
        elif not abstained and verdict == "ok" and (
            citable_ids is None or str(hit.get("chunk_id")) in citable_ids
        ):
            citable.append(hit)

    lines = [HEADER]
    if citable:
        lines.append(CITABLE_LEAD)
        lines += [
            f"- [ok] memo {h.get('memo_id')}{_validity(h)}: {_clip(h.get('text', ''))}" for h in citable
        ]
    else:
        lines.append(ABSTAIN_LINE)
    if retired:
        lines.append(RETIRED_LEAD)
        for h in retired:
            replaced = f", replaced by {h['superseded_by']}" if h.get("superseded_by") else ""
            lines.append(
                f"- [{h.get('verdict')}{replaced}] memo {h.get('memo_id')}{_validity(h)}: "
                f"{_clip(h.get('text', ''))}"
            )
    return Rendered(text="\n".join(lines), citable=len(citable))


def render_unavailable(reason: str) -> Rendered:
    """One line when the server refused the search (for example, a store not yet calibrated)."""
    return Rendered(text=f"{HEADER}\nRE-call memory could not answer this turn: {_clip(reason)}", citable=0)
