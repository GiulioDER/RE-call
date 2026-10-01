"""How much evidence to return, decided from what retrieval found.

A long PDF answers badly from five chunks. Measured on 114 held-out MMLongBench-Doc questions
(recall-lab PI-3, 2026-09-29), handing the reader the top 20 chunks instead of `recall_evidence`'s
default 5 raised judged accuracy from 0.290 to 0.421 (+0.132 [+0.061, +0.202]), and accuracy was
flat past 20. Memos, markdown and conversational memory showed no such effect elsewhere
(BEAM, 2026-09-24), so the wider budget applies only when the results come from paged documents.

The decision is made on the ranked pool BEFORE the trust gate, from metadata every hit already
carries, so it costs no extra I/O. It is deterministic and never reads the query or the corpus
beyond the hits in hand.

- A hit is **paged** when its extraction recorded `source_format` "pdf" or "pptx" (PPT and ODP are
  converted to PPTX by the extractor, so they record "pptx"). Rows indexed before extraction
  stamped the format fall back to the file suffix.
- A pool is **paged** when at least `PAGED_VOTE_THRESHOLD` of its first `PAGED_VOTE_WINDOW` hits
  are paged. The window is the default evidence depth, so the vote reads exactly the hits a
  standard answer would have shown.
- The switch is `RECALL_PAGED_EVIDENCE` (`off` or `on`), default `off`: the measured gain came from
  ungated retrieval, and the trust-gated path is validated before it becomes the default.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from recall.types import ScoredChunk

#: Extraction formats whose documents carry page or slide structure.
PAGED_SOURCE_FORMATS = frozenset({"pdf", "pptx"})
#: Fallback for rows without `source_format`: every suffix the extractor records as "pdf" or
#: "pptx" (`recall.extraction`; PPT and ODP convert to PPTX there).
PAGED_SUFFIXES = (".pdf", ".pptx", ".pptm", ".potx", ".potm", ".ppt", ".odp")
#: The vote reads the first five hits: the depth a standard evidence answer shows.
PAGED_VOTE_WINDOW = 5
#: A strict majority of the window must be paged, so a mixed pool keeps the standard depth.
PAGED_VOTE_THRESHOLD = 3

PagedEvidenceMode = Literal["off", "on"]
PagedSignal = Literal["metadata", "suffix", "none"]


@dataclass(frozen=True)
class PagedDepthDecision:
    """What the depth selector decided for one request, and on what evidence."""

    paged: bool
    depth: int
    standard_k: int
    paged_k: int
    paged_in_window: int
    window: int
    #: Which signal classified the paged hits: extraction metadata, the file suffix fallback,
    #: or none when no hit in the window was paged.
    signal: PagedSignal

    def as_dict(self) -> dict[str, object]:
        return {
            "paged": self.paged,
            "depth": self.depth,
            "standard_k": self.standard_k,
            "paged_k": self.paged_k,
            "paged_in_window": self.paged_in_window,
            "window": self.window,
            "signal": self.signal,
        }


def paged_evidence_mode(value: str | None) -> PagedEvidenceMode:
    """Parse `RECALL_PAGED_EVIDENCE`; unset, empty or blank means off, else it must be on or off.

    The one parser: startup validation (`recall_mcp.settings`) calls it too, so a value the
    server starts with can never be refused per request.
    """
    selected = (value or "").strip().casefold() or "off"
    if selected not in {"off", "on"}:
        raise ValueError("RECALL_PAGED_EVIDENCE must be off or on")
    return selected  # type: ignore[return-value]


def paged_evidence_enabled(env: Mapping[str, str]) -> bool:
    return paged_evidence_mode(env.get("RECALL_PAGED_EVIDENCE")) == "on"


def _hit_signal(hit: ScoredChunk) -> PagedSignal:
    metadata = hit.chunk.metadata or {}
    source_format = metadata.get("source_format")
    if isinstance(source_format, str) and source_format.strip():
        return "metadata" if source_format.strip().casefold() in PAGED_SOURCE_FORMATS else "none"
    # Only when extraction recorded no format: an explicit non-paged format is never overridden
    # by a suffix, so a markdown file named `notes.pdf.md` stays markdown.
    for name in (metadata.get("file"), hit.chunk.source):
        # A URI's query or fragment (`x.pdf?versionId=1`, `x.pdf#page=3`) is not its suffix.
        if isinstance(name, str) and _path_part(name).casefold().endswith(PAGED_SUFFIXES):
            return "suffix"
    return "none"


def _path_part(name: str) -> str:
    return name.split("#", 1)[0].split("?", 1)[0]


def decide_depth(
    hits: Sequence[ScoredChunk], *, standard_k: int, paged_k: int
) -> PagedDepthDecision:
    """The evidence depth for this ranked pool: `paged_k` for a paged pool, else `standard_k`.

    A pool shorter than the window votes on the hits it has against the same absolute
    threshold: three paged hits out of three is paged, two out of two is not.
    """
    if standard_k < 1 or paged_k < standard_k:
        raise ValueError("depths must satisfy 1 <= standard_k <= paged_k")
    window = list(hits[:PAGED_VOTE_WINDOW])
    signals = [_hit_signal(hit) for hit in window]
    paged_signals = [s for s in signals if s != "none"]
    paged = len(paged_signals) >= PAGED_VOTE_THRESHOLD
    signal: PagedSignal
    if not paged_signals:
        signal = "none"
    elif all(s == "metadata" for s in paged_signals):
        signal = "metadata"
    else:
        signal = "suffix"
    return PagedDepthDecision(
        paged=paged,
        depth=paged_k if paged else standard_k,
        standard_k=standard_k,
        paged_k=paged_k,
        paged_in_window=len(paged_signals),
        window=len(window),
        signal=signal,
    )


__all__ = [
    "PAGED_SOURCE_FORMATS",
    "PAGED_SUFFIXES",
    "PAGED_VOTE_THRESHOLD",
    "PAGED_VOTE_WINDOW",
    "PagedDepthDecision",
    "PagedEvidenceMode",
    "decide_depth",
    "paged_evidence_enabled",
    "paged_evidence_mode",
]
