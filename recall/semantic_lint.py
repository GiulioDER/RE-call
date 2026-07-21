"""Write-time SEMANTIC lint: catch the supersession edge you *forgot* to write.

The static `recall lint` verifies edges that exist (dangling / cycle / version-sibling). It is
structurally blind to the failure that actually hurts — the MISSING edge: a new memo that is
really about a prior settled decision but never declares `supersedes:`. That relation is not in
the frontmatter, it is in the meaning, so no syntactic check can see it. Retrieval can.

This is the write-time mirror of the anti-re-litigation guard: anti-re-litigation queries the
index before an agent *proposes* an idea; this queries the index before a memo is *committed*.
When a memo lands, search with its text; any high-similarity CLOSED decision it does not
reference is a candidate unlinked chain — surfaced for a one-keystroke confirm. Same embedder,
same index, same trust layer, pointed at the write path instead of the read path.

Honest limits (measured, not hidden):
- The similarity threshold inherits FINDINGS §2 — it does NOT transfer across embedders, so it
  must be calibrated per model. The cost asymmetry is friendly though: a false positive is a
  keystroke to dismiss, a false negative is a silent orphan, so tune it LOOSE and over-surface.
- "closed decision" is detected from the memo's own prose (a settled-status marker or closure
  wording). A fresh memo you are committing is typically unmarked, which conveniently orients
  the check: it flags the settled PRIOR, not the new memo. Requires the DB (opt-in), unlike the
  pure-filesystem `recall lint`.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from recall.embeddings import Embedder
from recall.frontmatter import parse_frontmatter
from recall.index import Indexer, chunk_text
from recall.lint import CLOSURE_MARKERS, DEFAULT_GLOB
from recall.retriever import HybridRetriever
from recall.store import PgVectorStore, _basename

#: A memo reads as a settled decision (something a new related memo should reference) when its
#: body carries a decision-status marker. An OPEN investigation is deliberately excluded — a
#: live hypothesis legitimately coexists with a fresh related memo and is not a missing edge.
_DECISION_STATUS = re.compile(
    r"\bstatus:\s*(adopted|closed|superseded|replaced|deprecated|obsolete|falsified|"
    r"rejected|decided|abandoned)\b",
    re.IGNORECASE,
)


#: Below any real cosine (which lies in [-1, 1]); the "no score yet, anything wins" floor.
_NO_SCORE = float("-inf")


def is_closed_decision(body: str) -> bool:
    """True if the memo body reads as a settled decision (status marker or closure prose)."""
    return bool(_DECISION_STATUS.search(body) or CLOSURE_MARKERS.search(body))


@dataclass(frozen=True)
class ChainCandidate:
    name: str            # basename of a retrieved prior memo
    cosine: float        # dense cosine of the prior memo to the new memo
    is_closed_decision: bool


@dataclass(frozen=True)
class UnlinkedChain:
    new_memo: str        # the memo being committed
    prior: str           # the unreferenced closed decision it is highly similar to
    cosine: float


def find_unlinked_chains(
    new_memo: str, supersedes: set[str], candidates: list[ChainCandidate], threshold: float
) -> list[UnlinkedChain]:
    """Pure core: which retrieved candidates are unlinked chains for `new_memo`.

    A candidate is flagged when it is a closed decision, scores at/above the threshold, is not
    the memo itself, and is not already referenced by the memo. Highest similarity first.
    """
    out = [
        UnlinkedChain(new_memo, c.name, c.cosine)
        for c in candidates
        if c.name != new_memo
        and c.name not in supersedes
        and c.is_closed_decision
        and c.cosine >= threshold
    ]
    out.sort(key=lambda u: u.cosine, reverse=True)
    return out


def semantic_lint(
    dsn: str, embedder: Embedder, corpus_dir: str | Path, threshold: float,
    glob: str = DEFAULT_GLOB, k: int = 10,
) -> list[UnlinkedChain]:
    """Sweep a corpus: for each memo, retrieve and report unreferenced high-similarity closed
    decisions (candidate missing `supersedes:` edges). Indexes into a throwaway table.

    A pair already linked in EITHER direction (M references C, or C references M) is skipped —
    the chain is complete regardless of which way the edge points.
    """
    root = Path(corpus_dir)
    files = sorted(root.glob(glob)) if root.is_dir() else [root]

    # This check works in BASENAME space (the `supersedes:` authoring convention), so a corpus
    # with the SAME basename in two directories cannot be disambiguated — the shadowed file
    # would be queried with the wrong body. Skip ambiguous basenames rather than mis-key them
    # (the static lint flags them as `ambiguous-supersedes-target`; here they are simply out of
    # scope). This mirrors lint.py's refusal to let same-named files shadow each other. Retrieval
    # now stamps metadata['file'] with the ROOT-RELATIVE path, so hits are reduced back to their
    # basename below; within the non-ambiguous set that basename is a unique key.
    name_count: dict[str, int] = {}
    for f in files:
        name_count[f.name] = name_count.get(f.name, 0) + 1
    ambiguous = {n for n, c in name_count.items() if c > 1}

    supersedes: dict[str, set[str]] = {}
    closed: dict[str, bool] = {}
    body_text: dict[str, str] = {}
    self_chunks: dict[str, int] = {}
    for f in files:
        if f.name in ambiguous:
            continue
        meta, body = parse_frontmatter(f.read_text(encoding="utf-8-sig"))
        target = meta.get("supersedes")
        supersedes[f.name] = {target} if target else set()
        closed[f.name] = is_closed_decision(body)
        body_text[f.name] = body
        self_chunks[f.name] = max(1, len(chunk_text(body)))

    table = "semlint_" + uuid.uuid4().hex[:8]
    store = PgVectorStore(dsn, dim=embedder.dim, table=table)
    findings: list[UnlinkedChain] = []
    try:
        store.ensure_schema()
        Indexer(store, embedder).index_path(corpus_dir, glob=glob)
        retr = HybridRetriever(store, embedder)
        for f in files:
            name = f.name
            if name in ambiguous:
                continue
            # The query is the memo's own body, so its own chunks are the top hits and are
            # dropped by the self-filter BELOW — after the retriever truncates to top-k. Ask for
            # k PLUS this memo's chunk count so a long multi-chunk memo can't crowd every real
            # prior out of the window before the self-filter runs.
            res = retr.search(body_text[name], k=k + self_chunks[name])
            best: dict[str, float] = {}  # other-file -> max dense cosine
            for h in res.hits:
                relfile = h.chunk.metadata.get("file")
                # reduce the root-relative id back to the basename this check keys on
                cname = _basename(relfile) if relfile else None
                if not cname or cname == name or cname in ambiguous:
                    continue
                # skip pairs already linked either way — the chain is complete
                if name in supersedes.get(cname, set()):
                    continue
                if h.score > best.get(cname, _NO_SCORE):
                    best[cname] = h.score
            candidates = [
                ChainCandidate(cn, sc, closed.get(cn, False)) for cn, sc in best.items()
            ]
            findings.extend(
                find_unlinked_chains(name, supersedes.get(name, set()), candidates, threshold)
            )
    finally:
        try:
            store.drop_table()
        except Exception:
            pass  # best-effort drop of the throwaway uuid table
        finally:
            store.close()

    # Two mutually-similar unlinked closed decisions each surface the other (M->C and C->M).
    # Report each missing link once, keeping the higher-cosine direction.
    best_by_pair: dict[frozenset[str], UnlinkedChain] = {}
    for u in findings:
        key = frozenset({u.new_memo, u.prior})
        if key not in best_by_pair or u.cosine > best_by_pair[key].cosine:
            best_by_pair[key] = u
    deduped = list(best_by_pair.values())
    deduped.sort(key=lambda u: u.cosine, reverse=True)
    return deduped
