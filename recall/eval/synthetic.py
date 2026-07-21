"""Generate a synthetic memory corpus + labelled query set at arbitrary scale.

The shipped eval corpus is 14 documents and 25 queries. That is enough to demonstrate a
mechanism and far too little to measure one: the headline superseded-trust rate rests on 6
queries, whose 95% Wilson interval is [0.00, 0.39]. An interval that wide cannot distinguish a
working trust layer from a mediocre one, and retrieval failure modes that only appear under
index pressure — HNSW recall under a selective filter, near-duplicate crowding, latency at p99 —
are invisible at 14 documents.

This module builds the same *shape* of corpus programmatically, so both axes scale: more
documents for index pressure, and more labelled queries for tighter intervals. Every document
is generated around a unique subject token, which is what makes the ground truth unambiguous —
a query about `quartz-ledger-0042` has exactly one right answer by construction.

Four query classes, mirroring `queries.json`:

- **answerable** — one document holds the fact; used for ranking metrics and calibration.
- **unanswerable** — the subject appears nowhere in the corpus; the gap guard should fire.
- **successor** (trust) — a v1/v2 supersession pair. The query is worded with v1's phrasing ON
  PURPOSE: the whole failure mode is a stale memory outranking its successor on similarity, so a
  query worded closer to the successor would measure nothing.
- **abstain** (trust) — a document whose `valid_until` has passed; nothing current answers it.

Filler documents create index pressure without polluting ground truth: they are drawn from a
disjoint vocabulary and packed many-chunks-per-file, because one file per chunk makes generating
a six-figure corpus a filesystem benchmark rather than a retrieval one.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from recall.index import DEFAULT_MAX_CHARS

#: Subject tokens are built from these so every document has a distinctive, unambiguous anchor.
#: Disjoint from _FILLER_NOUNS: filler must never look like an answer to a generated query.
_ADJECTIVES = [
    "quartz", "amber", "cobalt", "verdant", "onyx", "saffron", "indigo", "crimson",
    "granite", "azure", "cedar", "opal", "topaz", "slate", "ivory", "umber",
]
_NOUNS = [
    "ledger", "router", "beacon", "harbor", "spindle", "lantern", "conduit", "anvil",
    "trellis", "cistern", "pylon", "foundry", "quarry", "vault", "aqueduct", "kiln",
]
_FILLER_NOUNS = [
    "meadow", "thicket", "cove", "dune", "fjord", "glade", "isthmus", "marsh",
]

#: (aspect phrasing used by v1 and by the query, unit) — the successor deliberately avoids this
#: exact phrasing so the stale document stays the closer lexical match.
_ASPECTS = [
    ("retry budget", "attempts"),
    ("cache TTL", "seconds"),
    ("rate limit", "requests per second"),
    ("batch window", "milliseconds"),
    ("connection ceiling", "connections"),
    ("shard fanout", "shards"),
    ("replay horizon", "hours"),
    ("compaction interval", "minutes"),
]

_INCIDENTS = [
    "the checkout timeout incident", "the stale-quote incident", "the failover drill",
    "the capacity review", "the load-shedding postmortem",
]


@dataclass(frozen=True)
class SyntheticCorpus:
    """A generated corpus plus its labelled queries.

    `queries` follows the `queries.json` schema, with one extra key: `subject`, the unique token
    the query is about. It is not read by the harness — it exists so the corpus's own tests can
    assert that an unanswerable query's subject is genuinely absent.
    """

    root: Path
    queries: list[dict]
    queries_path: Path
    n_files: int
    n_chunks: int


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _subjects(rng: random.Random, n: int, nouns: list[str]) -> list[str]:
    """`n` distinct subject tokens. Uniqueness is what makes the ground truth unambiguous, so it
    is enforced by construction (an index suffix) rather than hoped for from sampling."""
    return [f"{rng.choice(_ADJECTIVES)}-{rng.choice(nouns)}-{i:04d}" for i in range(n)]


def generate(
    out_dir: str | Path,
    *,
    n_answerable: int = 200,
    n_unanswerable: int = 100,
    n_successor: int = 150,
    n_abstain: int = 100,
    n_filler_chunks: int = 0,
    filler_chunks_per_file: int = 200,
    seed: int = 1234,
) -> SyntheticCorpus:
    """Write a corpus under `out_dir` and return it with its labelled queries.

    Counts are exact, not approximate: an eval that silently generated fewer queries than asked
    would report a tighter interval than its sample supports.
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    queries: list[dict] = []
    n_files = 0

    # --- answerable: one document, one fact, one right answer ---------------------------------
    for i, subj in enumerate(_subjects(rng, n_answerable, _NOUNS)):
        aspect, unit = _ASPECTS[i % len(_ASPECTS)]
        value = rng.randint(2, 900)
        rel = f"facts/{subj}.md"
        _write(root, rel, (
            f"# {subj} {aspect}\n\n"
            f"The {aspect} for {subj} is {value} {unit}. This applies to every {subj} "
            f"deployment in the fleet. Status: adopted.\n"
        ))
        n_files += 1
        queries.append({
            "id": f"a{i:04d}",
            "query": f"what is the {aspect} for {subj}",
            "relevant_ids": [f"{rel}:0"],
            "answerable": True,
            "subject": subj,
        })

    # --- unanswerable: the subject is never written ------------------------------------------
    for i, subj in enumerate(_subjects(rng, n_unanswerable, _NOUNS)):
        aspect, _ = _ASPECTS[i % len(_ASPECTS)]
        # suffix keeps these tokens out of the answerable namespace even on a collision
        subj = f"{subj}-absent"
        queries.append({
            "id": f"u{i:04d}",
            "query": f"what is the {aspect} for {subj}",
            "relevant_ids": [],
            "answerable": False,
            "subject": subj,
        })

    # --- successor: adversarial v1/v2 supersession pairs ---------------------------------------
    for i, subj in enumerate(_subjects(rng, n_successor, _NOUNS)):
        aspect, unit = _ASPECTS[i % len(_ASPECTS)]
        old, new = rng.randint(2, 400), rng.randint(401, 900)
        incident = rng.choice(_INCIDENTS)
        v1_rel, v2_rel = f"pairs/{subj}_v1.md", f"pairs/{subj}_v2.md"
        # v1 carries the query's exact phrasing; v2 states the same fact in different words, so
        # similarity alone prefers the STALE document — the case the trust layer must win.
        _write(root, v1_rel, (
            f"# {subj} {aspect}\n\n"
            f"The {aspect} for {subj} is {old} {unit}. Applied fleet-wide to {subj}. "
            f"Status: adopted.\n"
        ))
        _write(root, v2_rel, (
            f"---\nsupersedes: {subj}_v1.md\n---\n"
            f"# {subj} revision\n\n"
            f"Following {incident}, {subj} now operates at {new} {unit} instead of the earlier "
            f"figure. This replaces the previous decision.\n"
        ))
        n_files += 2
        queries.append({
            "id": f"s{i:04d}",
            "query": f"what is the {aspect} for {subj}",
            "trust": True,
            "expect": "successor",
            "stale_ids": [f"{v1_rel}:0"],
            "successor_ids": [f"{v2_rel}:0"],
            "subject": subj,
        })

    # --- abstain: the only document about the subject has expired ------------------------------
    expired_on = (date.today() - timedelta(days=30)).isoformat()
    for i, subj in enumerate(_subjects(rng, n_abstain, _NOUNS)):
        aspect, unit = _ASPECTS[i % len(_ASPECTS)]
        value = rng.randint(2, 900)
        rel = f"expired/{subj}.md"
        _write(root, rel, (
            f"---\nvalid_until: {expired_on}\n---\n"
            f"# {subj} {aspect}\n\n"
            f"During the migration window the {aspect} for {subj} is {value} {unit}. "
            f"This is a temporary measure for {subj}.\n"
        ))
        n_files += 1
        queries.append({
            "id": f"x{i:04d}",
            "query": f"what is the {aspect} for {subj}",
            "trust": True,
            "expect": "abstain",
            "stale_ids": [f"{rel}:0"],
            "successor_ids": [],
            "subject": subj,
        })

    # --- filler: index pressure, disjoint vocabulary, packed many chunks per file ---------------
    written = 0
    while written < n_filler_chunks:
        batch = min(filler_chunks_per_file, n_filler_chunks - written)
        paras = []
        for _ in range(batch):
            noun = rng.choice(_FILLER_NOUNS)
            # Each paragraph must become EXACTLY one chunk. `chunk_text` packs paragraphs up to
            # max_chars (800), so a short paragraph would be merged with its neighbours and the
            # corpus would be a fraction of the requested size while still reporting the
            # requested number — scale silently overstated in every result computed from it.
            # Sized into (max_chars/2, max_chars) so two cannot pack and one cannot split.
            body = " ".join(
                f"Transect {rng.randint(1000, 9999)} of the {noun} survey recorded "
                f"{rng.randint(10, 99)} observations with no anomalies noted."
                for _ in range(5)
            )
            para = f"The {noun} survey archive. {body} The {noun} record was archived."
            # Enforced, not assumed. The safe band is (max_chars/2, max_chars): at or below the
            # half-way point two paragraphs pack into one chunk, and above max_chars one splits
            # into several. Either way the corpus size stops matching the requested one, which
            # is a silent misattribution of scale rather than a visible failure.
            if not (DEFAULT_MAX_CHARS / 2 < len(para) < DEFAULT_MAX_CHARS):
                raise AssertionError(
                    f"filler paragraph is {len(para)} chars, outside the one-chunk band "
                    f"({DEFAULT_MAX_CHARS / 2}, {DEFAULT_MAX_CHARS}) — it would not map 1:1 to "
                    f"chunks and the reported corpus size would be wrong"
                )
            paras.append(para)
        _write(root, f"filler/notes_{written:07d}.md", "\n\n".join(paras) + "\n")
        n_files += 1
        written += batch

    queries_path = root / "queries.json"
    queries_path.write_text(json.dumps(queries, indent=1), encoding="utf-8")
    # ground-truth docs are one chunk each by construction (see the corpus tests)
    n_chunks = n_answerable + 2 * n_successor + n_abstain + written
    return SyntheticCorpus(root, queries, queries_path, n_files, n_chunks)
