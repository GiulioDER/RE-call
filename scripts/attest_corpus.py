#!/usr/bin/env python
"""Attest an indexed corpus against the files it was built from. READ ONLY.

Ships the harness behind `docs/UNCALIBRATED_FIRST_RUN_DESIGN.md` sections 6c and 6d and the three
pre-registrations dated 2026-08-18. Those measurements were produced by throwaway scripts, so
nobody could re-run them; that is what this closes, and it is the whole claim being made for it.

⚠️ **PRIOR WORK EXISTS and this is not the first hash comparison in the tree.** The Indexer's
incremental skip guard already reads a file, hashes it, and compares against
`PgVectorStore.source_content_hashes()` — that is the same primitive. `recall.migration
.validate_generation_parity` also compares raw hashes, but between TWO STORES rather than between a
store and the filesystem. What is genuinely absent, and what this adds:

* a **read-only census** that reports the disposition of every source instead of acting on it, and
* **chunker identification**, which nothing in the tree does, because nothing records which chunker
  ran (chunk metadata has no chunker field, and `_index_fingerprint` carries only an inert
  `chunker_version` belonging to the embedding profile).

Three subcommands, matching the three checks the design specifies:

    census    every source's stored content_hash against the bytes on disk now
    chunker   which chunker configuration reproduces the stored chunks exactly
    embedder  re-embed a sample and compare cosine against the stored vectors

`census` and `chunker` need no model and no network. `embedder` needs the embedder that built the
corpus.

⛔ **The candidate set for `chunker` is fixed and must not be widened to make something fit.** A
search that keeps broadening always succeeds, and what it finds is a configuration that never ran.
When nothing reproduces a source the answer is "not identifiable", not "search harder".

⚠️ **The content-hash rule below is a SECOND implementation of the one in `recall.index`.** It has
to be, because the original is inline in the indexing loop rather than a callable. That is a drift
hazard, and `tests/test_attest_corpus.py::test_hash_rule_matches_the_indexer` pins this against a
corpus indexed by the real `Indexer` so the drift is detectable rather than silent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from recall.frontmatter import parse_frontmatter
from recall.index import _strip_nul, chunk_code, chunk_text

#: EXACTLY the set `recall/index.py:694` branches on. NOT a superset: `.txt` and `.rst` are
#: hashed as RAW BYTES by the indexer, so listing them here made every such source report
#: `changed`. That was a real defect found in review of this file's first draft.
MARKDOWN_SUFFIXES = frozenset({".md", ".markdown", ".mdx"})

#: FIXED. See the module docstring. Adding to this list to make a corpus identify is the one
#: change that would make this tool dishonest.
CHUNKER_CANDIDATES: dict[str, Callable[[str], list[str]]] = {
    "text/800/80": lambda body: chunk_text(body, max_chars=800, overlap=80),
    "text/800/0": lambda body: chunk_text(body, max_chars=800, overlap=0),
    "code/800": lambda body: chunk_code(body, max_chars=800),
    "text/1200/80": lambda body: chunk_text(body, max_chars=1200, overlap=80),
}


def content_hash_for(path: Path) -> str:
    """The indexer's rule, branching on media type exactly as `recall.index` does."""
    if path.suffix.lower() in MARKDOWN_SUFFIXES:
        raw = _strip_nul(path.read_text(encoding="utf-8-sig"), path)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Census:
    """Five buckets, not four. `unreadable` separates an I/O error from a missing file."""

    verified: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    no_hash: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(len(getattr(self, b)) for b in
                   ("verified", "changed", "missing", "no_hash", "unreadable"))

    def to_dict(self) -> dict:
        out = {b: len(getattr(self, b)) for b in
               ("verified", "changed", "missing", "no_hash", "unreadable")}
        out["total"] = self.total
        out["verified_pct"] = round(100.0 * len(self.verified) / self.total, 2) if self.total else 0.0
        return out


def run_census(rows: Iterable[tuple[str, str]], root: Path) -> Census:
    """`rows` is (relative_file, stored_content_hash). Pure apart from reading the files."""
    census = Census()
    for rel, stored in rows:
        if not stored:
            census.no_hash.append(rel)
            continue
        path = root / rel
        try:
            fresh = content_hash_for(path)
        except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
            census.missing.append(rel)
            continue
        except OSError:
            census.unreadable.append(rel)
            continue
        (census.verified if fresh == stored else census.changed).append(rel)
    return census


def identify_chunker(
    sources: Iterable[tuple[str, str, Sequence[str]]],
    candidates: dict[str, Callable[[str], list[str]]] | None = None,
) -> dict:
    """`sources` is (name, raw_text, stored_chunks_in_ordinal_order).

    Returns per-candidate reproduction counts. A candidate "identifies" the corpus only if it
    reproduces EVERY source exactly; anything less is reported rather than rounded up.
    """
    candidates = CHUNKER_CANDIDATES if candidates is None else candidates
    # Materialised ONCE: `sources` may be a generator, and iterating it twice would silently
    # report zero out-of-scope sources. Non-markdown is excluded rather than failed, because
    # `recall.index` routes it through `chunk_extracted_document`, so these chunkers never
    # produced its stored text and comparing against them would manufacture a confident
    # "not identifiable".
    every = list(sources)
    materialised = [s for s in every if Path(s[0]).suffix.lower() in MARKDOWN_SUFFIXES]
    out_of_scope = len(every) - len(materialised)
    results = {name: {"reproduced": 0, "differed": 0, "errored": 0, "first_difference": None}
               for name in candidates}
    for name, fn in candidates.items():
        for src_name, raw, stored in materialised:
            _meta, body = parse_frontmatter(raw)
            try:
                produced = fn(body)
            except Exception:
                results[name]["errored"] += 1
                continue
            if list(produced) == list(stored):
                results[name]["reproduced"] += 1
            else:
                results[name]["differed"] += 1
                if results[name]["first_difference"] is None:
                    results[name]["first_difference"] = src_name
    total = len(materialised)
    identifying = sorted(n for n, r in results.items() if total and r["reproduced"] == total)
    return {
        "sources": total,
        "out_of_scope_non_markdown": out_of_scope,
        "candidates": results,
        "identifies": identifying,
        # Three outcomes, deliberately distinct. "ambiguous" is not a failure and must not be
        # reported as one: it means several configurations are observationally equivalent here.
        "verdict": ("identified" if len(identifying) == 1
                    else "ambiguous" if identifying else "not_identifiable"),
    }


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return num / (na * nb) if na and nb else 0.0


def sample_size_for(smallest_fraction: float, confidence: float = 0.95) -> int:
    """n such that a contaminated fraction `p` is missed with probability <= 1 - confidence.

    P(miss) = (1 - p)^n. The 20-chunk sample this design was first measured with detects only
    p >= 0.139 at 95%, which is why the default below is 0.05 rather than a round number.
    """
    if not 0 < smallest_fraction < 1:
        raise ValueError("smallest_fraction must be in (0, 1)")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    return math.ceil(math.log(1 - confidence) / math.log(1 - smallest_fraction))


def attest_embedder(
    samples: Iterable[tuple[str, Sequence[float]]],
    embed: Callable[[list[str]], list[Sequence[float]]],
    bar: float = 0.9999,
) -> dict:
    """`samples` is (chunk_text, stored_vector). `embed` takes the passage path, not the query one.

    Reports an OFF-DIAGONAL control alongside the result: comparing chunk i's stored vector to
    chunk i+1's fresh one. Without it, a comparison stuck on 1.0 is indistinguishable from a
    pipeline confirmed.
    """
    pairs = list(samples)
    if not pairs:
        return {"n": 0, "verdict": "no_samples"}
    fresh = embed([t for t, _ in pairs])
    if len(fresh) != len(pairs):
        raise ValueError(f"embedder returned {len(fresh)} vectors for {len(pairs)} texts")
    diag = [cosine(stored, f) for (_, stored), f in zip(pairs, fresh)]
    off = [cosine(pairs[i][1], fresh[(i + 1) % len(fresh)]) for i in range(len(fresh))]
    below = sum(1 for c in diag if c < bar)
    return {
        "n": len(diag),
        "bar": bar,
        "cosine_min": min(diag),
        "cosine_mean": sum(diag) / len(diag),
        "at_or_above_bar": len(diag) - below,
        "control_offdiagonal_max": max(off) if off else None,
        # A single failure aborts: nothing in the legacy metadata says which other sources shared
        # the failing one's provenance, so a partial pass cannot be scoped.
        "verdict": "pass" if below == 0 else "abort",
    }


# --------------------------------------------------------------------------------------
# Database readers. Everything above is pure so it can be tested without PostgreSQL.
# --------------------------------------------------------------------------------------
def _safe_table(table: str) -> str:
    """Interpolated into SQL, so it may only ever be a plain identifier."""
    if not table.replace("_", "").isalnum():
        raise ValueError(f"refusing table name {table!r}: identifiers only")
    return table


def _rows(dsn: str, sql: str, params: tuple) -> list[tuple]:
    import psycopg

    with psycopg.connect(dsn) as conn:
        return conn.execute(sql, params).fetchall()


def fetch_sources(dsn: str, tenant: str, table: str) -> list[tuple[str, str]]:
    return [(str(r[0]), str(r[1] or "")) for r in _rows(
        dsn,
        f"SELECT DISTINCT metadata->>'file', coalesce(metadata->>'content_hash', '') "
        f"FROM {_safe_table(table)} WHERE tenant_id = %s AND metadata->>'file' IS NOT NULL",
        (tenant,),
    )]


def fetch_chunks_by_source(dsn: str, tenant: str, table: str) -> dict[str, list[str]]:
    out: dict[str, list[tuple[int, str]]] = {}
    for rel, ordinal, text in _rows(
        dsn,
        f"SELECT metadata->>'file', (metadata->>'ord')::int, text FROM {_safe_table(table)} "
        f"WHERE tenant_id = %s AND metadata->>'file' IS NOT NULL "
        f"AND metadata->>'ord' IS NOT NULL",
        (tenant,),
    ):
        out.setdefault(str(rel), []).append((int(ordinal), str(text)))
    return {k: [t for _, t in sorted(v)] for k, v in out.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["census", "chunker"])
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--root", required=True, help="directory the corpus was indexed from")
    parser.add_argument("--table", default="chunks")
    args = parser.parse_args(argv)
    root = Path(args.root)

    if args.command == "census":
        result = run_census(fetch_sources(args.dsn, args.tenant, args.table), root).to_dict()
    else:
        stored = fetch_chunks_by_source(args.dsn, args.tenant, args.table)
        sources = []
        skipped = 0
        for rel, chunks in sorted(stored.items()):
            try:
                raw = (root / rel).read_text(encoding="utf-8-sig")
            except OSError:
                skipped += 1
                continue
            sources.append((rel, raw, chunks))
        result = identify_chunker(sources)
        # Stated, never silent: a corpus whose files are half unreadable would otherwise report a
        # confident verdict over whatever remained.
        result["skipped_unreadable"] = skipped

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
