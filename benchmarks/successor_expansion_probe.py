"""Pre-registered probe: does fetching an absent successor recover the answer?

Record: `docs/preregistrations/2026-08-19-successor-directed-expansion.md`, committed at `af8c2abc`
before the implementation existed. Read it before reading any number this prints.

The design decision that makes the result readable is that the fixture is NOT tuned to make the
successor fall outside the pool. The baseline runs first, the supersession queries are partitioned
by whether the successor reached the pool at all, and the headline rate is reported over the absent
stratum only. Fixture difficulty therefore sets that stratum's SIZE, which is printed, instead of
silently setting the rate.

Two stratifications are printed, on purpose. The record says "absent from the baseline fused pool";
the implementation acts on `result.hits`, the top-k the caller actually receives, which is a subset
of that pool. Reporting both removes the judgement call about which one the record meant, and the
`hits` one is the operative definition because it is what the code can see.

Apparatus checks run BEFORE any quality number is read, and a failure of any of them means the
quality result must not be interpreted. See the four in the record.

    eval "$(scripts/session-db.sh up)"
    python -m benchmarks.successor_expansion_probe

⚠️ `-m`, from the worktree root, NOT `python benchmarks/successor_expansion_probe.py`. Run as a
script, Python puts the SCRIPT's directory on `sys.path[0]`, so `benchmarks/` goes on the path and
the worktree root does not. `import recall` then falls through to whatever is installed, which on
this machine is the MAIN CHECKOUT: the first run of this probe imported
`C:/Users/gde00/Documents/recall/recall/retriever.py` and died on a symbol that exists only here.
That failure was loud. The dangerous version is silent, a benchmark that runs happily and scores
the main checkout while reporting a number against your branch. The guard below the imports
turns the silent case back into a loud one.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import statistics
import sys
import tempfile
import time
import uuid
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.find_spec("recall")
_ORIGIN = Path(_SPEC.origin).resolve() if _SPEC is not None and _SPEC.origin else None
if _ORIGIN is None or _HERE not in _ORIGIN.parents:
    # Refuse to measure a `recall` that is not the one in this tree. `find_spec` rather than
    # `import recall`, so this can run BEFORE the imports it protects without executing the wrong
    # package first.
    raise SystemExit(
        f"refusing to run: `recall` resolves to {_ORIGIN}\n"
        f"                 but this probe lives under {_HERE}\n"
        "run it as `python -m benchmarks.successor_expansion_probe` from the worktree root"
    )

# E402 below is deliberate and is the entire point of the block above. The case this guard exists
# for is NOT the loud one that produced it, where a symbol was missing and the import died. It is
# the silent one, where every symbol resolves in both trees, the probe runs happily, and the number
# describes another checkout. That can only be caught before the first `recall` import.
from recall.calibration import from_samples  # noqa: E402
from recall.embeddings import FastEmbedEmbedder, embedding_profile_id  # noqa: E402
from recall.eval._research_trust import research_search  # noqa: E402
from recall.eval.calibrate import measure_top_cosines  # noqa: E402
from recall.eval.metrics import wilson_ci  # noqa: E402
from recall.index import Indexer  # noqa: E402
from recall.retriever import HybridRetriever, SuccessorExpansionPolicy  # noqa: E402
from recall.store import PgVectorStore  # noqa: E402
from recall.types import TrustedResult  # noqa: E402

from benchmarks.successor_fixture import (  # noqa: E402
    CALIBRATION_ANSWERABLE,
    CALIBRATION_UNANSWERABLE,
    PAIRS,
    UNANSWERABLE,
    documents,
)

#: The caller-facing depth. The library default, so the measurement describes the shipped shape.
K = 5
#: Enabled arm. `max_sources=2` is the policy default; a query rarely surfaces more than one
#: superseded document, so this is a ceiling rather than a working value.
TREATMENT = SuccessorExpansionPolicy(enabled=True, max_sources=2, chunks_per_source=3)
#: Repository prose indexed alongside the authored pairs, so the fused pool cannot hold the corpus.
#: Real text of the same register, which is what apparatus check 4 requires and what authoring
#: hundreds of distractors would have failed.
DISTRACTOR_DIR = Path(__file__).resolve().parent.parent / "docs"


def _ok_files(result: TrustedResult) -> list[str]:
    return [h.provenance.file or "" for h in result.hits if h.verdict == "ok"]


def _all_files(result: TrustedResult) -> set[str]:
    return {h.provenance.file or "" for h in result.hits}


# --- rank instrumentation, registered in 2026-08-20-successor-rank-hypothesis.md ----------------
#
# Recovery is top-1, so it collapses two different failures into one number: the successor was not
# promoted, and the successor was promoted but something else came first. Two records guessed
# between them in prose. These read it off instead.


def _verdict_of(result: TrustedResult, file: str) -> str:
    for hit in result.hits:
        if hit.provenance.file == file:
            return hit.verdict
    return "absent"


def _rank_among_ok(result: TrustedResult, file: str) -> int | None:
    """1-based position among verdict-`ok` hits, or None when the file is not `ok`."""
    ok = _ok_files(result)
    return ok.index(file) + 1 if file in ok else None


def _top_by_score(result: TrustedResult) -> str:
    """Which file would answer if `ok` hits were ordered by cosine instead of pool position.

    The record predicts this is NOT an improvement. Promotion exists for a successor whose own
    wording scores low, so ordering by cosine pushes exactly those back down.
    """
    ok = [h for h in result.hits if h.verdict == "ok"]
    if not ok:
        return ""
    return max(ok, key=lambda h: h.cosine).provenance.file or ""


def _top_promoted_first(result: TrustedResult) -> str:
    """Which file would answer if promoted successors were placed ahead of other `ok` hits.

    A promoted successor is identified as an `ok` hit whose file some OTHER hit names as its
    `superseded_by`, which is exactly the condition `evaluate` uses to promote it. `sorted` is
    stable, so pool order survives inside each group.
    """
    ok = [h for h in result.hits if h.verdict == "ok"]
    if not ok:
        return ""
    named = {h.validity.superseded_by for h in result.hits if h.validity.superseded_by}
    ranked = sorted(ok, key=lambda h: 0 if h.provenance.file in named else 1)
    return ranked[0].provenance.file or ""


def _rate(flags: list[bool]) -> str:
    if not flags:
        return "n/a (n=0)"
    lo, hi = wilson_ci(flags)
    return f"{sum(flags) / len(flags):.2f} [{lo:.2f}, {hi:.2f}] n={len(flags)}"


def _build_corpus(root: Path) -> int:
    for name, text in documents().items():
        (root / name).write_text(text, encoding="utf-8")
    copied = 0
    for doc in sorted(DISTRACTOR_DIR.glob("*.md")):
        shutil.copyfile(doc, root / f"docs__{doc.name}")
        copied += 1
    return copied


def main() -> int:
    dsn = os.environ.get("RECALL_TEST_DSN")
    if not dsn:
        # Same rule as `tests/conftest.py`: no default DSN, ever. A fallback to 5432 is what made
        # two checkouts drop each other's tables mid-run.
        print("RECALL_TEST_DSN is unset. Run: eval \"$(scripts/session-db.sh up)\"", file=sys.stderr)
        return 2

    embedder = FastEmbedEmbedder()
    # Disjoint from everything scored below. The first run fitted this from the ten
    # `Pair.query` strings and the six controls, which was both too small and a leak: it
    # calibrated on the evaluation set. See the 2026-08-20 record.
    calib_queries = [{"query": q, "answerable": True} for q in CALIBRATION_ANSWERABLE]
    calib_queries += [{"query": q, "answerable": False} for q in CALIBRATION_UNANSWERABLE]
    leak = ({p.query for p in PAIRS} | set(UNANSWERABLE)) & (
        set(CALIBRATION_ANSWERABLE) | set(CALIBRATION_UNANSWERABLE)
    )
    if leak:
        # Asserted rather than trusted. Disjointness is the whole point of this rerun, and it
        # is one careless paste away from silently not holding.
        raise SystemExit(f"calibration set overlaps the measured queries: {sorted(leak)}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        distractors = _build_corpus(root)
        store = PgVectorStore(dsn, dim=embedder.dim, table="successor_probe_" + uuid.uuid4().hex[:8])
        try:
            store.ensure_schema()
            stats = Indexer(store, embedder).index_path(root)
            cal = from_samples(
                embedding_profile_id(embedder), *measure_top_cosines(store, embedder, calib_queries)
            )
            retriever = HybridRetriever(store, embedder, gap_threshold=cal.threshold)

            rows = []
            for pair in PAIRS:
                successor = f"{pair.slug}_v2.md"
                stale = f"{pair.slug}_v1.md"

                started = time.perf_counter()
                base = research_search(store, embedder, pair.query, k=K, calibration=cal)
                base_ms = (time.perf_counter() - started) * 1000.0

                started = time.perf_counter()
                treat = research_search(
                    store, embedder, pair.query, k=K, calibration=cal, successor_expansion=TREATMENT
                )
                treat_ms = (time.perf_counter() - started) * 1000.0

                # The pre-truncation pool, for the record's own wording. Asked separately and
                # deliberately wide: `search(k=...)` truncates AFTER fusion, so a large k returns
                # what the pool held rather than what the caller was given.
                pool = retriever.search(pair.query, k=100)
                pool_files = {h.chunk.metadata.get("file") for h in pool.hits}

                base_ok = _ok_files(base)
                treat_ok = _ok_files(treat)
                rows.append(
                    {
                        "slug": pair.slug,
                        "in_hits": successor in _all_files(base),
                        "in_pool": successor in pool_files,
                        "base_recovered": bool(base_ok) and base_ok[0] == successor,
                        "treat_recovered": bool(treat_ok) and treat_ok[0] == successor,
                        "base_str": stale in base_ok,
                        "treat_str": stale in treat_ok,
                        "base_cov": bool(base_ok),
                        "treat_cov": bool(treat_ok),
                        "fetched": treat.diagnostics.stage_ms.get("successor_expansion_sources", 0.0) > 0,
                        "succ_verdict": _verdict_of(treat, successor),
                        "succ_rank": _rank_among_ok(treat, successor),
                        "outranked_by": (treat_ok[0] if treat_ok else ""),
                        "score_recovered": _top_by_score(treat) == successor,
                        "promoted_recovered": _top_promoted_first(treat) == successor,
                        "base_ms": base_ms,
                        "treat_ms": treat_ms,
                    }
                )

            controls = []
            for query in UNANSWERABLE:
                base = research_search(store, embedder, query, k=K, calibration=cal)
                treat = research_search(
                    store, embedder, query, k=K, calibration=cal, successor_expansion=TREATMENT
                )
                controls.append({"base": base.abstained, "treat": treat.abstained})
        finally:
            try:
                store.drop_table()
            finally:
                store.close()

    strat_b = [r for r in rows if not r["in_hits"]]
    strat_a = [r for r in rows if r["in_hits"]]
    pool_b = [r for r in rows if not r["in_pool"]]

    print("=" * 78)
    print("APPARATUS (checked before any quality number is read)")
    print("=" * 78)
    print(f"  corpus indexed          : {stats.chunks} chunks from {stats.files} files")
    print(f"                            ({len(documents())} authored, {distractors} repository docs)")
    print(f"  calibrated threshold    : {cal.threshold:.4f}")
    print(f"  1. stratum B (by hits)  : {len(strat_b)} of {len(rows)} supersession queries")
    print(f"     stratum B (by pool)  : {len(pool_b)} of {len(rows)}   <- the wording in the record")
    print(f"  2. baseline recovery on B: {_rate([r['base_recovered'] for r in strat_b])}")
    print(f"  3. known answer case    : stratum A holds {len(strat_a)}, "
          f"baseline correct on {sum(r['base_recovered'] for r in strat_a)}")
    print("  4. corpus defect        : authored prose, each successor reframed rather than "
          "renumbered")

    failed = []
    if not strat_b:
        failed.append("stratum B is empty: the fixture cannot exhibit the condition")
    if any(r["base_recovered"] for r in strat_b):
        failed.append("baseline recovered on stratum B: the stratification is reading the wrong pool")
    if not strat_a:
        failed.append("stratum A is empty: no known answer case to check the apparatus against")
    if failed:
        print()
        for line in failed:
            print(f"  APPARATUS FAILURE: {line}")
        print("\n  The quality result below is NOT interpretable. Fix the fixture and re-run.")

    print()
    print("=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  successor recovery, stratum B, treatment : {_rate([r['treat_recovered'] for r in strat_b])}")
    print(f"  successor recovery, stratum B, baseline  : {_rate([r['base_recovered'] for r in strat_b])}")
    print(f"  successor recovery, stratum A, treatment : {_rate([r['treat_recovered'] for r in strat_a])}")
    print(f"  successor recovery, stratum A, baseline  : {_rate([r['base_recovered'] for r in strat_a])}")
    print(f"  superseded trust rate, baseline          : {_rate([r['base_str'] for r in rows])}")
    print(f"  superseded trust rate, treatment         : {_rate([r['treat_str'] for r in rows])}")
    print(f"  trust coverage, baseline                 : {_rate([r['base_cov'] for r in rows])}")
    print(f"  trust coverage, treatment                : {_rate([r['treat_cov'] for r in rows])}")
    print(f"  abstention accuracy, baseline            : {_rate([c['base'] for c in controls])}")
    print(f"  abstention accuracy, treatment           : {_rate([c['treat'] for c in controls])}")

    # The confound the record names: a low recovery has two very different causes and they must
    # not be reported as one number.
    fetched_b = [r for r in strat_b if r["fetched"]]
    print()
    # ⚠️ This line said "were then promoted" and printed `treat_recovered`, which is top-1
    # RECOVERY. Promotion was never measured by it. That one wrong word survived two full
    # measurement cycles and produced two confidently wrong causal stories, both retracted in
    # docs/preregistrations/2026-08-20-successor-rank-hypothesis.md. Promotion is now read from
    # the verdict, and recovery is named as recovery.
    print(f"  of {len(strat_b)} stratum B queries: {len(fetched_b)} fetched a successor, "
          f"{sum(r['succ_verdict'] == 'ok' for r in fetched_b)} were promoted to ok, "
          f"{sum(r['treat_recovered'] for r in fetched_b)} then ranked FIRST among ok hits")
    print(f"  never fetched: {len(strat_b) - len(fetched_b)}  "
          "(the fetch did not fire, which is not the same as the fetch not helping)")

    triggering = [r for r in rows if r["fetched"]]
    quiet = [r for r in rows if not r["fetched"]]
    if triggering:
        ratio = statistics.median([r["treat_ms"] for r in triggering]) / max(
            statistics.median([r["base_ms"] for r in triggering]), 1e-9
        )
        print(f"  p50 latency ratio, triggering queries    : {ratio:.2f}x  n={len(triggering)}")
    if quiet:
        ratio = statistics.median([r["treat_ms"] for r in quiet]) / max(
            statistics.median([r["base_ms"] for r in quiet]), 1e-9
        )
        print(f"  p50 latency ratio, non triggering        : {ratio:.2f}x  n={len(quiet)}")

    print()
    print("  rank instrumentation, stratum B, treatment arm")
    print(f"    successor present            : "
          f"{sum(r['succ_verdict'] != 'absent' for r in strat_b)} of {len(strat_b)}")
    print(f"    successor verdict ok         : "
          f"{sum(r['succ_verdict'] == 'ok' for r in strat_b)} of {len(strat_b)}")
    missed = [r for r in strat_b if not r["treat_recovered"]]
    print(f"    of the {len(missed)} that did NOT recover:")
    for r in missed:
        rank = r["succ_rank"]
        print(f"      {r['slug']:<20} verdict={r['succ_verdict']:<16} "
              f"rank_among_ok={rank if rank is not None else '-'}  "
              f"outranked_by={r['outranked_by'] or '(nothing ok)'}")
    print()
    print("  counterfactual orderings, computed from the same result, nothing shipped changed")
    print(f"    recovery, pool order (as shipped) : {_rate([r['treat_recovered'] for r in strat_b])}")
    print(f"    recovery, score order             : {_rate([r['score_recovered'] for r in strat_b])}")
    print(f"    recovery, promoted first          : {_rate([r['promoted_recovered'] for r in strat_b])}")

    print()
    print("  per query")
    for r in rows:
        stratum = "A" if r["in_hits"] else "B"
        print(
            f"    [{stratum}] {r['slug']:<20} baseline={'ok' if r['base_recovered'] else '--'} "
            f"treatment={'ok' if r['treat_recovered'] else '--'} "
            f"fetched={'y' if r['fetched'] else 'n'}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
