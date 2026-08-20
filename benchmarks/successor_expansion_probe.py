"""Pre-registered probe for successor directed expansion and the ordering of promoted successors.

Records, in order, each committed before the run it describes:

1. `docs/preregistrations/2026-08-19-successor-directed-expansion.md` - does fetching help? Null.
2. `docs/preregistrations/2026-08-20-successor-expansion-recalibrated.md` - was it the threshold?
   No, and the null deepened.
3. `docs/preregistrations/2026-08-20-successor-rank-hypothesis.md` - successors are promoted 6 of 6
   and then outranked. Confirmed, and it retracted the diagnoses in 1 and 2.
4. `docs/preregistrations/2026-08-20-successor-ordering-regression.md` - THIS one. Three orderings
   over 30 pairs, plus the regression set that can say no.

The fourth record is the reason for the regression set. The first three ran on a fixture containing
only queries the successor was supposed to win, so they could not report the cost of promoting it.
A fixture that cannot produce a regression cannot report one, and a displacement rate of 0.00 from
such a fixture reads exactly like good news.

    eval "$(scripts/session-db.sh up)"
    python -m benchmarks.successor_expansion_probe

⚠️ `-m`, from the worktree root, NOT `python benchmarks/successor_expansion_probe.py`. Run as a
script, Python puts the SCRIPT's directory on `sys.path[0]`, so `benchmarks/` goes on the path and
the worktree root does not. `import recall` then falls through to whatever is installed, which on
a developer machine that is the MAIN CHECKOUT: the first run of this probe imported the main
checkout's `recall/retriever.py` and died on a symbol that exists only on the branch under test.
That failure was loud. The dangerous version is silent, a benchmark that runs happily and scores
the main checkout while reporting a number against your branch. The guard below the imports turns
the silent case back into a loud one.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import shutil
import tempfile
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
from recall.evidence import EvidencePolicy, build_evidence_bundle  # noqa: E402
from recall.eval.calibrate import measure_top_cosines  # noqa: E402
from recall.eval.metrics import wilson_ci  # noqa: E402
from recall.index import Indexer  # noqa: E402
from recall.retriever import SuccessorExpansionPolicy  # noqa: E402
from recall.store import PgVectorStore  # noqa: E402
from recall.types import TrustedResult  # noqa: E402

from benchmarks.successor_fixture import (  # noqa: E402
    CALIBRATION_ANSWERABLE,
    CALIBRATION_UNANSWERABLE,
    PAIRS,
    REGRESSIONS,
    UNANSWERABLE,
    documents,
)

#: The caller-facing depth. The library default, so the measurement describes the shipped shape.
K = 5
#: The three orderings, run as REAL behaviour selected by policy rather than as counterfactuals
#: computed after the fact. The third record's 1.00 was a counterfactual, which cannot exercise the
#: code path that would ship and so cannot be evidence for shipping it.
ARMS: dict[str, SuccessorExpansionPolicy] = {
    "pool": SuccessorExpansionPolicy(enabled=True, ordering="pool"),
    "promoted_first": SuccessorExpansionPolicy(enabled=True, ordering="promoted_first"),
    "inherit": SuccessorExpansionPolicy(enabled=True, ordering="inherit"),
}
DISTRACTOR_DIR = Path(__file__).resolve().parent.parent / "docs"


def _ok_files(result: TrustedResult) -> list[str]:
    return [h.provenance.file or "" for h in result.hits if h.verdict == "ok"]


def _top_ok(result: TrustedResult) -> str:
    ok = _ok_files(result)
    return ok[0] if ok else ""


def _all_files(result: TrustedResult) -> set[str]:
    return {h.provenance.file or "" for h in result.hits}


#: The DEFAULT policy, deliberately. The question this measures is what the shipped consumer gets,
#: so any tuning here would answer a different question well.
BUNDLE_POLICY = EvidencePolicy()


def _in_bundle(result: TrustedResult, filename: str) -> bool:
    """Does `filename` reach the evidence bundle a generator would actually receive?

    Calls the real `build_evidence_bundle` rather than comparing a rank against `max_items`. The
    two agree today, and reimplementing the selection rule here is precisely the counterfactual
    mistake the third record made: a model of the code cannot exercise the code.

    Matches on the basename of `source`, because `EvidenceItem` carries `source` and not `file`.
    """
    bundle = build_evidence_bundle(result, BUNDLE_POLICY)
    return any(Path(item.source).name == filename for item in bundle.items)


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
        print('RECALL_TEST_DSN is unset. Run: eval "$(scripts/session-db.sh up)"', file=sys.stderr)
        return 2

    embedder = FastEmbedEmbedder()
    calib = [{"query": q, "answerable": True} for q in CALIBRATION_ANSWERABLE]
    calib += [{"query": q, "answerable": False} for q in CALIBRATION_UNANSWERABLE]
    measured = {p.query for p in PAIRS} | set(UNANSWERABLE) | {r.query for r in REGRESSIONS}
    leak = measured & (set(CALIBRATION_ANSWERABLE) | set(CALIBRATION_UNANSWERABLE))
    if leak:
        # Asserted rather than trusted. Disjointness is one careless paste away from silently not
        # holding, and the second record exists because it silently did not.
        raise SystemExit(f"calibration set overlaps the measured queries: {sorted(leak)}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        distractors = _build_corpus(root)
        table = "successor_probe_" + uuid.uuid4().hex[:8]
        store = PgVectorStore(dsn, dim=embedder.dim, table=table)
        try:
            store.ensure_schema()
            stats = Indexer(store, embedder).index_path(root)
            cal = from_samples(
                embedding_profile_id(embedder), *measure_top_cosines(store, embedder, calib)
            )

            def search(query: str, policy: SuccessorExpansionPolicy | None) -> TrustedResult:
                return research_search(
                    store, embedder, query, k=K, calibration=cal, successor_expansion=policy
                )

            pairs = []
            for pair in PAIRS:
                successor, stale = f"{pair.slug}_v2.md", f"{pair.slug}_v1.md"
                base = search(pair.query, None)
                row = {
                    "slug": pair.slug,
                    "in_hits": successor in _all_files(base),
                    "base_recovered": _top_ok(base) == successor,
                    "base_str": stale in _ok_files(base),
                }
                for name, policy in ARMS.items():
                    result = search(pair.query, policy)
                    row[f"{name}_recovered"] = _top_ok(result) == successor
                    row[f"{name}_bundled"] = _in_bundle(result, successor)
                    row[f"{name}_str"] = stale in _ok_files(result)
                pairs.append(row)

            regressions = []
            for reg in REGRESSIONS:
                gold = f"{reg.slug}.md"
                base = search(reg.query, None)
                row = {
                    "slug": reg.slug,
                    # APPARATUS. A regression query tests nothing unless a superseded document is
                    # actually retrieved, and that is a property of the embedder, not of intent.
                    "pulled_stale": reg.expects_stale in _all_files(base),
                    "base_gold_top": _top_ok(base) == gold,
                }
                for name, policy in ARMS.items():
                    result = search(reg.query, policy)
                    row[f"{name}_gold_top"] = _top_ok(result) == gold
                    row[f"{name}_gold_bundled"] = _in_bundle(result, gold)
                regressions.append(row)

            controls = []
            for query in UNANSWERABLE:
                row = {"base": search(query, None).abstained}
                for name, policy in ARMS.items():
                    row[name] = search(query, policy).abstained
                controls.append(row)
        finally:
            try:
                store.drop_table()
            finally:
                store.close()

    strat_b = [r for r in pairs if not r["in_hits"]]
    strat_a = [r for r in pairs if r["in_hits"]]
    # Only a regression query that BOTH drags in a superseded document AND gets the gold answer
    # right under shipped ordering can show a displacement. The other two counts are printed rather
    # than folded away, because a small denominator here is the difference between "no cost" and
    # "no measurement".
    usable = [r for r in regressions if r["pulled_stale"] and r["base_gold_top"]]
    no_stale = [r for r in regressions if not r["pulled_stale"]]
    no_gold = [r for r in regressions if r["pulled_stale"] and not r["base_gold_top"]]

    print("=" * 78)
    print("APPARATUS (checked before any quality number is read)")
    print("=" * 78)
    print(f"  corpus                  : {stats.chunks} chunks / {stats.files} files "
          f"({len(documents())} authored, {distractors} repository docs)")
    print(f"  calibration             : threshold {cal.threshold:.4f}, "
          f"{len(CALIBRATION_ANSWERABLE)} answerable / {len(CALIBRATION_UNANSWERABLE)} unanswerable")
    print(f"  supersession pairs      : {len(pairs)}")
    print(f"  stratum B (absent)      : {len(strat_b)}   stratum A (present): {len(strat_a)}")
    print(f"  baseline recovery on B  : {_rate([bool(r['base_recovered']) for r in strat_b])}")
    print(f"  regression set          : {len(regressions)} authored")
    print(f"    usable                : {len(usable)}")
    print(f"    excluded, no stale doc retrieved : {len(no_stale)} "
          f"{[str(r['slug']) for r in no_stale] or ''}")
    print(f"    excluded, gold not top at baseline: {len(no_gold)} "
          f"{[str(r['slug']) for r in no_gold] or ''}")

    failed = []
    if not strat_b:
        failed.append("stratum B is empty: the fixture cannot exhibit the condition")
    if any(r["base_recovered"] for r in strat_b):
        failed.append("baseline recovered on stratum B: stratification is reading the wrong pool")
    if len(usable) < len(regressions) / 2:
        failed.append(
            f"only {len(usable)} of {len(regressions)} regression queries are usable: the "
            "displacement column measured almost nothing and 0.00 there is not good news"
        )
    if failed:
        print()
        for line in failed:
            print(f"  APPARATUS FAILURE: {line}")
        print("\n  The displacement column below is NOT interpretable.")

    print()
    print("=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  {'arm':<16} {'recovery (stratum B)':<26} {'gold kept (regression)':<26} str_trust")
    print(f"  {'baseline':<16} {_rate([bool(r['base_recovered']) for r in strat_b]):<26} "
          f"{_rate([bool(r['base_gold_top']) for r in usable]):<26} "
          f"{_rate([bool(r['base_str']) for r in pairs])}")
    for name in ARMS:
        print(f"  {name:<16} {_rate([bool(r[f'{name}_recovered']) for r in strat_b]):<26} "
              f"{_rate([bool(r[f'{name}_gold_top']) for r in usable]):<26} "
              f"{_rate([bool(r[f'{name}_str']) for r in pairs])}")

    print()
    print("  BUNDLE MEMBERSHIP: what build_evidence_bundle(EvidencePolicy()) actually delivers")
    print(f"    {'arm':<16} {'successor in bundle':<28} gold in bundle")
    for name in ARMS:
        print(f"    {name:<16} {_rate([bool(r[f'{name}_bundled']) for r in strat_b]):<28} "
              f"{_rate([bool(r[f'{name}_gold_bundled']) for r in usable])}")
    print("    (compare against the top-1 rows above: if these are all high, the ordering is")
    print("     near-irrelevant to the consumer that ships, and top-1 was the wrong metric)")

    print()
    print("  displacement: gold was top under shipped ordering and is NOT top under the arm")
    for name in ARMS:
        displaced = [r for r in usable if not r[f"{name}_gold_top"]]
        print(f"    {name:<16} {_rate([not r[f'{name}_gold_top'] for r in usable])}   "
              f"{[str(r['slug']) for r in displaced] or ''}")

    print()
    print("  stratum A recovery (must not move) and abstention accuracy (must not fall)")
    print(f"    {'baseline':<16} A={_rate([bool(r['base_recovered']) for r in strat_a]):<26} "
          f"abstain={_rate([bool(c['base']) for c in controls])}")
    for name in ARMS:
        print(f"    {name:<16} A={_rate([bool(r[f'{name}_recovered']) for r in strat_a]):<26} "
              f"abstain={_rate([bool(c[name]) for c in controls])}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
