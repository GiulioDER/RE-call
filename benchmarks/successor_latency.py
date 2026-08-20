"""What does the successor fetch cost per query? Registered in 2026-08-20-successor-latency.md.

Separate from `successor_expansion_probe` on purpose, so the quality probe keeps producing exactly
the numbers the earlier six records were measured with.

⚠️ The three latency figures already in those records are worthless and each says so: both arms ran
sequentially in one process with the treatment second, so cache warmth favoured it, and the same
code measured 0.90x, 1.66x and 1.81x. This script exists to fix that design, not to run it again.

Four things and what each is for:

* **warm up** every query through both arms and discard it, so cold model load and cold page cache
  fall outside the measurement instead of landing on whichever arm happened to run first;
* **alternate** which arm runs first by query index, so any residual second-mover advantage splits
  across the sample rather than being handed to one arm;
* **pair by query** and take the median of RATIOS, because query-to-query variance dwarfs the
  effect and a pooled p50 would be comparing different queries to each other;
* **separate the populations**, since only the queries that actually fetch pay anything, and
  averaging them with the ones that do not would hide the entire cost.

    eval "$(scripts/session-db.sh up)"
    python -m benchmarks.successor_latency

⚠️ `-m` from the worktree root. See the guard below and the note in `successor_expansion_probe`.
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
    raise SystemExit(
        f"refusing to run: `recall` resolves to {_ORIGIN}\n"
        f"                 but this script lives under {_HERE}\n"
        "run it as `python -m benchmarks.successor_latency` from the worktree root"
    )

from recall.calibration import from_samples  # noqa: E402
from recall.embeddings import FastEmbedEmbedder, embedding_profile_id  # noqa: E402
from recall.eval._research_trust import research_search  # noqa: E402
from recall.eval.calibrate import measure_top_cosines  # noqa: E402
from recall.index import Indexer  # noqa: E402
from recall.retriever import SuccessorExpansionPolicy  # noqa: E402
from recall.store import PgVectorStore  # noqa: E402

from benchmarks.successor_fixture import (  # noqa: E402
    CALIBRATION_ANSWERABLE,
    CALIBRATION_UNANSWERABLE,
    PAIRS,
    documents,
)

K = 5
#: The policy a default-on change would switch on: enabled, with the shipped `pool` ordering.
TREATMENT = SuccessorExpansionPolicy(enabled=True)
#: Timed repeats per query per arm. Five, so a single scheduling hiccup cannot decide a query's
#: number, and the per-query median is taken before any ratio is formed.
REPEATS = 5
DISTRACTOR_DIR = Path(__file__).resolve().parent.parent / "docs"


def main() -> int:
    dsn = os.environ.get("RECALL_TEST_DSN")
    if not dsn:
        print('RECALL_TEST_DSN is unset. Run: eval "$(scripts/session-db.sh up)"', file=sys.stderr)
        return 2

    embedder = FastEmbedEmbedder()
    calib = [{"query": q, "answerable": True} for q in CALIBRATION_ANSWERABLE]
    calib += [{"query": q, "answerable": False} for q in CALIBRATION_UNANSWERABLE]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, text in documents().items():
            (root / name).write_text(text, encoding="utf-8")
        for doc in sorted(DISTRACTOR_DIR.glob("*.md")):
            shutil.copyfile(doc, root / f"docs__{doc.name}")

        store = PgVectorStore(dsn, dim=embedder.dim, table="lat_" + uuid.uuid4().hex[:8])
        try:
            store.ensure_schema()
            Indexer(store, embedder).index_path(root)
            cal = from_samples(
                embedding_profile_id(embedder), *measure_top_cosines(store, embedder, calib)
            )

            def run(query: str, policy: SuccessorExpansionPolicy | None) -> float:
                started = time.perf_counter()
                research_search(
                    store, embedder, query, k=K, calibration=cal, successor_expansion=policy
                )
                return (time.perf_counter() - started) * 1000.0

            # Population split, from the baseline exactly as the quality runs define it.
            rows = []
            for pair in PAIRS:
                successor = f"{pair.slug}_v2.md"
                base = research_search(store, embedder, pair.query, k=K, calibration=cal)
                present = {h.provenance.file for h in base.hits}
                rows.append({"slug": pair.slug, "query": pair.query,
                             "triggers": successor not in present})

            # WARM UP, discarded. Cold model load and cold page cache land here.
            for row in rows:
                run(row["query"], None)
                run(row["query"], TREATMENT)

            for index, row in enumerate(rows):
                base_ms: list[float] = []
                treat_ms: list[float] = []
                for _ in range(REPEATS):
                    # Alternate which arm goes first. This is the whole correction: the previous
                    # three measurements always ran the treatment second and always flattered it.
                    if index % 2 == 0:
                        base_ms.append(run(row["query"], None))
                        treat_ms.append(run(row["query"], TREATMENT))
                    else:
                        treat_ms.append(run(row["query"], TREATMENT))
                        base_ms.append(run(row["query"], None))
                row["base"] = statistics.median(base_ms)
                row["treat"] = statistics.median(treat_ms)
                row["ratio"] = row["treat"] / row["base"] if row["base"] else float("nan")
        finally:
            try:
                store.drop_table()
            finally:
                store.close()

    trig = [r for r in rows if r["triggers"]]
    quiet = [r for r in rows if not r["triggers"]]

    print("=" * 78)
    print("APPARATUS")
    print("=" * 78)
    print(f"  repeats per query per arm : {REPEATS}, order alternated by query index")
    print("  warm up                   : both arms over every query, discarded")
    print(f"  triggering queries        : {len(trig)}   non triggering: {len(quiet)}")
    if (len(trig), len(quiet)) != (16, 14):
        print("  APPARATUS FAILURE: the split moved from the quality runs' 16 and 14, so these")
        print("                     numbers are not comparable to them.")

    print()
    print("=" * 78)
    print("RESULT   median of PER-QUERY ratios, not a ratio of pooled medians")
    print("=" * 78)
    for name, group in (("triggering", trig), ("non triggering", quiet)):
        if not group:
            print(f"  {name:<16} n/a (n=0)")
            continue
        ratios = sorted(r["ratio"] for r in group)
        base = statistics.median([r["base"] for r in group])
        treat = statistics.median([r["treat"] for r in group])
        print(f"  {name:<16} ratio {statistics.median(ratios):.2f}x  "
              f"[min {ratios[0]:.2f}, max {ratios[-1]:.2f}]  n={len(group)}")
        print(f"  {'':<16} baseline {base:7.1f} ms   treatment {treat:7.1f} ms   "
              f"added {treat - base:+7.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
