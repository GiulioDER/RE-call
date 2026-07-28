"""Can the abstention floor be derived at index time, with no labels and no user queries?

::

    # $0 — retrieval and arithmetic over an already-indexed corpus. No LLM.
    python -m benchmarks.beam.selfquery_probe --out selfquery_probe.json

The idea being tested (H4)
-------------------------
The regime sweep established that an absolute cosine floor is the wrong TYPE of statistic: 0.50
sits at the 0th percentile of five distributions and the 16th of a sixth, and model and corpus
INTERACT, so no stored per-model constant can work. Whatever sets the floor has to see the actual
corpus with the actual embedder.

The cheapest way to do that without asking the user for anything: after indexing, take a sample of
the corpus's own chunks, use them as queries, and read the top-1 distribution off the result. A low
quantile of that becomes the floor. No labels, no query log, no calibration run.

The obvious flaw, stated first
------------------------------
A chunk used as a query retrieves ITSELF at ~1.0. That is not retrieval, it is identity, so
self-matches are excluded — the probe takes the best NON-self hit. Even then the distribution is
expected to sit HIGH relative to real user questions: a chunk is written in the corpus's own
register and vocabulary, while a user's question is not. If the derived floor comes out far above
the floor implied by real queries, the method is too strict and H4 dies.

That comparison is the whole experiment, and it is possible precisely because the real-query floors
for these six conditions are already measured (`regime_sweep.json`).

Interpretation rule, fixed before running
-----------------------------------------
Agreement is judged on the STARVE RATE the derived floor would produce on real queries, not on the
closeness of the two floor values. A floor 0.05 too high is harmless where the distribution is
sparse there and catastrophic where it is dense; only the rate says which. Within ~3 points of the
target 5 % is usable; beyond ~10 % it is not.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

from benchmarks.systems import resolve_embedder
from recall.eval.locomo import DEFAULT_DSN
from recall.store import PgVectorStore

#: Conditions measured by the regime sweep, as (label, table, tenant). The real-query floors for
#: each already exist, which is what makes this comparable rather than merely descriptive.
CONDITIONS = (
    ("memory/bge-small", "regime_mem_bge_small", "regime-mem", "fastembed"),
    ("memory/bge-large", "regime_mem_bge_large", "regime-mem", "fastembed:BAAI/bge-large-en-v1.5"),
    (
        "memory/text-embedding-3-small",
        "regime_mem_text_embedding_3_small",
        "regime-mem",
        "router:openai/text-embedding-3-small",
    ),
)

#: How many chunks to sample as pseudo-queries. Large enough for a stable low quantile, small
#: enough that a deployment could afford it at index time.
SAMPLE = 300

#: The quantile taken as the floor, matching the regime sweep so the two are comparable.
QUANTILE = 0.05

SEED = 1234


def _self_query_scores(
    dsn: str, table: str, tenant: str, embedder: Any, sample: int, k: int
) -> tuple[list[float], int]:
    """Top-1 NON-self scores for `sample` chunks used as their own queries.

    Returns (scores, skipped). A chunk whose only hit is itself contributes nothing rather than a
    zero: it means the corpus has no second opinion on that text, which is a property of the
    corpus, not a low-similarity observation, and scoring it 0.0 would drag the floor down.
    """
    with PgVectorStore(dsn, dim=embedder.dim, tenant=tenant, table=table) as store:
        chunks = [c for c in store.iter_chunks()]
    rng = random.Random(SEED)
    picked = rng.sample(chunks, min(sample, len(chunks)))

    scores: list[float] = []
    skipped = 0
    for chunk in picked:
        vector = embedder.embed([chunk.text])[0]
        with PgVectorStore(dsn, dim=embedder.dim, tenant=tenant, table=table) as store:
            hits = store.query_dense(vector, k=k)
        # Drop the identity match — a chunk retrieving itself measures nothing.
        others = [h for h in hits if h.chunk.id != chunk.id]
        if not others:
            skipped += 1
            continue
        scores.append(max(h.score for h in others))
    return scores, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--sample", type=int, default=SAMPLE)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--sweep", type=Path, default=Path("regime_sweep.json"))
    parser.add_argument("--out", type=Path, default=Path("selfquery_probe.json"))
    args = parser.parse_args()

    sweep = json.loads(args.sweep.read_text(encoding="utf-8"))
    report: dict[str, Any] = {"sample": args.sample, "quantile": QUANTILE, "conditions": {}}

    for label, table, tenant, spec in CONDITIONS:
        embedder = resolve_embedder(spec)
        scores, skipped = _self_query_scores(
            args.dsn, table, tenant, embedder, args.sample, args.k
        )
        if not scores:
            print(f"  {label}: no usable self-queries, skipped", flush=True)
            continue
        s = sorted(scores)
        derived = s[min(int(QUANTILE * len(s)), len(s) - 1)]

        # The real-query distribution for the same condition, already measured.
        real = sweep["conditions"][label]
        real_floor = real["quantile_floors"][str(QUANTILE)]

        # The number that decides it: applying the DERIVED floor to REAL queries, how much is
        # starved? The target is the quantile itself (5 %).
        # The sweep stored summary stats, not raw scores, so the rate is bounded rather than
        # computed exactly: it is >= 0 and, if derived <= real_floor, at most the quantile.
        report["conditions"][label] = {
            "n_self_queries": len(s),
            "skipped_no_other_hit": skipped,
            "self_query_median": round(statistics.median(s), 4),
            "real_query_median": real["median"],
            "derived_floor": round(derived, 4),
            "real_query_floor": real_floor,
            "floor_delta": round(derived - real_floor, 4),
            "derived_is_stricter": derived > real_floor,
        }
        print(f"  {label}: {report['conditions'][label]}", flush=True)
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
