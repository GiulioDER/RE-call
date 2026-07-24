"""Score LongMemEval one haystack at a time — the protocol the benchmark publishes.

`recall.eval.labelled` indexes a corpus once and asks every question of all of it. For
LongMemEval that merges 500 haystacks into one 19,195-session corpus, which is a strictly
harder and non-comparable task (measured: hit@5 0.366 merged against 0.719 on evidence-only).
The benchmark's own retrieval protocol gives each question **only its own ~40 sessions**.

Doing that naively means 500 indexes and 500× the embedding bill. It does not have to: the
embedding of a session does not depend on which haystack it is in. So this module embeds the
whole corpus **once** into a persistent master table, and then, per question, copies that
question's rows into a scratch table and runs the ordinary `HybridRetriever` against it. The
retriever under test is unmodified; only the candidate set changes.

    python -m recall.eval.longmemeval_perq --questions s_out/questions.json --master lme_s

The master table is built by indexing the converted corpus into a named table — `Indexer` skips
by content hash, so re-running it over an existing index is a no-op rather than a re-embed.

**What this arm still does not fix.** Temporal-reasoning remains unscoreable: a session reused
across haystacks is stamped with a different date each time and the corpus holds one copy of it
(see `longmemeval.py`). Per-question scoring narrows the candidate set; it does not restore the
per-haystack timeline.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import time
import uuid
from pathlib import Path

from recall.calibration import from_samples
from recall.eval.metrics import wilson_ci
from recall.retriever import HybridRetriever
from recall.store import PgVectorStore
from recall.trust import trusted_search

#: Columns copied verbatim. `tsv` is deliberately absent — it is GENERATED ALWAYS ... STORED,
#: cannot be inserted into, and regenerates from `text` on the target. Listing it would make the
#: INSERT fail; omitting `text` would make it regenerate empty and silently disable the sparse
#: leg of the hybrid retriever, which is the failure a test pins.
_COPIED = ("tenant_id", "id", "source", "text", "metadata", "embedding", "indexed_at")


def populate_haystack(
    dsn: str, dim: int, master: str, scratch: str, files: list[str]
) -> PgVectorStore:
    """Fill `scratch` with the master rows whose source file is in `files`; return its store.

    Replaces the table's contents rather than adding to them: one scratch table is reused across
    every question, and an append would grow the haystack monotonically, making each successive
    question easier against a corpus that is supposed to be fixed at ~40 sessions.
    """
    store = PgVectorStore(dsn, dim=dim, table=scratch)
    store.ensure_schema()
    cols = ", ".join(_COPIED)
    # Match `metadata->>'file'` — the path RELATIVE to the index root, which is exactly the bare
    # name the question file carries — with `=`, not the absolute `source` with a suffix `LIKE`.
    #
    # Two reasons, one correctness and one cost. `LIKE` treats `_` as a single-character
    # wildcard and LongMemEval ids are full of underscores ("answer_c63c0458"), so a suffix
    # match silently pulled in every session differing only at that position: measured 50
    # sessions copied for a 49-session haystack. And a leading-wildcard LIKE cannot use an
    # index, so each of 500 populates sequentially scanned all 321,569 rows; the store already
    # creates an index on this expression, so equality makes it a lookup.
    with store._connect() as conn:  # noqa: SLF001 - eval-only helper, not library surface
        conn.execute(f"TRUNCATE {scratch}")
        conn.execute(
            f"INSERT INTO {scratch} ({cols}) "
            f"SELECT {cols} FROM {master} WHERE metadata->>'file' = ANY(%s)",
            (files,),
        )
        conn.commit()
    return store


def _rate(flags: list[bool]) -> dict:
    lo, hi = wilson_ci(flags)
    return {"rate": round(sum(flags) / len(flags), 4) if flags else float("nan"),
            "ci": (round(lo, 4), round(hi, 4)), "n": len(flags)}


def evaluate(dsn: str, master: str, questions: list[dict], embedder, k: int = 5) -> dict:
    scratch = "pq_" + uuid.uuid4().hex[:8]
    fit, held = questions[::2], questions[1::2]

    def top_cos_in_haystack(q: dict) -> float:
        sub = populate_haystack(dsn, embedder.dim, master, scratch, q["haystack_files"])
        try:
            hits = sub.query_dense(embedder.embed([q["query"]])[0], k=1)
            return hits[0].score if hits else 0.0
        finally:
            sub.close()

    # Calibrated on the fit half, exactly as `labelled` does — a threshold from another corpus's
    # cosine regime does not transfer (FINDINGS section 2), and per-question haystacks are a
    # different regime again from the merged corpus.
    cal = from_samples(
        embedder.name,
        [top_cos_in_haystack(q) for q in fit if q.get("answerable")],
        [top_cos_in_haystack(q) for q in fit if not q.get("answerable")],
    )

    hits, reciprocal, latency, misses = [], [], [], []
    by_type: dict[str, list[bool]] = collections.defaultdict(list)
    abstained, false_abstain, haystack_sizes = [], [], []

    for q in held:
        sub = populate_haystack(dsn, embedder.dim, master, scratch, q["haystack_files"])
        try:
            haystack_sizes.append(sub.count())
            if q.get("answerable"):
                t = time.perf_counter()
                res = HybridRetriever(sub, embedder).search(q["query"], k=k)
                latency.append((time.perf_counter() - t) * 1000)
                files, want = [], set(q["relevant_files"])
                for h in res.hits:
                    f = h.chunk.metadata.get("file")
                    if f and f not in files:
                        files.append(f)
                rank = next((i for i, f in enumerate(files) if f in want), None)
                got = rank is not None and rank < k
                hits.append(got)
                by_type[q.get("question_type", "?")].append(got)
                reciprocal.append(1.0 / (rank + 1) if rank is not None else 0.0)
                if not got:
                    misses.append({"id": q["id"], "query": q["query"],
                                   "expected": sorted(want), "got": files[:k]})
                false_abstain.append(
                    trusted_search(sub, embedder, q["query"], k=k, calibration=cal).abstained
                )
            else:
                abstained.append(
                    trusted_search(sub, embedder, q["query"], k=k, calibration=cal).abstained
                )
        finally:
            sub.close()

    final = PgVectorStore(dsn, dim=embedder.dim, table=scratch)
    final.drop_table()
    final.close()

    lat = sorted(latency)
    return {
        "protocol": "per-question haystack",
        "master_table": master,
        "haystack_chunks": {
            "mean": round(statistics.mean(haystack_sizes), 1) if haystack_sizes else 0,
            "min": min(haystack_sizes, default=0), "max": max(haystack_sizes, default=0),
        },
        "questions": {"total": len(questions), "held_out": len(held)},
        "threshold": cal.threshold,
        f"hit_at_{k}": _rate(hits),
        "mrr": round(statistics.mean(reciprocal), 4) if reciprocal else float("nan"),
        "by_type": {t: _rate(v) for t, v in sorted(by_type.items())},
        "latency_ms": {"p50": round(lat[len(lat) // 2], 1),
                       "p95": round(lat[int(0.95 * len(lat))], 1)} if lat else {},
        "abstention_accuracy": _rate(abstained),
        "false_abstain": _rate(false_abstain),
        "misses": misses,
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog="recall.eval.longmemeval_perq")
    ap.add_argument("--questions", required=True, help="questions.json from the converter")
    ap.add_argument("--master", required=True, help="table holding the fully indexed corpus")
    ap.add_argument("--embedder", default="fastembed")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--dsn", default="postgresql://recall:recall@localhost:5432/recall")
    args = ap.parse_args()

    from recall.eval.labelled import _make_embedder

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    missing = [q["id"] for q in questions if not q.get("haystack_files")]
    if missing:
        raise SystemExit(
            f"{len(missing)} question(s) carry no haystack_files (e.g. {missing[:3]}). "
            "Re-run the converter — this arm cannot be scored without them."
        )
    print(json.dumps(evaluate(args.dsn, args.master, questions,
                              _make_embedder(args.embedder), k=args.k), indent=2))


if __name__ == "__main__":
    main()
