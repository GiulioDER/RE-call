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

**Which questions the retrieval metrics count.** `--score-retrieval-on held|all`, defaulting to
`held` — the same flag, default and meaning as `recall.eval.labelled`, defined once in
`recall.eval._scoring`. Every LongMemEval figure this repo publishes was computed under `held`.
`all` scores the fit half's answerable questions too, doubling the retrieval sample at the cost
of one extra haystack populate per fit answerable question; abstention stays on the held half in
both modes, because its threshold is fitted on the other one. `haystack_chunks` follows the
retrieval population, so it is not comparable across modes — it publishes its own `n`.

**What this arm still does not fix.** Temporal-reasoning remains unscoreable: a session reused
across haystacks is stamped with a different date each time and the corpus holds one copy of it
(see `longmemeval.py`). Per-question scoring narrows the candidate set; it does not restore the
per-haystack timeline.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psycopg

from recall.calibration import from_samples
from recall.embeddings import Embedder, embed_query, embedding_profile_id
from recall.eval._scoring import (
    DEFAULT_SCORE_RETRIEVAL_ON,
    SCORE_RETRIEVAL_ON,
    check_score_retrieval_on,
    scores_retrieval,
)
from recall.eval.metrics import latency_report, wilson_ci
from recall.retriever import HybridRetriever
from recall.store import PgVectorStore
from recall.trust import evaluate as trust_evaluate
from recall.eval._research_trust import research_search

DEFAULT_DSN = os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall")

#: Columns copied verbatim. `tsv` is deliberately absent — it is GENERATED ALWAYS ... STORED,
#: cannot be inserted into, and regenerates from `text` on the target. Listing it would make the
#: INSERT fail; omitting `text` would make it regenerate empty and silently disable the sparse
#: leg of the hybrid retriever, which is the failure a test pins.
#:
#: BOTH time columns. Omitting `first_indexed_at` let it take the scratch table's `DEFAULT now()`,
#: i.e. the populate instant, while `indexed_at` kept the master's older value — a first write
#: AFTER the last write, so a `known_as_of` replay in this arm read the whole haystack as
#: `not_yet_known`. Unlike `tsv` it is an ordinary stored column with nothing stopping the insert.
_COPIED = (
    "tenant_id", "id", "source", "text", "metadata", "embedding",
    "indexed_at", "first_indexed_at",
)


def populate_haystack(
    dsn: str, dim: int, master: str, scratch: str, files: list[str], *,
    store: PgVectorStore | None = None,
) -> PgVectorStore:
    """Fill `scratch` with the master rows whose source file is in `files`; return its store.

    Pass `store` to reuse one `PgVectorStore` across every question (the caller owns its lifecycle):
    `evaluate` builds the scratch store once and only the rows change per question, so constructing
    a fresh store + running `ensure_schema` each time was ~1000 connection opens and ~500 redundant
    DDL passes over a full LongMemEval run. Without `store`, a new store is created and its schema
    ensured — the standalone path the tests exercise.

    Replaces the table's contents rather than adding to them: one scratch table is reused across
    every question, and an append would grow the haystack monotonically, making each successive
    question easier against a corpus that is supposed to be fixed at ~40 sessions.
    """
    if store is None:
        store = PgVectorStore(dsn, dim=dim, table=scratch)
        store.ensure_schema()
    # The MASTER table is never migrated by this arm (`ensure_schema` runs on the SCRATCH table
    # only), so a master built before `first_indexed_at` existed would make this SELECT die on a
    # raw UndefinedColumn. Degrade to its last write, which is what the read path does anyway.
    with psycopg.connect(dsn) as probe:
        has_first = probe.execute(
            "SELECT 1 FROM pg_attribute WHERE attrelid = %s::regclass "
            "AND attname = 'first_indexed_at' AND NOT attisdropped",
            (master,),
        ).fetchone() is not None
    cols = ", ".join(_COPIED)
    select_cols = cols if has_first else ", ".join(
        c if c != "first_indexed_at" else "indexed_at AS first_indexed_at" for c in _COPIED
    )
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
            f"SELECT {select_cols} FROM {master} WHERE metadata->>'file' = ANY(%s)",
            (files,),
        )
        conn.commit()
    return store


def _rate(flags: list[bool]) -> dict:
    lo, hi = wilson_ci(flags)
    return {"rate": round(sum(flags) / len(flags), 4) if flags else float("nan"),
            "ci": (round(lo, 4), round(hi, 4)), "n": len(flags)}


def _assert_master_covers(store: PgVectorStore, master: str, questions: list[dict]) -> dict:
    """Refuse to score against a master missing sessions the questions reference.

    This arm never indexes — it copies rows out of `master` — so it cannot repair a partial index
    the way `labelled` does, and it cannot notice one either: `populate_haystack` matches
    `metadata->>'file' = ANY(files)`, so a session absent from the master simply yields a smaller
    haystack, and the run reports a hit rate computed against a corpus that is not the one being
    claimed, with no error anywhere.

    Postgres makes this concrete rather than theoretical. It is crash-safe, so a reset mid-build
    never leaves corrupt rows — it leaves *fewer* rows, each committed and valid. A kept table
    (`labelled --table`) is precisely where such a remnant survives to be reused.
    """
    required = {f for q in questions for f in q.get("haystack_files", [])}
    with store._connect() as conn:  # noqa: SLF001 - eval-only helper, not library surface
        rows = conn.execute(f"SELECT DISTINCT metadata->>'file' FROM {master}").fetchall()
    present = {r[0] for r in rows if r[0]}
    missing = sorted(required - present)
    if missing:
        raise ValueError(
            f"master table {master!r} is missing {len(missing)} of {len(required)} session(s) the "
            f"questions reference — e.g. {missing[:5]}. Scoring against it would measure a smaller "
            f"haystack than the one reported. This is what a Postgres reset mid-index leaves "
            f"behind (crash-safe means fewer rows, not corrupt rows). Repair it by re-running "
            f"`recall.eval.labelled --table {master}`: indexing skips by stored content hash, so "
            f"only the missing sessions are embedded."
        )
    return {"sessions_required": len(required), "sessions_present": len(required)}


def evaluate(dsn: str, master: str, questions: list[dict], embedder: Embedder,
             k: int = 5, score_retrieval_on: str = DEFAULT_SCORE_RETRIEVAL_ON) -> dict:
    """Score LongMemEval per haystack. `score_retrieval_on` means what it means in `labelled`.

    The flag is threaded through rather than left out. Both harnesses report a key called
    `hit_at_k`, and once one of them could be widened to every answerable question while the
    other could not, the same key named two different populations depending on which module
    produced the JSON — with nothing in either artifact saying which. The default is `"held"`
    here for the same reason it is there: it is what every already-published RATE was computed
    under, and a default is what decides what a published number MEANS.

    Abstention is unaffected, in both modes and in both harnesses: `false_abstain` and
    `abstention_accuracy` stay on the held half, because the threshold is fitted on the other one.

    Cost of `"all"`: one extra haystack populate (a TRUNCATE plus an indexed INSERT ... SELECT
    against the master) and one extra retrieval per fit ANSWERABLE question. On LongMemEval-S
    that is roughly +40% of the run's populates rather than a doubling, because the calibration
    pass already populates every fit haystack — and it re-populates them, which is redundant work
    this implementation does not yet avoid.
    """
    check_score_retrieval_on(score_retrieval_on)
    scratch = "pq_" + uuid.uuid4().hex[:8]
    fit, held = questions[::2], questions[1::2]

    # One scratch store for the whole run: build it (and its schema) once, then only refill the
    # rows per question. Rebuilding a PgVectorStore + ensure_schema per question was ~1000
    # connection opens and ~500 redundant DDL passes over a full LongMemEval run.
    store = PgVectorStore(dsn, dim=embedder.dim, table=scratch)
    store.ensure_schema()
    try:
        # BEFORE any scoring: a partial master must stop the run, not shrink its haystacks.
        coverage = _assert_master_covers(store, master, questions)
        def top_cos_in_haystack(q: dict) -> float:
            populate_haystack(dsn, embedder.dim, master, scratch, q["haystack_files"], store=store)
            hits = store.query_dense(embed_query(embedder, q["query"]), k=1)
            return hits[0].score if hits else 0.0

        # Calibrated on the fit half, exactly as `labelled` does — a threshold from another
        # corpus's cosine regime does not transfer (FINDINGS section 2), and per-question
        # haystacks are a different regime again from the merged corpus.
        cal = from_samples(
            embedding_profile_id(embedder),
            [top_cos_in_haystack(q) for q in fit if q.get("answerable")],
            [top_cos_in_haystack(q) for q in fit if not q.get("answerable")],
        )

        hits, reciprocal, latency, misses = [], [], [], []
        by_type: dict[str, list[bool]] = collections.defaultdict(list)
        abstained, false_abstain, haystack_sizes = [], [], []

        # Index parity IS the fit/held boundary here, read straight off `enumerate`, and widening
        # the retrieval sample must not change that: selecting the held subset back out BY ID is
        # what lets a duplicate id in `labelled` score abstention on a fitted question (see
        # `labelled.check_question_ids`). Note the narrower claim than "ids are unused in this
        # arm" — `q["id"]` is still read for the miss report below and in `main()`, so a question
        # set with a missing id fails there, late, rather than at entry.
        #
        # Under the `"held"` default the loop visits exactly `questions[1::2]`, in order, doing
        # exactly what the single `for q in held` pass did — same questions, same order, scored
        # identically — and `"all"` is the only thing that adds work.
        for pos, q in enumerate(questions):
            is_held = pos % 2 == 1
            answerable = bool(q.get("answerable"))
            # Stated positively, and through the rule `labelled` uses: an answerable question is
            # retrieved when the shared predicate says so, an unanswerable one only on the held
            # half (it feeds `abstention_accuracy`, which never widens).
            retrieved = scores_retrieval(is_held=is_held, mode=score_retrieval_on)
            if not (retrieved if answerable else is_held):
                continue
            populate_haystack(dsn, embedder.dim, master, scratch, q["haystack_files"], store=store)
            # Descriptive, and therefore reported over whatever was actually scored: under
            # `"all"` that includes the fit half's haystacks. It publishes its own `n` below,
            # because that makes it the fourth population in this report, not the fourth
            # unnamed denominator.
            haystack_sizes.append(store.count())
            if answerable:
                t = time.perf_counter()
                res = HybridRetriever(store, embedder).search(q["query"], k=k)
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
                # Reuse the hybrid `res` already retrieved above for the abstain decision instead
                # of a second full retrieval (PERF-003): abstained depends only on the hits +
                # supersession + threshold, so this equals research_search at the same pool.
                #
                # HELD ONLY, in both modes. Widening retrieval must never widen this: the
                # threshold was fitted on the fit half, so scoring `false_abstain` over a fit
                # question is fitting and scoring on the same data.
                if is_held:
                    sup, unres = store.supersession() if res.hits else ({}, frozenset())
                    false_abstain.append(
                        trust_evaluate(res, sup, cal, datetime.now(timezone.utc), unres).abstained
                    )
            else:
                abstained.append(
                    research_search(store, embedder, q["query"], k=k, calibration=cal).abstained
                )

        return {
            "protocol": "per-question haystack",
            "master_table": master,
            # Recorded, not merely checked: a reviewer reading this JSON can see the master was
            # verified complete rather than assumed to be.
            "master_coverage": coverage,
            "haystack_chunks": {
                # `n` because this population MOVES with `score_retrieval_on` (it is the
                # retrieval-scored questions plus the held unanswerable ones), and an aggregate
                # whose population moves while its key does not is two numbers wearing one name.
                "n": len(haystack_sizes),
                "mean": round(statistics.mean(haystack_sizes), 1) if haystack_sizes else 0,
                "min": min(haystack_sizes, default=0), "max": max(haystack_sizes, default=0),
            },
            # FOUR metric families, FOUR denominators, each read off the list that was actually
            # scored rather than recomputed from a definition — the same contract `labelled`
            # publishes, so a reader can compare the two artifacts without guessing. The fourth
            # is `haystack_chunks.n` above, kept beside its aggregate.
            "questions": {"total": len(questions), "held_out": len(held),
                          "score_retrieval_on": score_retrieval_on,
                          "retrieval_scored_on": len(hits),
                          "false_abstain_scored_on": len(false_abstain),
                          "abstention_accuracy_scored_on": len(abstained)},
            "threshold": cal.threshold,
            f"hit_at_{k}": _rate(hits),
            "mrr": round(statistics.mean(reciprocal), 4) if reciprocal else float("nan"),
            "by_type": {t: _rate(v) for t, v in sorted(by_type.items())},
            "latency_ms": latency_report(latency),
            "abstention_accuracy": _rate(abstained),
            "false_abstain": _rate(false_abstain),
            "misses": misses,
        }
    finally:
        store.drop_table()
        store.close()


def main() -> None:
    ap = argparse.ArgumentParser(prog="recall.eval.longmemeval_perq")
    ap.add_argument("--questions", required=True, help="questions.json from the converter")
    ap.add_argument("--master", required=True, help="table holding the fully indexed corpus")
    ap.add_argument("--embedder", default="fastembed")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--score-retrieval-on", default=DEFAULT_SCORE_RETRIEVAL_ON,
                    choices=list(SCORE_RETRIEVAL_ON),
                    help="which questions the RETRIEVAL metrics are scored on — the same flag, "
                         "the same default and the same meaning as recall.eval.labelled. 'held' "
                         "(default) reproduces every published RATE; 'all' scores every "
                         "answerable question, doubling the retrieval sample at the cost of one "
                         "extra haystack populate and one extra retrieval per fit ANSWERABLE "
                         "question (unanswerable fit questions are skipped in both modes). "
                         "Abstention always uses the held half.")
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    args = ap.parse_args()

    from recall.eval.labelled import _make_embedder, check_question_ids

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    # Before the comprehension below dereferences `q["id"]` itself. `evaluate` does not call this
    # (index parity, not ids, carries the split in this arm), but the CLI reads ids in two places
    # and a bare `KeyError` from a comprehension is not a report.
    check_question_ids(questions)
    missing = [q["id"] for q in questions if not q.get("haystack_files")]
    if missing:
        raise SystemExit(
            f"{len(missing)} question(s) carry no haystack_files (e.g. {missing[:3]}). "
            "Re-run the converter — this arm cannot be scored without them."
        )
    print(json.dumps(evaluate(args.dsn, args.master, questions,
                              _make_embedder(args.embedder), k=args.k,
                              score_retrieval_on=args.score_retrieval_on), indent=2))


if __name__ == "__main__":
    main()
