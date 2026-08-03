"""Can an embedder-independent abstention rule beat an absolute cosine floor?

::

    # $0 of generator/judge spend — retrieval and arithmetic only.
    python -m benchmarks.beam.threshold_probe --data <shards> --conversations 0-14 \
        --table bench_beam_cal --embedder router:openai/text-embedding-3-small

The problem, measured
---------------------
`DEFAULT_GAP_THRESHOLD` is an ABSOLUTE cosine, 0.50, and it is right for bge-small whose
similarities run high. On `text-embedding-3-small` the top-1 cosines of this corpus run 0.41-0.76,
so the floor sits inside the distribution and returned empty retrieval — a guaranteed zero — for
18 of 300 questions. `recall calibrate` is the intended remedy and does not rescue it: fitted on
100 held-out questions it produced 0.617, uncertified, which abstains MORE. Wiring the loader
(PR #101) made calibrating work; it did not make the number easy to get right.

So the question is whether a rule with no absolute constant does better. Two candidates, each
with a failure mode predicted BEFORE running:

  **percentile** — abstain unless the top score clears the p-th percentile of this query's own
  score distribution. Embedder-independent by construction. Predicted failure: a percentile
  abstains on a fixed FRACTION of queries by definition, so on a corpus that does contain every
  answer it still discards p% of them.

  **gap** — abstain unless the top score exceeds the k-th by some margin, i.e. look at separation
  rather than level. Invariant to the cosine regime without fixing an abstention rate. Predicted
  failure: it flattens when a corpus restates one fact many ways — exactly the TTL case, where 24
  paraphrases of the stale value crushed the spread. That is the case it most needs to survive.

What this measures, and what it cannot
--------------------------------------
For every question: would the rule have retrieved anything, and was the question answerable? That
yields recovered-answerable and lost-abstentions per rule per parameter — the two quantities whose
difference decided the last change. It does NOT produce a score: converting a retrieval rule into
a BEAM cell needs the paid arm, and §9h/§9i are the standing record of what happens when a
mechanism measured on a proxy is promoted to a recommendation before the cell is run.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from benchmarks.beam.dataset import iter_conversations, parse_conversation_indices
from benchmarks.beam.systems import BEAM_TABLE, BeamRecallSystem, require_indexed
from recall.eval.locomo import DEFAULT_DSN
from recall.embeddings import embedding_profile_id
from recall.store import PgVectorStore

#: Percentile candidates, as the fraction of the query's own scores that must be cleared.
PERCENTILES = (0.0, 0.10, 0.25, 0.50)

#: Gap candidates: top1 - median(scores), as a fraction of top1. Scale-free, so a model whose
#: cosines cluster at 0.9 and one clustered at 0.5 are judged on separation, not level.
GAP_RATIOS = (0.0, 0.02, 0.05, 0.10)

#: The shipped absolute floor, kept as the baseline every candidate must beat, plus one lower
#: candidate.
#:
#: Deliberately a LITERAL, with the coupling to the library enforced by a test
#: (`tests/test_bench_threshold_probe_floor.py`) rather than by an import. Importing
#: `DEFAULT_GAP_THRESHOLD` here looks tidier and is worse: this is a candidate GRID, results are
#: keyed `f"{rule}@{param}"`, and if the library default ever moved to 0.40 the tuple would become
#: `(0.4, 0.4)` — two grid points silently collapsing onto one key, taking the baseline row with
#: them and raising nothing. Lowering that default is a live proposal, so this is not hypothetical.
#: A test can say "these drifted apart"; a shared reference can only make them agree by erasing one.
ABSOLUTE_FLOORS = (0.50, 0.40)

#: Corpus-quantile candidates. The floor is the q-th quantile of the TOP-1 scores observed across
#: many queries on this corpus — unsupervised (no labels) and, in principle, adapting to whatever
#: cosine regime the embedder produces.
#:
#: Its structural cost is visible before running: a quantile of the top-1 distribution abstains on
#: ~q of queries BY CONSTRUCTION. That is a defensible policy only if the worst q% really are the
#: unanswerable ones. On BEAM they are not (§9h: unanswerable questions score HIGHER), so this is
#: expected to fail HERE while possibly being right on a corpus that is not adversarial by design.
CORPUS_QUANTILES = (0.02, 0.05, 0.10)

UNANSWERABLE = "abstention"


def _rule_retrieves(scores: list[float], rule: str, param: float) -> bool:
    """Would this rule return anything for a query with these candidate scores?

    `param` is the rule's parameter, already resolved to a comparable number by the caller —
    for `corpus_quantile` that means the floor derived from the corpus, not the quantile itself.
    """
    if not scores:
        return False
    top = max(scores)
    if rule in ("absolute", "corpus_quantile"):
        return top >= param
    if rule == "percentile":
        # `param` of the mass must lie BELOW the top score. At 0.0 this never abstains, which is
        # the honest floor of the family and is included so the curve shows what it costs.
        below = sum(1 for s in scores if s < top) / len(scores)
        return below >= param
    if rule == "gap":
        med = statistics.median(scores)
        return top > 0 and (top - med) / top >= param
    raise ValueError(rule)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--chat-size", default="1M")
    parser.add_argument("--conversations", default="0-14")
    parser.add_argument("--embedder", default="router:openai/text-embedding-3-small")
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--table", default=BEAM_TABLE)
    parser.add_argument("--candidates", type=int, default=45)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    indices = parse_conversation_indices(args.conversations)

    system = BeamRecallSystem(
        args.dsn, embedder_name=args.embedder, k=args.candidates, table=args.table
    )
    embedder = system._embedder  # noqa: SLF001 - probe
    rows: list[dict[str, Any]] = []

    chunks_per_tenant: dict[str, int] = {}
    for conv in iter_conversations(args.data, args.chat_size, indices):
        tenant = f"beam-{conv.chat_size}-{conv.index}".lower()
        # Fail BEFORE scoring, not after. This probe does not index; on an empty tenant every
        # candidate rule below "correctly starves" every unanswerable question and serves none of
        # the answerable ones, and the resulting table is complete enough to publish.
        with PgVectorStore(args.dsn, dim=embedder.dim, tenant=tenant, table=args.table) as store:
            chunks_per_tenant[tenant] = require_indexed(
                store, tenant=tenant, table=args.table, what="threshold_probe"
            )
        for q in conv.questions:
            vector = embedder.embed([q.question])[0]
            # RAW candidate scores, before any threshold: a probe that sampled through the live
            # rule would only ever see what the current rule already admits, and every candidate
            # would look identical to the incumbent.
            with PgVectorStore(
                args.dsn, dim=embedder.dim, tenant=tenant, table=args.table
            ) as store:
                hits = store.query_dense(vector, k=args.candidates)
            rows.append(
                {
                    "question_id": q.question_id,
                    "type": q.question_type,
                    "answerable": q.question_type != UNANSWERABLE,
                    "scores": [round(h.score, 4) for h in hits],
                }
            )
        print(f"  conversation {conv.index}: {len(rows)} questions", flush=True)

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]

    # The corpus-level floor is derived from the observed top-1 distribution, which is what makes
    # it unsupervised. Computed over ALL queries here; a deployment would sample at index time.
    tops = sorted(max(r["scores"]) for r in rows if r["scores"])
    corpus_floors = {
        q: (tops[min(int(q * len(tops)), len(tops) - 1)] if tops else 0.0) for q in CORPUS_QUANTILES
    }

    results: dict[str, Any] = {}
    for rule, params in (
        ("absolute", ABSOLUTE_FLOORS),
        ("percentile", PERCENTILES),
        ("gap", GAP_RATIOS),
        ("corpus_quantile", tuple(CORPUS_QUANTILES)),
    ):
        for p in params:
            resolved = corpus_floors[p] if rule == "corpus_quantile" else p
            served_ans = sum(1 for r in answerable if _rule_retrieves(r["scores"], rule, resolved))
            served_unans = sum(
                1 for r in unanswerable if _rule_retrieves(r["scores"], rule, resolved)
            )
            results[f"{rule}@{p}"] = {
                "resolved_floor": round(resolved, 4),
                # An answerable question served nothing is a guaranteed zero; served is the
                # chance to be right. An unanswerable question served nothing is CORRECT.
                "answerable_served": served_ans,
                "answerable_starved": len(answerable) - served_ans,
                "unanswerable_correctly_starved": len(unanswerable) - served_unans,
                "n_answerable": len(answerable),
                "n_unanswerable": len(unanswerable),
            }

    report = {
        "candidates_per_query": args.candidates,
        # Recorded so an empty or half-built tenant is visible in the artifact, not
        # only in the guard that refused to run against one.
        "chunks_per_tenant": chunks_per_tenant,
        "conversations": args.conversations,
        "embedder": embedding_profile_id(embedder),
        # The regime itself — the thing an absolute floor cannot adapt to, published so two runs
        # on different embedders can be compared directly.
        "top1_distribution": {
            "min": round(tops[0], 4) if tops else None,
            "q05": round(tops[int(0.05 * len(tops))], 4) if tops else None,
            "median": round(statistics.median(tops), 4) if tops else None,
            "max": round(tops[-1], 4) if tops else None,
        },
        "corpus_floors": {str(q): round(v, 4) for q, v in corpus_floors.items()},
        "rules": results,
        "per_question": [{k: v for k, v in r.items() if k != "scores"} for r in rows],
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_question"}, indent=1))


if __name__ == "__main__":
    main()
