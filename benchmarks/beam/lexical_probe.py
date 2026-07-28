"""Does lexical absence separate what cosine cannot?

::

    # $0 — retrieval and string matching. No LLM, no embedding beyond the query.
    python -m benchmarks.beam.lexical_probe --data <shards> --conversations 0-14 \
        --table bench_beam_cal --embedder router:openai/text-embedding-3-small

Why this and not another threshold
----------------------------------
Four rules have now been measured and killed: per-query percentile, score gap, corpus quantile on
real queries, corpus quantile from self-queries (§9n). They failed for one shared reason — each is
a MONOTONE function of the dense score, and §9h established that on BEAM the unanswerable questions
score HIGHER than the answerable ones. A monotone transform preserves order, and the order is what
is wrong.

So the criterion for anything worth testing next is that it must not be a function of that score.
Lexical overlap qualifies: whether a rare term from the question appears ANYWHERE in the retrieved
text is independent of how similar the embedder thought they were. The retriever already carries a
BM25 leg, so this signal is present and unused for abstention.

The hypothesis: a question whose distinctive terms appear nowhere in the retrieved chunks is a
question the corpus cannot answer, regardless of how confident the cosine was.

Predicted failure, written first
--------------------------------
**This is expected to fail on BEAM specifically.** Its unanswerable questions are built by asking
for details never stated about topics that WERE discussed — "tell me about Omar's professional
background" when Omar is discussed at length and his background never is. The entities therefore
DO appear; it is the requested attribute that is missing. Lexical presence cannot see that
distinction, so the separation should be weak here.

That prediction is the point. If it fails on BEAM and the mechanism is sound, the next measurement
belongs on a corpus whose unanswerable questions are about genuinely absent subjects — which is
the ordinary case, and the one a deployment actually meets.

Reported both ways: for answerable and unanswerable questions separately, so a weak separation is
visible as a weak separation rather than hidden inside an average.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.beam.dataset import iter_conversations
from benchmarks.beam.systems import BEAM_TABLE, BeamRecallSystem
from recall.eval.locomo import DEFAULT_DSN
from recall.store import PgVectorStore

UNANSWERABLE = "abstention"

#: Terms this common carry no evidence either way; matching on them would make every question look
#: covered. Deliberately short — an aggressive stoplist would do the discriminating itself.
STOPWORDS = frozenset("""
a an and are as at be been but by can did do does for from had has have how i if in into is it its
me my of on or our so than that the their them then there these they this to was we were what when
where which who why will with would you your about more most some such only just also
""".split())

TOKEN = re.compile(r"[a-z0-9][a-z0-9.\-]{2,}")


def _terms(text: str) -> set[str]:
    return {t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS}


def _rare_terms(question: str, corpus_df: Counter[str], max_df: float, n_chunks: int) -> set[str]:
    """Question terms that are RARE in this corpus.

    A term present in most chunks says nothing about whether this question is answerable — it is
    background. Rarity is what makes presence informative, so the filter is on document frequency
    rather than on a fixed vocabulary.
    """
    return {
        t for t in _terms(question)
        if corpus_df.get(t, 0) <= max_df * n_chunks
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--chat-size", default="1M")
    parser.add_argument("--conversations", default="0-14")
    parser.add_argument("--embedder", default="router:openai/text-embedding-3-small")
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--table", default=BEAM_TABLE)
    parser.add_argument("--k", type=int, default=45)
    parser.add_argument("--max-df", type=float, default=0.10,
                        help="A term in more than this fraction of chunks is background, not signal.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    indices: list[int] = []
    for part in args.conversations.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.extend(range(int(lo), int(hi) + 1))
        elif part.strip():
            indices.append(int(part))

    system = BeamRecallSystem(
        args.dsn, embedder_name=args.embedder, k=args.k, table=args.table
    )
    embedder = system._embedder  # noqa: SLF001 - probe
    rows: list[dict[str, Any]] = []

    for conv in iter_conversations(args.data, args.chat_size, sorted(set(indices))):
        tenant = f"beam-{conv.chat_size}-{conv.index}".lower()

        # Document frequency over THIS conversation's chunks — rarity is a property of the corpus
        # being searched, not of English.
        with PgVectorStore(args.dsn, dim=embedder.dim, tenant=tenant, table=args.table) as store:
            chunk_terms = [_terms(c.text) for c in store.iter_chunks()]
        df: Counter[str] = Counter()
        for ts in chunk_terms:
            df.update(ts)
        n_chunks = len(chunk_terms)

        for q in conv.questions:
            rare = _rare_terms(q.question, df, args.max_df, n_chunks)
            vector = embedder.embed([q.question])[0]
            with PgVectorStore(
                args.dsn, dim=embedder.dim, tenant=tenant, table=args.table
            ) as store:
                hits = store.query_dense(vector, k=args.k)
            retrieved = set()
            for h in hits:
                retrieved |= _terms(h.chunk.text)

            covered = rare & retrieved
            rows.append(
                {
                    "question_id": q.question_id,
                    "type": q.question_type,
                    "answerable": q.question_type != UNANSWERABLE,
                    "n_rare_terms": len(rare),
                    "n_covered": len(covered),
                    # The statistic: what fraction of the question's distinctive vocabulary the
                    # retrieved text actually contains. 1.0 = everything asked about is present.
                    "coverage": round(len(covered) / len(rare), 4) if rare else None,
                    "top1": round(max((h.score for h in hits), default=0.0), 4),
                }
            )
        print(f"  conversation {conv.index}: {len(rows)} questions", flush=True)

    scored = [r for r in rows if r["coverage"] is not None]
    ans = [r["coverage"] for r in scored if r["answerable"]]
    una = [r["coverage"] for r in scored if not r["answerable"]]

    report: dict[str, Any] = {
        "n": len(scored),
        "no_rare_terms": len(rows) - len(scored),
        "coverage_answerable": {
            "n": len(ans), "mean": round(statistics.mean(ans), 4),
            "median": round(statistics.median(ans), 4),
        },
        "coverage_unanswerable": {
            "n": len(una), "mean": round(statistics.mean(una), 4),
            "median": round(statistics.median(una), 4),
        },
        # The comparison that matters: cosine put the unanswerable HIGHER (§9h). Does coverage
        # put them lower? If the gap has the same sign as cosine's, this signal is no better.
        "separation": round(statistics.mean(ans) - statistics.mean(una), 4),
        "rules": {
            f"abstain_if_coverage_below_{t}": {
                "correct_abstain": round(sum(1 for v in una if v < t) / len(una), 4),
                "false_abstain": round(sum(1 for v in ans if v < t) / len(ans), 4),
            }
            for t in (0.2, 0.4, 0.5, 0.6, 0.8)
        },
        "per_question": rows,
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_question"}, indent=1))


if __name__ == "__main__":
    main()
