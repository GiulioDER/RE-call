"""Where in the ranking does the answer-bearing chunk actually sit?

::

    # ~$0.05 — retrieval plus one embedding per gold nugget. No answerer, no judge.
    python -m benchmarks.beam.rank_probe --data <shards> --conversations 0-9 \
        --table bench_beam_probe --embedder router:openai/text-embedding-3-small

Why this question decides the next experiment
---------------------------------------------
The BEAM arm sends the answerer ~200 retrieved chunks, measured at ~30,000 tokens per question
against Mem0's ~6,700 — 4.5x the context, for a lower score. Two very different explanations fit
that fact, and they imply opposite fixes:

  1. **Dilution.** The answer IS retrieved, high in the ranking, and the other ~190 chunks bury
     it. Then the fix is ours and cheap: send fewer, better chunks. Nothing about Mem0's
     LLM-extraction architecture is required to close the gap.
  2. **Density.** The answer is genuinely spread across many chunks, or ranked low, and 200 is
     what it takes to contain it. Then the context cost is intrinsic to retrieving raw turns
     rather than distilled facts, and only changing WHAT is stored can help.

This measures which. For every question it finds the rank of the highest-ranked retrieved chunk
that matches the gold rubric, and reports the distribution. If the mass sits in the top 20, the
answer is (1) and a k-sweep is the experiment to run. If it is spread to rank 150, it is (2).

Matching is deliberately crude and TWO-WAY: a normalised substring test for the literal gold
value, and a cosine test against the embedded nugget. The substring test is precise but misses
paraphrase; the cosine test catches paraphrase but can be fooled by topical similarity. Reporting
both, rather than picking one, keeps a single matcher's blind spot from becoming the finding —
the two disagreeing IS informative and is printed as such.

It scores nothing and spends no generator or judge tokens, so it cannot be confused with a cell.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

from benchmarks.beam.dataset import iter_conversations
from benchmarks.beam.systems import BEAM_TABLE, BeamRecallSystem
from recall.eval.locomo import DEFAULT_DSN

#: Cosine at or above which an embedded chunk counts as carrying the nugget. Deliberately high:
#: this probe is looking for the chunk that ANSWERS, not one that is merely on-topic, and BEAM's
#: whole difficulty (§9h) is that on-topic and answering come apart.
NUGGET_MATCH_COSINE = 0.75

#: Ranks reported as a cumulative curve. These are the candidate k values a sweep would test.
RANK_BUCKETS = (1, 3, 5, 10, 20, 50, 100, 200)


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower())


def _gold_terms(nugget: str) -> list[str]:
    """Distinctive literals from a rubric nugget: numbers and quoted/technical tokens.

    Rubrics read "LLM response should state: 20 minutes", so the discriminating content is the
    VALUE, not the boilerplate. Matching on the whole nugget string would never hit; matching on
    its numbers and rare words does.
    """
    body = nugget.split(":", 1)[-1]
    terms = re.findall(r"\d+(?:\.\d+)?\s*[a-z%]*|\b[A-Z][A-Za-z0-9.]{2,}\b", body)
    return [t.strip() for t in terms if t.strip()]


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return num / (na * nb) if na and nb else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--chat-size", default="1M")
    parser.add_argument("--conversations", default="0-9")
    parser.add_argument("--embedder", default="router:openai/text-embedding-3-small")
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--table", default=BEAM_TABLE)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--reindex", action="store_true",
                        help="Ingest each conversation first. Omit when --table is already built.")
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
    embedder = system._embedder  # noqa: SLF001 - probe, deliberately reaching past the façade
    per_question: list[dict[str, Any]] = []

    for conv in iter_conversations(args.data, args.chat_size, sorted(set(indices))):
        if args.reindex:
            system.ingest(conv)
        else:
            system._tenant = f"beam-{conv.chat_size}-{conv.index}".lower()  # noqa: SLF001
            system._dates = {}  # noqa: SLF001
        for q in conv.questions:
            if not q.rubric or q.question_type == "abstention":
                continue  # nothing to locate for an unanswerable question
            memories = system.retrieve(q.question)
            texts = [m["memory"] for m in memories]
            if not texts:
                per_question.append(
                    {"question_id": q.question_id, "type": q.question_type,
                     "retrieved": 0, "rank_substring": None, "rank_cosine": None}
                )
                continue

            terms = [t for n in q.rubric for t in _gold_terms(n)]
            rank_sub = None
            if terms:
                haystacks = [_normalise(t) for t in texts]
                for i, hay in enumerate(haystacks, 1):
                    if any(_normalise(t) in hay for t in terms):
                        rank_sub = i
                        break

            nugget_vec = embedder.embed([q.rubric[0]])[0]
            chunk_vecs = embedder.embed(texts)
            sims = [_cosine(nugget_vec, v) for v in chunk_vecs]
            # Rank IN RETRIEVAL ORDER of the best-matching chunk — that is the number which
            # decides k. Its rank in a re-sorted-by-similarity list would answer a different
            # question (how good is the matcher), not this one (how deep must k go).
            best = max(range(len(sims)), key=lambda j: sims[j])
            rank_cos = best + 1 if sims[best] >= NUGGET_MATCH_COSINE else None

            per_question.append(
                {
                    "question_id": q.question_id,
                    "type": q.question_type,
                    "retrieved": len(texts),
                    "gold_terms": terms[:5],
                    "rank_substring": rank_sub,
                    "rank_cosine": rank_cos,
                    "best_cosine": round(sims[best], 4),
                }
            )
        print(f"  conversation {conv.index}: probed", flush=True)

    def _curve(key: str) -> dict[str, Any]:
        found = [r[key] for r in per_question if r[key] is not None]
        return {
            "n_located": len(found),
            "n_total": len(per_question),
            "median_rank": statistics.median(found) if found else None,
            "cumulative_within": {
                f"top_{b}": round(sum(1 for v in found if v <= b) / len(per_question), 4)
                for b in RANK_BUCKETS
            },
        }

    report = {
        "conversations": args.conversations,
        "k": args.k,
        "substring_match": _curve("rank_substring"),
        "cosine_match": _curve("rank_cosine"),
        "per_question": per_question,
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_question"}, indent=1))


if __name__ == "__main__":
    main()
