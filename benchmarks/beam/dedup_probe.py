"""Does collapsing near-duplicate chunks let the CURRENT fact outvote the superseded one?

::

    # $0 of generator/judge spend — embeddings and cosines only.
    python -m benchmarks.beam.dedup_probe --data <shards> --conversations 0-14 \
        --table bench_beam_probe --embedder router:openai/text-embedding-3-small

The mechanism this targets, measured
------------------------------------
On the Redis-TTL question the retriever did everything right: the CURRENT value ("20 minutes")
came back at rank 1. The answerer still said "15 minutes" — because the superseded value appears
**24 times** in that conversation against the current value's **7**, and at k=200 enough of those
repetitions reach the prompt to outvote rank 1. Confirmed by the k sweep: at k=5, where the stale
copies fall outside the window, the same system answers "20 minutes" and scores 1.00.

So the failure is not retrieval and not ordering (the ISO date fix was a null result, p=0.727).
It is REPETITION: a stale fact restated many times beats a current fact stated once.

RE-call's supersession is the right instrument and cannot fire here. It is annotation-driven —
`supersedes:` frontmatter, plus `recall lint`'s regex over prose markers ("superseded by",
"replaces"). BEAM's users never write those words; they just say a new number. `recall fix`'s own
docstring reports proposing ZERO edges on a real 792-memo corpus. The relation is semantic, not
declared, so no amount of parsing prose will find it.

What this probe tests instead
-----------------------------
Collapse near-identical chunks and keep only the NEWEST of each cluster, then ask whether the
surviving set still contains the stale value. No LLM: clustering is cosine over embeddings we
already compute, and "newest" is the ISO date already attached to every chunk.

It reports, per question, how many chunks survive and whether the stale/current values are still
present — NOT a score. Turning a promising collapse into a benchmark cell requires the paid arm,
and conflating the two is how a mechanism becomes an overclaim (see §9h/§9i, where exactly that
went wrong at n=2).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

from benchmarks.beam.dataset import iter_conversations
from benchmarks.beam.systems import (
    BEAM_TABLE,
    BeamRecallSystem,
    rebuild_dates,
    require_indexed,
)
from recall.eval.locomo import DEFAULT_DSN
from recall.store import PgVectorStore

#: Cosine at or above which two chunks are treated as restatements of one another. High on
#: purpose: this must collapse "the TTL is 15 minutes" against "I set the TTL to 15 minutes",
#: and must NOT collapse two genuinely different facts that share vocabulary. Swept below so the
#: choice is reported as a curve rather than asserted as a constant.
DEDUP_COSINES = (0.98, 0.95, 0.92, 0.90)


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return num / (na * nb) if na and nb else 0.0


def collapse(
    chunks: list[dict[str, str]], vectors: list[list[float]], threshold: float
) -> list[int]:
    """Indices surviving a greedy newest-wins collapse, in original retrieval order.

    Greedy and rank-ordered: walk the ranking, and drop a chunk when it is within `threshold` of
    a chunk already kept. When the dropped one is NEWER than the keeper, the keeper's slot takes
    the newer chunk instead — the point is to keep the current statement of a fact, not the
    highest-ranked statement of it.

    Undated chunks never displace a dated keeper: an empty date is unknown, not old, and letting
    it win would silently promote whatever the harness failed to stamp.
    """
    kept: list[int] = []
    for i in range(len(chunks)):
        dup_of = None
        for slot, j in enumerate(kept):
            if _cosine(vectors[i], vectors[j]) >= threshold:
                dup_of = slot
                break
        if dup_of is None:
            kept.append(i)
            continue
        j = kept[dup_of]
        di, dj = chunks[i].get("created_at", ""), chunks[j].get("created_at", "")
        if di and di > dj:
            kept[dup_of] = i
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--chat-size", default="1M")
    parser.add_argument("--conversations", default="0-14")
    parser.add_argument("--embedder", default="router:openai/text-embedding-3-small")
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--table", default=BEAM_TABLE)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--types", default="knowledge_update,temporal_reasoning")
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    indices: list[int] = []
    for part in args.conversations.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.extend(range(int(lo), int(hi) + 1))
        elif part.strip():
            indices.append(int(part))
    wanted = {t.strip() for t in args.types.split(",")}

    system = BeamRecallSystem(args.dsn, embedder_name=args.embedder, k=args.k, table=args.table)
    embedder = system._embedder  # noqa: SLF001 - probe
    rows: list[dict[str, Any]] = []

    for conv in iter_conversations(args.data, args.chat_size, sorted(set(indices))):
        if args.reindex:
            system.ingest(conv)
        else:
            tenant = f"beam-{conv.chat_size}-{conv.index}".lower()
            system._tenant = tenant  # noqa: SLF001
            # Rebuilt from the store, NOT blanked. This used to be `system._dates = {}`, which
            # made `retrieve` stamp `created_at: ""` on every memory — and `collapse`'s
            # newest-wins comparison `if di and di > dj` cannot fire on empty strings. In its own
            # documented invocation (which does not pass --reindex) this probe was therefore
            # measuring keep-the-highest-RANKED and publishing it as a newest-wins curve.
            with PgVectorStore(
                args.dsn, dim=embedder.dim, tenant=tenant, table=args.table
            ) as store:
                require_indexed(store, tenant=tenant, table=args.table, what="dedup_probe")
                system._dates = rebuild_dates(store)  # noqa: SLF001
            if not any(system._dates.values()):  # noqa: SLF001
                raise SystemExit(
                    f"dedup_probe: no chunk in tenant {tenant!r} carries a parseable date, so the "
                    f"newest-wins collapse this probe exists to measure cannot fire. Re-run with "
                    f"--reindex rather than publish a rank-wins curve under a newest-wins label."
                )
        for q in conv.questions:
            if q.question_type not in wanted or not q.rubric:
                continue
            mems = system.retrieve(q.question)
            if not mems:
                continue
            vecs = embedder.embed([m["memory"] for m in mems])
            # The gold value, as literals — same extraction the rank probe uses.
            gold = re.findall(r"\d+(?:\.\d+)?\s*[a-z%]*", q.rubric[0].split(":", 1)[-1].lower())
            row: dict[str, Any] = {
                "question_id": q.question_id,
                "type": q.question_type,
                "retrieved": len(mems),
                "gold_terms": gold[:3],
                "by_threshold": {},
            }
            for thr in DEDUP_COSINES:
                keep = collapse(mems, vecs, thr)
                texts = " ".join(mems[i]["memory"].lower() for i in keep)
                row["by_threshold"][str(thr)] = {
                    "survivors": len(keep),
                    "gold_present": bool(gold) and any(g.strip() in texts for g in gold),
                }
            rows.append(row)
        print(f"  conversation {conv.index}: {len(rows)} questions so far", flush=True)

    report: dict[str, Any] = {"k": args.k, "types": sorted(wanted), "n": len(rows), "thresholds": {}}
    for thr in DEDUP_COSINES:
        s = [r["by_threshold"][str(thr)] for r in rows]
        report["thresholds"][str(thr)] = {
            "median_survivors": statistics.median(r["survivors"] for r in s),
            "mean_survivors": round(statistics.mean(r["survivors"] for r in s), 1),
            "gold_still_present": round(sum(1 for r in s if r["gold_present"]) / len(s), 4),
        }
    report["baseline_gold_present"] = round(
        sum(1 for r in rows if r["by_threshold"][str(DEDUP_COSINES[0])]["gold_present"]) / len(rows), 4
    )
    report["per_question"] = rows
    if args.out:
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_question"}, indent=1))


if __name__ == "__main__":
    main()
