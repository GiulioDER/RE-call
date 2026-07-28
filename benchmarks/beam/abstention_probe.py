"""Does entailment COUNT separate BEAM's answerable questions from its unanswerable ones?

::

    # $0 — retrieval + a local cross-encoder. No LLM call is made.
    python -m benchmarks.beam.abstention_probe --data <shards> --conversations 0-14 \
        --embedder router:openai/text-embedding-3-small --table bench_beam_probe

What this tests and why it is not the benchmark
-----------------------------------------------
§9f established that a cosine threshold CANNOT gate abstention on BEAM: its unanswerable questions
score HIGHER than its answerable ones (median 0.676 vs 0.641), so every threshold abstains on the
wrong class first. §9g found, on a single conversation, that the count of ENTAILED hits recovers
the ordering that similarity inverts — but on n=2 unanswerable questions, which is a direction,
not a result.

This probe measures the same quantity across every abstention question in a conversation range,
which is 2 per conversation. It reports the separation and what each candidate abstention policy
would cost, INCLUDING the policy RE-call currently ships (`abstain iff zero hits entail`).

It deliberately does NOT score anything. No answerer, no judge, no rubric: this measures the
retrieval-side signal available to an abstention decision, not the end-to-end cell. Turning a
promising separation here into a benchmark claim requires the paid arm, and conflating the two is
how a mechanism becomes an overclaim.

The table is a parameter and defaults to a probe-only name, because the accuracy arms clear and
re-index their tenants as they walk the corpus: sharing a table with a run in flight would let
each delete the other's chunks mid-question.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from benchmarks.beam.dataset import iter_conversations
from benchmarks.beam.systems import BeamRecallSystem
from recall.eval.locomo import DEFAULT_DSN

UNANSWERABLE_TYPE = "abstention"

#: Candidate abstention policies, as "answer only if at least N of the judged hits entail".
#: N = 1 is what `apply_entailment` ships today (abstain iff zero entail).
POLICIES = (1, 2, 3, 4, 5)


def _parse_indices(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--chat-size", default="1M")
    parser.add_argument("--conversations", default="0-14")
    parser.add_argument("--embedder", default="router:openai/text-embedding-3-small")
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--table", default="bench_beam_probe")
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument(
        "--judge-top",
        type=int,
        default=10,
        help="How many top-ranked hits the entailment judge sees per question.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from recall.entailment import QnliEntailmentJudge

    system = BeamRecallSystem(
        args.dsn, embedder_name=args.embedder, k=args.k, table=args.table
    )
    judge = QnliEntailmentJudge()
    per_question: list[dict[str, Any]] = []

    for conv in iter_conversations(args.data, args.chat_size, _parse_indices(args.conversations)):
        system.ingest(conv)
        for q in conv.questions:
            memories = system.retrieve(q.question)
            texts = [m["memory"] for m in memories[: args.judge_top]]
            entailed = sum(judge.judge(q.question, texts)) if texts else 0
            per_question.append(
                {
                    "question_id": q.question_id,
                    "type": q.question_type,
                    "unanswerable": q.question_type == UNANSWERABLE_TYPE,
                    "retrieved": len(memories),
                    "judged": len(texts),
                    "entailed": entailed,
                }
            )
        done = sum(1 for r in per_question if r["question_id"].startswith(f"{args.chat_size}_{conv.index}_"))
        print(f"  conversation {conv.index}: {done} questions probed", flush=True)

    un = [r["entailed"] for r in per_question if r["unanswerable"]]
    an = [r["entailed"] for r in per_question if not r["unanswerable"]]
    if not un or not an:
        raise SystemExit("need both classes present to measure separation")

    policies = {}
    for n in POLICIES:
        policies[f"answer_if_entailed_ge_{n}"] = {
            # An unanswerable question is handled CORRECTLY when the policy abstains on it.
            "correct_abstain": round(sum(1 for v in un if v < n) / len(un), 4),
            "false_abstain": round(sum(1 for v in an if v < n) / len(an), 4),
            "ships_today": n == 1,
        }

    report = {
        "conversations": args.conversations,
        "judge_top": args.judge_top,
        "n_unanswerable": len(un),
        "n_answerable": len(an),
        "mean_entailed_unanswerable": round(statistics.mean(un), 3),
        "mean_entailed_answerable": round(statistics.mean(an), 3),
        "median_entailed_unanswerable": statistics.median(un),
        "median_entailed_answerable": statistics.median(an),
        "policies": policies,
        "per_question": per_question,
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_question"}, indent=1))


if __name__ == "__main__":
    main()
