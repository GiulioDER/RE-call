"""Token counter and cost projector for Bench'd harness runs.

Two subcommands, no harness import, stdlib plus optional tiktoken:

    python benchmarks/benchd/count_tokens.py manifest RUNS/run_x/manifest.signed.json \
        [--project 500] [--price-in 0.15] [--price-out 0.60]

        Recomputes answerer and judge token usage from the run's traces by re-rendering the
        harness's exact prompt templates (copied verbatim below from
        benchd_harness/scoring/llm_judge.py, clone of 2026-08-23). Counts with tiktoken's
        o200k_base when installed (the gpt-4o-mini tokenizer), otherwise falls back to the
        harness's own estimator (words * 4 // 3) and says so. Prints totals, per-question means,
        the recall-token distribution that feeds Bench'd's BMI efficiency term, and a linear
        projection to --project questions.

    python benchmarks/benchd/count_tokens.py openrouter

        Prints the OpenRouter key's cumulative usage counter (GET /api/v1/auth/key with
        OPENROUTER_API_KEY). Snapshot before and after a run; the difference is the measured
        spend, which is the number a results artifact should carry. An estimate never is.

Prices default to gpt-4o-mini list prices in USD per million tokens as of 2026-08-23
(re-check at openrouter.ai/openai/gpt-4o-mini before trusting a projection).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

ANSWERER_PROMPT = """You are answering a question using ONLY the retrieved memories below.

Retrieved memories:
{retrieved_memories}

Question: {question}

Instructions:
- Answer the question using ONLY information from the retrieved memories above.
- If the retrieved memories do not contain enough information to answer, say "Insufficient information in memory."
- Be concise and direct. Give the specific answer, not an explanation.
- Do not make up information not present in the retrieved memories."""

JUDGE_PROMPT = """You are a judge evaluating whether an answer is correct.

Question: {question}
Expected Answer: {expected_answer}
Given Answer: {given_answer}

Instructions:
- Determine if the Given Answer is semantically equivalent to the Expected Answer.
- The Given Answer does not need to match word-for-word. It needs to contain the correct information.
- If the Given Answer contains the correct fact even with extra context, judge it as CORRECT.
- If the Given Answer says "insufficient information" or equivalent, judge it as INCORRECT.
- If the Given Answer is partially correct but missing key information, judge it as INCORRECT.

Respond with EXACTLY one line in this format:
JUDGMENT: CORRECT
or
JUDGMENT: INCORRECT

Then on the next line, briefly explain your reasoning in one sentence."""

#: Per-message overhead the chat API adds around a single user message; small and constant.
CHAT_OVERHEAD_TOKENS = 7


def _make_counter():
    try:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        return (lambda s: len(enc.encode(s))), "tiktoken o200k_base"
    except Exception:
        return (lambda s: len(str(s).split()) * 4 // 3), "harness estimator (words * 4 // 3)"


def cmd_manifest(args: argparse.Namespace) -> int:
    count, counter_name = _make_counter()
    data = json.loads(open(args.path, encoding="utf-8").read())
    m = data.get("manifest", data)
    traces = m.get("traces", [])
    if not traces:
        print("no traces in manifest", file=sys.stderr)
        return 1

    ans_in = ans_out = jud_in = jud_out = 0
    recall_tokens = []
    for t in traces:
        recall_tokens.append(count(t.get("raw_recall", "")))
        ans_in += CHAT_OVERHEAD_TOKENS + count(
            ANSWERER_PROMPT.format(
                retrieved_memories=t.get("raw_recall", ""),
                question=t.get("query", ""),
            )
        )
        ans_out += count(t.get("generated_answer", ""))
        jud_in += CHAT_OVERHEAD_TOKENS + count(
            JUDGE_PROMPT.format(
                question=t.get("query", ""),
                expected_answer=t.get("expected_answer", ""),
                given_answer=t.get("generated_answer", ""),
            )
        )
        # The stored reasoning had the JUDGMENT line stripped; add it back (~4 tokens).
        jud_out += count(t.get("judge_reasoning", "")) + 4

    n = len(traces)
    total_in, total_out = ans_in + jud_in, ans_out + jud_out
    cost = total_in / 1e6 * args.price_in + total_out / 1e6 * args.price_out
    recall_tokens.sort()

    print(f"traces: {n}   counter: {counter_name}")
    print(f"answerer  in {ans_in:>10,}   out {ans_out:>8,}")
    print(f"judge     in {jud_in:>10,}   out {jud_out:>8,}")
    print(f"total     in {total_in:>10,}   out {total_out:>8,}")
    print(f"per question: in {total_in / n:,.0f}   out {total_out / n:,.0f}")
    print(
        "recall tokens per question (BMI efficiency input): "
        f"mean {sum(recall_tokens) / n:,.0f}   median {recall_tokens[n // 2]:,}   "
        f"p95 {recall_tokens[int(n * 0.95) - 1]:,}"
    )
    print(f"estimated cost at ${args.price_in}/M in, ${args.price_out}/M out: ${cost:.4f}")
    if args.project and args.project != n:
        scale = args.project / n
        print(
            f"projected to {args.project} questions: in {total_in * scale:,.0f}   "
            f"out {total_out * scale:,.0f}   cost ${cost * scale:.2f}"
        )
    return 0


def cmd_openrouter(_args: argparse.Namespace) -> int:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/auth/key",
        headers={"Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read())
    info = body.get("data", body)
    print(json.dumps({k: info.get(k) for k in ("label", "usage", "limit", "limit_remaining")}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pm = sub.add_parser("manifest", help="recount tokens from a run manifest")
    pm.add_argument("path")
    pm.add_argument("--project", type=int, default=None, help="project totals to N questions")
    pm.add_argument("--price-in", type=float, default=0.15, help="USD per million input tokens")
    pm.add_argument("--price-out", type=float, default=0.60, help="USD per million output tokens")
    pm.set_defaults(func=cmd_manifest)

    po = sub.add_parser("openrouter", help="print the OpenRouter key's usage counter")
    po.set_defaults(func=cmd_openrouter)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
