"""Score a completed run's answers with Ragas, as a separate cost surface.

⚠️ **This script does NOT run on the repository's own Python.** Ragas pulls in HuggingFace
`datasets`, which imports `pyarrow.parquet`, whose extension DLL is blocked on this machine by a
Windows Application Control policy. Only pyarrow 25 publishes a cp314 wheel and that is the blocked
one, so on Python 3.14 there is no version of the dependency that loads. Nothing here tries to work
around the policy; the scorer simply runs on a Python where a permitted wheel exists.

Build the interpreter once:

    py -3.12 -m venv .ragas-venv
    .ragas-venv/Scripts/python -m pip install \
        "git+https://github.com/vibrantlabsai/ragas@main" \
        "pyarrow==18.1.0" "langchain-community==0.2.19" "langchain-core<0.3" \
        "langchain<0.3" "langchain-openai<0.2"

then run this with it:

    .ragas-venv/Scripts/python scripts/agent_ab_score_ragas.py --run-id agent-ab-traps-002

The langchain pins are not cosmetic: the vibrantlabsai copy of Ragas imports
`langchain_community.chat_models.vertexai`, which current langchain no longer ships, so an
unpinned install imports and then fails at the first metric.

**What is scored, and what is deliberately not.** Ragas judges the ANSWER: is it factually right
against the written reference. It does not judge whether the session avoided the hazard, because
`benchmarks/agent_ab/traps.py` already decides that deterministically from the transcript, and a
deterministic checker beats an LLM on a question with a crisp answer. Reaching for `AspectCritic`
here would replace a measurement with an opinion.

Judge cost is recorded separately from the system under test. They are different budgets and
conflating them would make the memory layer look more expensive than it is.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JUDGE = "openai/gpt-4.1-mini"


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--judge",
        default=os.environ.get("AGENT_AB_JUDGE", DEFAULT_JUDGE),
        help="deliberately a DIFFERENT model family from the agent under test, so a model is "
        "not grading its own output",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    artifacts = REPO_ROOT / "benchmarks" / "artifacts" / "agent_ab" / args.run_id
    records_path = artifacts / "records.jsonl"
    if not records_path.is_file():
        raise SystemExit(f"no records at {records_path}; run the comparison first")

    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerCorrectness, FactualCorrectness

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is required to reach the judge")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    llm = llm_factory(args.judge, provider="openai", client=client)

    # weights=[1, 0] drops AnswerCorrectness's embedding half. The semantic-similarity term would
    # need a second provider and would reward an answer for sounding like the reference, which is
    # not the question being asked.
    correctness = AnswerCorrectness(llm=llm, weights=[1.0, 0.0])
    factual = FactualCorrectness(llm=llm)

    records = [r for r in load_records(records_path) if r.get("reference") and r.get("response")]
    print(f"scoring {len(records)} records with {args.judge}")

    semaphore = asyncio.Semaphore(args.concurrency)
    scored: list[dict[str, Any]] = []

    async def score_one(record: dict[str, Any]) -> None:
        async with semaphore:
            entry: dict[str, Any] = {
                "task_id": record["task_id"],
                "base_task_id": record.get("metadata", {}).get("base_task_id")
                or record["task_id"].split("#")[0],
                "variant": record["variant"],
                # None, never 0.0: a judge that failed and a judge that scored zero are different
                # facts, and the second is a finding about the agent.
                "answer_correctness": None,
                "factual_correctness": None,
                "error": None,
            }
            try:
                result = await correctness.ascore(
                    user_input=record.get("user_input", ""),
                    response=record["response"],
                    reference=record["reference"],
                )
                entry["answer_correctness"] = float(result.value)
                fact = await factual.ascore(
                    response=record["response"], reference=record["reference"]
                )
                entry["factual_correctness"] = float(fact.value)
            except Exception as error:  # noqa: BLE001 - one bad sample must not lose the run
                entry["error"] = f"{type(error).__name__}: {error}"[:300]
            scored.append(entry)
            done = len(scored)
            if done % 10 == 0 or done == len(records):
                print(f"  {done}/{len(records)}")

    await asyncio.gather(*(score_one(r) for r in records))

    failures = [s for s in scored if s["error"]]
    payload = {
        "run_id": args.run_id,
        "judge_model": args.judge,
        "judge_provider": "openrouter",
        "ragas_version": __import__("ragas").__version__,
        "python": sys.version.split()[0],
        "metrics": {
            "answer_correctness": "weights=[1.0, 0.0], factual half only",
            "factual_correctness": "default",
        },
        "scored": len(scored) - len(failures),
        "failed": len(failures),
        "scores": sorted(scored, key=lambda s: (s["task_id"], s["variant"])),
    }
    out = artifacts / "ragas-scores.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n{payload['scored']} scored, {payload['failed']} failed -> {out}")
    if failures:
        print("failures are recorded in the artifact rather than dropped:")
        for failure in failures[:5]:
            print(f"  {failure['task_id']} {failure['variant']}: {failure['error'][:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
