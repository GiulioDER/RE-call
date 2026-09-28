"""W2 and W4 at the answer level: session coalescing, T-1 content gate, T-1 v2 render.

Pre-registration: kept in the maintainer's private research log (W2 and W4 answer measurement,
2026-09-28, with amendment 1 dropping the speaker-mark arm K-1 had already measured).

Prior work: ``scripts/aml_t1k2_locomo.py`` (T-1's LoCoMo half: one stored collect, arms rendered
offline by the production functions, AML's LoCoMo prompts, the DeepSeek ``Reader``) and K-1 Stage 1
on branch ``claude/k1-census-final`` (AML's LongMemEval-S pipeline and prompt, copied here as
``load_lme_pipeline`` and ``lme_prompt`` so the two measurements ask the same question).

**Collect** builds C9 in process with the served Voyage retrieval and reads each Search from the
service itself, because the API never returns ``SearchItem.render_facts``; the compiler, T-1 and
dated content are off, so every arm can apply them offline in production order: compose (W2), then
``dated_items``, then T-1. **Arms:** B (served: T-1 v1 off the code route), B2 (B again), W4a (T-1
v1 with the content gate), W4b (T-1 v2 with the content gate), W2c (session coalescing, then B).

    python scripts/aml_w2w4_answers.py collect --dataset locomo --locomo locomo10.json --out C-locomo.json.gz
    python scripts/aml_w2w4_answers.py collect --dataset lme --x1-data-dir DIR --x1-draw draw.json --out C-lme.json.gz
    python scripts/aml_w2w4_answers.py mechanism --collected C-locomo.json.gz --out mech-locomo.json
    python scripts/aml_w2w4_answers.py run --dataset locomo --collected C-locomo.json.gz --locomo locomo10.json \\
        --draw locomo-draw.json --aml-repo aml-official --out A-locomo.jsonl
    python scripts/aml_w2w4_answers.py score --answers A-locomo.jsonl A-lme.jsonl --out score.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
import gzip
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import threading
import time
from types import ModuleType
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recall_aml.models import SearchItem  # noqa: E402
from recall_aml.temporal_render import resolve_relative_times  # noqa: E402
from recall_aml.window_compose import compose_items  # noqa: E402
from recall_aml.window_format import dated_items  # noqa: E402

ARMS = ("B", "B2", "W4a", "W4b", "W2c")
C9 = "C9_routed_specialists_grounded_graph_atomic"
SEED = 20260928
BOOTSTRAP_RESAMPLES = 10_000
#: T-1's inserted resolutions in both renders, v1 (``[= 2023-05-07]``) and v2 (``[= the week before
#: 2023-05-08]``), with the space that precedes them.
BRACKET = re.compile(r" \[(?:=|≈|week of|weekend of) [^\]]*\]")
LME_PIPELINE = Path("data") / "longmemeval-s" / "pipeline.py"


# ---------------------------------------------------------------------------------------------
# Rendering (the only part that decides what the reader sees)


def items_of(row: dict[str, Any]) -> list[SearchItem]:
    return [
        SearchItem(
            id=str(item["id"]),
            content=item["content"],
            created_at=datetime.fromisoformat(item["created_at"]) if item.get("created_at") else None,
            source=str(item.get("source") or "collected"),
            session_id=str(item.get("session_id") or ""),
            kind=str(item.get("kind") or "raw"),
            score=float(item.get("score") or 0.0),
            render_facts=item.get("render_facts"),
        )
        for item in row["items"]
    ]


def render(arm: str, row: dict[str, Any]) -> list[SearchItem]:
    """The items the reader sees in ``arm``, by the production functions in production order."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    items = items_of(row)
    if arm == "W2c":
        items = compose_items(items, speakers=False, coalesce=True)
    items = dated_items(items)
    if arm in ("W4a", "W4b"):
        return resolve_relative_times(items, skip_code=True, render="v2" if arm == "W4b" else "v1")
    if row["route"] != "code":
        return resolve_relative_times(items)
    return items


def memories(items: Sequence[SearchItem]) -> list[dict[str, Any]]:
    return [{"content": item.content, "created_at": item.created_at} for item in items if isinstance(item.content, str)]


def strip_resolutions(text: str) -> str:
    return BRACKET.sub("", text)


# ---------------------------------------------------------------------------------------------
# Collect


def check_environment(env: dict[str, str]) -> None:
    wanted = {
        "RECALL_AML_COMPILER": "0",
        "RECALL_AML_RESOLVE_RELATIVE_TIMES": "0",
        "RECALL_AML_SPEAKER_MARKS": "0",
        "RECALL_AML_SESSION_COALESCE": "0",
    }
    wrong = [f"{name}={value}" for name, value in wanted.items() if env.get(name, "").strip() != value]
    if wrong:
        raise SystemExit("collect refused; set " + ", ".join(wrong))


def load_questions(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(adds, questions) for the dataset, from the earlier harnesses' own builders."""
    if args.dataset == "locomo":
        from aml_locomo_route_compare import build_corpus

        adds, questions = build_corpus(json.loads(args.locomo.read_bytes()), f"w2w4-{args.run_id}", None)
        return adds, [
            {"id": q["question_id"], "user_id": q["user_id"], "query": q["query"], "category": q["category"]}
            for q in questions
        ]
    import aml_x1_sources as x1

    draw = json.loads(args.x1_draw.read_text(encoding="utf-8"))
    adds, questions = [], []
    for tenant in x1.tenants_for("longmemeval_s", args.x1_data_dir, draw, f"w2w4-{args.run_id}"):
        adds.extend(add for add, _ in x1.build_adds(tenant, x1.SourceStats()))
        for question in tenant.questions:
            questions.append(
                {
                    "id": str(question["question_id"]),
                    "user_id": tenant.user_id,
                    "query": str(question["search"]["query"]),
                    "category": str(question["category"]),
                }
            )
    return adds, questions


def collect(args: argparse.Namespace) -> dict[str, Any]:
    from starlette.testclient import TestClient

    import recall_aml.__main__ as hosted_main
    from recall_aml.models import SearchRequest

    check_environment(dict(os.environ))
    served_variant = hosted_main.variant
    hosted_main.variant = lambda name: replace(served_variant(name), dated_search_content=False)  # type: ignore[assignment]
    captured: dict[str, Any] = {}
    served_create_app = hosted_main.create_app

    def capture(service: Any, *more: Any, **options: Any) -> Any:
        captured["service"] = service
        return served_create_app(service, *more, **options)

    hosted_main.create_app = capture  # type: ignore[assignment]
    adds, questions = load_questions(args)
    headers = {"Authorization": f"Bearer {os.environ['RECALL_AML_API_KEY']}"}
    failures: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    users = sorted({add["user_id"] for add in adds})
    with TestClient(hosted_main.build_app()) as client:
        version = client.get("/version", headers=headers).json()
        if version.get("variant") != C9:
            raise SystemExit(f"served variant {version.get('variant')!r}")
        service = captured["service"]
        for user in users:
            client.post("/v1/delete", json={"user_id": user}, headers=headers)
        try:
            for position, add in enumerate(adds, start=1):
                for attempt in range(4):
                    response = client.post("/v1/add", json=add, headers=headers)
                    if response.status_code < 500 or attempt == 3:
                        break
                    time.sleep(2**attempt)
                if response.status_code != 200:
                    failures[f"add_{response.status_code}"] += 1
                if position % 200 == 0:
                    print(f"added {position}/{len(adds)}", file=sys.stderr, flush=True)
            for question in questions:
                request = SearchRequest.model_validate(
                    {"query": question["query"], "user_id": question["user_id"], "top_k": 100}
                )
                result = client.portal.call(service.search, request)
                rows.append(
                    {
                        **{k: question[k] for k in ("id", "category")},
                        "route": result.specialist_route,
                        "items": [
                            {
                                "id": item.id,
                                "content": item.content,
                                "created_at": item.created_at.isoformat() if item.created_at else None,
                                "session_id": item.session_id,
                                "kind": item.kind,
                                "score": item.score,
                                "render_facts": item.render_facts,
                            }
                            for item in result.data
                        ],
                    }
                )
        finally:
            for user in users:
                client.post("/v1/delete", json={"user_id": user}, headers=headers)
    return {
        "dataset": args.dataset,
        "variant": version.get("variant"),
        "git_commit": version.get("git_commit") or os.environ.get("RECALL_AML_GIT_COMMIT"),
        "search_content_profile": version.get("search_content_profile"),
        "n_adds": len(adds),
        "failures": dict(failures),
        "seconds": round(time.perf_counter() - started, 1),
        "rows": rows,
    }


# ---------------------------------------------------------------------------------------------
# Mechanism (free)


def mechanism(collected: dict[str, Any]) -> dict[str, Any]:
    """Per arm: questions whose rendered text differs from B; T-1 round trip; missing facts."""
    changed: Counter[str] = Counter()
    round_trip_failures: Counter[str] = Counter()
    raw_without_facts = 0
    raw_total = 0
    for row in collected["rows"]:
        for item in row["items"]:
            if item.get("kind") == "raw" and isinstance(item.get("content"), str):
                raw_total += 1
                raw_without_facts += int(not item.get("render_facts"))
        base = [m["content"] for m in memories(render("B", row))]
        for arm in ARMS[1:]:
            text = [m["content"] for m in memories(render(arm, row))]
            changed[arm] += int(text != base)
            if arm in ("W4a", "W4b") and [strip_resolutions(t) for t in text] != [strip_resolutions(t) for t in base]:
                round_trip_failures[arm] += 1
    return {
        "dataset": collected["dataset"],
        "questions": len(collected["rows"]),
        "routes": dict(Counter(row["route"] for row in collected["rows"])),
        "questions_changed_vs_B": dict(changed),
        "round_trip_failures": dict(round_trip_failures),
        "raw_items": raw_total,
        "raw_items_without_render_facts": raw_without_facts,
    }


# ---------------------------------------------------------------------------------------------
# Run (paid)


def load_lme_pipeline(repo: Path) -> ModuleType:
    """AML's LongMemEval-S pipeline from the pinned checkout (as K-1 Stage 1 loaded it)."""
    from aml_locomo_loss_diagnosis import AML_COMMIT

    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != AML_COMMIT:
        raise SystemExit(f"AML checkout is at {head}, expected {AML_COMMIT}")
    spec = importlib.util.spec_from_file_location("aml_lme_pipeline", repo / LME_PIPELINE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(args: argparse.Namespace) -> None:
    from aml_locomo_loss_diagnosis import load_aml_pipeline, qa_index, render_memories
    from aml_t1k2_locomo import ANSWER_MAX_TOKENS, CREDIT_FLOOR_USD, JUDGE_MAX_TOKENS, Reader

    collected = json.loads(gzip.decompress(args.collected.read_bytes()))
    if collected["dataset"] != args.dataset:
        raise SystemExit("the collected file is for another dataset")
    rows = {row["id"]: row for row in collected["rows"]}
    if args.dataset == "locomo":
        pipeline = load_aml_pipeline(args.aml_repo)
        qas = qa_index(json.loads(args.locomo.read_bytes()))
        ids = [i for i in json.loads(args.draw.read_text(encoding="utf-8"))["ids"] if i in rows]
    else:
        pipeline = load_lme_pipeline(args.aml_repo)
        lme = {str(q["question_id"]): q for q in json.loads(args.lme_data.read_text(encoding="utf-8"))}
        ids = sorted(rows)
    arms = tuple(args.arms.split(","))
    done: set[tuple[str, str]] = set()
    spent = 0.0
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done.add((record["id"], record["arm"]))
            spent += float(record.get("cost") or 0.0)
    reader = Reader(spent, args.max_usd)
    balance = reader.balance()
    print(json.dumps({"questions": len(ids), "done_pairs": len(done), "spent_usd": round(spent, 4), "balance_usd": round(balance, 2)}), flush=True)
    if balance < CREDIT_FLOOR_USD:
        raise SystemExit(f"balance {balance:.2f} below the {CREDIT_FLOOR_USD:.0f} USD floor")
    lock = threading.Lock()
    finished = {"n": 0}

    def prompt(ident: str, arm: str) -> tuple[str, str, str]:
        rendered = render_memories(memories(render(arm, rows[ident])), dated=False)
        if args.dataset == "locomo":
            qa = qas[ident]
            speaker_a, speaker_b = qa["speakers"]
            return (
                pipeline.render_answer_prompt({
                    "question": qa["question"], "speaker_1_name": f"{speaker_a} and {speaker_b}",
                    "speaker_1_memories": rendered, "speaker_2_name": "(none)",
                    "speaker_2_memories": "(all memories are listed above)",
                }),
                str(qa["question"]),
                str(qa["answer"]),
            )
        question = lme[ident]
        return (
            pipeline.render_answer_prompt({
                "question": question["question"], "speaker_1_name": "user", "speaker_1_memories": rendered,
                "speaker_2_name": "(none)", "speaker_2_memories": "(all memories are listed above)",
            }),
            str(question["question"]),
            str(question["answer"]),
        )

    def work(index_ident: tuple[int, str]) -> None:
        index, ident = index_ident
        order = arms[index % len(arms):] + arms[: index % len(arms)]
        for arm in order:
            if (ident, arm) in done or reader.stopped.is_set():
                continue
            text, question, gold = prompt(ident, arm)
            try:
                generated, answer_usage = reader.complete(text, ANSWER_MAX_TOKENS)
                verdict, judge_usage = reader.complete(
                    pipeline.render_accuracy_prompt({"question": question, "gold_answer": gold}, generated),
                    JUDGE_MAX_TOKENS,
                )
            except RuntimeError:
                return
            try:
                label = pipeline.parse_judge_label(verdict)
            except (ValueError, json.JSONDecodeError):
                label = "UNPARSED"
            record = {
                "dataset": args.dataset, "id": ident, "arm": arm, "category": rows[ident]["category"],
                "route": rows[ident]["route"], "generated_answer": generated, "label": label,
                "judge_response": verdict,
                "cost": float(answer_usage.get("cost") or 0.0) + float(judge_usage.get("cost") or 0.0),
            }
            with lock, args.out.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        with lock:
            finished["n"] += 1
            if finished["n"] % 50 == 0:
                current = reader.balance()
                print(json.dumps({"questions_done": finished["n"], "spent_usd": round(reader.spent, 4), "balance_usd": round(current, 2)}), flush=True)
                if current < CREDIT_FLOOR_USD:
                    reader.stopped.set()
                    print(f"STOP: balance {current:.2f} below the floor", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, enumerate(ids)))
    print(json.dumps({"spent_usd": round(reader.spent, 4), "stopped": reader.stopped.is_set()}), flush=True)


# ---------------------------------------------------------------------------------------------
# Score


def paired(control: Sequence[bool], treatment: Sequence[bool]) -> dict[str, Any]:
    """Accuracy difference in points with a 95% percentile bootstrap, resampling questions."""
    if len(control) != len(treatment) or not control:
        return {"n": 0}
    deltas = [int(t) - int(c) for c, t in zip(control, treatment, strict=True)]
    n = len(deltas)
    rng = random.Random(SEED)
    means = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP_RESAMPLES))
    return {
        "n": n,
        "diff_points": round(100 * sum(deltas) / n, 2),
        "ci95_points": [round(100 * means[int(0.025 * BOOTSTRAP_RESAMPLES)], 2), round(100 * means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1], 2)],
        "control_accuracy": round(100 * sum(control) / n, 2),
        "arm_accuracy": round(100 * sum(treatment) / n, 2),
        "wins": sum(1 for d in deltas if d > 0),
        "losses": sum(1 for d in deltas if d < 0),
    }


#: The pre-registered contrasts: (arm, control, dataset, subset). A subset of None is every question.
CONTRASTS = (
    ("W4a", "B", "locomo", "2"), ("W4a", "B", "locomo", None), ("W4a", "B", "lme", "temporal-reasoning"), ("W4a", "B", "lme", None),
    ("W4b", "W4a", "locomo", "2"), ("W4b", "W4a", "locomo", None), ("W4b", "W4a", "lme", "temporal-reasoning"), ("W4b", "W4a", "lme", None),
    ("W2c", "B", "locomo", None), ("W2c", "B", "locomo", "1"), ("W2c", "B", "lme", None),
    ("B2", "B", "locomo", None), ("B2", "B", "locomo", "2"), ("B2", "B", "locomo", "1"),
    ("B2", "B", "lme", None), ("B2", "B", "lme", "temporal-reasoning"),
)
#: Primary subsets per option (the rule's first and third conditions).
PRIMARY = {"W4a": [("locomo", "2")], "W4b": [("locomo", "2")], "W2c": [("locomo", None), ("lme", None)]}
GUARD_POINTS = -1.5


def score(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    category: dict[tuple[str, str], str] = {}
    unparsed: Counter[str] = Counter()
    for record in records:
        key = (record["dataset"], record["id"])
        labels[key][record["arm"]] = record["label"] == "CORRECT"
        category[key] = str(record["category"])
        unparsed[record["arm"]] += int(record["label"] == "UNPARSED")
    out: dict[str, Any] = {"unparsed_per_arm": dict(unparsed), "contrasts": {}}
    for arm, control, dataset, subset in CONTRASTS:
        keys = sorted(k for k in labels if k[0] == dataset and arm in labels[k] and control in labels[k]
                      and (subset is None or category[k] == subset))
        out["contrasts"][f"{arm}-vs-{control}:{dataset}:{subset or 'all'}"] = paired(
            [labels[k][control] for k in keys], [labels[k][arm] for k in keys]
        )
    decisions = {}
    for arm, primaries in PRIMARY.items():
        results = out["contrasts"]
        control = "W4a" if arm == "W4b" else "B"
        primary_ok = all(results.get(f"{arm}-vs-{control}:{d}:{s or 'all'}", {}).get("diff_points", -1) > 0 for d, s in primaries)
        guard_ok = all(results.get(f"{arm}-vs-{control}:{d}:all", {}).get("ci95_points", [-99])[0] > GUARD_POINTS for d in ("locomo", "lme")
                       if results.get(f"{arm}-vs-{control}:{d}:all", {}).get("n"))
        floor_ok = all(
            results.get(f"{arm}-vs-{control}:{d}:{s or 'all'}", {}).get("diff_points", 0)
            > abs(results.get(f"B2-vs-B:{d}:{s or 'all'}", {}).get("diff_points", 0))
            for d, s in primaries
        )
        decisions[arm] = {"primary_positive": primary_ok, "guard_holds": guard_ok, "beats_noise_floor": floor_ok,
                          "recommended": primary_ok and guard_ok and floor_ok}
    out["decisions"] = decisions
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    col = sub.add_parser("collect")
    col.add_argument("--dataset", choices=("locomo", "lme"), required=True)
    col.add_argument("--locomo", type=Path)
    col.add_argument("--x1-data-dir", type=Path)
    col.add_argument("--x1-draw", type=Path)
    col.add_argument("--run-id", default=datetime.now().strftime("%Y%m%dT%H%M%S"))
    col.add_argument("--out", type=Path, required=True)
    mech = sub.add_parser("mechanism")
    mech.add_argument("--collected", type=Path, required=True)
    mech.add_argument("--out", type=Path, required=True)
    ru = sub.add_parser("run")
    ru.add_argument("--dataset", choices=("locomo", "lme"), required=True)
    ru.add_argument("--collected", type=Path, required=True)
    ru.add_argument("--aml-repo", type=Path, required=True)
    ru.add_argument("--locomo", type=Path)
    ru.add_argument("--draw", type=Path)
    ru.add_argument("--lme-data", type=Path)
    ru.add_argument("--arms", default=",".join(ARMS))
    ru.add_argument("--workers", type=int, default=2)
    ru.add_argument("--max-usd", type=float, default=14.0)
    ru.add_argument("--out", type=Path, required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--answers", type=Path, nargs="+", required=True)
    sc.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "collect":
        result = collect(args)
        args.out.write_bytes(gzip.compress(json.dumps(result).encode("utf-8")))
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    elif args.mode == "mechanism":
        result = mechanism(json.loads(gzip.decompress(args.collected.read_bytes())))
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    elif args.mode == "run":
        run(args)
    else:
        records = [json.loads(line) for path in args.answers for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        result = score(records)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
