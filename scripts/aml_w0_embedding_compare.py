"""W0: C9's retrieval with its Voyage spaces against Alibaba `text-embedding-v4`, three datasets.

Why: the AML organisers recommend `text-embedding-v4` for every second Full run (their reply of
2026-09-26). Moving C9 onto it replaces both of its text embedding spaces, so every retrieval
number moves, and this measures by how much before any Full depends on it. The pre-registration
is kept in the maintainer's private research log (W0, 2026-09-28) and is committed before any
paid call.

Prior work: `scripts/aml_locomo_route_compare.py` (LoCoMo corpus and turn scoring),
`scripts/aml_x1_sources.py` (the LongMemEval-S draw of 120 questions) and
`scripts/aml_c9_coding_window_check.py` (the frozen Agent Memory Bench coding corpus and its
session scoring). This file imports their loaders and scorers rather than re-deriving them, so a
W0 number means what the earlier C9 numbers meant.

One `collect` is one arm on one dataset: C9 built in process from the `RECALL_AML_*` environment
(the arm is the variant, `C9_routed_specialists_grounded_graph_atomic`, `C9_v4` or
`C9_v4_instruct`), every session added, each question searched once at top_k 100 through the
served router. With `--no-instruction-pass` each question is searched a second time with the
DashScope query instruction removed, which is arm B0 on the same stored vectors (v4 embeds
documents without an instruction, so only the query differs).

Two render features are held OFF for every arm, because they change the returned text and not
the ranking: compiled records need the compiler (``RECALL_AML_COMPILER=0``, which is also what
keeps the measurement free of gpt-4o-mini), and T-1 writes date annotations inside the returned
text, which would break the verbatim turn matching below (``RECALL_AML_RESOLVE_RELATIVE_TIMES=0``).

    python scripts/aml_w0_embedding_compare.py preflight --profile dashscope-text-embedding-v4-1024-memory-instruct-v1
    python scripts/aml_w0_embedding_compare.py collect --dataset locomo --data locomo10.json \\
        --arm A --expected-variant C9_routed_specialists_grounded_graph_atomic --out A-locomo.json.gz
    python scripts/aml_w0_embedding_compare.py collect --dataset lme --x1-data-dir DIR --x1-draw draw.json \\
        --arm B --expected-variant C9_v4_instruct --no-instruction-pass --out B-lme.json.gz
    python scripts/aml_w0_embedding_compare.py collect --dataset coding --amb-root <agent-memory-bench> \\
        --arm B --expected-variant C9_v4_instruct --no-instruction-pass --out B-coding.json.gz
    python scripts/aml_w0_embedding_compare.py report --control A --arms A-*.json.gz B-*.json.gz
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterator, Sequence
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

DATASETS = ("locomo", "lme", "coding")
DEPTHS = (5, 10, 20, 100)
#: The pre-registered primary metric per dataset, and how a loss is bounded.
PRIMARY = {"locomo": "turn_hit@10", "lme": "session_hit@10", "coding": "rr"}
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260928
RETRY_ATTEMPTS = 4


@dataclass(frozen=True)
class Question:
    question_id: str
    category: str
    user_id: str
    query: str
    gold_sessions: frozenset[str]
    gold_turns: tuple[str, ...] = ()


@dataclass
class Corpus:
    dataset: str
    adds: list[dict[str, Any]]
    questions: list[Question]
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------
# Loaders: each reuses the earlier harness's own corpus builder.


def load_locomo(path: Path, run_id: str, limit: int | None) -> Corpus:
    from aml_locomo_route_compare import PINNED_DATA_SHA256, build_corpus

    raw = path.read_bytes()
    adds, questions = build_corpus(json.loads(raw), f"w0-{run_id}", limit)
    digest = hashlib.sha256(raw).hexdigest()
    return Corpus(
        "locomo",
        adds,
        [
            Question(
                question_id=str(q["question_id"]),
                category=str(q["category"]),
                user_id=q["user_id"],
                query=q["query"],
                gold_sessions=frozenset(q["gold_sessions"]),
                gold_turns=tuple(q["gold_turns"]),
            )
            for q in questions
        ],
        {"data_sha256": digest, "data_matches_pinned": digest == PINNED_DATA_SHA256},
    )


def lme_evidence_turns(item: dict[str, Any]) -> tuple[str, ...]:
    """The contents of every haystack turn LongMemEval marks ``has_answer``, in order."""
    return tuple(
        str(turn["content"])
        for session in item["haystack_sessions"]
        for turn in session
        if turn.get("has_answer")
    )


def load_lme(data_dir: Path, draw_path: Path, run_id: str, limit: int | None) -> Corpus:
    import aml_x1_sources as x1

    draw = json.loads(draw_path.read_text(encoding="utf-8"))
    wanted = x1.drawn_ids(draw, "longmemeval_s")
    turns: dict[str, tuple[str, ...]] = {}
    for item in x1.iter_json_array(data_dir / x1.PINNED["longmemeval_s"][0].local):
        if str(item["question_id"]) in wanted:
            turns[str(item["question_id"])] = lme_evidence_turns(item)
    adds: list[dict[str, Any]] = []
    questions: list[Question] = []
    for position, tenant in enumerate(x1.tenants_for("longmemeval_s", data_dir, draw, f"w0-{run_id}")):
        if limit is not None and position >= limit:
            break
        stats = x1.SourceStats()
        adds.extend(add for add, _facts in x1.build_adds(tenant, stats))
        for question in tenant.questions:
            evidence = (question.get("evidence") or {}).get("sessions")
            if not evidence:  # abstention: LongMemEval scores retrieval on answerable questions
                continue
            questions.append(
                Question(
                    question_id=str(question["question_id"]),
                    category=str(question["category"]),
                    user_id=tenant.user_id,
                    query=str(question["search"]["query"]),
                    gold_sessions=frozenset(str(s) for s in evidence),
                    gold_turns=turns.get(str(question["question_id"]), ()),
                )
            )
    return Corpus("lme", adds, questions, {"draw_sha256": hashlib.sha256(draw_path.read_bytes()).hexdigest()})


def load_coding(amb_root: Path, run_id: str) -> Corpus:
    from aml_c7_qualification import load_frozen_corpus
    from aml_c9_coding_window_check import event_messages

    corpus = load_frozen_corpus(amb_root)
    user_id = f"w0-coding-{run_id}"
    adds = [
        {
            "request_id": f"w0-{run_id}-{position:04d}",
            "user_id": user_id,
            "session_id": relative,
            "messages": event_messages(corpus.corpus_root / relative),
        }
        for position, relative in enumerate(sorted(corpus.sessions), start=1)
    ]
    questions = [
        Question(task_id, "coding", user_id, prompt, frozenset(gold))
        for task_id, prompt, gold in corpus.tasks
    ]
    return Corpus("coding", adds, questions, {"sessions": len(adds)})


# ---------------------------------------------------------------------------------------------
# Scoring


def item_text(item: dict[str, Any]) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return ""


def kept_items(items: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """The served items as R1-Coding re-orders them: session id and text, in the served order."""
    return [{"session_id": str(item.get("session_id", "")), "content": item_text(item)} for item in items]


def score_items(items: Sequence[dict[str, Any]], question: Question) -> dict[str, float]:
    """Session hits, verbatim turn hits (where the dataset labels turns) and reciprocal rank."""
    from aml_locomo_route_compare import turn_present

    sessions = [str(item.get("session_id", "")) for item in items]
    row: dict[str, float] = {}
    for depth in DEPTHS:
        row[f"session_hit@{depth}"] = float(any(s in question.gold_sessions for s in sessions[:depth]))
        if question.gold_turns:
            row[f"turn_hit@{depth}"] = float(
                any(
                    turn_present(turn, item_text(item))
                    for item in items[:depth]
                    for turn in question.gold_turns
                )
            )
    first = next((rank for rank, s in enumerate(sessions, start=1) if s in question.gold_sessions), None)
    row["rr"] = 0.0 if first is None else 1.0 / first
    return row


# ---------------------------------------------------------------------------------------------
# Collect


def check_environment(env: dict[str, str], *, no_instruction_pass: bool, allow_compile: bool) -> None:
    """Refuse a collect whose environment would measure something other than the embedder."""
    problems = []
    if not allow_compile and env.get("RECALL_AML_COMPILER", "").strip() != "0":
        problems.append("RECALL_AML_COMPILER=0 is required (compiled records are held off)")
    if env.get("RECALL_AML_RESOLVE_RELATIVE_TIMES", "").strip() != "0":
        problems.append("RECALL_AML_RESOLVE_RELATIVE_TIMES=0 is required (T-1 edits returned text)")
    if problems:
        raise SystemExit("W0 collect refused: " + "; ".join(problems))


@contextlib.contextmanager
def queries_bypass_the_cache() -> Iterator[None]:
    """Embed every query live for the whole collect, so the passage cache can stay on.

    The embedding cache keys a query by its text and profile, not by whether the instruction was
    sent, so a cached query would answer the no-instruction pass with the instructed vector.
    Passages are unaffected, and they are where the cache saves money: C9 embeds every window and
    view once for each of its two tenants.
    """
    from recall_aml.embedding_lock import CachedEmbedder

    original = CachedEmbedder.embed_query

    def live(self: Any, text: str) -> list[float]:
        return list(self._inner.embed_query(text))

    CachedEmbedder.embed_query = live  # type: ignore[method-assign]
    try:
        yield
    finally:
        CachedEmbedder.embed_query = original  # type: ignore[method-assign]


@contextlib.contextmanager
def without_query_instruction() -> Iterator[None]:
    """Make every instructed query embedding drop its instruction, and restore it after.

    Both classes that can send an instruction are patched: DashScope (v4) and the
    OpenAI-compatible client (the Qwen3-Embedding proxy). Patching only one would let the other
    answer the second pass with the instructed vector, silently.
    """
    from recall.dashscope import DashScopeEmbedder
    from recall.embeddings import OpenAICompatEmbedder

    classes: tuple[Any, ...] = (DashScopeEmbedder, OpenAICompatEmbedder)
    originals = [(cls, cls.embed_query) for cls in classes]
    for cls, _ in originals:
        cls.embed_query = cls.embed_query_without_instruction
    try:
        yield
    finally:
        for cls, original in originals:
            cls.embed_query = original


def load_corpus(args: argparse.Namespace) -> Corpus:
    if args.dataset == "locomo":
        return load_locomo(args.data, args.run_id, args.limit)
    if args.dataset == "lme":
        return load_lme(args.x1_data_dir, args.x1_draw, args.run_id, args.limit)
    return load_coding(args.amb_root, args.run_id)


def collect(args: argparse.Namespace) -> dict[str, Any]:
    from starlette.testclient import TestClient

    import recall_aml.__main__ as hosted_main

    check_environment(
        dict(os.environ), no_instruction_pass=args.no_instruction_pass, allow_compile=args.allow_compile
    )
    corpus = load_corpus(args)
    headers = {"Authorization": f"Bearer {os.environ['RECALL_AML_API_KEY']}"}
    failures: Counter[str] = Counter()
    retries: Counter[str] = Counter()

    def post(client: Any, path: str, body: dict[str, Any]) -> Any:
        for attempt in range(RETRY_ATTEMPTS):
            response = client.post(path, json=body, headers=headers)
            if response.status_code < 500 or attempt == RETRY_ATTEMPTS - 1:
                return response
            retries[path] += 1
            time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def search(client: Any, question: Question) -> tuple[dict[str, float] | None, dict[str, Any]]:
        tick = time.perf_counter()
        response = post(client, "/v1/search", {"query": question.query, "user_id": question.user_id, "top_k": 100})
        facts = {
            "route": response.headers.get("X-Recall-Specialist-Route"),
            "ms": round(1_000 * (time.perf_counter() - tick), 1),
            "status": response.status_code,
        }
        if response.status_code != 200:
            failures[f"search_{response.status_code}"] += 1
            return None, facts
        items = response.json()["data"]
        facts["items"] = len(items)
        facts["kinds"] = dict(Counter(str(item.get("kind", "")) for item in items[:10]))
        if args.keep_items:
            facts["kept"] = kept_items(items)
        return score_items(items, question), facts

    started = time.perf_counter()
    users = sorted({add["user_id"] for add in corpus.adds})
    with queries_bypass_the_cache(), TestClient(hosted_main.build_app()) as client:
        version = client.get("/version", headers=headers).json()
        if version.get("variant") != args.expected_variant:
            raise SystemExit(f"served variant {version.get('variant')!r} is not {args.expected_variant!r}")
        if args.no_instruction_pass and "instruct" not in str(version.get("embedding_profile", "")):
            raise SystemExit("--no-instruction-pass needs a variant whose profile sends an instruction")
        for user in users:
            client.post("/v1/delete", json={"user_id": user}, headers=headers)
        rows: list[dict[str, Any]] = []
        try:
            add_ms: list[float] = []
            for position, add in enumerate(corpus.adds, start=1):
                tick = time.perf_counter()
                response = post(client, "/v1/add", add)
                add_ms.append(1_000 * (time.perf_counter() - tick))
                if response.status_code != 200:
                    failures[f"add_{response.status_code}"] += 1
                if position % 100 == 0:
                    print(f"added {position}/{len(corpus.adds)}", file=sys.stderr, flush=True)
            add_seconds = time.perf_counter() - started
            for position, question in enumerate(corpus.questions, start=1):
                served, facts = search(client, question)
                row: dict[str, Any] = {
                    "question_id": question.question_id,
                    "category": question.category,
                    "served": served,
                    "served_facts": facts,
                }
                if args.no_instruction_pass:
                    with without_query_instruction():
                        row["no_instruction"], row["no_instruction_facts"] = search(client, question)
                rows.append(row)
                if position % 50 == 0:
                    print(f"searched {position}/{len(corpus.questions)}", file=sys.stderr, flush=True)
        finally:
            for user in users:
                client.post("/v1/delete", json={"user_id": user}, headers=headers)
    return {
        "w0": "text-embedding-v4 migration, collect",
        "arm": args.arm,
        "dataset": corpus.dataset,
        "run_id": args.run_id,
        "variant": version.get("variant"),
        "git_commit": version.get("git_commit") or os.environ.get("RECALL_AML_GIT_COMMIT"),
        "embedding_profile": version.get("embedding_profile"),
        "context_embedding_profile": version.get("context_embedding_profile"),
        "search_content_profile": version.get("search_content_profile"),
        "no_instruction_pass": bool(args.no_instruction_pass),
        "corpus": corpus.meta,
        "n_adds": len(corpus.adds),
        "n_questions": len(corpus.questions),
        "add_p50_ms": round(statistics.median(add_ms), 1) if add_ms else None,
        "add_seconds": round(add_seconds, 1),
        "total_seconds": round(time.perf_counter() - started, 1),
        "failures": dict(failures),
        "retries": dict(retries),
        "route_counts": dict(Counter(row["served_facts"]["route"] for row in rows)),
        "rows": rows,
    }


# ---------------------------------------------------------------------------------------------
# Report


def paired_bootstrap(control: Sequence[float], treatment: Sequence[float], *, scale: float) -> dict[str, float]:
    """Percentile 95% interval of mean(treatment - control), resampling questions, times ``scale``."""
    if len(control) != len(treatment) or not control:
        raise ValueError("paired samples must be non-empty and equal in length")
    deltas = [t - c for c, t in zip(control, treatment)]
    n = len(deltas)
    rng = random.Random(BOOTSTRAP_SEED)
    means = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP_RESAMPLES))
    return {
        "delta": scale * sum(deltas) / n,
        "ci95_low": scale * means[int(0.025 * BOOTSTRAP_RESAMPLES)],
        "ci95_high": scale * means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1],
        "better": sum(1 for d in deltas if d > 0),
        "worse": sum(1 for d in deltas if d < 0),
        "n": n,
    }


def passes(dataset: str, interval: dict[str, float], *, margin_points: float, margin_mrr: float) -> bool:
    """The pre-registered rule: the interval must rule out a loss as large as the margin."""
    bound = margin_mrr if dataset == "coding" else margin_points
    return interval["ci95_low"] > -bound


def report(runs: list[dict[str, Any]], control: str, *, margin_points: float, margin_mrr: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dataset in DATASETS:
        by_arm = {run["arm"]: run for run in runs if run["dataset"] == dataset}
        if control not in by_arm:
            continue
        base = {row["question_id"]: row["served"] for row in by_arm[control]["rows"] if row["served"]}
        metric = PRIMARY[dataset]
        scale = 1.0 if metric == "rr" else 100.0
        for arm, run in sorted(by_arm.items()):
            passes_to_check = [("served", arm)]
            if run.get("no_instruction_pass"):
                passes_to_check.append(("no_instruction", f"{arm}0"))
            for key, label in passes_to_check:
                if label == control:
                    continue
                treated = {row["question_id"]: row.get(key) for row in run["rows"]}
                shared = sorted(q for q in base if treated.get(q))
                if not shared:
                    continue
                interval = paired_bootstrap(
                    [base[q][metric] for q in shared], [treated[q][metric] for q in shared], scale=scale
                )
                out[f"{dataset}:{label}-vs-{control}"] = {
                    "metric": metric if metric != "rr" else "MRR",
                    **{k: round(v, 4) if isinstance(v, float) else v for k, v in interval.items()},
                    "control_mean": round(scale * statistics.fmean(base[q][metric] for q in shared), 3),
                    "arm_mean": round(scale * statistics.fmean(treated[q][metric] for q in shared), 3),
                    "passes": passes(dataset, interval, margin_points=margin_points, margin_mrr=margin_mrr),
                }
    return out


# ---------------------------------------------------------------------------------------------
# Preflight: the only mode that talks to DashScope without a corpus.


def preflight(profile_id: str) -> dict[str, Any]:
    """Build the profile against the live endpoint and check what the documentation cannot."""
    from recall.embeddings import resolve_registered_embedder

    embedder: Any = resolve_registered_embedder(profile_id, dict(os.environ))
    result: dict[str, Any] = {"profile": profile_id, "dim": embedder.dim, "width_and_norm": "checked at construction"}
    if hasattr(embedder, "instruction_changes_query_vector") and "instruct" in profile_id:
        result["instruction_honoured"] = embedder.instruction_changes_query_vector()
    long_text = "word " * 12_000
    result["long_text_embedded"] = len(embedder.embed_passages([long_text])[0]) == embedder.dim
    result["texts_cut_after_refusal"] = getattr(embedder, "truncated_inputs", None)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--profile", required=True)
    col = sub.add_parser("collect")
    col.add_argument("--dataset", choices=DATASETS, required=True)
    col.add_argument("--arm", required=True)
    col.add_argument("--expected-variant", required=True)
    col.add_argument("--out", type=Path, required=True)
    col.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    col.add_argument("--limit", type=int, default=None, help="conversations (locomo) or tenants (lme)")
    col.add_argument("--data", type=Path, help="locomo10.json")
    col.add_argument("--x1-data-dir", type=Path)
    col.add_argument("--x1-draw", type=Path)
    col.add_argument("--amb-root", type=Path)
    col.add_argument("--no-instruction-pass", action="store_true")
    col.add_argument("--allow-compile", action="store_true")
    col.add_argument("--keep-items", action="store_true", help="store each served item's session id and text")
    rep = sub.add_parser("report")
    rep.add_argument("--control", required=True)
    rep.add_argument("--arms", type=Path, nargs="+", required=True)
    rep.add_argument("--margin-points", type=float, default=2.0)
    rep.add_argument("--margin-mrr", type=float, default=0.05)
    rep.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "preflight":
        print(json.dumps(preflight(args.profile), indent=2))
        return 0
    if args.mode == "collect":
        needed = {"locomo": ("data",), "lme": ("x1_data_dir", "x1_draw"), "coding": ("amb_root",)}[args.dataset]
        missing = [name for name in needed if getattr(args, name) is None]
        if missing:
            parser.error(f"--dataset {args.dataset} needs " + ", ".join("--" + m.replace("_", "-") for m in missing))
        result = collect(args)
        args.out.write_bytes(gzip.compress(json.dumps(result).encode("utf-8")))
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2, default=str))
        return 0
    runs = [json.loads(gzip.decompress(path.read_bytes())) for path in args.arms]
    result = report(runs, args.control, margin_points=args.margin_points, margin_mrr=args.margin_mrr)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
