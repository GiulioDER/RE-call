"""gpt-4o-mini reasoning arms for C8 atomic rescue on the frozen CAMBench corpus.

Builds on ``scripts/c8_atomizer_reference.py``, whose windows, vector cache, fusion and scoring
are imported unchanged so every arm shares one control path. Three subcommands, run on VPS3:

``atoms``
    Add-time reasoning. Pinned ``openai/gpt-4o-mini-2024-07-18`` reads each hosted window and
    proposes up to six self-contained fact statements, each with a verbatim supporting quote.
    ``recall.atomizer.ground_quote`` keeps a fact only when its quote is an exact word range of
    that window, so a paraphrased quote can never become a view.

``decompose``
    Search-time reasoning. The same model rewrites each query as one to three short search
    statements. Each becomes an extra probe vector for the gated selector.

``evaluate``
    Replays the reference control path for six arms and scores them against ``off``.

Model outputs, probe texts and per-query rows are private and stay outside the repository.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recall.atomizer import ground_quote  # noqa: E402
from recall.types import ScoredChunk  # noqa: E402
from scripts import c8_atomizer_reference as ref  # noqa: E402


MODEL = ref.MODEL
PROVIDER_ROUTING = ref.PROVIDER_ROUTING
MAX_FACTS = 6
MAX_STATEMENT_WORDS = 30
MAX_SUBQUERIES = 3
ATOMS_BUDGET_USD = 1.50
DECOMPOSE_BUDGET_USD = 0.50
WORKERS = 6
GATE_RANK = 5
ARMS = ("off", "micro", "micro_gate", "llm_gate", "micro_gate_decomp", "llm_gate_decomp")

ATOMS_PROMPT = (
    "You extract atomic facts from one window of a software engineering session transcript. "
    "The transcript is lower-cased and flattened, so punctuation and spacing may be odd. Return "
    f"up to {MAX_FACTS} facts that a developer might later need to look up: a command that was "
    "run, a file or function that was touched, an error and its cause, a value, a decision, or a "
    "result. For each fact give: statement, one self-contained sentence of at most "
    f"{MAX_STATEMENT_WORDS} words that names the specific things involved (file names, commands, "
    "tools, values) so it can be understood without the window; and quote, the exact words from "
    "WINDOW that state the fact, copied verbatim as 3 to 25 consecutive words, with no changes, "
    "no reordering and no ellipses. Skip boilerplate, greetings and generic remarks. Return an "
    "empty list when the window states no such fact."
)
ATOMS_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "window_atoms",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["facts"],
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["statement", "quote"],
                        "properties": {
                            "statement": {"type": "string"},
                            "quote": {"type": "string"},
                        },
                    },
                }
            },
        },
    },
}
DECOMPOSE_PROMPT = (
    "You prepare search queries for a memory of past software engineering sessions. Rewrite the "
    f"developer's question as 1 to {MAX_SUBQUERIES} short, self-contained search statements, each "
    "targeting one concrete thing to find: a file, command, error, value, decision or result. "
    "Use the question's own specific terms. Do not answer the question and do not add facts the "
    "question does not contain."
)
DECOMPOSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "query_decomposition",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["subqueries"],
            "properties": {"subqueries": {"type": "array", "items": {"type": "string"}}},
        },
    },
}


# --------------------------------------------------------------------------- model calls


class Budget:
    """A thread-safe spend meter that refuses new calls once the cap is reached."""

    def __init__(self, cap_usd: float) -> None:
        self.cap = cap_usd
        self.spent = 0.0
        self.served: set[str] = set()
        self._lock = threading.Lock()

    def check(self) -> None:
        with self._lock:
            if self.spent >= self.cap:
                raise RuntimeError(f"budget exhausted: USD {self.spent:.4f} of {self.cap:.2f}")

    def charge(self, response: Mapping[str, Any]) -> None:
        usage = response.get("usage") or {}
        cost = usage.get("cost")
        if not isinstance(cost, (int, float)):
            cost = ref.PRICE_IN * usage.get("prompt_tokens", 0) + ref.PRICE_OUT * usage.get(
                "completion_tokens", 0
            )
        with self._lock:
            self.spent += float(cost)
            self.served.add(f"{response.get('model')}|{response.get('provider')}")


def _payload(system: str, user: Mapping[str, str], response_format: Mapping[str, Any], max_tokens: int) -> dict[str, Any]:
    return {
        "model": MODEL,
        "provider": PROVIDER_ROUTING,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": response_format,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(dict(user), ensure_ascii=False)},
        ],
        "usage": {"include": True},
    }


def _content(response: Mapping[str, Any]) -> Any:
    return json.loads(response["choices"][0]["message"]["content"])


def ground_facts(window_text: str, facts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first MAX_FACTS proposals, grounding each quote and bounding each statement."""

    grounded: list[dict[str, Any]] = []
    for fact in list(facts)[:MAX_FACTS]:
        statement = " ".join(str(fact.get("statement", "")).split())
        quote = str(fact.get("quote", ""))
        span = ground_quote(window_text, quote)
        reason = None
        if not statement or len(statement.split()) > MAX_STATEMENT_WORDS:
            reason = "statement_length"
        elif span is None:
            reason = "ungrounded"
        grounded.append(
            {
                "statement": statement,
                "quote": quote,
                "span": list(span) if span is not None else None,
                "rejected": reason,
            }
        )
    return grounded


def generate_atoms(
    windows: Sequence[ref.Window],
    out: Path,
    *,
    call: Callable[[dict[str, Any]], dict[str, Any]],
    budget_usd: float = ATOMS_BUDGET_USD,
    workers: int = WORKERS,
) -> dict[str, Any]:
    budget = Budget(budget_usd)

    def one(window: ref.Window) -> dict[str, Any]:
        budget.check()
        response = call(_payload(ATOMS_PROMPT, {"WINDOW": window.chunk.text}, ATOMS_FORMAT, 900))
        budget.charge(response)
        try:
            facts = _content(response)["facts"]
            if not isinstance(facts, list):
                raise TypeError("facts is not a list")
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            return {"chunk_id": window.chunk.id, "malformed": True, "facts": []}
        return {
            "chunk_id": window.chunk.id,
            "session": window.session,
            "segment": window.segment,
            "malformed": False,
            "facts": ground_facts(window.chunk.text, facts),
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, windows))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    proposed = [fact for row in rows for fact in row["facts"]]
    kept = [fact for fact in proposed if fact["rejected"] is None]
    return {
        "windows": len(rows),
        "malformed_windows": sum(1 for row in rows if row["malformed"]),
        "facts_proposed": len(proposed),
        "facts_grounded": len(kept),
        "grounding_rate": round(len(kept) / len(proposed), 4) if proposed else 0.0,
        "rejections": {
            reason: sum(1 for fact in proposed if fact["rejected"] == reason)
            for reason in ("ungrounded", "statement_length")
        },
        "windows_with_a_fact": sum(
            1 for row in rows if any(fact["rejected"] is None for fact in row["facts"])
        ),
        "usd": round(budget.spent, 6),
        "served": sorted(budget.served),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }


def generate_decompositions(
    queries: Sequence[tuple[str, str]],
    out: Path,
    *,
    call: Callable[[dict[str, Any]], dict[str, Any]],
    budget_usd: float = DECOMPOSE_BUDGET_USD,
    workers: int = WORKERS,
) -> dict[str, Any]:
    budget = Budget(budget_usd)

    def one(item: tuple[str, str]) -> dict[str, Any]:
        query_id, text = item
        budget.check()
        started = time.perf_counter()
        response = call(_payload(DECOMPOSE_PROMPT, {"QUESTION": text}, DECOMPOSE_FORMAT, 300))
        latency_ms = (time.perf_counter() - started) * 1000.0
        budget.charge(response)
        try:
            raw = _content(response)["subqueries"]
            subqueries = [" ".join(str(value).split()) for value in raw if str(value).strip()]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            subqueries = []
        return {
            "query_id": query_id,
            "subqueries": subqueries[:MAX_SUBQUERIES],
            "latency_ms": round(latency_ms, 1),
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, queries))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    latencies = sorted(row["latency_ms"] for row in rows)
    return {
        "queries": len(rows),
        "empty": sum(1 for row in rows if not row["subqueries"]),
        "subqueries_mean": round(statistics.mean(len(row["subqueries"]) for row in rows), 3),
        "latency_ms_p50": latencies[len(latencies) // 2] if latencies else None,
        "latency_ms_p95": latencies[max(0, int(0.95 * len(latencies)) - 1)] if latencies else None,
        "usd": round(budget.spent, 6),
        "served": sorted(budget.served),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }


# --------------------------------------------------------------------------- artifacts and replay


def llm_views(windows: Sequence[ref.Window], atoms_path: Path) -> tuple[list[str], list[dict[str, object]]]:
    """Grounded statements as views, each bound to the window its quote was grounded in."""

    by_id = {window.chunk.id: window for window in windows}
    ordinal: dict[str, int] = {}
    texts: list[str] = []
    rows: list[dict[str, object]] = []
    records = [json.loads(line) for line in atoms_path.read_text(encoding="utf-8").splitlines()]
    for record in sorted(records, key=lambda row: (row.get("session", ""), row.get("segment", 0))):
        window = by_id.get(record["chunk_id"])
        if window is None:
            raise RuntimeError("atoms file names a window this corpus does not have")
        seen: set[str] = set()
        for fact in record["facts"]:
            if fact["rejected"] is not None or fact["statement"] in seen:
                continue
            seen.add(fact["statement"])
            index = ordinal.get(window.session, 0)
            ordinal[window.session] = index + 1
            texts.append(fact["statement"])
            rows.append(
                {
                    "chunk_id": window.chunk.id,
                    "source": window.session,
                    "parent_ordinal": window.segment,
                    "view_ordinal": index,
                }
            )
    if not rows:
        raise RuntimeError("no grounded atoms")
    return texts, rows


def write_artifact(name: str, texts: Sequence[str], rows: Sequence[Mapping[str, object]], matrix: Any, windows: Sequence[ref.Window], directory: Path) -> Any:
    from recall.atomic_rescue import load_atomic_rescue_artifact, write_atomic_rescue_artifact

    manifest = write_atomic_rescue_artifact(
        directory / name,
        matrix=matrix,
        views=[dict(row) for row in rows],
        generation_id="cambench-reference",
        calibration_id="uncalibrated-reference",
        pipeline_fingerprint=f"atomizer-{name}",
        corpus_fingerprint=hashlib.sha256(
            "\n".join(window.chunk.id for window in windows).encode("utf-8")
        ).hexdigest(),
        embedding_profile=ref.EMBEDDING_PROFILE,
        embedding_fingerprint="reference",
        ordinary_chunk_count=len(windows),
        source_commit=os.environ.get("RECALL_SOURCE_COMMIT", "unknown"),
    )
    return load_atomic_rescue_artifact(manifest)


def gated_replay(
    vector: Any,
    sub_vectors: Sequence[Any],
    window_matrix: Any,
    windows: Sequence[ref.Window],
    lexical: Sequence[ScoredChunk],
    artifact: Any,
) -> ref.ArmOutcome:
    from recall.atomic_rescue import (
        AtomicRescueArtifactError,
        AtomicRescueSelectionError,
        insert_gated_atomic_rescue_dense,
    )

    dense = ref.dense_top(window_matrix, vector, windows)
    by_id = {window.chunk.id: window.chunk for window in windows}
    probes = [(vector, dense)] + [
        (sub, ref.dense_top(window_matrix, sub, windows)) for sub in sub_vectors
    ]
    started = time.perf_counter()
    try:
        ranked, gated = insert_gated_atomic_rescue_dense(
            artifact,
            [([float(value) for value in probe], hits) for probe, hits in probes],
            dense,
            lambda chunk_id, score: ScoredChunk(by_id[chunk_id], score) if chunk_id in by_id else None,
            gate_rank=GATE_RANK,
        )
    except (AtomicRescueArtifactError, AtomicRescueSelectionError):
        return ref.ArmOutcome(ref.fuse(dense, lexical), None, None, 0.0, True)
    elapsed = (time.perf_counter() - started) * 1000.0
    rescued = gated.selection.chunk_id if gated is not None else None
    prior_ids = [hit.chunk.id for hit in dense]
    prior = prior_ids.index(rescued) + 1 if rescued in prior_ids else None
    return ref.ArmOutcome(ref.fuse(ranked, lexical), rescued, prior, elapsed, False)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from recall.embeddings import embed_passages, embed_query, resolve_registered_embedder
    from scripts.aml_c7_qualification import load_frozen_corpus

    corpus = load_frozen_corpus(args.amb_root)
    rendered = dict(corpus.rendered)
    windows = ref.build_windows(rendered)
    session_of = {window.chunk.id: window.session for window in windows}
    ids = {(window.session, window.segment): window.chunk.id for window in windows}

    queries: list[ref.Query] = []
    for line in args.probes.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if args.split != "all" and row["split"] != args.split:
            continue
        queries.append(
            ref.Query(
                row["probe_id"],
                "probe",
                row["question"],
                frozenset(ids[(row["session"], segment)] for segment in row["gold_segments"]),
                frozenset({row["session"]}),
            )
        )
    for task_id, prompt, gold in corpus.tasks:
        gold_ids = frozenset(window.chunk.id for window in windows if window.session in gold)
        queries.append(ref.Query(task_id, "task", prompt, gold_ids, frozenset(gold)))
    decomposed = {
        row["query_id"]: row["subqueries"]
        for row in (json.loads(line) for line in args.decompositions.read_text(encoding="utf-8").splitlines())
    }
    missing = [query.query_id for query in queries if query.query_id not in decomposed]
    if missing:
        raise RuntimeError(f"{len(missing)} queries have no decomposition")

    embedder = resolve_registered_embedder(ref.EMBEDDING_PROFILE, dict(os.environ))

    def documents(batch: list[str]) -> list[list[float]]:
        return embed_passages(embedder, batch)

    def questions(batch: list[str]) -> list[list[float]]:
        return [embed_query(embedder, text) for text in batch]

    cache = ref.VectorCache(args.cache)
    sub_texts = sorted({text for query in queries for text in decomposed[query.query_id]})
    llm_texts, llm_rows = llm_views(windows, args.atoms)
    spend = {
        "windows": cache.ensure("document", [w.chunk.text for w in windows], documents),
        "queries": cache.ensure("query", [q.text for q in queries], questions),
        "subqueries": cache.ensure("query", sub_texts, questions),
        "llm_views": cache.ensure("document", llm_texts, documents),
    }
    window_matrix = cache.matrix("document", [window.chunk.text for window in windows])
    query_matrix = cache.matrix("query", [query.text for query in queries])
    sub_vectors = {
        query.query_id: (
            cache.matrix("query", decomposed[query.query_id])
            if decomposed[query.query_id]
            else []
        )
        for query in queries
    }
    chunks = [window.chunk for window in windows]
    lexical = [
        ref.rank_bm25_chunks(chunks, query.text, k=ref.CANDIDATE_K, stable_ties=True)
        for query in queries
    ]

    outcomes: dict[str, list[ref.ArmOutcome]] = {}
    artifacts: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="c8-reasoning-") as scratch:
        micro, artifacts["micro"] = ref.build_artifact(
            "micro", windows, rendered, cache, documents, Path(scratch)
        )
        llm = write_artifact(
            "llm", llm_texts, llm_rows, cache.matrix("document", llm_texts), windows, Path(scratch)
        )
        artifacts["llm"] = {
            "views": len(llm_rows),
            "parents": len({row["chunk_id"] for row in llm_rows}),
            "parent_coverage": round(len({row["chunk_id"] for row in llm_rows}) / len(windows), 4),
        }
        for arm in args.arms:
            rows: list[ref.ArmOutcome] = []
            for index, query in enumerate(queries):
                vector = query_matrix[index]
                if arm == "off":
                    rows.append(ref.replay(query, vector, window_matrix, windows, lexical[index], None))
                elif arm == "micro":
                    rows.append(ref.replay(query, vector, window_matrix, windows, lexical[index], micro))
                else:
                    artifact = llm if arm.startswith("llm") else micro
                    subs = sub_vectors[query.query_id] if arm.endswith("_decomp") else []
                    rows.append(
                        gated_replay(vector, list(subs), window_matrix, windows, lexical[index], artifact)
                    )
            outcomes[arm] = rows
    report: dict[str, Any] = {
        "protocol": "2026-09-22-c8-atomizer-reasoning",
        "split": args.split,
        "windows": len(windows),
        "newly_embedded": spend,
        "artifacts": artifacts,
    }
    for kind in ("probe", "task"):
        selected = [index for index, query in enumerate(queries) if query.kind == kind]
        report[kind] = ref.summarise(
            [queries[index] for index in selected],
            {arm: [rows[index] for index in selected] for arm, rows in outcomes.items()},
            session_of,
        )
    if args.rows is not None:
        with args.rows.open("w", encoding="utf-8", newline="\n") as handle:
            for index, query in enumerate(queries):
                handle.write(
                    json.dumps(
                        {
                            "query_id": query.query_id,
                            "kind": query.kind,
                            "arms": {
                                arm: {
                                    "exact_rank": ref.first_hit(rows[index].ranked, query.gold_ids),
                                    "rescued": rows[index].rescued,
                                    "rescued_prior_rank": rows[index].rescued_prior_rank,
                                    "rescued_is_gold": rows[index].rescued in query.gold_ids,
                                }
                                for arm, rows in outcomes.items()
                            },
                        }
                    )
                    + "\n"
                )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    atoms = sub.add_parser("atoms")
    atoms.add_argument("--amb-root", type=Path, required=True)
    atoms.add_argument("--out", type=Path, required=True)
    atoms.add_argument("--budget-usd", type=float, default=ATOMS_BUDGET_USD)
    decompose = sub.add_parser("decompose")
    decompose.add_argument("--amb-root", type=Path, required=True)
    decompose.add_argument("--probes", type=Path, required=True)
    decompose.add_argument("--out", type=Path, required=True)
    decompose.add_argument("--budget-usd", type=float, default=DECOMPOSE_BUDGET_USD)
    run = sub.add_parser("evaluate")
    run.add_argument("--amb-root", type=Path, required=True)
    run.add_argument("--probes", type=Path, required=True)
    run.add_argument("--atoms", type=Path, required=True)
    run.add_argument("--decompositions", type=Path, required=True)
    run.add_argument("--cache", type=Path, required=True)
    run.add_argument("--split", choices=("dev", "confirm", "all"), required=True)
    run.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    run.add_argument("--rows", type=Path)
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from scripts.aml_c7_qualification import load_frozen_corpus

    corpus = load_frozen_corpus(args.amb_root)
    if args.command in {"atoms", "decompose"}:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY is required")

        def call(payload: dict[str, Any]) -> dict[str, Any]:
            return ref._openrouter(payload, api_key)

    if args.command == "atoms":
        summary = generate_atoms(
            ref.build_windows(dict(corpus.rendered)), args.out, call=call, budget_usd=args.budget_usd
        )
    elif args.command == "decompose":
        items = [
            (row["probe_id"], row["question"])
            for row in (json.loads(line) for line in args.probes.read_text(encoding="utf-8").splitlines())
        ] + [(task_id, prompt) for task_id, prompt, _ in corpus.tasks]
        summary = generate_decompositions(items, args.out, call=call, budget_usd=args.budget_usd)
    else:
        if "off" not in args.arms:
            raise SystemExit("the off control arm is mandatory")
        summary = evaluate(args)
        args.out.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
