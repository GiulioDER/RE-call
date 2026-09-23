"""Round 3 of the C8 atomizer study: fresh dev questions and a view-level admission gate.

Imports the reference and reasoning harnesses unchanged, so the control path, the artifacts,
the vector cache and the scoring are identical to rounds 1 and 2. Two subcommands, run on VPS3:

``probes``
    Fresh dev questions. Spans are drawn with a new seed from **dev-half sessions only**, never
    overlapping a span already used, so the untouched 149 confirm probes stay the final test.
    The writer is a non-OpenAI model (Llama 3.3 70B) so that the questions do not share a model
    with the gpt-4o-mini atoms; prompt and rejection rules are the reference record's.

``evaluate``
    Arms ``off``, ``micro``, ``llm``, ``micro_vgate`` and ``llm_vgate`` on one probe set (the fresh
    dev set, or the original confirm split) plus the 34-task sentinel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recall.types import ScoredChunk  # noqa: E402
from scripts import c8_atomizer_reasoning as reasoning  # noqa: E402
from scripts import c8_atomizer_reference as ref  # noqa: E402


SEED = 20260923
WRITER_MODEL = "meta-llama/llama-3.3-70b-instruct"
WRITER_ROUTING = {"require_parameters": True, "allow_fallbacks": True}
PROBE_BUDGET_USD = 0.50
ARMS = ("off", "micro", "llm", "micro_vgate", "llm_vgate")


def fresh_spans(
    rendered: Mapping[str, str],
    used: Mapping[str, Sequence[tuple[int, int]]],
    *,
    seed: int = SEED,
) -> list[ref.Span]:
    """Seeded spans from dev-half sessions only, disjoint from every span already used."""

    rng = random.Random(seed)
    spans: list[ref.Span] = []
    for session in sorted(rendered):
        if ref.split_for(session) != "dev":
            continue
        words = rendered[session].split()
        if len(words) < ref.SPAN_MAX_WORDS * 2:
            continue
        taken: list[tuple[int, int]] = list(used.get(session, ()))
        chosen: list[tuple[int, int]] = []
        attempts = 0
        while len(chosen) < ref.SPANS_PER_SESSION and attempts < 80:
            attempts += 1
            length = rng.randint(ref.SPAN_MIN_WORDS, ref.SPAN_MAX_WORDS)
            start = rng.randrange(0, len(words) - length + 1)
            end = start + length
            if any(start < other_end and other_start < end for other_start, other_end in taken):
                continue
            if sum(1 for word in words[start:end] if ref._ALNUM.search(word)) < ref.MIN_SPAN_CONTENT_WORDS:
                continue
            taken.append((start, end))
            chosen.append((start, end))
        for start, end in sorted(chosen):
            spans.append(
                ref.Span(
                    session,
                    start,
                    end,
                    " ".join(words[start:end]),
                    tuple(ref.gold_segments(len(words), start, end)),
                )
            )
    return spans


def generate_fresh_probes(
    rendered: Mapping[str, str],
    used: Mapping[str, Sequence[tuple[int, int]]],
    out: Path,
    *,
    call: Callable[[dict[str, Any]], dict[str, Any]],
    budget_usd: float = PROBE_BUDGET_USD,
) -> dict[str, Any]:
    spans = fresh_spans(rendered, used)
    budget = reasoning.Budget(budget_usd)
    reasons: dict[str, int] = {}
    kept = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for span in spans:
            budget.check()
            words = rendered[span.session].split()
            payload = {
                "model": WRITER_MODEL,
                "provider": WRITER_ROUTING,
                "temperature": 0,
                "max_tokens": 200,
                "response_format": ref.RESPONSE_FORMAT,
                "messages": [
                    {"role": "system", "content": ref.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"CONTEXT": ref._context(words, span.start, span.end), "TARGET": span.text},
                            ensure_ascii=False,
                        ),
                    },
                ],
                "usage": {"include": True},
            }
            response = call(payload)
            budget.charge(response)
            try:
                decoded = json.loads(response["choices"][0]["message"]["content"])
                answerable = bool(decoded["answerable"])
                question = str(decoded["question"]).strip()
            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                reasons["malformed"] = reasons.get("malformed", 0) + 1
                continue
            reason = ref.probe_rejection(answerable, question, span.text)
            if reason is not None:
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            kept += 1
            handle.write(
                json.dumps(
                    {
                        "probe_id": "r3-"
                        + hashlib.sha256(
                            f"{span.session}\x00{span.start}\x00{span.end}".encode("utf-8")
                        ).hexdigest()[:16],
                        "session": span.session,
                        "split": "dev3",
                        "span_start": span.start,
                        "span_end": span.end,
                        "span_text": span.text,
                        "question": question,
                        "gold_segments": list(span.gold),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return {
        "spans": len(spans),
        "kept": kept,
        "rejections": reasons,
        "usd": round(budget.spent, 6),
        "served": sorted(budget.served),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }


def view_gated_replay(
    vector: Any,
    window_matrix: Any,
    windows: Sequence[ref.Window],
    lexical: Sequence[ScoredChunk],
    artifact: Any,
) -> ref.ArmOutcome:
    from recall.atomic_rescue import (
        AtomicRescueArtifactError,
        AtomicRescueSelectionError,
        insert_view_gated_atomic_rescue_dense,
    )

    dense = ref.dense_top(window_matrix, vector, windows)
    by_id = {window.chunk.id: window.chunk for window in windows}
    started = time.perf_counter()
    try:
        ranked, gated = insert_view_gated_atomic_rescue_dense(
            artifact,
            [float(value) for value in vector],
            dense,
            lambda chunk_id, score: ScoredChunk(by_id[chunk_id], score) if chunk_id in by_id else None,
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
        if row["split"] != args.split:
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
    if not queries:
        raise RuntimeError(f"no probes for split {args.split!r}")
    for task_id, prompt, gold in corpus.tasks:
        gold_ids = frozenset(window.chunk.id for window in windows if window.session in gold)
        queries.append(ref.Query(task_id, "task", prompt, gold_ids, frozenset(gold)))

    embedder = resolve_registered_embedder(ref.EMBEDDING_PROFILE, dict(os.environ))

    def documents(batch: list[str]) -> list[list[float]]:
        return embed_passages(embedder, batch)

    def questions(batch: list[str]) -> list[list[float]]:
        return [embed_query(embedder, text) for text in batch]

    cache = ref.VectorCache(args.cache)
    llm_texts, llm_rows = reasoning.llm_views(windows, args.atoms)
    spend = {
        "windows": cache.ensure("document", [w.chunk.text for w in windows], documents),
        "queries": cache.ensure("query", [q.text for q in queries], questions),
        "llm_views": cache.ensure("document", llm_texts, documents),
    }
    window_matrix = cache.matrix("document", [window.chunk.text for window in windows])
    query_matrix = cache.matrix("query", [query.text for query in queries])
    chunks = [window.chunk for window in windows]
    lexical = [
        ref.rank_bm25_chunks(chunks, query.text, k=ref.CANDIDATE_K, stable_ties=True)
        for query in queries
    ]
    outcomes: dict[str, list[ref.ArmOutcome]] = {}
    artifacts: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="c8-round3-") as scratch:
        micro, artifacts["micro"] = ref.build_artifact(
            "micro", windows, rendered, cache, documents, Path(scratch)
        )
        llm = reasoning.write_artifact(
            "llm", llm_texts, llm_rows, cache.matrix("document", llm_texts), windows, Path(scratch)
        )
        artifacts["llm"] = {"views": len(llm_rows), "parents": len({r["chunk_id"] for r in llm_rows})}
        for arm in args.arms:
            rows: list[ref.ArmOutcome] = []
            for index, query in enumerate(queries):
                vector = query_matrix[index]
                if arm == "off":
                    rows.append(ref.replay(query, vector, window_matrix, windows, lexical[index], None))
                elif arm in {"micro", "llm"}:
                    artifact = micro if arm == "micro" else llm
                    rows.append(ref.replay(query, vector, window_matrix, windows, lexical[index], artifact))
                else:
                    artifact = micro if arm.startswith("micro") else llm
                    rows.append(view_gated_replay(vector, window_matrix, windows, lexical[index], artifact))
            outcomes[arm] = rows
    report: dict[str, Any] = {
        "protocol": "2026-09-23-c8-atomizer-round3",
        "split": args.split,
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
    probes = sub.add_parser("probes")
    probes.add_argument("--amb-root", type=Path, required=True)
    probes.add_argument("--used-probes", type=Path, required=True)
    probes.add_argument("--out", type=Path, required=True)
    probes.add_argument("--budget-usd", type=float, default=PROBE_BUDGET_USD)
    run = sub.add_parser("evaluate")
    run.add_argument("--amb-root", type=Path, required=True)
    run.add_argument("--probes", type=Path, required=True)
    run.add_argument("--atoms", type=Path, required=True)
    run.add_argument("--cache", type=Path, required=True)
    run.add_argument("--split", choices=("dev3", "confirm"), required=True)
    run.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    run.add_argument("--rows", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "probes":
        from scripts.aml_c7_qualification import load_frozen_corpus

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY is required")
        used: dict[str, list[tuple[int, int]]] = {}
        for line in args.used_probes.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            used.setdefault(row["session"], []).append((row["span_start"], row["span_end"]))
        summary = generate_fresh_probes(
            dict(load_frozen_corpus(args.amb_root).rendered),
            used,
            args.out,
            call=lambda payload: ref._openrouter(payload, api_key),
            budget_usd=args.budget_usd,
        )
    else:
        if "off" not in args.arms:
            raise SystemExit("the off control arm is mandatory")
        summary = evaluate(args)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
