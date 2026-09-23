"""Memory-tenant check of the C8-confirmed micro atomizer against production's memo atomizer.

Runs on VPS2 against the memory tenant's active generation, read only, pinned to one generation.
It mirrors production's memory path: Context4 query embedding, ``query_dense``, the dense top five
kept and one atomic winner inserted at slot six (``insert_atomic_rescue_dense``). Arms:

``off``    dense only.
``prod``   the production memo paragraph atomizer, built for the pinned generation by the unchanged
           ``scripts.build_atomic_fact_production_artifact`` into a private directory.
``micro``  24-word windows with a 12-word stride cut inside each stored chunk, parent = that
           chunk, embedded with Context4 grouped by source as production groups its views.

Two subcommands: ``probes`` writes fresh questions about production-disjoint sources with a
non-OpenAI writer; ``evaluate`` replays one split. Probe texts, views and per-query rows are
private and stay on VPS2.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recall.types import Chunk, ScoredChunk  # noqa: E402
from scripts import atomizer_study_common as common  # noqa: E402


SEED = 20260924
SOURCES_PER_RUN = 240
MIN_CHUNK_WORDS = 40
MICRO_SIZE = 24
MICRO_STRIDE = 12
MIN_CONTENT_WORDS = 4
CANDIDATE_K = 100
CUTOFFS = (1, 5, 6, 10)
ARMS = ("off", "prod", "micro")
PROBE_BUDGET_USD = 0.50
_ALNUM = re.compile(r"[0-9A-Za-z]")
_SPACE = re.compile(r"\s+")

MEMORY_PROMPT = (
    "You write retrieval test questions about a project's engineering memory notes. You receive "
    "CONTEXT, with the TARGET passage marked between [[ and ]], and the TARGET alone. Write exactly "
    "one question that an engineer on this project might ask later, whose answer is stated in "
    "TARGET. Ask about the specific fact, value, command, file, decision, cause or result that "
    "TARGET states. Do not copy more than four consecutive words from TARGET. Do not mention the "
    "target, the passage, the note or the context. If TARGET states no specific checkable fact, set "
    "answerable to false and question to an empty string."
)


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    source: str
    ordinal: int
    text: str


def _norm(text: str) -> str:
    return _SPACE.sub(" ", text).strip().casefold()


def split_for_source(source: str) -> str:
    return "dev" if hashlib.sha256(source.encode("utf-8")).digest()[0] % 2 == 0 else "confirm"


def excluded_sources(records: Iterable[Any]) -> set[str]:
    """Every source string named as gold anywhere in prior atomic evaluation records."""

    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"gold_sources", "gold_source", "source", "sources"}:
                    for text in item if isinstance(item, list) else [item]:
                        if isinstance(text, str) and "/" in text:
                            found.add(text)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for record in records:
        walk(record)
    return found


@dataclass(frozen=True)
class MemorySpan:
    source: str
    chunk_id: str
    text: str
    gold_chunk_ids: tuple[str, ...]


def sample_memory_spans(
    chunks: Sequence[StoredChunk],
    excluded: set[str],
    *,
    seed: int = SEED,
    sources: int = SOURCES_PER_RUN,
) -> list[MemorySpan]:
    """One seeded span per sampled production-disjoint source; gold is every containing chunk."""

    by_source: dict[str, list[StoredChunk]] = defaultdict(list)
    for chunk in chunks:
        by_source[chunk.source].append(chunk)
    eligible = sorted(
        source
        for source, items in by_source.items()
        if source not in excluded and any(len(item.text.split()) >= MIN_CHUNK_WORDS for item in items)
    )
    rng = random.Random(seed)
    rng.shuffle(eligible)
    spans: list[MemorySpan] = []
    for source in eligible[:sources]:
        candidates = sorted(
            (item for item in by_source[source] if len(item.text.split()) >= MIN_CHUNK_WORDS),
            key=lambda item: item.ordinal,
        )
        chunk = candidates[rng.randrange(len(candidates))]
        words = chunk.text.split()
        for _ in range(50):
            length = rng.randint(common.SPAN_MIN_WORDS, common.SPAN_MAX_WORDS)
            start = rng.randrange(0, len(words) - length + 1)
            span_words = words[start : start + length]
            if sum(1 for word in span_words if _ALNUM.search(word)) >= common.MIN_SPAN_CONTENT_WORDS:
                break
        else:
            continue
        text = " ".join(span_words)
        needle = _norm(text)
        gold = tuple(
            sorted(item.chunk_id for item in by_source[source] if needle in _norm(item.text))
        )
        if chunk.chunk_id not in gold:
            raise RuntimeError("a sampled span is not found in its own chunk")
        spans.append(MemorySpan(source, chunk.chunk_id, text, gold))
    return spans


def span_records(
    chunks: Sequence[StoredChunk],
    excluded: set[str],
    *,
    seed: int = SEED,
    sources: int = SOURCES_PER_RUN,
) -> list[dict[str, Any]]:
    """The sampled spans with their chunk as context: all the writer needs, and nothing else.

    VPS2 holds the corpus but no OpenRouter key; VPS3 holds the key but not the corpus. This record
    is what crosses between them, so no credential and no database access has to move.
    """

    text_of = {chunk.chunk_id: chunk.text for chunk in chunks}
    records = []
    for span in sample_memory_spans(chunks, excluded, seed=seed, sources=sources):
        chunk_text = text_of[span.chunk_id]
        context = (
            chunk_text.replace(span.text, f"[[ {span.text} ]]", 1)
            if span.text in chunk_text
            else chunk_text
        )
        records.append(
            {
                "source": span.source,
                "chunk_id": span.chunk_id,
                "span_text": span.text,
                "context": context,
                "gold_chunk_ids": list(span.gold_chunk_ids),
            }
        )
    return records


def generate_memory_probes(
    spans: Sequence[Mapping[str, Any]],
    out: Path,
    *,
    call: Callable[[dict[str, Any]], dict[str, Any]],
    budget_usd: float = PROBE_BUDGET_USD,
) -> dict[str, Any]:
    budget = common.Budget(budget_usd)
    reasons: dict[str, int] = {}
    kept = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for span in spans:
            budget.check()
            payload = {
                "model": common.WRITER_MODEL,
                "provider": common.WRITER_ROUTING,
                "temperature": 0,
                "max_tokens": 200,
                "response_format": common.RESPONSE_FORMAT,
                "messages": [
                    {"role": "system", "content": MEMORY_PROMPT},
                    {"role": "user", "content": json.dumps({"CONTEXT": span["context"], "TARGET": span["span_text"]}, ensure_ascii=False)},
                ],
                "usage": {"include": True},
            }
            response = call(payload)
            budget.charge(response)
            try:
                decoded = json.loads(response["choices"][0]["message"]["content"])
                answerable, question = bool(decoded["answerable"]), str(decoded["question"]).strip()
            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                reasons["malformed"] = reasons.get("malformed", 0) + 1
                continue
            reason = common.probe_rejection(answerable, question, span["span_text"])
            if reason is not None:
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            kept += 1
            handle.write(
                json.dumps(
                    {
                        "probe_id": "mem-"
                        + hashlib.sha256(
                            f"{span['chunk_id']}\x00{span['span_text']}".encode("utf-8")
                        ).hexdigest()[:16],
                        "source": span["source"],
                        "split": split_for_source(span["source"]),
                        "chunk_id": span["chunk_id"],
                        "span_text": span["span_text"],
                        "question": question,
                        "gold_chunk_ids": list(span["gold_chunk_ids"]),
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


def micro_chunk_views(chunks: Sequence[StoredChunk]) -> list[tuple[StoredChunk, int, str]]:
    """24/12 word windows inside each stored chunk, deduplicated within a source."""

    views: list[tuple[StoredChunk, int, str]] = []
    seen: dict[str, set[str]] = defaultdict(set)
    ordinal: dict[str, int] = defaultdict(int)
    for chunk in sorted(chunks, key=lambda item: (item.source, item.ordinal)):
        words = chunk.text.split()
        for start, end in common.window_bounds(len(words), size=MICRO_SIZE, stride=MICRO_STRIDE):
            span = words[start:end]
            if sum(1 for word in span if _ALNUM.search(word)) < MIN_CONTENT_WORDS:
                continue
            text = " ".join(span)
            if text in seen[chunk.source]:
                continue
            seen[chunk.source].add(text)
            views.append((chunk, ordinal[chunk.source], text))
            ordinal[chunk.source] += 1
    return views


def paired(
    ranks_a: Sequence[int | None], ranks_b: Sequence[int | None], cutoff: int
) -> dict[str, int]:
    """Gains and losses of arm b against arm a at one cutoff, over the same queries."""

    def hit(rank: int | None) -> bool:
        return rank is not None and rank <= cutoff

    gains = sum(1 for a, b in zip(ranks_a, ranks_b, strict=True) if hit(b) and not hit(a))
    losses = sum(1 for a, b in zip(ranks_a, ranks_b, strict=True) if hit(a) and not hit(b))
    return {"gains": gains, "losses": losses, "net": gains - losses}


def _first(ranked: Sequence[str], wanted: Iterable[str]) -> int | None:
    targets = set(wanted)
    for rank, item in enumerate(ranked, start=1):
        if item in targets:
            return rank
    return None


def load_chunks(store: Any) -> list[StoredChunk]:
    chunks: list[StoredChunk] = []
    for chunk in store.iter_chunks():
        source, ordinal = chunk.metadata.get("file"), chunk.metadata.get("ord")
        if not isinstance(source, str) or not isinstance(ordinal, int) or isinstance(ordinal, bool):
            raise RuntimeError("chunk lacks file or integer ord metadata")
        chunks.append(StoredChunk(chunk.id, source, ordinal, chunk.text))
    return chunks


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    from recall.atomic_rescue import (
        AtomicRescueArtifactError,
        AtomicRescueSelectionError,
        insert_atomic_rescue_dense,
        load_atomic_rescue_artifact,
        write_atomic_rescue_artifact,
    )
    from recall.embeddings import (
        embed_document_groups,
        embed_query,
        embedding_profile,
        embedding_profile_id,
        resolve_embedder,
    )
    from recall.generation_store import GenerationStore

    embedder = resolve_embedder(args.embedder)
    rows = [json.loads(line) for line in args.probes.read_text(encoding="utf-8").splitlines()]
    probes = [row for row in rows if row["split"] == args.split]
    if not probes:
        raise RuntimeError(f"no probes for split {args.split!r}")
    prod = load_atomic_rescue_artifact(args.prod_artifact)
    report: dict[str, Any] = {"protocol": "2026-09-23-memory-atomizer-check", "split": args.split}
    with GenerationStore(args.dsn, embedder.dim, tenant=args.tenant) as raw_store:
        store: Any = raw_store
        store.set_fixed_generation(args.generation)
        binding = store.generation_binding()
        prod.assert_lineage(
            generation_id=args.generation,
            calibration_id=prod.calibration_id,
            pipeline_fingerprint=str(binding["pipeline_fingerprint"]),
            corpus_fingerprint=str(binding["corpus_fingerprint"]),
            embedder=embedder,
        )
        chunks = load_chunks(store)
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        manifest = args.micro_directory / "manifest.json"
        if not manifest.exists():
            views = micro_chunk_views(chunks)
            groups: dict[str, list[int]] = defaultdict(list)
            for index, (chunk, _, _) in enumerate(views):
                groups[chunk.source].append(index)
            ordered = sorted(groups)
            started = time.perf_counter()
            vectors = embed_document_groups(embedder, [[views[i][2] for i in groups[s]] for s in ordered])
            matrix: Any = np.zeros((len(views), embedder.dim), dtype=np.float32)
            for source, group_vectors in zip(ordered, vectors, strict=True):
                for index, vector in zip(groups[source], group_vectors, strict=True):
                    matrix[index] = vector
            matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
            report["micro_embed_s"] = round(time.perf_counter() - started, 1)
            write_atomic_rescue_artifact(
                args.micro_directory,
                matrix=matrix,
                views=[
                    {"chunk_id": chunk.chunk_id, "source": chunk.source, "parent_ordinal": chunk.ordinal, "view_ordinal": view_ordinal}
                    for chunk, view_ordinal, _ in views
                ],
                generation_id=args.generation,
                calibration_id=prod.calibration_id,
                pipeline_fingerprint=str(binding["pipeline_fingerprint"]),
                corpus_fingerprint=str(binding["corpus_fingerprint"]),
                embedding_profile=embedding_profile_id(embedder),
                embedding_fingerprint=embedding_profile(embedder).fingerprint(),
                ordinary_chunk_count=len(chunks),
                source_commit=os.environ.get("RECALL_SOURCE_COMMIT", "unknown"),
            )
        micro = load_atomic_rescue_artifact(manifest)
        micro.assert_lineage(
            generation_id=args.generation,
            calibration_id=prod.calibration_id,
            pipeline_fingerprint=str(binding["pipeline_fingerprint"]),
            corpus_fingerprint=str(binding["corpus_fingerprint"]),
            embedder=embedder,
        )
        report["artifacts"] = {
            name: {"views": artifact.view_count, "parents": artifact.parent_count, "load_ms": round(artifact.load_ms, 1), "matrix_mb": round(artifact.matrix.nbytes / 1e6, 1)}
            for name, artifact in (("prod", prod), ("micro", micro))
        }

        def loader(chunk_id: str, score: float) -> ScoredChunk | None:
            chunk = by_id.get(chunk_id)
            return ScoredChunk(Chunk(chunk.chunk_id, chunk.source, chunk.text, {}), score) if chunk else None

        ranks: dict[str, dict[str, list[int | None]]] = {arm: {"exact": [], "source": []} for arm in ARMS}
        rescue: dict[str, dict[str, int]] = {arm: {"gold": 0, "admitted": 0, "fallback": 0} for arm in ("prod", "micro")}
        latency: dict[str, list[float]] = {"prod": [], "micro": []}
        private_rows: list[dict[str, Any]] = []
        for probe in probes:
            vector = embed_query(embedder, probe["question"])
            dense = store.query_dense(vector, k=CANDIDATE_K)
            gold = set(probe["gold_chunk_ids"])
            row: dict[str, Any] = {"probe_id": probe["probe_id"]}
            for arm in ARMS:
                ranked = [hit.chunk.id for hit in dense]
                if arm != "off":
                    artifact = prod if arm == "prod" else micro
                    started = time.perf_counter()
                    try:
                        inserted = insert_atomic_rescue_dense(artifact, vector, dense, loader)
                        latency[arm].append((time.perf_counter() - started) * 1000.0)
                        ranked = [hit.chunk.id for hit in inserted]
                        rescue[arm]["admitted"] += 1
                        rescue[arm]["gold"] += int(inserted[5].chunk.id in gold)
                    except (AtomicRescueArtifactError, AtomicRescueSelectionError):
                        rescue[arm]["fallback"] += 1
                exact = _first(ranked, gold)
                source_rank = _first(
                    [by_id[item].source if item in by_id else "" for item in ranked], [probe["source"]]
                )
                ranks[arm]["exact"].append(exact)
                ranks[arm]["source"].append(source_rank)
                row[arm] = {"exact": exact, "source": source_rank}
            private_rows.append(row)
        report["active_generation_unchanged"] = store.active_generation_id() == args.generation

    report["queries"] = len(probes)
    report["arms"] = {
        arm: {f"{kind}@{cutoff}": sum(1 for r in ranks[arm][kind] if r is not None and r <= cutoff) for kind in ("exact", "source") for cutoff in CUTOFFS}
        for arm in ARMS
    }
    report["paired"] = {
        f"{b}_vs_{a}": {f"{kind}@{cutoff}": paired(ranks[a][kind], ranks[b][kind], cutoff) for kind in ("exact", "source") for cutoff in (1, 6)}
        for a, b in (("off", "prod"), ("off", "micro"), ("prod", "micro"))
    }
    report["rescue"] = rescue
    report["selector_ms_p95"] = {
        arm: round(sorted(values)[max(0, int(0.95 * len(values)) - 1)], 2) if values else None
        for arm, values in latency.items()
    }
    args.rows.write_text("\n".join(json.dumps(row) for row in private_rows) + "\n", encoding="utf-8", newline="\n")
    os.chmod(args.rows, 0o600)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=os.environ.get("RECALL_DSN"))
    parser.add_argument("--tenant", default="memory")
    parser.add_argument("--generation")
    parser.add_argument("--embedder", default="voyage-context:voyage-context-4")
    sub = parser.add_subparsers(dest="command", required=True)
    spans = sub.add_parser("spans", help="VPS2: sample production-disjoint spans with context")
    spans.add_argument("--evals-root", type=Path, required=True)
    spans.add_argument("--out", type=Path, required=True)
    write = sub.add_parser("write", help="VPS3: turn spans into questions with the writer model")
    write.add_argument("--spans", type=Path, required=True)
    write.add_argument("--out", type=Path, required=True)
    write.add_argument("--budget-usd", type=float, default=PROBE_BUDGET_USD)
    run = sub.add_parser("evaluate")
    run.add_argument("--probes", type=Path, required=True)
    run.add_argument("--prod-artifact", type=Path, required=True)
    run.add_argument("--micro-directory", type=Path, required=True)
    run.add_argument("--split", choices=("dev", "confirm"), required=True)
    run.add_argument("--rows", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "write":
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY is required")
        spans_in = [json.loads(line) for line in args.spans.read_text(encoding="utf-8").splitlines()]
        summary = generate_memory_probes(
            spans_in,
            args.out,
            call=lambda payload: common._openrouter(payload, api_key),
            budget_usd=args.budget_usd,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if not args.dsn or not args.generation:
        raise SystemExit("--dsn (or RECALL_DSN) and --generation are required")
    if args.command == "spans":
        from recall.generation_store import GenerationStore

        records = []
        for path in sorted(args.evals_root.glob("atomic-fact-*/*.json")):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        excluded = excluded_sources(records)
        with GenerationStore(args.dsn, 1024, tenant=args.tenant) as raw_store:
            store: Any = raw_store
            store.set_fixed_generation(args.generation)
            chunks = load_chunks(store)
        rows = span_records(chunks, excluded)
        args.out.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
            newline="\n",
        )
        os.chmod(args.out, 0o600)
        summary = {
            "spans": len(rows),
            "excluded_sources": len(excluded),
            "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        }
    else:
        summary = evaluate(args)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
