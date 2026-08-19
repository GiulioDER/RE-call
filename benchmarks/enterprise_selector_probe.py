"""Measure real document selection with retrieval coverage held constant by gold injection."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.enterprise_rag import load_questions
from recall.calibration import Calibration
from recall.embeddings import embedding_profile_id, resolve_embedder
from recall.evidence import AnswerSlot, EvidencePolicy, build_evidence_bundle
from recall.store import PgVectorStore
from recall.trust_policy import TrustPolicy

from benchmarks._trust import bench_search
from recall.types import Chunk, Provenance, TrustedHit, Validity

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / ".benchdata" / "enterprise-rag-v1.0.0"
LABELS = ROOT / "benchmarks" / "enterprise_real_answer_slots.jsonl"
QUESTION_IDS = (
    "qst_0303", "qst_0304", "qst_0309", "qst_0310", "qst_0311", "qst_0312",
    "qst_0313", "qst_0316", "qst_0318", "qst_0320", "qst_0322", "qst_0323",
    "qst_0324", "qst_0325", "qst_0326", "qst_0327", "qst_0328", "qst_0330",
    "qst_0331", "qst_0337", "qst_0338", "qst_0340",
)
ARMS = (
    "current_retrieval", "document_grouping", "gold_document_grouping",
    "gold_answer_slots", "gold_bundle_beam",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.enterprise_selector_probe")
    parser.add_argument("--questions", type=Path, default=DEFAULT_DATA / "questions.jsonl")
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dsn", default=os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall"))
    parser.add_argument("--table", default="bench_five_arm_enterprise")
    parser.add_argument("--tenant", default="five-arm-enterprise")
    parser.add_argument("--embedder", default="hashing")
    parser.add_argument("--threshold", type=float, default=0.219)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--allow-development", action="store_true")
    return parser


def _load_labels(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(payload["question_id"]): payload
        for payload in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    }


def _slots(payload: dict[str, Any]) -> tuple[AnswerSlot, ...]:
    return tuple(
        AnswerSlot(str(slot["name"]), tuple(str(term) for term in slot["terms"]), int(slot.get("min_matches", 1)))
        for slot in payload.get("answer_slots", [])
    )


def _term_matches(text: str, term: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(term.strip().casefold())}(?!\w)", text.casefold()) is not None


def _slots_covered(slots: tuple[AnswerSlot, ...], texts: list[str]) -> bool:
    return all(
        any(sum(_term_matches(text, term) for term in slot.terms) >= slot.min_matches for text in texts)
        for slot in slots
    )


def _gold_hit(chunk: Chunk, now: datetime) -> TrustedHit:
    return TrustedHit(
        chunk=chunk,
        cosine=1.0,
        confidence=1.0,
        verdict="ok",
        provenance=Provenance(
            source=chunk.source,
            file=str(chunk.metadata.get("file", chunk.source)),
            ord=chunk.metadata.get("ord") if isinstance(chunk.metadata.get("ord"), int) else None,
            indexed_at=now,
        ),
        validity=Validity(valid_from=None, valid_until=None, superseded_by=None),
    )


def _with_gold(result: Any, chunks: list[Chunk], now: datetime) -> Any:
    seen = {hit.chunk.id for hit in result.hits}
    hits = list(result.hits)
    hits.extend(_gold_hit(chunk, now) for chunk in chunks if chunk.id not in seen)
    return replace(result, hits=hits, abstained=False, reason="")


def _run_arm(
    result: Any,
    slots: tuple[AnswerSlot, ...],
    expected_docs: set[str],
    arm: str,
    k: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    policy = EvidencePolicy(
        max_items=k,
        bundle_mode="document" if arm != "current_retrieval" else "retrieval",
        max_documents=2,
        answer_slots=slots if arm in {"gold_answer_slots", "gold_bundle_beam"} else (),
        selection_mode="beam" if arm == "gold_bundle_beam" else "prefix",
        beam_width=8,
    )
    bundle = build_evidence_bundle(result, policy)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    texts = [item.text for item in bundle.items]
    selected_docs = {item.source for item in bundle.items}
    return {
        "arm": arm,
        "decision": bundle.decision,
        "reason_code": bundle.reason_code,
        "selected_chunk_ids": sorted(item.chunk_id for item in bundle.items),
        "selected_doc_ids": sorted(selected_docs),
        "complete_slots": _slots_covered(slots, texts),
        "selected_non_gold": len(selected_docs - expected_docs),
        "elapsed_ms": elapsed_ms,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        timings = sorted(float(row["elapsed_ms"]) for row in subset)
        result[arm] = {
            "questions": len(subset),
            "complete_slot_recall": sum(bool(row["complete_slots"]) for row in subset),
            "non_gold_chunks": sum(int(row["selected_non_gold"]) for row in subset),
            "mean_ms": sum(timings) / len(timings),
            "p95_ms": timings[max(0, (95 * len(timings) + 99) // 100 - 1)],
        }
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.allow_development:
        raise SystemExit("gold conditioned diagnostic requires --allow-development")
    labels = _load_labels(args.labels)
    questions = {question.question_id: question for question in load_questions(args.questions)}
    missing = [qid for qid in QUESTION_IDS if qid not in questions or qid not in labels]
    if missing:
        raise SystemExit(f"missing fixed questions or labels: {', '.join(missing)}")
    embedder = resolve_embedder(args.embedder)
    calibration = Calibration(embedder=embedding_profile_id(embedder), threshold=args.threshold, scale=0.05)
    now = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    with PgVectorStore(args.dsn, dim=embedder.dim, table=args.table, tenant=args.tenant) as store:
        chunks_by_source: dict[str, list[Chunk]] = {}
        for chunk in store.iter_chunks():
            chunks_by_source.setdefault(chunk.source, []).append(chunk)
        for question_id in QUESTION_IDS:
            question = questions[question_id]
            slots = _slots(labels[question_id])
            expected_docs = {str(value) for value in question.raw.get("expected_doc_ids", [])}
            gold_chunks = [chunk for source in expected_docs for chunk in chunks_by_source.get(source, [])]
            if not gold_chunks:
                raise SystemExit(f"no indexed gold chunks for {question_id}")
            if not _slots_covered(slots, [chunk.text for chunk in gold_chunks]):
                raise SystemExit(f"label coverage audit failed for {question_id}")
            retrieved = bench_search(
                store, embedder, question.question, k=8, candidate_k=args.candidate_k,
                calibration=calibration, policy=TrustPolicy.development(),
            )
            gold_result = _with_gold(retrieved, gold_chunks, now)
            arm_results = {
                "current_retrieval": retrieved,
                "document_grouping": retrieved,
                "gold_document_grouping": gold_result,
                "gold_answer_slots": gold_result,
                "gold_bundle_beam": gold_result,
            }
            for arm in ARMS:
                row = _run_arm(arm_results[arm], slots, expected_docs, arm, 8)
                row["question_id"] = question_id
                rows.append(row)
    output = {
        "benchmark": "enterprise_selector_probe",
        "questions": list(QUESTION_IDS),
        "labels": str(args.labels),
        "retrieval": {"embedder": embedding_profile_id(embedder), "candidate_k": args.candidate_k, "threshold": args.threshold},
        "trust": {"mode": "development", "calibrated": False, "reason": "gold conditioned diagnostic uses uncertified calibration"},
        "summary": _summary(rows),
        "details": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "summary": output["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
