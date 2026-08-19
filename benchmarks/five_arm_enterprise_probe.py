"""Run the five bundle arms against real EnterpriseRAG documents and labels.

This runner intentionally requires ``--allow-development`` because the local benchmark calibration
is not a production certified artifact. Its results measure retrieval and selection on real data,
but they are not a launch approval.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from benchmarks.enterprise_rag import (
    EnterpriseDoc,
    _doc_from_text_file,
    index_documents,
    load_documents,
    load_questions,
)
from recall.calibration import Calibration
from recall.embeddings import embedding_profile_id, resolve_embedder
from recall.evidence import AnswerSlot, EvidencePolicy, build_evidence_bundle
from recall.retriever import (
    DocumentExpansionPolicy,
    StructuralExpansionPolicy,
)
from recall.store import PgVectorStore
from recall.trust import trusted_search
from recall.trust_policy import TrustPolicy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / ".benchdata" / "enterprise-rag-v1.0.0"
ARMS = ("current_retrieval", "document_grouping", "structural_expansion", "answer_slots", "bundle_beam")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.five_arm_enterprise_probe")
    parser.add_argument("--documents", type=Path, default=DEFAULT_DATA / "calibration_20_docs.zip")
    parser.add_argument("--questions", type=Path, default=DEFAULT_DATA / "questions.jsonl")
    parser.add_argument("--labels", type=Path, help="optional JSONL slot and forbidden chunk labels")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dsn", default=os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall"))
    parser.add_argument("--table", default="bench_five_arm_enterprise")
    parser.add_argument("--tenant", default="five-arm-enterprise")
    parser.add_argument("--embedder", default="hashing")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--limit-questions", type=int)
    parser.add_argument("--reset-index", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument(
        "--gold-doc-filter",
        action="store_true",
        help="when reading a zip archive, index only document ids referenced by the questions",
    )
    parser.add_argument("--allow-development", action="store_true")
    return parser


def _load_labels(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    labels: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        question_id = str(payload.get("question_id", "")).strip()
        if not question_id:
            raise ValueError(f"{path}:{line_number}: question_id is required")
        if question_id in labels:
            raise ValueError(f"{path}:{line_number}: duplicate question_id {question_id}")
        labels[question_id] = payload
    return labels


def _load_documents(
    path: Path, questions: list[Any], gold_doc_filter: bool
) -> Iterator[EnterpriseDoc]:
    if not gold_doc_filter:
        yield from load_documents([path])
        return
    expected_ids = _expected_doc_ids(questions)
    if path.suffix.lower() != ".zip":
        yield from load_documents([path])
        return
    with ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".txt"):
                continue
            match = re.search(r"(dsid_[A-Za-z0-9]+)", Path(name).name)
            if match is None or match.group(1) not in expected_ids:
                continue
            content = archive.read(name).decode("utf-8", errors="replace")
            yield _doc_from_text_file(name, content)


def _expected_doc_ids(questions: list[Any]) -> set[str]:
    return {
        str(doc_id)
        for question in questions
        for doc_id in question.raw.get("expected_doc_ids", [])
    }


def _slots(payload: dict[str, Any]) -> tuple[AnswerSlot, ...]:
    return tuple(
        AnswerSlot(
            str(slot["name"]),
            tuple(str(term) for term in slot["terms"]),
            int(slot.get("min_matches", 1)),
        )
        for slot in payload.get("answer_slots", [])
    )


def _run_case(
    store: PgVectorStore,
    embedder: Any,
    question: Any,
    labels: dict[str, Any] | None,
    calibration: Calibration,
    arm: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    slots = _slots(labels or {})
    if arm in {"answer_slots", "bundle_beam"} and not slots:
        return {"question_id": question.question_id, "arm": arm, "measured": False}

    started = time.perf_counter()
    expansion = None
    structural = None
    if arm == "structural_expansion":
        structural = StructuralExpansionPolicy(
            enabled=True, max_sources=2, chunks_per_source=8, radius=2
        )
    elif arm in {"answer_slots", "bundle_beam"}:
        expansion = DocumentExpansionPolicy(enabled=True, max_sources=2, chunks_per_source=8)

    result = trusted_search(
        store,
        embedder,
        question.question,
        k=args.k,
        candidate_k=args.candidate_k,
        calibration=calibration,
        policy=TrustPolicy.development(),
        document_expansion=expansion,
        structural_expansion=structural,
    )
    policy = EvidencePolicy(
        max_items=args.k,
        bundle_mode="document" if arm != "current_retrieval" else "retrieval",
        max_documents=2,
        answer_slots=slots if arm in {"answer_slots", "bundle_beam"} else (),
        selection_mode="beam" if arm == "bundle_beam" else "prefix",
        beam_width=8,
    )
    bundle = build_evidence_bundle(result, policy)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    selected_docs = {item.source for item in bundle.items}
    expected_docs = {str(value) for value in question.raw.get("expected_doc_ids", [])}
    required_chunks = {str(value) for value in (labels or {}).get("required_chunk_ids", [])}
    selected_chunks = {item.chunk_id for item in bundle.items}
    forbidden = {str(value) for value in (labels or {}).get("forbidden_chunk_ids", [])}
    return {
        "question_id": question.question_id,
        "arm": arm,
        "measured": True,
        "decision": bundle.decision,
        "reason_code": bundle.reason_code,
        "trust_state": result.trust_state,
        "selected_doc_ids": sorted(selected_docs),
        "selected_chunk_ids": sorted(selected_chunks),
        "expected_doc_ids": sorted(expected_docs),
        "complete_documents": expected_docs <= selected_docs if expected_docs else True,
        "required_chunk_ids": sorted(required_chunks),
        "complete_chunks": required_chunks <= selected_chunks if required_chunks else None,
        "complete_slots": (
            bundle.decision == "answer" if arm in {"answer_slots", "bundle_beam"} else None
        ),
        "forbidden_selected": len(forbidden & selected_chunks),
        "false_positive": not expected_docs and bundle.decision == "answer",
        "elapsed_ms": elapsed_ms,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for arm in ARMS:
        subset = [row for row in rows if row["arm"] == arm and row["measured"]]
        timings = [float(row["elapsed_ms"]) for row in subset]
        if not subset:
            report[arm] = {"measured": False}
            continue
        report[arm] = {
            "measured_questions": len(subset),
            "complete_document_recall": sum(bool(row["complete_documents"]) for row in subset),
            "complete_chunk_recall": sum(bool(row["complete_chunks"]) for row in subset if row["complete_chunks"] is not None),
            "complete_slot_recall": sum(bool(row["complete_slots"]) for row in subset if row["complete_slots"] is not None),
            "forbidden_selected": sum(int(row["forbidden_selected"]) for row in subset),
            "false_positives": sum(bool(row["false_positive"]) for row in subset),
            "trust_states": dict(Counter(str(row["trust_state"]) for row in subset)),
            "mean_ms": sum(timings) / len(timings),
            "p95_ms": sorted(timings)[max(0, (95 * len(timings) + 99) // 100 - 1)],
        }
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.allow_development:
        raise SystemExit(
            "real benchmark requires --allow-development because its explicit calibration is not "
            "a production certified artifact"
        )
    if args.k < 1 or args.candidate_k < 1:
        raise SystemExit("--k and --candidate-k must be positive")
    labels = _load_labels(args.labels)
    questions = load_questions(args.questions, limit=args.limit_questions)
    embedder = resolve_embedder(args.embedder)
    calibration = Calibration(
        embedder=embedding_profile_id(embedder), threshold=args.threshold, scale=0.05
    )
    stats: dict[str, Any] = {}
    with PgVectorStore(args.dsn, dim=embedder.dim, table=args.table, tenant=args.tenant) as store:
        store.ensure_schema()
        if not args.skip_index:
            stats = index_documents(
                store,
                embedder,
                _load_documents(args.documents, questions, args.gold_doc_filter),
                batch_chunks=256,
                chunk_chars=800,
                chunk_overlap=80,
                reset=args.reset_index,
            )
        if args.gold_doc_filter:
            expected_ids = _expected_doc_ids(questions)
            indexed_ids = {chunk.source for chunk in store.iter_chunks()}
            missing_ids = sorted(expected_ids - indexed_ids)
            if missing_ids:
                raise SystemExit(
                    "gold filtered index is incomplete: "
                    f"missing {len(missing_ids)} of {len(expected_ids)} expected sources; "
                    "rerun with --reset-index"
                )
        rows = [
            _run_case(store, embedder, question, labels.get(question.question_id), calibration, arm, args)
            for question in questions
            for arm in ARMS
        ]
    output = {
        "benchmark": "five_arm_enterprise_probe",
        "data": {"documents": str(args.documents), "questions": str(args.questions), "labels": str(args.labels) if args.labels else None},
        "index": {"table": args.table, "tenant": args.tenant, **stats},
        "retrieval": {"embedder": embedding_profile_id(embedder), "k": args.k, "candidate_k": args.candidate_k, "threshold": args.threshold},
        "trust": {"mode": "development", "calibrated": False, "reason": "explicit benchmark calibration is not certified or generation bound"},
        "summary": _summary(rows),
        "details": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "summary": output["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
