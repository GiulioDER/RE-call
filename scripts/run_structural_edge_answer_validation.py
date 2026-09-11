"""Replay measured structural-edge contexts through the configured answer provider.

This runner deliberately measures the answer boundary, not retrieval. It consumes one immutable
structural-edge retrieval artifact, builds the same evidence bundle and prompt used by production,
and records the raw envelope, structural validation, provider metadata, and citation coverage.
Factual correctness remains a separate judge-dependent measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recall.answer_provider import resolve_answer_provider  # noqa: E402
from recall.evidence import (  # noqa: E402
    AnswerEnvelope,
    EvidenceBundle,
    EvidenceItem,
    normalize_citations,
    parse_answer_envelope,
    render_evidence_prompt,
    validate_answer,
)


ARMS = (
    "baseline",
    "semantic_order_more_direct",
    "semantic_order_most_direct",
    "category_selective",
)


def _bundle(row: dict[str, Any]) -> EvidenceBundle:
    items: list[EvidenceItem] = []
    for context in row.get("context") or []:
        reference = str(context.get("id") or "")
        source, separator, ordinal_text = reference.partition(":")
        ordinal = int(ordinal_text) if separator and ordinal_text.isdigit() else None
        items.append(
            EvidenceItem(
                chunk_id=str(context["chunk_id"]),
                text=str(context.get("text") or ""),
                source=source,
                ordinal=ordinal,
                indexed_at=None,
                valid_from=None,
                valid_until=None,
                cosine=0.0,
                confidence=1.0,
            )
        )
    supported = bool(items)
    return EvidenceBundle(
        query=str(row["question"]),
        decision="answer" if supported else "abstain",
        reason_code=None if supported else "empty_context",
        decision_state="supported" if supported else "corpus_gap",
        calibrated=True,
        stale=False,
        embedding_profile="structural-edge-evaluation",
        retrieval_profile="hybrid-top20-context10",
        index_generation="structural-edge-evaluation",
        items=tuple(items),
    )


def _citation_stats(row: dict[str, Any], envelope: AnswerEnvelope) -> dict[str, object]:
    by_chunk = {str(item["chunk_id"]): str(item.get("id") or "") for item in row.get("context") or []}
    gold = {str(value) for value in row.get("gold") or []}
    resolved = [by_chunk[citation] for citation in envelope.citations if citation in by_chunk]
    matched = sum(reference in gold for reference in resolved)
    return {
        "gold_ids": sorted(gold),
        "cited_evidence_ids": resolved,
        "cited_gold_count": matched,
        "cited_gold_recall": matched / len(gold) if gold else None,
    }


def _request_key(row: dict[str, Any]) -> str:
    bundle = _bundle(row)
    system, user = render_evidence_prompt(bundle)
    return hashlib.sha256((system + "\0" + user).encode("utf-8")).hexdigest()


def _answer_row(
    row: dict[str, Any],
    provider: Any,
    *,
    arm: str,
    source_commit: str | None,
) -> dict[str, object]:
    bundle = _bundle(row)
    system, user = render_evidence_prompt(bundle)
    record: dict[str, object] = {
        "id": str(row["id"]),
        "question": str(row["question"]),
        "category": row.get("category"),
        "arm": arm,
        "source_commit": source_commit,
        "context_items": len(bundle.items),
        "added_items": len(row.get("additions") or []),
        "provider_invoked": False,
        "provider_cache_hit": False,
        "answer": None,
        "citations": [],
        "insufficient_evidence": None,
        "answer_valid": False,
        "answer_errors": [],
        "citation_metrics": None,
        "raw_response": None,
        "provider": None,
    }
    if not bundle.items:
        envelope = AnswerEnvelope(None, (), True)
        validation = validate_answer(envelope, bundle)
        record.update(
            {
                "insufficient_evidence": True,
                "answer_valid": validation.valid,
                "answer_errors": list(validation.errors),
                "citation_metrics": _citation_stats(row, envelope),
            }
        )
        return record

    record["provider_invoked"] = True
    try:
        raw = provider(system, user)
        envelope = normalize_citations(parse_answer_envelope(raw))
        validation = validate_answer(envelope, bundle)
        metadata = provider.provider_metadata().to_dict()
        record.update(
            {
                "answer": envelope.answer,
                "citations": list(envelope.citations),
                "insufficient_evidence": envelope.insufficient_evidence,
                "answer_valid": validation.valid,
                "answer_errors": list(validation.errors),
                "citation_metrics": _citation_stats(row, envelope),
                "raw_response": raw,
                "provider": metadata,
            }
        )
    except Exception as exc:  # noqa: BLE001 - preserve the failure in the artifact
        record["answer_errors"] = [f"{type(exc).__name__}: {exc}"]
        try:
            record["provider"] = provider.provider_metadata().to_dict()
        except Exception:  # pragma: no cover - provider metadata is best effort on total failure
            pass
    return record


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("arms"), dict):
        raise ValueError("input must be a structural-edge result object with arms")
    missing = [arm for arm in ARMS if arm not in payload["arms"]]
    if missing:
        raise ValueError(f"input is missing arms: {', '.join(missing)}")
    return cast(dict[str, Any], payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="same first N questions per arm")
    parser.add_argument("--workers", type=int, default=8, help="bounded concurrent provider calls")
    parser.add_argument("--resume", action="store_true", help="reuse completed prompt groups from the checkpoint")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.workers < 1 or args.workers > 16:
        parser.error("--workers must be between 1 and 16")
    payload = _load(args.input)
    provider = resolve_answer_provider(os.environ)
    if provider is None:
        raise SystemExit("answer provider is disabled; set RECALL_REASONING_ANSWER_ENABLED=1")
    source_commit = payload.get("git_revision")
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        rows = payload["arms"][arm].get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"arm {arm} has no rows")
        selected = rows[: args.limit] if args.limit is not None else rows
        rows_by_arm[arm] = selected
    baseline_ids = [str(row["id"]) for row in rows_by_arm["baseline"]]
    for arm, rows in rows_by_arm.items():
        ids = [str(row["id"]) for row in rows]
        if ids != baseline_ids[: len(ids)]:
            raise ValueError(f"arm {arm} is not paired with the baseline question order")

    groups: dict[str, tuple[str, dict[str, Any]]] = {}
    for arm, rows in rows_by_arm.items():
        for row in rows:
            groups.setdefault(_request_key(row), (arm, row))
    checkpoint = args.output.with_suffix(args.output.suffix + ".partial.jsonl")
    completed: dict[str, dict[str, object]] = {}
    if args.resume and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            saved = json.loads(line)
            if saved.get("_type") == "row" and isinstance(saved.get("_prompt_key"), str):
                completed[saved["_prompt_key"]] = saved
    pending = [(key, arm, row) for key, (arm, row) in groups.items() if key not in completed]
    print(
        f"provider groups {len(groups)}, completed checkpoint {len(completed)}, pending {len(pending)}, "
        f"workers {args.workers}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _answer_row,
                row,
                provider,
                arm=arm,
                source_commit=source_commit,
            ): (key, arm, row)
            for key, arm, row in pending
        }
        with checkpoint.open("a", encoding="utf-8") as handle:
            for index, future in enumerate(as_completed(futures), start=1):
                key, arm, row = futures[future]
                result = future.result()
                result["_type"] = "row"
                result["_prompt_key"] = key
                completed[key] = result
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"completed {index}/{len(pending)} {arm} {row['id']}", flush=True)

    result_rows: list[dict[str, object]] = []
    for arm, rows in rows_by_arm.items():
        for row in rows:
            key = _request_key(row)
            source = dict(completed[key])
            source.pop("_type", None)
            source.pop("_prompt_key", None)
            source.update(
                {
                    "id": str(row["id"]),
                    "question": str(row["question"]),
                    "category": row.get("category"),
                    "arm": arm,
                    "source_commit": source_commit,
                    "added_items": len(row.get("additions") or []),
                    "provider_cache_hit": source.get("id") != str(row["id"])
                    or source.get("arm") != arm,
                }
            )
            result_rows.append(source)
    metadata = provider.provider_metadata().to_dict()
    artifact = {
        "artifact": "RE-call structural-edge answer-stage validation",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "source_commit": source_commit,
        "arms": list(ARMS),
        "questions_per_arm": len(rows_by_arm["baseline"]),
        "model": metadata.get("model_id"),
        "provider": "OpenRouter configured answer provider",
        "answer_prompt_digest": metadata.get("prompt_digest"),
        "rows": result_rows,
        "provider_requests": len(groups),
        "provider_cache_hits": sum(1 for row in result_rows if row["provider_cache_hit"]),
        "provider_summary": metadata,
        "judge": "not run; factual correctness requires a separate fixed judge",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(result_rows), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
