"""Build the untouched exact-span pool for query-anchor admission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_guarded_spare_slot_extractive_pool import (  # noqa: E402
    SKIP_NAMES,
    _extract,
    _pool_sources,
    _root,
    _sha256,
    _text_sha256,
    _trace_sources,
)


SYLLABLES = (
    "ba",
    "ce",
    "di",
    "fo",
    "gu",
    "ha",
    "ji",
    "ko",
    "lu",
    "me",
    "na",
    "pi",
    "ro",
    "su",
    "ta",
    "ve",
)


def _nonce(seed: str, index: int, source_sha256: str) -> str:
    digest = hashlib.sha256(f"{seed}\0{index}\0{source_sha256}".encode()).digest()
    return "".join(SYLLABLES[value & 15] for value in digest[:6])


def _candidate_sources(roots: list[tuple[str, Path]]) -> str:
    values: list[str] = []
    for _, root in roots:
        for path in root.rglob("*.md"):
            if path.name not in SKIP_NAMES:
                values.append(path.read_text(encoding="utf-8", errors="replace").casefold())
    return "\n".join(values)


def build_pool(
    roots: list[tuple[str, Path]],
    trace_path: Path,
    old_pool_path: Path,
    consumed_pool_path: Path,
    *,
    count: int,
    seed: str,
) -> dict[str, Any]:
    excluded = _trace_sources(trace_path) | _pool_sources(old_pool_path)
    consumed_sources = _pool_sources(consumed_pool_path)
    excluded |= consumed_sources
    candidates: list[dict[str, Any]] = []
    corpus = _candidate_sources(roots)
    for prefix, root in roots:
        for path in root.rglob("*.md"):
            if path.name in SKIP_NAMES:
                continue
            source = f"{prefix}/{path.relative_to(root).as_posix()}"
            if source in excluded:
                continue
            item = _extract(path, source)
            if item is None:
                continue
            item["order"] = _text_sha256(f"{seed}\0{source}")
            candidates.append(item)
    candidates.sort(key=lambda item: (item["order"], item["source"]))

    rows: list[dict[str, Any]] = []
    used_questions: set[str] = set()
    used_nonces: set[str] = set()
    for item in candidates:
        if len(rows) == count * 2:
            break
        question = str(item["question"])
        normalized_question = question.casefold()
        if normalized_question in used_questions:
            continue
        index = len(rows) // 2 + 1
        nonce = _nonce(seed, index, str(item["source_sha256"]))
        if nonce in used_nonces or nonce.casefold() in corpus:
            raise ValueError("generated alphabetic nonce is not unique and corpus absent")
        used_nonces.add(nonce)
        control = f"According to the {nonce} revision, {question[0].lower()}{question[1:]}"
        if control.casefold() in used_questions:
            raise ValueError("control query construction produced a duplicate")
        used_questions.add(normalized_question)
        used_questions.add(control.casefold())
        rows.extend(
            [
                {
                    "id": f"anchor-positive-{index:03d}",
                    "query": question,
                    "expected_answerability": "answerable",
                    "gold_sources": [item["source"]],
                    "source_sha256": item["source_sha256"],
                    "gold_ordinal": item["source_ordinal"],
                    "answer_span": item["answer_span"],
                    "answer_span_sha256": item["answer_span_sha256"],
                    "construction": item["construction"],
                },
                {
                    "id": f"anchor-control-{index:03d}",
                    "query": control,
                    "expected_answerability": "unanswerable",
                    "gold_sources": [],
                    "source_sha256": None,
                    "gold_ordinal": None,
                    "answer_span": "NOT_FOUND",
                    "answer_span_sha256": _text_sha256("NOT_FOUND"),
                    "construction": "matched_absent_alphabetic_anchor_control",
                },
            ]
        )
    if len(rows) != count * 2:
        raise ValueError(f"INSUFFICIENT_POOL: unique questions left {len(rows) // 2} pairs")
    rows.sort(key=lambda item: (_text_sha256(f"{seed}\0{item['id']}"), item["id"]))
    return {
        "schema_version": 1,
        "protocol": "2026-09-14-query-anchor-spare-slot-pool",
        "seed": seed,
        "trace_sha256": _sha256(trace_path),
        "old_pool_sha256": _sha256(old_pool_path),
        "consumed_pool_sha256": _sha256(consumed_pool_path),
        "excluded_sources": len(excluded),
        "eligible_sources": len(candidates),
        "answerable_queries": count,
        "unanswerable_queries": count,
        "queries": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", type=_root, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--old-pool", type=Path, required=True)
    parser.add_argument("--consumed-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--seed", default="query-anchor-spare-slot-v1")
    args = parser.parse_args()
    payload = build_pool(
        args.source_root,
        args.trace,
        args.old_pool,
        args.consumed_pool,
        count=args.count,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "answerable_queries": payload["answerable_queries"],
                "unanswerable_queries": payload["unanswerable_queries"],
                "eligible_sources": payload["eligible_sources"],
                "sha256": _sha256(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
