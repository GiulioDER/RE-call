"""Build the preregistered 96 source atomic fact release confirmation pool."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Protocol, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.llm import OpenRouterLLM  # noqa: E402
from scripts.build_atomic_fact_blind_source_census import (  # noqa: E402
    MAX_TOKENS,
    restricted_ntfs_acl,
)
from scripts.build_atomic_fact_blind_source_holdout import (  # noqa: E402
    MODEL,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    _normalize,
    _sha256,
    _source_path,
    _source_root,
    _text_sha256,
    _write_private,
    _write_public,
    build_source_views,
    validate_question,
    writer_user_prompt,
)
from scripts.inventory_atomic_fact_release_confirmation import (  # noqa: E402
    _restrict_private_acl,
    _snapshot_sources,
    manifest_sha256,
)


PROTOCOL = "2026-09-16-atomic-fact-release-confirmation-construction"
TARGET_ROWS = 96
MAX_ATTEMPTS = 160
EXPECTED_INVENTORY_SHA256 = (
    "a5df11069bce5eb4041e87ca5ebf3e3b1d8b35fb3b1814b61cd1725e46eba768"
)
EXPECTED_CANDIDATE_MANIFEST_SHA256 = (
    "fdd00ba7d356b4b07d376c5b2d87b146c2f58c45bf35ea55b17d85e21fcf8e62"
)
EXPECTED_SOURCE_SNAPSHOT_SHA256 = (
    "12885c6ea1cde2f81758d3cc247b77e3127ba10484f1c257289f9d3946b7722d"
)


class Writer(Protocol):
    def complete(self, system: str, user: str) -> str: ...


def load_inventory(
    path: Path, roots: Sequence[tuple[str, Path]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    digest = _sha256(path)
    if digest != EXPECTED_INVENTORY_SHA256:
        raise ValueError(f"INVENTORY_HASH_MISMATCH:{digest}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("INVENTORY_SHAPE_MISMATCH")
    rows = payload.get("candidates")
    if not isinstance(rows, list):
        raise ValueError("INVENTORY_CANDIDATES_MISSING")
    if manifest_sha256(rows) != EXPECTED_CANDIDATE_MANIFEST_SHA256:
        raise ValueError("CANDIDATE_MANIFEST_MISMATCH")
    _, snapshot_sha256 = _snapshot_sources(roots)
    if (
        payload.get("source_snapshot_sha256") != EXPECTED_SOURCE_SNAPSHOT_SHA256
        or snapshot_sha256 != EXPECTED_SOURCE_SNAPSHOT_SHA256
    ):
        raise ValueError(f"SOURCE_SNAPSHOT_MISMATCH:{snapshot_sha256}")
    return payload, hydrate_candidates(rows, roots)


def hydrate_candidates(
    rows: Sequence[Mapping[str, Any]], roots: Sequence[tuple[str, Path]]
) -> list[dict[str, Any]]:
    root_map = dict(roots)
    hydrated: list[dict[str, Any]] = []
    for row in rows:
        source = str(row["source"])
        path = _source_path(source, root_map)
        if path is None or not path.is_file() or _sha256(path) != row["source_sha256"]:
            raise ValueError(f"CANDIDATE_SOURCE_MISMATCH:{source}")
        _, views = build_source_views(path.read_text(encoding="utf-8", errors="replace"), source)
        matches = [
            view
            for view in views
            if int(view["parent_ordinal"]) == int(row["gold_ordinal"])
            and _text_sha256(str(view["content"])) == row["answer_span_sha256"]
            and str(view["construction"]) == row["construction"]
        ]
        if len(matches) != 1:
            raise ValueError(f"CANDIDATE_VIEW_MISMATCH:{source}:{len(matches)}")
        hydrated.append({**dict(row), "path": path, "view": matches[0]})
    return hydrated


def generate_confirmation_rows(
    candidates: Sequence[Mapping[str, Any]],
    writer: Writer,
    *,
    target_rows: int = TARGET_ROWS,
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    accepted: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    rejections: Counter[str] = Counter()
    attempts = 0
    for candidate in candidates[:max_attempts]:
        attempts += 1
        view = candidate["view"]
        content = str(view["content"])
        response = writer.complete(SYSTEM_PROMPT, writer_user_prompt(content))
        question, rejection = validate_question(response, content)
        if rejection is not None or question is None:
            rejections[rejection or "unknown"] += 1
            continue
        normalized_question = _normalize(question)
        if normalized_question in seen_questions:
            rejections["duplicate_question"] += 1
            continue
        seen_questions.add(normalized_question)
        accepted.append(
            {
                "id": f"atomic-release-confirmation-{len(accepted) + 1:03d}",
                "query": question,
                "expected_answerability": "answerable",
                "gold_sources": [str(candidate["source"])],
                "source_sha256": str(candidate["source_sha256"]),
                "gold_ordinal": int(candidate["gold_ordinal"]),
                "answer_span": content,
                "answer_span_sha256": str(candidate["answer_span_sha256"]),
                "construction": str(candidate["construction"]),
                "candidate_order": str(candidate["order"]),
            }
        )
        if len(accepted) == target_rows:
            break
    return accepted, rejections, attempts


def audit_pool(
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    roots: Sequence[tuple[str, Path]],
    rejections: Mapping[str, int],
) -> dict[str, Any]:
    candidate_by_source = {str(row["source"]): row for row in candidates}
    root_counts = Counter(str(row["gold_sources"][0]).partition("/")[0] for row in rows)
    construction_counts = Counter(str(row["construction"]) for row in rows)
    integrity_errors = 0
    validation_errors = 0
    seen_questions: set[str] = set()
    for row in rows:
        source = str(row["gold_sources"][0])
        candidate = candidate_by_source.get(source)
        if candidate is None:
            integrity_errors += 1
            continue
        expected = {
            "source_sha256": candidate["source_sha256"],
            "gold_ordinal": candidate["gold_ordinal"],
            "answer_span_sha256": candidate["answer_span_sha256"],
            "construction": candidate["construction"],
            "candidate_order": candidate["order"],
        }
        if any(row.get(key) != value for key, value in expected.items()):
            integrity_errors += 1
        path = _source_path(source, dict(roots))
        if path is None or not path.is_file() or _sha256(path) != row["source_sha256"]:
            integrity_errors += 1
        content = str(row["answer_span"])
        if _text_sha256(content) != row["answer_span_sha256"]:
            integrity_errors += 1
        question, rejection = validate_question(
            json.dumps({"question": row["query"]}), content
        )
        normalized = _normalize(str(row["query"]))
        if rejection is not None or question is None or normalized in seen_questions:
            validation_errors += 1
        seen_questions.add(normalized)

    selected_sources = [str(row["gold_sources"][0]) for row in rows]
    checks = {
        "exactly_96_distinct_sources": len(rows) == TARGET_ROWS
        and len(set(selected_sources)) == TARGET_ROWS,
        "source_and_answer_integrity": integrity_errors == 0,
        "question_validation": validation_errors == 0,
        "root_distribution": len(root_counts) >= 3
        and bool(rows)
        and max(root_counts.values(), default=0) / len(rows) <= 0.80,
    }
    return {
        "checks": checks,
        "decision": (
            "READY_TO_PREREGISTER_ATOMIC_RELEASE_RETRIEVAL"
            if all(checks.values())
            else "STOP_ATOMIC_RELEASE_CONFIRMATION_CONSTRUCTION"
        ),
        "accepted_rows": len(rows),
        "distinct_sources": len(set(selected_sources)),
        "root_counts": dict(sorted(root_counts.items())),
        "construction_counts": dict(sorted(construction_counts.items())),
        "integrity_errors": integrity_errors,
        "validation_errors": validation_errors,
        "rejections": dict(sorted(rejections.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", type=_source_root, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required")
    try:
        args.private_output.resolve().relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise RuntimeError("private output must remain outside the repository")

    inventory, candidates = load_inventory(args.inventory, args.source_root)
    writer = OpenRouterLLM(
        model=MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0.0,
        max_tokens=MAX_TOKENS,
    )
    rows, rejections, attempts = generate_confirmation_rows(candidates, writer)
    provider = asdict(writer.provider_metadata())
    provider["calls"] = writer.usage()["calls"]
    pool = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system_prompt": SYSTEM_PROMPT,
        "user_template": USER_TEMPLATE,
        "inventory_sha256": _sha256(args.inventory),
        "candidate_manifest_sha256": manifest_sha256(inventory["candidates"]),
        "source_snapshot_sha256": inventory["source_snapshot_sha256"],
        "attempted_sources": attempts,
        "provider": provider,
        "queries": rows,
    }
    audit = audit_pool(rows, candidates, args.source_root, rejections)
    _write_private(args.private_output, pool)
    _restrict_private_acl(args.private_output)
    acl_restricted = restricted_ntfs_acl(args.private_output)
    audit["checks"]["attempt_ceiling"] = attempts <= MAX_ATTEMPTS
    audit["checks"]["provider_call_accounting"] = attempts == int(provider.get("calls", 0))
    audit["checks"]["restricted_ntfs_acl"] = acl_restricted
    audit["decision"] = (
        "READY_TO_PREREGISTER_ATOMIC_RELEASE_RETRIEVAL"
        if all(audit["checks"].values())
        else "STOP_ATOMIC_RELEASE_CONFIRMATION_CONSTRUCTION"
    )
    public = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "pool_sha256": _sha256(args.private_output),
        "inventory_sha256": pool["inventory_sha256"],
        "candidate_manifest_sha256": pool["candidate_manifest_sha256"],
        "source_snapshot_sha256": pool["source_snapshot_sha256"],
        "system_prompt_sha256": _text_sha256(SYSTEM_PROMPT),
        "user_template_sha256": _text_sha256(USER_TEMPLATE),
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "attempted_sources": attempts,
        "provider": provider,
        "restricted_ntfs_acl": acl_restricted,
        **audit,
    }
    _write_public(args.public_output, public)
    print(json.dumps(public, ensure_ascii=False))
    if public["decision"] != "READY_TO_PREREGISTER_ATOMIC_RELEASE_RETRIEVAL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
