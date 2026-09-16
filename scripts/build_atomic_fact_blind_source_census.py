"""Build the preregistered exhaustive 45 source blind question census."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.llm import OpenRouterLLM  # noqa: E402
from scripts.build_atomic_fact_blind_source_holdout import (  # noqa: E402
    MODEL,
    SEED,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    Writer,
    _normalize,
    _sha256,
    _source_path,
    _source_root,
    _text_sha256,
    _write_private,
    _write_public,
    build_source_views,
    candidate_manifest,
    candidate_rows,
    load_exclusions,
    validate_question,
    writer_user_prompt,
)


PROTOCOL = "2026-09-16-atomic-fact-blind-source-census"
EXPECTED_CANDIDATES = 45
EXPECTED_MANIFEST_SHA256 = "edfd11a46216b4b10b3b063c65623456bfd75876c37667f43e57505c616d1b60"
MIN_ACCEPTED = 32


def generate_census_rows(
    candidates: Sequence[Mapping[str, Any]], writer: Writer
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    accepted: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    rejections: Counter[str] = Counter()
    attempts = 0
    for candidate in candidates:
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
        source = str(candidate["source"])
        path = Path(candidate["path"])
        accepted.append(
            {
                "id": f"atomic-fact-census-{len(accepted) + 1:03d}",
                "query": question,
                "expected_answerability": "answerable",
                "gold_sources": [source],
                "source_sha256": _sha256(path),
                "gold_ordinal": int(view["parent_ordinal"]),
                "answer_span": content,
                "answer_span_sha256": _text_sha256(content),
                "construction": str(view["construction"]),
            }
        )
    return accepted, rejections, attempts


def audit_census(
    pool: Mapping[str, Any],
    roots: Sequence[tuple[str, Path]],
    excluded: set[str],
    rejections: Mapping[str, int],
) -> dict[str, Any]:
    root_map = dict(roots)
    rows = list(pool.get("queries", []))
    sources = [str(row.get("gold_sources", [""])[0]) for row in rows]
    root_counts = Counter(source.partition("/")[0] for source in sources)
    integrity_errors = 0
    validation_errors = 0
    for row in rows:
        source = str(row.get("gold_sources", [""])[0])
        path = _source_path(source, root_map)
        if path is None or not path.is_file() or _sha256(path) != row.get("source_sha256"):
            integrity_errors += 1
            continue
        content = str(row.get("answer_span", ""))
        if _text_sha256(content) != row.get("answer_span_sha256"):
            integrity_errors += 1
            continue
        _, views = build_source_views(path.read_text(encoding="utf-8", errors="replace"), source)
        matches = [
            view
            for view in views
            if _normalize(str(view["content"])) == _normalize(content)
            and int(view["parent_ordinal"]) == int(row.get("gold_ordinal", -1))
        ]
        if len(matches) != 1:
            integrity_errors += 1
        accepted_question, rejection = validate_question(
            json.dumps({"question": row.get("query")}), content
        )
        if rejection is not None or accepted_question is None:
            validation_errors += 1

    accepted_count = len(rows)
    checks = {
        "candidate_manifest": int(pool.get("candidate_count", 0)) == EXPECTED_CANDIDATES
        and pool.get("candidate_manifest_sha256") == EXPECTED_MANIFEST_SHA256,
        "no_excluded_sources": not (set(sources) & excluded),
        "accepted_gte_32": accepted_count >= MIN_ACCEPTED,
        "source_and_answer_integrity": integrity_errors == 0,
        "question_validation": validation_errors == 0
        and len({_normalize(str(row.get("query", ""))) for row in rows}) == accepted_count,
        "root_distribution": len(root_counts) >= 3
        and accepted_count > 0
        and max(root_counts.values(), default=0) / accepted_count <= 0.80,
    }
    return {
        "checks": checks,
        "decision": (
            "READY_TO_PREREGISTER_EXPLORATORY_RETRIEVAL"
            if all(checks.values())
            else "STOP_BLIND_SOURCE_CENSUS"
        ),
        "accepted_rows": accepted_count,
        "distinct_sources": len(set(sources)),
        "root_counts": dict(sorted(root_counts.items())),
        "integrity_errors": integrity_errors,
        "validation_errors": validation_errors,
        "rejections": dict(sorted(rejections.items())),
    }


def restricted_ntfs_acl(path: Path) -> bool:
    completed = subprocess.run(
        ["icacls", str(path)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        return False
    output = completed.stdout.casefold()
    forbidden = ("everyone", "builtin\\users", "authenticated users")
    return all(value not in output for value in forbidden)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", type=_source_root, required=True)
    parser.add_argument("--exclusion-input", action="append", type=Path, required=True)
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

    excluded, input_hashes = load_exclusions(args.exclusion_input)
    candidates, _, parsed_sources = candidate_rows(args.source_root, excluded)
    _, manifest_sha256 = candidate_manifest(candidates)
    if len(candidates) != EXPECTED_CANDIDATES or manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"CANDIDATE_MANIFEST_MISMATCH:{len(candidates)}:{manifest_sha256}"
        )

    writer = OpenRouterLLM(
        model=MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0.0,
        max_tokens=120,
    )
    rows, rejections, attempts = generate_census_rows(candidates, writer)
    provider = asdict(writer.provider_metadata())
    provider["calls"] = writer.usage()["calls"]
    pool = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "seed": SEED,
        "model": MODEL,
        "system_prompt": SYSTEM_PROMPT,
        "user_template": USER_TEMPLATE,
        "input_hashes": input_hashes,
        "excluded_sources": len(excluded),
        "parsed_unexcluded_sources": parsed_sources,
        "candidate_count": len(candidates),
        "candidate_manifest_sha256": manifest_sha256,
        "attempted_sources": attempts,
        "provider": provider,
        "queries": rows,
    }
    audit = audit_census(pool, args.source_root, excluded, rejections)
    _write_private(args.private_output, pool)
    acl_restricted = restricted_ntfs_acl(args.private_output)
    audit["checks"]["provider_call_accounting"] = (
        attempts == EXPECTED_CANDIDATES == int(provider.get("calls", 0))
    )
    audit["checks"]["restricted_ntfs_acl"] = acl_restricted
    audit["decision"] = (
        "READY_TO_PREREGISTER_EXPLORATORY_RETRIEVAL"
        if all(audit["checks"].values())
        else "STOP_BLIND_SOURCE_CENSUS"
    )
    public = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "pool_sha256": _sha256(args.private_output),
        "system_prompt_sha256": _text_sha256(SYSTEM_PROMPT),
        "user_template_sha256": _text_sha256(USER_TEMPLATE),
        "model": MODEL,
        "input_hashes": input_hashes,
        "excluded_sources": len(excluded),
        "parsed_unexcluded_sources": parsed_sources,
        "candidate_count": len(candidates),
        "candidate_manifest_sha256": manifest_sha256,
        "attempted_sources": attempts,
        "provider": provider,
        "restricted_ntfs_acl": acl_restricted,
        **audit,
    }
    _write_public(args.public_output, public)
    print(json.dumps(public, ensure_ascii=False))
    if public["decision"] != "READY_TO_PREREGISTER_EXPLORATORY_RETRIEVAL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
