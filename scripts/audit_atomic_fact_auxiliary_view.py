"""Audit a bounded atomic fact auxiliary retrieval view without embedding it."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable

from recall.context import document_title
from recall.document import parse_document
from recall.index import chunk_text


FROZEN_POOL_SHA256 = "66ec82a058c9b06cf80314a1001779a1608e5144e097ace28b91664a48ede855"
FIELD_ORDER = (
    "FACT",
    "VERDICT",
    "RESULT",
    "OUTCOME",
    "DECISION",
    "APPLY",
    "WHY",
    "STATUS",
    "OBJECTIVE",
    "ROOT CAUSE",
)
FIELD_PATTERN = re.compile(
    r"^\s*(?:[-*]\s+)?(?:\*\*)?(?P<field>"
    + "|".join(re.escape(value) for value in FIELD_ORDER)
    + r")(?:\*\*)?\s*:\s*(?P<value>.+)$",
    re.IGNORECASE | re.DOTALL,
)
HEADING_PATTERN = re.compile(r"^(?P<marks>#{1,6})\s+(?P<heading>.+?)\s*$", re.DOTALL)
WORD = re.compile(r"\b[\w'-]+\b", re.UNICODE)
WHITESPACE = re.compile(r"\s+")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collapse(value: str) -> str:
    return WHITESPACE.sub(" ", value).strip()


def _normalize(value: str) -> str:
    return _collapse(unicodedata.normalize("NFKC", value).casefold())


def _clean_structure(value: str) -> str:
    clean = "".join(ch for ch in value if not unicodedata.category(ch).startswith("C"))
    return _collapse(clean)


def _nearest_rank(values: list[int]) -> dict[str, int | None]:
    if not values:
        return {name: None for name in ("min", "p50", "p90", "p95", "max")}
    ordered = sorted(values)

    def percentile(fraction: float) -> int:
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]

    return {
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def _source_path(source: str, roots: dict[str, Path]) -> Path | None:
    prefix, separator, relative = source.partition("/")
    root = roots.get(prefix)
    if not separator or root is None or not relative:
        return None
    resolved_root = root.resolve()
    candidate = (resolved_root / Path(relative)).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError:
        return None
    return candidate


def _paragraphs(body: str) -> Iterable[str]:
    return (value.strip() for value in re.split(r"\n\s*\n", body) if value.strip())


def _eligible(paragraph: str, normalized_source: str) -> bool:
    value = _collapse(paragraph)
    lowered = value.casefold()
    return (
        40 <= len(value) <= 320
        and len(WORD.findall(value)) >= 8
        and "http://" not in lowered
        and "https://" not in lowered
        and "```" not in value
        and "|" not in value
        and normalized_source.count(value) == 1
    )


def _render(title: str, headings: list[str], field: str | None, content: str) -> str:
    lines = [f"title: {title}"]
    if headings:
        lines.append(f"section: {' > '.join(headings)}")
    if field:
        lines.append(f"field: {field}")
    lines.append(f"content: {content}")
    return "\n".join(lines)


def build_source_views(raw: str, source: str) -> tuple[list[str], list[dict[str, Any]]]:
    """Build frozen auxiliary views and map each one to one ordinary parent chunk."""
    parsed = parse_document(raw)
    chunks = chunk_text(raw)
    normalized_chunks = [_collapse(chunk) for chunk in chunks]
    normalized_source = _collapse(raw)
    title = document_title(raw, parsed.human_body, source)
    headings: list[str] = []
    heading_levels: list[int] = []
    views: list[dict[str, Any]] = []

    for paragraph in _paragraphs(parsed.human_body):
        heading_match = HEADING_PATTERN.fullmatch(paragraph)
        if heading_match is not None:
            level = len(heading_match.group("marks"))
            while heading_levels and heading_levels[-1] >= level:
                heading_levels.pop()
                headings.pop()
            heading_levels.append(level)
            headings.append(_clean_structure(heading_match.group("heading")))
            continue
        if not _eligible(paragraph, normalized_source):
            continue

        normalized_paragraph = _collapse(paragraph)
        parents = [
            ordinal
            for ordinal, chunk in enumerate(normalized_chunks)
            if normalized_paragraph in chunk
        ]
        if len(parents) != 1:
            continue
        field_match = FIELD_PATTERN.fullmatch(paragraph)
        field = _clean_structure(field_match.group("field")).upper() if field_match else None
        content = _collapse(field_match.group("value") if field_match else paragraph)
        rendered = _render(title, headings, field, content)
        views.append(
            {
                "source": source,
                "parent_ordinal": parents[0],
                "title": title,
                "headings": list(headings),
                "field": field,
                "content": content,
                "rendered": rendered,
            }
        )
    return chunks, views


def _duplicate_excess(values: Iterable[str]) -> int:
    return sum(count - 1 for count in Counter(values).values() if count > 1)


def audit_atomic_fact_view(
    pool_path: Path,
    roots: dict[str, Path],
    *,
    enforce_frozen_hash: bool = True,
) -> dict[str, Any]:
    pool_sha256 = _sha256(pool_path)
    if enforce_frozen_hash and pool_sha256 != FROZEN_POOL_SHA256:
        raise ValueError(f"POOL_HASH_MISMATCH: {pool_sha256}")
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in pool.get("queries", [])
        if row.get("expected_answerability") == "answerable"
    ]

    missing_sources = 0
    source_hash_mismatches = 0
    reconstruction_failures = 0
    source_views: dict[str, list[dict[str, Any]]] = {}
    source_chunks: dict[str, list[str]] = {}
    for row in rows:
        sources = [str(value) for value in row.get("gold_sources", [])]
        source = sources[0] if len(sources) == 1 else ""
        path = _source_path(source, roots)
        if path is None or not path.is_file():
            missing_sources += 1
            continue
        if _sha256(path) != row.get("source_sha256"):
            source_hash_mismatches += 1
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        try:
            chunks, views = build_source_views(raw, source)
        except (TypeError, ValueError):
            reconstruction_failures += 1
            continue
        source_chunks[source] = chunks
        source_views[source] = views

    all_views = [view for values in source_views.values() for view in values]
    ordinary_chunks = [chunk for values in source_chunks.values() for chunk in values]
    views_per_source = [len(source_views.get(source, [])) for source in source_chunks]
    views_by_parent: Counter[tuple[str, int]] = Counter(
        (str(view["source"]), int(view["parent_ordinal"])) for view in all_views
    )
    views_per_parent = [
        views_by_parent.get((source, ordinal), 0)
        for source, chunks in source_chunks.items()
        for ordinal in range(len(chunks))
    ]

    within_source_fact_duplicates = sum(
        _duplicate_excess(_normalize(str(view["content"])) for view in views)
        for views in source_views.values()
    )
    within_source_rendered_collisions = sum(
        _duplicate_excess(_normalize(str(view["rendered"])) for view in views)
        for views in source_views.values()
    )
    fact_sources: dict[str, set[str]] = defaultdict(set)
    rendered_sources: dict[str, set[str]] = defaultdict(set)
    for view in all_views:
        source = str(view["source"])
        fact_sources[_normalize(str(view["content"]))].add(source)
        rendered_sources[_normalize(str(view["rendered"]))].add(source)
    cross_source_fact_collisions = sum(
        len(sources) - 1 for sources in fact_sources.values() if len(sources) > 1
    )
    cross_source_rendered_collisions = sum(
        len(sources) - 1 for sources in rendered_sources.values() if len(sources) > 1
    )

    gold_rows = 0
    gold_unique_view = 0
    gold_parent_matches = 0
    gold_complete_content = 0
    family_totals: Counter[str] = Counter()
    family_covered: Counter[str] = Counter()
    for row in rows:
        sources = [str(value) for value in row.get("gold_sources", [])]
        source = sources[0] if len(sources) == 1 else ""
        if source not in source_views:
            continue
        gold_rows += 1
        family = str(row.get("construction", "unknown"))
        family_totals[family] += 1
        answer = _normalize(str(row.get("answer_span", "")))
        matches = [
            view for view in source_views[source] if answer in _normalize(str(view["content"]))
        ]
        if len(matches) != 1:
            continue
        gold_unique_view += 1
        family_covered[family] += 1
        match = matches[0]
        if int(match["parent_ordinal"]) == int(row.get("gold_ordinal", -1)):
            gold_parent_matches += 1
        if _normalize(str(match["content"])) == answer:
            gold_complete_content += 1

    view_count = len(all_views)
    chunk_count = len(ordinary_chunks)
    ordinary_characters = sum(len(chunk) for chunk in ordinary_chunks)
    rendered_characters = sum(len(str(view["rendered"])) for view in all_views)
    row_growth = view_count / chunk_count if chunk_count else math.inf
    character_growth = rendered_characters / ordinary_characters if ordinary_characters else math.inf
    oversized_views = sum(len(str(view["rendered"])) > 800 for view in all_views)
    family_coverage = {
        family: {
            "total": total,
            "covered": family_covered[family],
            "rate": round(family_covered[family] / total, 6) if total else 0.0,
        }
        for family, total in sorted(family_totals.items())
    }
    source_count = len(rows)
    zero_view_sources = sum(not source_views.get(source) for source in source_chunks)

    parent_distribution = _nearest_rank(views_per_parent)
    checks = {
        "source_integrity": (
            source_count == 250
            and missing_sources == 0
            and source_hash_mismatches == 0
            and reconstruction_failures == 0
        ),
        "unique_gold_view": gold_unique_view == 250,
        "gold_parent_matches": gold_parent_matches == 250,
        "gold_family_coverage": (
            set(family_coverage)
            == {"extractive_fallback", "extractive_field", "extractive_heading"}
            and all(value["rate"] == 1.0 for value in family_coverage.values())
        ),
        "no_zero_view_sources": zero_view_sources == 0,
        "view_length": oversized_views == 0,
        "within_source_fact_duplicates": (
            within_source_fact_duplicates / view_count <= 0.01 if view_count else False
        ),
        "cross_source_rendered_collisions": (
            cross_source_rendered_collisions / view_count <= 0.01 if view_count else False
        ),
        "row_growth": row_growth <= 2.0,
        "character_growth": character_growth <= 1.5,
        "parent_crowding": (
            parent_distribution["p95"] is not None
            and parent_distribution["p95"] <= 4
            and parent_distribution["max"] is not None
            and parent_distribution["max"] <= 8
        ),
    }
    decision = (
        "GO_BUILD_ATOMIC_FACT_AUXILIARY_SHADOW"
        if all(checks.values())
        else "STOP_ATOMIC_FACT_AUXILIARY_SHADOW"
    )
    failed_gates = sorted(name for name, passed in checks.items() if not passed)

    return {
        "schema_version": 1,
        "protocol": "2026-09-16-atomic-fact-auxiliary-view-audit",
        "pool_sha256": pool_sha256,
        "decision": decision,
        "failed_gates": failed_gates,
        "gates": checks,
        "metrics": {
            "integrity": {
                "answerable_rows": source_count,
                "missing_sources": missing_sources,
                "source_hash_mismatches": source_hash_mismatches,
                "reconstruction_failures": reconstruction_failures,
            },
            "size": {
                "sources": len(source_chunks),
                "ordinary_chunks": chunk_count,
                "auxiliary_views": view_count,
                "zero_view_sources": zero_view_sources,
                "auxiliary_row_growth": round(row_growth, 6),
                "rendered_character_growth": round(character_growth, 6),
                "ordinary_characters": ordinary_characters,
                "rendered_characters": rendered_characters,
            },
            "distribution": {
                "views_per_source": _nearest_rank(views_per_source),
                "views_per_parent_chunk": parent_distribution,
                "rendered_length": _nearest_rank(
                    [len(str(view["rendered"])) for view in all_views]
                ),
                "fact_content_length": _nearest_rank(
                    [len(str(view["content"])) for view in all_views]
                ),
            },
            "coverage": {
                "title": sum(bool(view["title"]) for view in all_views),
                "heading": sum(bool(view["headings"]) for view in all_views),
                "field": sum(bool(view["field"]) for view in all_views),
                "oversized_views": oversized_views,
            },
            "duplicates": {
                "within_source_fact_excess": within_source_fact_duplicates,
                "within_source_rendered_excess": within_source_rendered_collisions,
                "cross_source_fact_excess": cross_source_fact_collisions,
                "cross_source_rendered_excess": cross_source_rendered_collisions,
            },
            "gold": {
                "rows_evaluated": gold_rows,
                "unique_view": gold_unique_view,
                "parent_ordinal_match": gold_parent_matches,
                "complete_content": gold_complete_content,
                "family_coverage": family_coverage,
            },
        },
    }


def _root(value: str) -> tuple[str, Path]:
    prefix, separator, raw_path = value.partition("=")
    if not separator or not prefix.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("source root must be PREFIX=PATH")
    return prefix.strip().strip("/"), Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--source-root", action="append", type=_root, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_atomic_fact_view(args.pool, dict(args.source_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"output": str(args.output), "decision": result["decision"]}))


if __name__ == "__main__":
    main()
