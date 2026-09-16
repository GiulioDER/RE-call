"""Build and audit a fresh production aligned atomic fact auxiliary view."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from recall.context import (
    SECTION_DEGRADED_MAX_CHARS,
    SECTION_MAX_CHARS,
    document_title,
)
from recall.document import parse_document
from recall.index import chunk_text


INPUT_HASHES = {
    "2026-09-14-guarded-spare-slot-extractive-pool.json": (
        "66ec82a058c9b06cf80314a1001779a1608e5144e097ace28b91664a48ede855"
    ),
    "2026-09-14-query-anchor-spare-slot-pool.json": (
        "6dd9485b12bf0c88d166cc29114cd03ada1e732e47b11592a4725e91d7324e68"
    ),
    "2026-09-14-guarded-spare-slot-fresh-pool.json": (
        "af7c74d4d2b232cb79de72d5fdfd67e1ccf1d6ad814fb634ba61f98999632c75"
    ),
    "2026-09-13-live-source-admission-trace-capture.json": (
        "facdac77945c820c80af76e8adaa3fd106f34598c88dd56a291d001f3fa2bfd9"
    ),
}
DEFAULT_SEED = "atomic-fact-fresh-audit-v1"
SKIP_NAMES = {
    "MEMORY.md",
    "project_index.md",
    "feedback_index.md",
    "archived_index.md",
    "closed_hypotheses_index.md",
    "EXECUTION_LOG.md",
}
SKIP_DATE_PREFIXES = ("2026-09-15", "2026-09-16")
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
FIELD_TEMPLATES = {
    "FACT": "What fact was recorded for {title}?",
    "VERDICT": "What verdict was recorded for {title}?",
    "RESULT": "What result was recorded for {title}?",
    "OUTCOME": "What outcome was recorded for {title}?",
    "DECISION": "What decision was recorded for {title}?",
    "APPLY": "How should {title} be applied?",
    "WHY": "Why does {title} matter?",
    "STATUS": "What status was recorded for {title}?",
    "OBJECTIVE": "What was the objective of {title}?",
    "ROOT CAUSE": "What root cause was recorded for {title}?",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _paragraphs(body: str) -> Iterable[str]:
    return (value.strip() for value in re.split(r"\n\s*\n", body) if value.strip())


def _render(
    *, title: str, section: str, field: str | None, content: str
) -> tuple[str, str]:
    def representation(selected_title: str, selected_section: str) -> str:
        lines: list[str] = []
        if selected_title:
            lines.append(f"title: {selected_title}")
        if selected_section:
            lines.append(f"section: {selected_section}")
        if field:
            lines.append(f"field: {field}")
        lines.append(f"content: {content}")
        return "\n".join(lines)

    full_section = section[:SECTION_MAX_CHARS]
    candidates = (
        ("full", representation(title, full_section)),
        (
            "shorten-section",
            representation(title, full_section[:SECTION_DEGRADED_MAX_CHARS]),
        ),
        ("drop-section", representation(title, "")),
        ("drop-title", representation("", "")),
        ("content-only", content),
    )
    return next((rung, text) for rung, text in candidates if len(text) <= 800)


def build_source_views(raw: str, source: str) -> tuple[list[str], list[dict[str, Any]]]:
    parsed = parse_document(raw)
    chunks = chunk_text(parsed.human_body)
    normalized_chunks = [_collapse(chunk) for chunk in chunks]
    normalized_body = _collapse(parsed.human_body)
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

        field_match = FIELD_PATTERN.fullmatch(paragraph)
        field = _clean_structure(field_match.group("field")).upper() if field_match else None
        content = _collapse(field_match.group("value") if field_match else paragraph)
        lowered = content.casefold()
        if not (
            40 <= len(content) <= 320
            and len(WORD.findall(content)) >= 8
            and "http://" not in lowered
            and "https://" not in lowered
            and "```" not in content
            and "|" not in content
            and normalized_body.count(content) == 1
        ):
            continue

        normalized_paragraph = _collapse(paragraph)
        parents = [
            ordinal
            for ordinal, chunk in enumerate(normalized_chunks)
            if normalized_paragraph in chunk
        ]
        if len(parents) != 1:
            continue
        section = _clean_structure(" > ".join(headings))
        rung, rendered = _render(
            title=title,
            section=section,
            field=field,
            content=content,
        )
        construction = "extractive_field" if field else (
            "extractive_heading" if headings else "extractive_fallback"
        )
        views.append(
            {
                "source": source,
                "parent_ordinal": parents[0],
                "title": title,
                "headings": list(headings),
                "field": field,
                "content": content,
                "rendered": rendered,
                "rung": rung,
                "construction": construction,
            }
        )
    return chunks, views


def _question(view: Mapping[str, Any]) -> str:
    title = str(view["title"])
    field = view.get("field")
    if field:
        return FIELD_TEMPLATES[str(field)].format(title=title)
    headings = [str(value) for value in view.get("headings", [])]
    if headings:
        return f"What does {title} record under {headings[-1]}?"
    return f"What key statement is recorded in {title}?"


def _pool_sources(payload: Mapping[str, Any]) -> set[str]:
    return {
        str(source)
        for row in payload.get("queries", [])
        for source in row.get("gold_sources", [])
    }


def _trace_sources(payload: Mapping[str, Any]) -> set[str]:
    return {
        str(item["source"])
        for row in payload.get("rows", [])
        for item in row.get("trace", {}).get("pool", [])
    }


def _load_exclusions(paths: Sequence[Path], *, enforce_hashes: bool) -> tuple[set[str], dict[str, str]]:
    sources: set[str] = set()
    hashes: dict[str, str] = {}
    for path in paths:
        digest = _sha256(path)
        hashes[path.name] = digest
        expected = INPUT_HASHES.get(path.name)
        if enforce_hashes and digest != expected:
            raise ValueError(f"INPUT_HASH_MISMATCH:{path.name}:{digest}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "rows" in payload:
            sources.update(_trace_sources(payload))
        else:
            sources.update(_pool_sources(payload))
    return sources, hashes


def build_fresh_pool(
    roots: Sequence[tuple[str, Path]],
    exclusion_paths: Sequence[Path],
    *,
    count: int = 250,
    seed: str = DEFAULT_SEED,
    enforce_hashes: bool = True,
) -> dict[str, Any]:
    excluded, input_hashes = _load_exclusions(exclusion_paths, enforce_hashes=enforce_hashes)
    candidates: list[dict[str, Any]] = []
    for prefix, root in roots:
        for path in root.rglob("*.md"):
            if path.name in SKIP_NAMES or path.name.startswith(SKIP_DATE_PREFIXES):
                continue
            source = f"{prefix}/{path.relative_to(root).as_posix()}"
            if source in excluded:
                continue
            raw = path.read_text(encoding="utf-8", errors="replace")
            chunks, views = build_source_views(raw, source)
            if not views:
                continue
            gold = views[0]
            candidates.append(
                {
                    "source": source,
                    "source_sha256": _sha256(path),
                    "gold_ordinal": gold["parent_ordinal"],
                    "answer_span": gold["content"],
                    "answer_span_sha256": _text_sha256(str(gold["content"])),
                    "query": _question(gold),
                    "construction": gold["construction"],
                    "ordinary_chunks": len(chunks),
                    "order": _text_sha256(f"{seed}\0{source}"),
                }
            )
    candidates.sort(key=lambda item: (item["order"], item["source"]))

    selected: list[dict[str, Any]] = []
    used_queries: set[str] = set()
    for candidate in candidates:
        query_key = _normalize(str(candidate["query"]))
        if query_key in used_queries:
            continue
        used_queries.add(query_key)
        selected.append(candidate)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError(f"INSUFFICIENT_FRESH_POOL: need {count}, found {len(selected)}")

    queries = [
        {
            "id": f"atomic-fact-fresh-{index:03d}",
            "query": item["query"],
            "expected_answerability": "answerable",
            "gold_sources": [item["source"]],
            "source_sha256": item["source_sha256"],
            "gold_ordinal": item["gold_ordinal"],
            "answer_span": item["answer_span"],
            "answer_span_sha256": item["answer_span_sha256"],
            "construction": item["construction"],
        }
        for index, item in enumerate(selected, start=1)
    ]
    return {
        "schema_version": 1,
        "protocol": "2026-09-16-production-aligned-atomic-fact-fresh-audit",
        "seed": seed,
        "input_hashes": dict(sorted(input_hashes.items())),
        "excluded_sources": len(excluded),
        "eligible_sources": len(candidates),
        "selected_sources": len(queries),
        "queries": queries,
    }


def _source_path(source: str, roots: Mapping[str, Path]) -> Path | None:
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


def _duplicate_excess(values: Iterable[str]) -> int:
    return sum(count - 1 for count in Counter(values).values() if count > 1)


def audit_fresh_pool(pool_path: Path, roots: Mapping[str, Path]) -> dict[str, Any]:
    pool_sha256 = _sha256(pool_path)
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    rows = list(pool.get("queries", []))
    missing_sources = 0
    source_hash_mismatches = 0
    source_chunks: dict[str, list[str]] = {}
    source_views: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        source = str(row.get("gold_sources", [""])[0])
        path = _source_path(source, roots)
        if path is None or not path.is_file():
            missing_sources += 1
            continue
        if _sha256(path) != row.get("source_sha256"):
            source_hash_mismatches += 1
            continue
        chunks, views = build_source_views(
            path.read_text(encoding="utf-8", errors="replace"), source
        )
        source_chunks[source] = chunks
        source_views[source] = views

    all_views = [view for views in source_views.values() for view in views]
    all_chunks = [chunk for chunks in source_chunks.values() for chunk in chunks]
    views_by_parent: Counter[tuple[str, int]] = Counter(
        (str(view["source"]), int(view["parent_ordinal"])) for view in all_views
    )
    views_per_parent = [
        views_by_parent.get((source, ordinal), 0)
        for source, chunks in source_chunks.items()
        for ordinal in range(len(chunks))
    ]
    views_per_source = [len(source_views.get(source, [])) for source in source_chunks]

    within_source_fact_duplicates = sum(
        _duplicate_excess(_normalize(str(view["content"])) for view in views)
        for views in source_views.values()
    )
    rendered_sources: dict[str, set[str]] = defaultdict(set)
    for view in all_views:
        rendered_sources[_normalize(str(view["rendered"]))].add(str(view["source"]))
    cross_source_rendered_collisions = sum(
        len(sources) - 1 for sources in rendered_sources.values() if len(sources) > 1
    )

    unique_gold = 0
    parent_matches = 0
    complete_content = 0
    family_totals: Counter[str] = Counter()
    family_covered: Counter[str] = Counter()
    for row in rows:
        source = str(row.get("gold_sources", [""])[0])
        family = str(row.get("construction", "unknown"))
        family_totals[family] += 1
        answer = _normalize(str(row.get("answer_span", "")))
        matches = [
            view
            for view in source_views.get(source, [])
            if _normalize(str(view["content"])) == answer
        ]
        if len(matches) != 1:
            continue
        unique_gold += 1
        family_covered[family] += 1
        match = matches[0]
        if int(match["parent_ordinal"]) == int(row.get("gold_ordinal", -1)):
            parent_matches += 1
        if str(match["content"]) == str(row.get("answer_span", "")):
            complete_content += 1

    family_coverage = {
        family: {
            "total": total,
            "covered": family_covered[family],
            "rate": round(family_covered[family] / total, 6) if total else 0.0,
        }
        for family, total in sorted(family_totals.items())
    }
    view_count = len(all_views)
    chunk_count = len(all_chunks)
    ordinary_characters = sum(len(chunk) for chunk in all_chunks)
    rendered_characters = sum(len(str(view["rendered"])) for view in all_views)
    row_growth = view_count / chunk_count if chunk_count else math.inf
    character_growth = rendered_characters / ordinary_characters if ordinary_characters else math.inf
    parent_distribution = _nearest_rank(views_per_parent)
    oversized = sum(len(str(view["rendered"])) > 800 for view in all_views)
    zero_view_sources = sum(not source_views.get(source) for source in source_chunks)
    input_hashes_match = pool.get("input_hashes") == dict(sorted(INPUT_HASHES.items()))

    checks = {
        "fresh_pool": len(rows) == 250 and input_hashes_match,
        "source_integrity": missing_sources == 0 and source_hash_mismatches == 0,
        "unique_gold_view": unique_gold == 250,
        "gold_parent_matches": parent_matches == 250,
        "gold_family_coverage": (
            set(family_coverage)
            == {"extractive_fallback", "extractive_field", "extractive_heading"}
            and all(value["rate"] == 1.0 for value in family_coverage.values())
        ),
        "no_zero_view_sources": zero_view_sources == 0,
        "view_length": oversized == 0,
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
        "GO_BUILD_ATOMIC_FACT_CONTEXT4_SHADOW"
        if all(checks.values())
        else "STOP_PRODUCTION_ALIGNED_ATOMIC_FACT_VIEW"
    )
    return {
        "schema_version": 1,
        "protocol": "2026-09-16-production-aligned-atomic-fact-fresh-audit",
        "pool_sha256": pool_sha256,
        "decision": decision,
        "failed_gates": sorted(name for name, passed in checks.items() if not passed),
        "gates": checks,
        "metrics": {
            "pool": {
                "excluded_sources": int(pool.get("excluded_sources", 0)),
                "eligible_sources": int(pool.get("eligible_sources", 0)),
                "selected_sources": len(rows),
                "construction_counts": dict(sorted(family_totals.items())),
            },
            "integrity": {
                "missing_sources": missing_sources,
                "source_hash_mismatches": source_hash_mismatches,
            },
            "size": {
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
                "oversized_views": oversized,
                "rung_counts": dict(
                    sorted(Counter(str(view["rung"]) for view in all_views).items())
                ),
            },
            "duplicates": {
                "within_source_fact_excess": within_source_fact_duplicates,
                "cross_source_rendered_excess": cross_source_rendered_collisions,
            },
            "gold": {
                "unique_view": unique_gold,
                "parent_ordinal_match": parent_matches,
                "complete_content": complete_content,
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
    parser.add_argument("--source-root", action="append", type=_root, required=True)
    parser.add_argument("--exclusion-input", action="append", type=Path, required=True)
    parser.add_argument("--pool-output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=250)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()

    roots = list(args.source_root)
    pool = build_fresh_pool(
        roots,
        args.exclusion_input,
        count=args.count,
        seed=args.seed,
    )
    args.pool_output.parent.mkdir(parents=True, exist_ok=True)
    args.pool_output.write_text(
        json.dumps(pool, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    result = audit_fresh_pool(args.pool_output, dict(roots))
    args.result_output.parent.mkdir(parents=True, exist_ok=True)
    args.result_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "pool_sha256": result["pool_sha256"],
                "eligible_sources": pool["eligible_sources"],
                "selected_sources": pool["selected_sources"],
                "decision": result["decision"],
            }
        )
    )


if __name__ == "__main__":
    main()
