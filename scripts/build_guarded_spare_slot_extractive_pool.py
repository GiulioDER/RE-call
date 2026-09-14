"""Build an untouched exact-span memory retrieval holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from recall.index import chunk_text


SKIP_NAMES = {
    "MEMORY.md",
    "project_index.md",
    "feedback_index.md",
    "archived_index.md",
    "closed_hypotheses_index.md",
    "EXECUTION_LOG.md",
}
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
HEADING_PATTERN = re.compile(r"^#{2,6}\s+(?P<heading>.+?)\s*$")
MARKDOWN = re.compile(r"[`*_>#]+")
LEADING_DATE = re.compile(r"^\d{4}[-_]\d{2}[-_]\d{2}[-_ ]+")
WORD = re.compile(r"\b[\w'-]+\b", re.UNICODE)
WHITESPACE = re.compile(r"\s+")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize(value: str) -> str:
    return WHITESPACE.sub(" ", value).strip()


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return text
    lines = text.splitlines()
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[index + 1 :])
    return text


def _title(path: Path, body: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            value = _normalize(MARKDOWN.sub("", stripped))
            if len(value) >= 12 and len(value.split()) >= 3:
                return value[:180]
    value = LEADING_DATE.sub("", path.stem).replace("_", " ").replace("-", " ")
    return _normalize(value)[:180]


def _eligible(value: str, normalized_source: str) -> str | None:
    span = _normalize(value)
    if not 40 <= len(span) <= 320:
        return None
    if len(WORD.findall(span)) < 8:
        return None
    lowered = span.casefold()
    if "http://" in lowered or "https://" in lowered:
        return None
    if "```" in span or "|" in span or span.startswith("#"):
        return None
    if normalized_source.count(span) != 1:
        return None
    return span


def _paragraphs(body: str) -> list[str]:
    return [value.strip() for value in re.split(r"\n\s*\n", body) if value.strip()]


def _span_candidates(body: str) -> Iterable[tuple[str, str | None, str]]:
    paragraphs = _paragraphs(body)
    for field in FIELD_ORDER:
        for paragraph in paragraphs:
            match = FIELD_PATTERN.match(paragraph)
            if match and match.group("field").upper() == field:
                yield "field", field, match.group("value")

    heading: str | None = None
    for paragraph in paragraphs:
        match = HEADING_PATTERN.match(paragraph)
        if match:
            heading = _normalize(MARKDOWN.sub("", match.group("heading")))[:120]
            continue
        if heading is not None:
            yield "heading", heading, paragraph

    for paragraph in paragraphs:
        if paragraph.startswith("#") or FIELD_PATTERN.match(paragraph):
            continue
        yield "fallback", None, paragraph


def _question(title: str, kind: str, label: str | None) -> str:
    if kind == "heading":
        return f"What does {title} record under {label}?"
    if kind == "fallback":
        return f"What key statement is recorded in {title}?"
    templates = {
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
    return templates[str(label)].format(title=title)


def _extract(path: Path, source: str) -> dict[str, Any] | None:
    raw = path.read_text(encoding="utf-8", errors="replace")
    body = _strip_frontmatter(raw)
    title = _title(path, body)
    if len(title) < 12:
        return None
    normalized_source = _normalize(raw)
    chunks = chunk_text(raw)
    for kind, label, raw_span in _span_candidates(body):
        span = _eligible(raw_span, normalized_source)
        if span is None:
            continue
        question = _question(title, kind, label)
        if span.casefold() in _normalize(question).casefold():
            continue
        ordinals = [
            ordinal for ordinal, chunk in enumerate(chunks) if span in _normalize(chunk)
        ]
        if len(ordinals) != 1:
            continue
        return {
            "source": source,
            "source_sha256": _sha256(path),
            "source_ordinal": ordinals[0],
            "title": title,
            "question": question,
            "answer_span": span,
            "answer_span_sha256": _text_sha256(span),
            "construction": f"extractive_{kind}",
        }
    return None


def _root(value: str) -> tuple[str, Path]:
    prefix, separator, raw_path = value.partition("=")
    if not separator or not prefix.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("source root must be PREFIX=PATH")
    path = Path(raw_path)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"source root does not exist: {path}")
    return prefix.strip().strip("/"), path


def _trace_sources(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(item["source"])
        for row in payload["rows"]
        for item in row["trace"]["pool"]
    }


def _pool_sources(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(source)
        for row in payload["queries"]
        for source in row.get("gold_sources", [])
    }


def _candidates(
    roots: list[tuple[str, Path]],
    trace_path: Path,
    old_pool_path: Path,
    seed: str,
    negative_prefix: str,
) -> tuple[list[dict[str, Any]], int]:
    excluded = _trace_sources(trace_path) | _pool_sources(old_pool_path)
    nonce = negative_prefix.casefold()
    values: list[dict[str, Any]] = []
    for prefix, root in roots:
        for path in root.rglob("*.md"):
            if path.name in SKIP_NAMES:
                continue
            relative = path.relative_to(root).as_posix()
            source = f"{prefix}/{relative}"
            if source in excluded:
                continue
            raw = path.read_text(encoding="utf-8", errors="replace")
            if nonce in raw.casefold():
                raise ValueError(f"negative prefix already occurs in {source}")
            extracted = _extract(path, source)
            if extracted is None:
                continue
            extracted["order"] = _text_sha256(f"{seed}\0{source}")
            values.append(extracted)
    values.sort(key=lambda item: (item["order"], item["source"]))
    return values, len(excluded)


def _build_rows(
    candidates: list[dict[str, Any]], count: int, seed: str, negative_prefix: str
) -> list[dict[str, Any]]:
    if len(candidates) < count:
        raise ValueError(f"INSUFFICIENT_POOL: need {count} eligible sources, found {len(candidates)}")
    rows: list[dict[str, Any]] = []
    used_queries: set[str] = set()
    for index, item in enumerate(candidates[:count], start=1):
        positive_query = str(item["question"])
        if positive_query.casefold() in used_queries:
            continue
        nonce = f"{negative_prefix}-{index:04d}"
        negative_query = f"According to memory item {nonce}, {positive_query[0].lower()}{positive_query[1:]}"
        if negative_query.casefold() in used_queries:
            raise ValueError("negative query construction produced a duplicate")
        used_queries.add(positive_query.casefold())
        used_queries.add(negative_query.casefold())
        rows.extend(
            [
                {
                    "id": f"extractive-positive-{index:03d}",
                    "query": positive_query,
                    "expected_answerability": "answerable",
                    "gold_sources": [item["source"]],
                    "source_sha256": item["source_sha256"],
                    "gold_ordinal": item["source_ordinal"],
                    "answer_span": item["answer_span"],
                    "answer_span_sha256": item["answer_span_sha256"],
                    "construction": item["construction"],
                },
                {
                    "id": f"extractive-negative-{index:03d}",
                    "query": negative_query,
                    "expected_answerability": "unanswerable",
                    "gold_sources": [],
                    "source_sha256": None,
                    "gold_ordinal": None,
                    "answer_span": "NOT_FOUND",
                    "answer_span_sha256": _text_sha256("NOT_FOUND"),
                    "construction": "matched_absent_identifier_control",
                },
            ]
        )
    if len(rows) != count * 2:
        raise ValueError(f"INSUFFICIENT_POOL: duplicate questions left {len(rows) // 2} pairs")
    rows.sort(key=lambda item: (_text_sha256(f"{seed}\0{item['id']}"), item["id"]))
    return rows


def build_pool(
    roots: list[tuple[str, Path]],
    trace_path: Path,
    old_pool_path: Path,
    *,
    count: int,
    seed: str,
    negative_prefix: str,
) -> dict[str, Any]:
    candidates, excluded_sources = _candidates(
        roots, trace_path, old_pool_path, seed, negative_prefix
    )
    rows = _build_rows(candidates, count, seed, negative_prefix)
    return {
        "schema_version": 1,
        "protocol": "2026-09-14-guarded-spare-slot-extractive-pool",
        "seed": seed,
        "negative_prefix": negative_prefix,
        "trace_sha256": _sha256(trace_path),
        "old_pool_sha256": _sha256(old_pool_path),
        "excluded_sources": excluded_sources,
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--answerable-count", type=int, default=250)
    parser.add_argument("--seed", default="guarded-spare-slot-extractive-v1")
    parser.add_argument("--negative-prefix", default="ZXQXACT")
    args = parser.parse_args()
    payload = build_pool(
        args.source_root,
        args.trace,
        args.old_pool,
        count=args.answerable_count,
        seed=args.seed,
        negative_prefix=args.negative_prefix,
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
                "queries": len(payload["queries"]),
                "eligible_sources": payload["eligible_sources"],
                "sha256": _sha256(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
