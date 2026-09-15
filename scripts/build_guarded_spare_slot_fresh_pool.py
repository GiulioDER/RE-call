"""Build a treatment-blind guarded spare-slot query pool from uninspected memory sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


SKIP_NAMES = {
    "MEMORY.md",
    "project_index.md",
    "archived_index.md",
    "closed_hypotheses_index.md",
    "EXECUTION_LOG.md",
}
MARKDOWN = re.compile(r"[`*_>#]+")
LEADING_DATE = re.compile(r"^\d{4}[-_]\d{2}[-_]\d{2}[-_ ]+")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trace_sources(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(item["source"])
        for row in payload["rows"]
        for item in row["trace"]["pool"]
    }


def _title(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            value = MARKDOWN.sub("", stripped).strip()
            if len(value) >= 12 and len(value.split()) >= 4:
                return value[:180]
    value = LEADING_DATE.sub("", path.stem).replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", value).strip()[:180]


def _root(value: str) -> tuple[str, Path]:
    prefix, separator, raw_path = value.partition("=")
    if not separator or not prefix.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("source root must be PREFIX=PATH")
    path = Path(raw_path)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"source root does not exist: {path}")
    return prefix.strip().strip("/"), path


def _source_candidates(
    roots: list[tuple[str, Path]], excluded: set[str], seed: str, negative_prefix: str
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    nonce = negative_prefix.casefold()
    for prefix, root in roots:
        for path in root.rglob("*.md"):
            if path.name in SKIP_NAMES:
                continue
            relative = path.relative_to(root).as_posix()
            source = f"{prefix}/{relative}"
            if source in excluded:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if nonce in text.casefold():
                raise ValueError(f"negative prefix already occurs in {source}")
            title = _title(path)
            if len(title) < 12:
                continue
            candidates.append(
                {
                    "source": source,
                    "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "title": title,
                    "order": hashlib.sha256(f"{seed}\0{source}".encode()).hexdigest(),
                }
            )
    candidates.sort(key=lambda item: (item["order"], item["source"]))
    return candidates


def _pool(
    candidates: list[dict[str, str]], count: int, seed: str, negative_prefix: str
) -> list[dict[str, Any]]:
    if len(candidates) < count:
        raise ValueError(f"need {count} unused sources, found {len(candidates)}")
    rows: list[dict[str, Any]] = []
    used_queries: set[str] = set()
    for index, item in enumerate(candidates[:count], start=1):
        title = item["title"]
        positive_query = f"What should I remember about {title}?"
        if positive_query.casefold() in used_queries:
            positive_query = f"What should I remember about {title} in {item['source']}?"
        used_queries.add(positive_query.casefold())
        negative_query = (
            f"What was the measured outcome of {negative_prefix}-{index:04d} for {title}?"
        )
        if negative_query.casefold() in used_queries:
            raise ValueError("negative query construction produced a duplicate")
        used_queries.add(negative_query.casefold())
        rows.extend(
            [
                {
                    "id": f"fresh-positive-{index:03d}",
                    "query": positive_query,
                    "expected_answerability": "answerable",
                    "gold_sources": [item["source"]],
                    "source_sha256": item["source_sha256"],
                    "construction": "unused_memory_source_title",
                },
                {
                    "id": f"fresh-negative-{index:03d}",
                    "query": negative_query,
                    "expected_answerability": "unanswerable",
                    "gold_sources": [],
                    "source_sha256": None,
                    "construction": "matched_absent_project_control",
                },
            ]
        )
    rows.sort(
        key=lambda item: (
            hashlib.sha256(f"{seed}\0{item['id']}".encode()).hexdigest(),
            item["id"],
        )
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", type=_root, required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--answerable-count", type=int, default=250)
    parser.add_argument("--seed", default="guarded-spare-slot-fresh-v1")
    parser.add_argument("--negative-prefix", default="ZXQMEM")
    args = parser.parse_args()

    trace_path = Path(args.trace)
    excluded = _trace_sources(trace_path)
    candidates = _source_candidates(
        args.source_root, excluded, args.seed, args.negative_prefix
    )
    queries = _pool(candidates, args.answerable_count, args.seed, args.negative_prefix)
    payload = {
        "schema_version": 1,
        "protocol": "2026-09-14-guarded-spare-slot-fresh-pool",
        "seed": args.seed,
        "negative_prefix": args.negative_prefix,
        "trace_sha256": _sha256(trace_path),
        "excluded_trace_sources": len(excluded),
        "answerable_queries": args.answerable_count,
        "unanswerable_queries": args.answerable_count,
        "queries": queries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "queries": len(queries),
                "excluded_trace_sources": len(excluded),
                "unused_source_candidates": len(candidates),
                "sha256": _sha256(output),
            }
        )
    )


if __name__ == "__main__":
    main()
