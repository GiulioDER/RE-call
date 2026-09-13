"""Validate that every essential fact label is coverable by one canonical source chunk."""

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

from recall.document import parse_document  # noqa: E402
from recall.index import chunk_text  # noqa: E402


def _covered(text: str, terms: list[str], min_matches: int) -> bool:
    lowered = text.casefold()
    return sum(term.casefold() in lowered for term in terms) >= min_matches


def validate(
    labels: list[dict[str, Any]], source_roots: dict[str, Path]
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    uncovered: list[str] = []
    for label in labels:
        source = str(label["source"])
        namespace, relative = source.split("/", 1)
        source_path = source_roots[namespace] / relative
        raw = source_path.read_bytes()
        body = parse_document(raw.decode("utf-8-sig")).human_body
        chunks = chunk_text(body)
        fact_rows: list[dict[str, object]] = []
        for fact in label["facts"]:
            terms = [str(term) for term in fact["terms"]]
            min_matches = int(fact["min_matches"])
            ordinals = [
                ordinal
                for ordinal, chunk in enumerate(chunks)
                if _covered(chunk, terms, min_matches)
            ]
            key = f"{label['query_id']}:{fact['name']}"
            if not ordinals:
                uncovered.append(key)
            fact_rows.append(
                {
                    "name": fact["name"],
                    "supporting_ordinals": ordinals,
                }
            )
        rows.append(
            {
                "query_id": label["query_id"],
                "source": source,
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "chunk_count": len(chunks),
                "facts": fact_rows,
            }
        )
    return {
        "labels": len(labels),
        "facts": sum(len(label["facts"]) for label in labels),
        "uncovered": uncovered,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--recall-root", required=True)
    parser.add_argument("--sentiment-root", required=True)
    args = parser.parse_args()

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    result = validate(
        labels,
        {
            "recall": Path(args.recall_root),
            "sentiment-agent": Path(args.sentiment_root),
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["uncovered"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
