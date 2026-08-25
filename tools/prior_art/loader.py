"""Load the checked in prior art evidence corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "docs" / "prior_art"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        records.append(value)
    return records


def load_dataset(root: Path = DATA_ROOT) -> dict[str, Any]:
    """Load all canonical prior art files from ``root``."""

    return {
        "root": root,
        "taxonomy": _load_json(root / "taxonomy.json"),
        "report_config": _load_json(root / "report_config.json"),
        "sources": _load_jsonl(root / "sources.jsonl"),
        "systems": _load_jsonl(root / "systems.jsonl"),
        "claims": _load_jsonl(root / "claims.jsonl"),
        "reviews": _load_jsonl(root / "reviews.jsonl"),
    }
