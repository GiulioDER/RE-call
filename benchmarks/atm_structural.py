"""ATM memory preparation with explicit structural graph metadata.

The published ATM reproduction harness is intentionally frozen. This module provides the new
annotation path for fresh ATM graph experiments without changing that historical program.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.atm_bench import build_memory_items, load_json
from benchmarks.structural_edges import atm_structural_metadata


def build_memory_items_with_structural_edges(
    image_file: Path, video_file: Path, email_file: Path
) -> list[tuple[str, str, str, dict[str, Any]]]:
    """Return ATM items with deterministic, source-backed ``recall_graph`` relations."""
    base = build_memory_items(image_file, video_file, email_file)
    raw_records: list[dict[str, Any]] = []
    for path, modality in ((email_file, "email"), (image_file, "image"), (video_file, "video")):
        records = load_json(path)
        if not isinstance(records, list):
            continue
        for record in records:
            if isinstance(record, dict):
                enriched = dict(record)
                enriched["modality"] = modality
                if modality == "email":
                    enriched["evidence_id"] = str(record.get("id") or "").strip()
                else:
                    media_path = str(record.get(f"{modality}_path") or "").strip()
                    enriched["evidence_id"] = Path(media_path).stem
                raw_records.append(enriched)
    graph_by_id = atm_structural_metadata(raw_records)
    return [
        (
            evidence_id,
            modality,
            text,
            {
                **metadata,
                "file": evidence_id,
                "recall_graph": graph_by_id.get(
                    evidence_id,
                    {"schema_version": 1, "relations": []},
                ),
            },
        )
        for evidence_id, modality, text, metadata in base
    ]


__all__ = ["build_memory_items_with_structural_edges"]
