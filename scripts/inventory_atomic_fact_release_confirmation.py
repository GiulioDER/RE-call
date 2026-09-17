"""Inventory atomic-development-disjoint facts for release confirmation."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_atomic_fact_blind_source_census import (  # noqa: E402
    EXPECTED_CANDIDATES as OLD_EXPECTED_CANDIDATES,
    EXPECTED_MANIFEST_SHA256 as OLD_EXPECTED_MANIFEST_SHA256,
    restricted_ntfs_acl,
)
from scripts.build_atomic_fact_blind_source_holdout import (  # noqa: E402
    _input_sources,
    _normalize,
    _sha256,
    _source_root,
    _text_sha256,
    _write_private,
    _write_public,
    candidate_manifest as old_candidate_manifest,
    candidate_rows as old_candidate_rows,
    load_exclusions as load_old_exclusions,
)
from scripts.run_production_atomic_fact_fresh_audit import (  # noqa: E402
    SKIP_NAMES,
    build_source_views,
)


PROTOCOL = "2026-09-16-atomic-fact-release-confirmation-inventory"
SEED = "atomic-fact-release-confirmation-v1"
MIN_CANDIDATES = 160
SKIP_DATE_PREFIXES = ("2026-09-15", "2026-09-16")
EXPECTED_INPUT_HASHES = {
    "extractive_pool": "66ec82a058c9b06cf80314a1001779a1608e5144e097ace28b91664a48ede855",
    "pilot_pool": "720272504e128d861b09b7d29df8570eadfe3362865a256951c0b404bf448aa9",
    "source_gold": "06e5cfb2a345d3108ee5ae9e2d0bc2cd74fba455d46f56658bf496b2447e088f",
    "census_pool": "97f77c71c1feb278b9d5297511e8b4fb913002208eaae2b3bbd2dc65d578fd18",
}


def _read_verified(path: Path, key: str) -> object:
    digest = _sha256(path)
    expected = EXPECTED_INPUT_HASHES[key]
    if digest != expected:
        raise ValueError(f"INPUT_HASH_MISMATCH:{key}:{digest}")
    return json.loads(path.read_text(encoding="utf-8"))


def _query_sources(payload: object) -> set[str]:
    return _input_sources(payload)


def reconstruct_old_population(
    roots: Sequence[tuple[str, Path]], old_exclusion_paths: Sequence[Path]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    excluded, input_hashes = load_old_exclusions(old_exclusion_paths)
    candidates, _, _ = old_candidate_rows(roots, excluded)
    _, manifest_sha256 = old_candidate_manifest(candidates)
    if (
        len(candidates) != OLD_EXPECTED_CANDIDATES
        or manifest_sha256 != OLD_EXPECTED_MANIFEST_SHA256
    ):
        raise RuntimeError(
            f"OLD_CANDIDATE_MANIFEST_MISMATCH:{len(candidates)}:{manifest_sha256}"
        )
    return candidates, input_hashes


def development_exclusions(
    *,
    extractive_pool: object,
    pilot_pool: object,
    source_gold: object,
    old_candidates: Sequence[Mapping[str, Any]],
) -> tuple[set[str], dict[str, int]]:
    populations = {
        "extractive_pool": _query_sources(extractive_pool),
        "pilot_pool": _query_sources(pilot_pool),
        "source_gold": _query_sources(source_gold),
        "old_blind_population": {str(row["source"]) for row in old_candidates},
    }
    excluded: set[str] = set()
    for sources in populations.values():
        excluded.update(sources)
    counts = {key: len(value) for key, value in sorted(populations.items())}
    counts["union"] = len(excluded)
    return excluded, counts


def _snapshot_sources(
    roots: Sequence[tuple[str, Path]],
) -> tuple[list[dict[str, Any]], str]:
    if len(dict(roots)) != len(roots):
        raise ValueError("source root prefixes must be unique")
    sources: list[dict[str, Any]] = []
    for prefix, root in roots:
        if not root.is_dir():
            raise ValueError(f"missing source root: {root}")
        for path in sorted(root.rglob("*.md")):
            if path.name in SKIP_NAMES or path.name.startswith(SKIP_DATE_PREFIXES):
                continue
            sources.append(
                {
                    "source": f"{prefix}/{path.relative_to(root).as_posix()}",
                    "path": path,
                    "source_sha256": _sha256(path),
                }
            )
    sources.sort(key=lambda row: str(row["source"]))
    snapshot_rows = [
        {"source": row["source"], "source_sha256": row["source_sha256"]}
        for row in sources
    ]
    canonical = json.dumps(
        snapshot_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sources, hashlib.sha256(canonical).hexdigest()


def build_inventory(
    roots: Sequence[tuple[str, Path]], excluded: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources, snapshot_sha256 = _snapshot_sources(roots)
    content_sources: dict[str, set[str]] = {}
    source_views: dict[str, list[dict[str, Any]]] = {}
    for item in sources:
        source = str(item["source"])
        path = Path(item["path"])
        _, views = build_source_views(
            path.read_text(encoding="utf-8", errors="replace"), source
        )
        source_views[source] = views
        for view in views:
            content_sources.setdefault(_normalize(str(view["content"])), set()).add(source)

    candidates: list[dict[str, Any]] = []
    for item in sources:
        source = str(item["source"])
        if source in excluded:
            continue
        unique_view = next(
            (
                view
                for view in source_views[source]
                if content_sources.get(_normalize(str(view["content"]))) == {source}
            ),
            None,
        )
        if unique_view is None:
            continue
        candidates.append(
            {
                "source": source,
                "source_sha256": str(item["source_sha256"]),
                "gold_ordinal": int(unique_view["parent_ordinal"]),
                "answer_span_sha256": _text_sha256(str(unique_view["content"])),
                "construction": str(unique_view["construction"]),
                "order": _text_sha256(f"{SEED}\0{source}"),
            }
        )
    candidates.sort(key=lambda row: (str(row["order"]), str(row["source"])))
    return candidates, {
        "source_snapshot_sha256": snapshot_sha256,
        "snapshot_sources": len(sources),
        "parsed_sources_with_views": sum(bool(views) for views in source_views.values()),
    }


def manifest_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    canonical = json.dumps(
        list(rows), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def audit_inventory(
    rows: Sequence[Mapping[str, Any]],
    *,
    excluded: set[str],
    accepted_census_sources: set[str],
    old_population_sources: set[str],
    roots: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    root_counts = Counter(str(row["source"]).partition("/")[0] for row in rows)
    construction_counts = Counter(str(row["construction"]) for row in rows)
    integrity_errors = 0
    root_map = dict(roots)
    for row in rows:
        source = str(row["source"])
        prefix, separator, relative = source.partition("/")
        path = root_map.get(prefix, Path()) / relative if separator and prefix in root_map else None
        if path is None or not path.is_file() or _sha256(path) != row["source_sha256"]:
            integrity_errors += 1
            continue
        _, views = build_source_views(path.read_text(encoding="utf-8", errors="replace"), source)
        matches = [
            view
            for view in views
            if int(view["parent_ordinal"]) == int(row["gold_ordinal"])
            and _text_sha256(str(view["content"])) == row["answer_span_sha256"]
            and str(view["construction"]) == row["construction"]
        ]
        if len(matches) != 1:
            integrity_errors += 1

    candidate_count = len(rows)
    checks = {
        "old_population_manifest": len(old_population_sources) == OLD_EXPECTED_CANDIDATES,
        "accepted_census_subset": accepted_census_sources <= old_population_sources,
        "candidate_count_gte_160": candidate_count >= MIN_CANDIDATES,
        "no_development_sources": not ({str(row["source"]) for row in rows} & excluded),
        "root_distribution": len(root_counts) >= 3
        and candidate_count > 0
        and max(root_counts.values(), default=0) / candidate_count <= 0.80,
        "source_and_view_integrity": integrity_errors == 0,
    }
    return {
        "checks": checks,
        "decision": (
            "READY_TO_BUILD_ATOMIC_RELEASE_CONFIRMATION"
            if all(checks.values())
            else "STOP_ATOMIC_RELEASE_CONFIRMATION_INVENTORY"
        ),
        "candidate_count": candidate_count,
        "root_counts": dict(sorted(root_counts.items())),
        "construction_counts": dict(sorted(construction_counts.items())),
        "integrity_errors": integrity_errors,
    }


def _restrict_private_acl(path: Path) -> None:
    if sys.platform != "win32":
        return
    username = subprocess.run(
        ["whoami"], check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()
    completed = subprocess.run(
        ["icacls", str(path), "/inheritance:r", "/grant:r", f"{username}:(F)"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"PRIVATE_ACL_UPDATE_FAILED:{completed.stderr.strip()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", type=_source_root, required=True)
    parser.add_argument("--old-exclusion-input", action="append", type=Path, required=True)
    parser.add_argument("--extractive-pool", type=Path, required=True)
    parser.add_argument("--pilot-pool", type=Path, required=True)
    parser.add_argument("--source-gold", type=Path, required=True)
    parser.add_argument("--census-pool", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        args.private_output.resolve().relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise RuntimeError("private output must remain outside the repository")

    extractive_pool = _read_verified(args.extractive_pool, "extractive_pool")
    pilot_pool = _read_verified(args.pilot_pool, "pilot_pool")
    source_gold = _read_verified(args.source_gold, "source_gold")
    census_pool = _read_verified(args.census_pool, "census_pool")
    old_candidates, old_input_hashes = reconstruct_old_population(
        args.source_root, args.old_exclusion_input
    )
    excluded, exclusion_counts = development_exclusions(
        extractive_pool=extractive_pool,
        pilot_pool=pilot_pool,
        source_gold=source_gold,
        old_candidates=old_candidates,
    )
    rows, snapshot = build_inventory(args.source_root, excluded)
    old_population_sources = {str(row["source"]) for row in old_candidates}
    accepted_census_sources = _query_sources(census_pool)
    audit = audit_inventory(
        rows,
        excluded=excluded,
        accepted_census_sources=accepted_census_sources,
        old_population_sources=old_population_sources,
        roots=args.source_root,
    )
    private = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "seed": SEED,
        "source_snapshot_sha256": snapshot["source_snapshot_sha256"],
        "old_input_hashes": old_input_hashes,
        "input_hashes": {
            "extractive_pool": _sha256(args.extractive_pool),
            "pilot_pool": _sha256(args.pilot_pool),
            "source_gold": _sha256(args.source_gold),
            "census_pool": _sha256(args.census_pool),
        },
        "development_exclusion_counts": exclusion_counts,
        "candidates": rows,
    }
    _write_private(args.private_output, private)
    _restrict_private_acl(args.private_output)
    acl_restricted = restricted_ntfs_acl(args.private_output)
    audit["checks"]["restricted_ntfs_acl"] = acl_restricted
    audit["decision"] = (
        "READY_TO_BUILD_ATOMIC_RELEASE_CONFIRMATION"
        if all(audit["checks"].values())
        else "STOP_ATOMIC_RELEASE_CONFIRMATION_INVENTORY"
    )
    public = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "seed": SEED,
        "private_manifest_sha256": _sha256(args.private_output),
        "candidate_manifest_sha256": manifest_sha256(rows),
        "source_snapshot_sha256": snapshot["source_snapshot_sha256"],
        "snapshot_sources": snapshot["snapshot_sources"],
        "parsed_sources_with_views": snapshot["parsed_sources_with_views"],
        "old_candidate_count": len(old_candidates),
        "old_candidate_manifest_sha256": OLD_EXPECTED_MANIFEST_SHA256,
        "old_input_hashes": old_input_hashes,
        "input_hashes": private["input_hashes"],
        "development_exclusion_counts": exclusion_counts,
        "restricted_ntfs_acl": acl_restricted,
        **audit,
    }
    _write_public(args.public_output, public)
    print(json.dumps(public, ensure_ascii=False))
    if public["decision"] != "READY_TO_BUILD_ATOMIC_RELEASE_CONFIRMATION":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
