"""Audit frozen extractive gold for leakage resistant selector training."""

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

from recall.index import chunk_text


FROZEN_POOL_SHA256 = "66ec82a058c9b06cf80314a1001779a1608e5144e097ace28b91664a48ede855"
FROZEN_COMPARISON_SHA256 = (
    "6dd9485b12bf0c88d166cc29114cd03ada1e732e47b11592a4725e91d7324e68"
)
DEFAULT_SEED = "extractive-selector-source-split-v1"
PAIR_ID = re.compile(r"^extractive-(positive|negative)-(\d{3})$")
CONTROL_IDENTIFIER = re.compile(r"ZXQXACT-\d{4}", re.IGNORECASE)
WORD = re.compile(r"\b[\w'-]+\b", re.UNICODE)
WHITESPACE = re.compile(r"\s+")
FAMILY_PREFIXES = (
    "closed-hypothesis",
    "todo-autorun",
    "review-due",
    "project",
    "feedback",
    "reference",
    "lesson",
    "incident",
)
TEMPLATE_PREFIXES = {
    "field_fact": "What fact was recorded for ",
    "field_verdict": "What verdict was recorded for ",
    "field_result": "What result was recorded for ",
    "field_outcome": "What outcome was recorded for ",
    "field_decision": "What decision was recorded for ",
    "field_apply": "How should ",
    "field_why": "Why does ",
    "field_status": "What status was recorded for ",
    "field_objective": "What was the objective of ",
    "field_root_cause": "What root cause was recorded for ",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize(value: str) -> str:
    return WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value).casefold()).strip()


def _collapse_ws(value: str) -> str:
    return WHITESPACE.sub(" ", value).strip()


def _nearest_rank(values: list[int]) -> dict[str, int | None]:
    if not values:
        return {name: None for name in ("min", "p10", "p25", "p50", "p75", "p90", "max")}
    ordered = sorted(values)

    def percentile(fraction: float) -> int:
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]

    return {
        "min": ordered[0],
        "p10": percentile(0.10),
        "p25": percentile(0.25),
        "p50": percentile(0.50),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "max": ordered[-1],
    }


def _split(source_sha256: str, seed: str) -> str:
    digest = hashlib.sha256(f"{seed}\0{source_sha256}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket <= 69:
        return "train"
    if bucket <= 84:
        return "validation"
    return "internal_test"


def _template(query: str, construction: str) -> str:
    if construction == "extractive_heading":
        return "heading"
    if construction == "extractive_fallback":
        return "fallback"
    for name, prefix in TEMPLATE_PREFIXES.items():
        if query.startswith(prefix):
            return name
    return "unknown"


def _source_family(source: str) -> str:
    root, _, relative = source.partition("/")
    stem = Path(relative).stem.casefold().replace("_", "-")
    for prefix in FAMILY_PREFIXES:
        if stem == prefix or stem.startswith(prefix + "-"):
            return f"{root}/{prefix}"
    return f"{root}/other"


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


def _rule_metrics(labels: list[bool], predictions: list[bool]) -> dict[str, float | int]:
    answerable = sum(not label for label in labels)
    unanswerable = sum(labels)
    false_positive = sum(pred and not label for label, pred in zip(labels, predictions))
    false_negative = sum(not pred and label for label, pred in zip(labels, predictions))
    correct = len(labels) - false_positive - false_negative
    true_positive_rate = (unanswerable - false_negative) / unanswerable if unanswerable else 0.0
    true_negative_rate = (answerable - false_positive) / answerable if answerable else 0.0
    return {
        "accuracy": round(correct / len(labels), 6) if labels else 0.0,
        "balanced_accuracy": round((true_positive_rate + true_negative_rate) / 2, 6),
        "answerable_false_positives": false_positive,
        "unanswerable_false_negatives": false_negative,
    }


def _count_overlap(left: Iterable[str], right: Iterable[str]) -> int:
    return len(set(left) & set(right))


def _comparison_sets(payload: dict[str, Any]) -> tuple[set[str], set[str]]:
    sources: set[str] = set()
    source_hashes: set[str] = set()
    for row in payload.get("queries", []):
        if row.get("expected_answerability") != "answerable":
            continue
        sources.update(str(value) for value in row.get("gold_sources", []))
        if row.get("source_sha256"):
            source_hashes.add(str(row["source_sha256"]))
    return sources, source_hashes


def audit_pool(
    pool_path: Path,
    comparison_path: Path,
    roots: dict[str, Path],
    *,
    seed: str = DEFAULT_SEED,
    enforce_frozen_hashes: bool = True,
) -> dict[str, Any]:
    pool_sha256 = _sha256(pool_path)
    comparison_sha256 = _sha256(comparison_path)
    if enforce_frozen_hashes and pool_sha256 != FROZEN_POOL_SHA256:
        raise ValueError(f"POOL_HASH_MISMATCH: {pool_sha256}")
    if enforce_frozen_hashes and comparison_sha256 != FROZEN_COMPARISON_SHA256:
        raise ValueError(f"COMPARISON_HASH_MISMATCH: {comparison_sha256}")

    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    rows = list(pool.get("queries", []))
    answerable = [row for row in rows if row.get("expected_answerability") == "answerable"]
    controls = [row for row in rows if row.get("expected_answerability") == "unanswerable"]

    pairs: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"positive": [], "negative": []}
    )
    malformed_pair_ids = 0
    for row in rows:
        match = PAIR_ID.fullmatch(str(row.get("id", "")))
        if match is None:
            malformed_pair_ids += 1
            continue
        pairs[int(match.group(2))][match.group(1)].append(row)
    complete_pairs = sum(
        len(pair["positive"]) == 1
        and len(pair["negative"]) == 1
        and pair["positive"][0].get("expected_answerability") == "answerable"
        and pair["negative"][0].get("expected_answerability") == "unanswerable"
        for pair in pairs.values()
    )
    malformed_pairs = sum(
        len(pair["positive"]) != 1 or len(pair["negative"]) != 1 for pair in pairs.values()
    ) + malformed_pair_ids

    source_missing = 0
    source_hash_mismatches = 0
    answer_hash_mismatches = 0
    containment_mismatches = 0
    ordinal_mismatches = 0
    answer_char_counts: list[int] = []
    answer_word_counts: list[int] = []
    boundary_clearance: list[int] = []
    boundary_below_40 = 0
    construction_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    source_root_counts: Counter[str] = Counter()
    source_family_counts: Counter[str] = Counter()
    ordinal_counts: Counter[str] = Counter()

    for row in answerable:
        query = str(row.get("query", ""))
        span = str(row.get("answer_span", ""))
        construction = str(row.get("construction", ""))
        sources = [str(value) for value in row.get("gold_sources", [])]
        source = sources[0] if len(sources) == 1 else ""
        source_root = source.partition("/")[0] if source else "unknown"
        construction_counts[construction] += 1
        template_counts[_template(query, construction)] += 1
        source_root_counts[source_root] += 1
        source_family_counts[_source_family(source) if source else "unknown"] += 1
        ordinal_counts[str(row.get("gold_ordinal"))] += 1
        answer_char_counts.append(len(span))
        answer_word_counts.append(len(WORD.findall(span)))
        if row.get("answer_span_sha256") != _text_sha256(span):
            answer_hash_mismatches += 1

        path = _source_path(source, roots)
        if path is None or not path.is_file():
            source_missing += 1
            continue
        if _sha256(path) != row.get("source_sha256"):
            source_hash_mismatches += 1
        raw = path.read_text(encoding="utf-8", errors="replace")
        chunks = [_collapse_ws(chunk) for chunk in chunk_text(raw)]
        normalized_span = _collapse_ws(span)
        matching = [index for index, chunk in enumerate(chunks) if normalized_span in chunk]
        if len(matching) != 1:
            containment_mismatches += 1
            continue
        ordinal = int(row.get("gold_ordinal", -1))
        if matching[0] != ordinal:
            ordinal_mismatches += 1
        chunk = chunks[matching[0]]
        start = chunk.find(normalized_span)
        clearance = min(start, len(chunk) - start - len(normalized_span))
        boundary_clearance.append(clearance)
        if clearance < 40:
            boundary_below_40 += 1

    answerable_sources = {
        str(row["gold_sources"][0])
        for row in answerable
        if len(row.get("gold_sources", [])) == 1
    }
    answerable_source_hashes = {
        str(row["source_sha256"]) for row in answerable if row.get("source_sha256")
    }
    answerable_questions = {_normalize(str(row.get("query", ""))) for row in answerable}
    answerable_answer_hashes = {
        str(row["answer_span_sha256"]) for row in answerable if row.get("answer_span_sha256")
    }
    comparison_sources, comparison_source_hashes = _comparison_sets(comparison)

    split_rows: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "internal_test": [],
    }
    pair_split: dict[int, str] = {}
    for row in answerable:
        source_hash = str(row.get("source_sha256", ""))
        split = _split(source_hash, seed)
        split_rows[split].append(row)
        match = PAIR_ID.fullmatch(str(row.get("id", "")))
        if match:
            pair_split[int(match.group(2))] = split
    split_all_rows: dict[str, list[dict[str, Any]]] = {
        name: list(values) for name, values in split_rows.items()
    }
    for row in controls:
        match = PAIR_ID.fullmatch(str(row.get("id", "")))
        if match and int(match.group(2)) in pair_split:
            split_all_rows[pair_split[int(match.group(2))]].append(row)

    split_metrics: dict[str, Any] = {}
    split_sets: dict[str, dict[str, set[str]]] = {}
    for name in ("train", "validation", "internal_test"):
        positives = split_rows[name]
        all_split = split_all_rows[name]
        split_metrics[name] = {
            "answerable": len(positives),
            "controls": len(all_split) - len(positives),
            "construction_counts": dict(sorted(Counter(str(row["construction"]) for row in positives).items())),
            "source_family_counts": dict(
                sorted(
                    Counter(
                        _source_family(str(row["gold_sources"][0]))
                        for row in positives
                        if len(row.get("gold_sources", [])) == 1
                    ).items()
                )
            ),
        }
        split_sets[name] = {
            "query": {_normalize(str(row.get("query", ""))) for row in all_split},
            "source": {
                str(row["gold_sources"][0])
                for row in positives
                if len(row.get("gold_sources", [])) == 1
            },
            "source_sha256": {
                str(row["source_sha256"]) for row in positives if row.get("source_sha256")
            },
            "answer_span_sha256": {
                str(row["answer_span_sha256"])
                for row in positives
                if row.get("answer_span_sha256")
            },
        }

    split_overlaps: dict[str, dict[str, int]] = {}
    split_pairs = (
        ("train", "validation"),
        ("train", "internal_test"),
        ("validation", "internal_test"),
    )
    for left, right in split_pairs:
        split_overlaps[f"{left}_{right}"] = {
            key: _count_overlap(split_sets[left][key], split_sets[right][key])
            for key in ("query", "source", "source_sha256", "answer_span_sha256")
        }

    labels = [row.get("expected_answerability") == "unanswerable" for row in rows]
    identifier_predictions = [bool(CONTROL_IDENTIFIER.search(str(row.get("query", "")))) for row in rows]
    intro_predictions = [
        str(row.get("query", "")).startswith("According to memory item ") for row in rows
    ]
    positive_tokens = {
        token
        for row in answerable
        for token in WORD.findall(_normalize(str(row.get("query", ""))))
        if len(token) >= 3
    }
    control_token_sets = [
        {
            token
            for token in WORD.findall(_normalize(str(row.get("query", ""))))
            if len(token) >= 3
        }
        for row in controls
    ]
    universal_control_tokens = (
        set.intersection(*control_token_sets) - positive_tokens if control_token_sets else set()
    )
    identifier_rule_metrics = _rule_metrics(labels, identifier_predictions)
    intro_rule_metrics = _rule_metrics(labels, intro_predictions)
    control_metrics = {
        "identifier_rule": identifier_rule_metrics,
        "intro_rule": intro_rule_metrics,
        "class_exclusive_universal_token_count": len(universal_control_tokens),
        "class_exclusive_universal_tokens": sorted(universal_control_tokens),
    }

    all_cross_source_overlap_zero = all(
        values["source"] == 0 for values in split_overlaps.values()
    )
    all_cross_hash_overlap_zero = all(
        values["source_sha256"] == 0 for values in split_overlaps.values()
    )
    all_cross_query_overlap_zero = all(
        values["query"] == 0 for values in split_overlaps.values()
    )
    positive_checks = {
        "complete_pairs": (
            len(answerable) == 250
            and len(controls) == 250
            and complete_pairs == 250
            and malformed_pairs == 0
        ),
        "sources_exist": source_missing == 0,
        "source_hashes_match": source_hash_mismatches == 0,
        "span_containment_matches": containment_mismatches == 0 and ordinal_mismatches == 0,
        "distinct_source_hashes": len(answerable_source_hashes) >= 240,
        "split_sizes": (
            len(split_rows["train"]) >= 150
            and len(split_rows["validation"]) >= 25
            and len(split_rows["internal_test"]) >= 25
        ),
        "cross_split_source_paths": all_cross_source_overlap_zero,
        "cross_split_source_hashes": all_cross_hash_overlap_zero,
        "cross_split_queries": all_cross_query_overlap_zero,
        "comparison_source_paths": not (answerable_sources & comparison_sources),
        "comparison_source_hashes": not (answerable_source_hashes & comparison_source_hashes),
        "construction_diversity": all(
            len({str(row["construction"]) for row in split_rows[name]}) >= 2
            for name in split_rows
        ),
    }
    positive_passed = all(positive_checks.values())
    control_checks = {
        "identifier_rule_balanced_accuracy": (
            identifier_rule_metrics["balanced_accuracy"] <= 0.60
        ),
        "intro_rule_balanced_accuracy": (
            intro_rule_metrics["balanced_accuracy"] <= 0.60
        ),
        "no_class_exclusive_universal_tokens": len(universal_control_tokens) == 0,
    }
    control_passed = all(control_checks.values())
    if not positive_passed:
        decision = "STOP_CURRENT_POOL"
    elif control_passed:
        decision = "GO_FULL_SELECTOR_WITH_NULL"
    else:
        decision = "GO_POSITIVE_SELECTOR_ONLY"

    return {
        "schema_version": 1,
        "protocol": "2026-09-16-extractive-gold-shape-source-split-audit",
        "pool_sha256": pool_sha256,
        "comparison_pool_sha256": comparison_sha256,
        "split_seed": seed,
        "decision": decision,
        "gates": {
            "positive_supervision": {"passed": positive_passed, "checks": positive_checks},
            "control_supervision": {"passed": control_passed, "checks": control_checks},
        },
        "metrics": {
            "integrity": {
                "answerable_rows": len(answerable),
                "control_rows": len(controls),
                "complete_pairs": complete_pairs,
                "malformed_pairs": malformed_pairs,
                "missing_sources": source_missing,
                "source_hash_mismatches": source_hash_mismatches,
                "answer_hash_mismatches": answer_hash_mismatches,
                "containment_mismatches": containment_mismatches,
                "ordinal_mismatches": ordinal_mismatches,
            },
            "answer_shape": {
                "character_count": _nearest_rank(answer_char_counts),
                "word_count": _nearest_rank(answer_word_counts),
                "boundary_clearance": _nearest_rank(boundary_clearance),
                "boundary_clearance_below_40": boundary_below_40,
            },
            "construction_counts": dict(sorted(construction_counts.items())),
            "template_counts": dict(sorted(template_counts.items())),
            "source_root_counts": dict(sorted(source_root_counts.items())),
            "source_family_counts": dict(sorted(source_family_counts.items())),
            "gold_ordinal_counts": dict(sorted(ordinal_counts.items())),
            "uniques": {
                "normalized_questions": len(answerable_questions),
                "source_paths": len(answerable_sources),
                "source_sha256": len(answerable_source_hashes),
                "answer_span_sha256": len(answerable_answer_hashes),
            },
            "comparison_overlap": {
                "source_paths": len(answerable_sources & comparison_sources),
                "source_sha256": len(answerable_source_hashes & comparison_source_hashes),
            },
            "splits": split_metrics,
            "split_overlaps": split_overlaps,
            "controls": control_metrics,
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
    parser.add_argument("--comparison-pool", type=Path, required=True)
    parser.add_argument("--source-root", action="append", type=_root, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()
    result = audit_pool(
        args.pool,
        args.comparison_pool,
        dict(args.source_root),
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"output": str(args.output), "decision": result["decision"]}))


if __name__ == "__main__":
    main()
