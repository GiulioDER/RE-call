"""Measure one hop benchmark structural edge reachability on LoCoMo.

Prior work: [[2026-09-10-graph-activation-diagnostic]] — graph activation requires seed coverage;
no prior result measured these new benchmark structural relations on raw LoCoMo questions.

This is an offline mechanism probe. It uses a fixed lexical seed ranking and compares baseline
evidence reachability with one hop expansion from the exact same seeds. It does not claim an
embedding, answer quality, or production serving improvement.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from benchmarks.structural_edges import locomo_structural_metadata

TOKEN_RE = re.compile(r"[a-z0-9]+")
PROTOCOL = "2026-09-10-structural-edge-effect-v1"
K = 5


def _tokens(text: Any) -> set[str]:
    return set(TOKEN_RE.findall(str(text).casefold()))


def _turns(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for key in sorted(
        (key for key in conversation if re.fullmatch(r"session_\d+", str(key))),
        key=lambda item: int(item.split("_")[1]),
    ):
        for turn in conversation.get(key, ()):
            if isinstance(turn, dict) and isinstance(turn.get("dia_id"), str):
                turns.append(turn)
    return turns


def _baseline_ids(turns: list[dict[str, Any]], question: str) -> tuple[str, ...]:
    query_tokens = _tokens(question)
    ranked = sorted(
        enumerate(turns),
        key=lambda indexed_turn: (
            -len(query_tokens & _tokens(indexed_turn[1].get("text", ""))),
            indexed_turn[0],
        ),
    )
    return tuple(str(turn["dia_id"]) for _index, turn in ranked[:K])


def _structural_neighbors(
    conversation: dict[str, Any],
) -> tuple[dict[str, tuple[tuple[str, str, str], ...]], Counter[str]]:
    metadata = locomo_structural_metadata(conversation)
    neighbors: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for source, value in metadata.items():
        for relation in value.get("relations", ()):
            target_file = relation.get("object")
            if not isinstance(target_file, str):
                continue
            target = target_file.removesuffix(".md").replace("_", ":", 1)
            edge_type = str(relation.get("structural_type", "unknown"))
            edge_key = str(relation.get("structural_key", ""))
            source_id = source.removesuffix(".md").replace("_", ":", 1)
            neighbors[source_id].append((target, edge_type, edge_key))
            neighbors[target].append((source_id, edge_type, edge_key))
    return (
        {key: tuple(value) for key, value in neighbors.items()},
        Counter(
            relation.get("structural_type", "unknown")
            for value in metadata.values()
            for relation in value.get("relations", ())
        ),
    )


def _expanded_ids(
    baseline: tuple[str, ...],
    turns: list[dict[str, Any]],
    neighbors: dict[str, tuple[tuple[str, str, str], ...]],
) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    turn_order = {str(turn["dia_id"]): index for index, turn in enumerate(turns)}
    additions: dict[str, tuple[str, str, str]] = {}
    for seed in baseline:
        for target, edge_type, edge_key in neighbors.get(seed, ()):
            if target not in baseline and target in turn_order:
                candidate = (edge_type, edge_key, target)
                previous = additions.get(target)
                if previous is None or candidate < previous:
                    additions[target] = candidate
    ordered = sorted(
        additions,
        key=lambda target: (
            additions[target][0],
            additions[target][1],
            turn_order[target],
        ),
    )
    details = [
        {"dia_id": target, "structural_type": additions[target][0], "structural_key": additions[target][1]}
        for target in ordered
    ]
    return tuple((*baseline, *ordered)), details


def run(data_path: Path) -> dict[str, Any]:
    conversations = json.loads(data_path.read_text(encoding="utf-8"))
    observations: list[dict[str, Any]] = []
    edge_counts: Counter[str] = Counter()
    for conversation_index, conversation in enumerate(conversations):
        if not isinstance(conversation, dict):
            continue
        inner = conversation.get("conversation", conversation)
        if not isinstance(inner, dict):
            continue
        turns = _turns(inner)
        neighbors, counts = _structural_neighbors(inner)
        edge_counts.update(counts)
        for question_index, question_row in enumerate(conversation.get("qa", ())):
            if not isinstance(question_row, dict) or question_row.get("category") not in {1, 2, 3, 4}:
                continue
            evidence = tuple(item for item in question_row.get("evidence", ()) if isinstance(item, str))
            question = question_row.get("question")
            if not evidence or not isinstance(question, str):
                continue
            baseline = _baseline_ids(turns, question)
            expanded, additions = _expanded_ids(baseline, turns, neighbors)
            gold = set(evidence)
            baseline_hit = gold.issubset(baseline)
            treatment_hit = gold.issubset(expanded)
            observations.append(
                {
                    "question_id": f"conversation_{conversation_index}:qa_{question_index}",
                    "category": question_row["category"],
                    "evidence": list(evidence),
                    "baseline_ids": list(baseline),
                    "expanded_ids": list(expanded),
                    "additions": additions,
                    "baseline_hit": baseline_hit,
                    "treatment_hit": treatment_hit,
                }
            )
    baseline_hits = sum(item["baseline_hit"] for item in observations)
    treatment_hits = sum(item["treatment_hit"] for item in observations)
    baseline_misses = [item for item in observations if not item["baseline_hit"]]
    rescues = sum(item["treatment_hit"] for item in baseline_misses)
    expanded_precision = [
        len(set(item["evidence"]) & set(item["expanded_ids"])) / len(item["expanded_ids"])
        if item["expanded_ids"] else 1.0
        for item in observations
    ]
    return {
        "protocol": PROTOCOL,
        "data_sha256": sha256(data_path.read_bytes()).hexdigest(),
        "questions": len(observations),
        "conversations": len(conversations),
        "k": K,
        "edge_counts": dict(sorted(edge_counts.items())),
        "summary": {
            "baseline_hit_rate": baseline_hits / len(observations) if observations else None,
            "structural_one_hop_hit_rate": treatment_hits / len(observations) if observations else None,
            "absolute_delta": (treatment_hits - baseline_hits) / len(observations) if observations else None,
            "baseline_misses": len(baseline_misses),
            "rescues": rescues,
            "rescue_rate_among_baseline_misses": rescues / len(baseline_misses) if baseline_misses else None,
            "mean_expanded_candidates": sum(len(item["expanded_ids"]) for item in observations) / len(observations) if observations else None,
            "mean_expanded_precision": sum(expanded_precision) / len(expanded_precision) if expanded_precision else None,
        },
        "by_category": {
            str(category): {
                "questions": sum(item["category"] == category for item in observations),
                "baseline_hits": sum(item["category"] == category and item["baseline_hit"] for item in observations),
                "treatment_hits": sum(item["category"] == category and item["treatment_hit"] for item in observations),
            }
            for category in (1, 2, 3, 4)
        },
        "observations": observations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("locomo10.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    summary = result["summary"]
    print(
        f"n={result['questions']} baseline={summary['baseline_hit_rate']:.4f} "
        f"treatment={summary['structural_one_hop_hit_rate']:.4f} "
        f"delta={summary['absolute_delta']:.4f} rescues={summary['rescues']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
