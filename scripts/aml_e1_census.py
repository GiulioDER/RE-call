"""Stage 0 of the E-1 pre-registration: how often the ordering gate fires, and on what.

Pre-registration: docs/preregistrations/2026-09-25-aml-c9-gated-chronological-order.md.
No model, no service, no network. For each set, the share of questions for which
``recall_aml.order_gate.asks_for_order`` is true, by the set's own question type, with every
pattern's hit count and the text of each fire outside the target type and each miss inside it.

BEAM's ``probing_questions`` cell is parsed by the repository's own size-capped parser
(``benchmarks.beam.dataset._parse_probing``). ScriptMem's question field carries its lettered
options, exactly as AML's ScriptMem pipeline reads it, so the gate sees what a Search would.

    python scripts/aml_e1_census.py --beam 100K.parquet --locomo locomo10.json \\
        --lme longmemeval_s_cleaned.json --scriptmem ScriptMem/data/raw --out census.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.beam.dataset import _parse_probing  # noqa: E402
from recall_aml.order_gate import asks_for_order, matched_patterns  # noqa: E402

SCRIPTMEM_FILES = ("angry.json", "enemy.json", "friends.json", "man_earth.json")


def _tally(by_type: dict[str, list[str]], target: str | None) -> dict[str, Any]:
    types = {}
    patterns: Counter[str] = Counter()
    fires_outside: list[dict[str, str]] = []
    misses_inside: list[str] = []
    for kind, questions in sorted(by_type.items()):
        fired = 0
        for question in questions:
            hits = matched_patterns(question)
            patterns.update(hits)
            if hits:
                fired += 1
                if kind != target:
                    fires_outside.append({"type": kind, "patterns": ", ".join(hits), "question": question[:240]})
            elif kind == target:
                misses_inside.append(question[:240])
        types[kind] = {"n": len(questions), "fired": fired, "share": fired / len(questions) if questions else None}
    everything = [q for questions in by_type.values() for q in questions]
    outside = [q for kind, questions in by_type.items() if kind != target for q in questions]
    report: dict[str, Any] = {
        "n": len(everything),
        "fired": sum(asks_for_order(q) for q in everything),
        "by_type": types,
        "pattern_hits": dict(patterns.most_common()),
        "fires_outside_target": fires_outside,
    }
    if target is not None:
        inside = by_type.get(target, [])
        report["target_type"] = target
        report["recall_on_target"] = sum(asks_for_order(q) for q in inside) / len(inside) if inside else None
        report["false_fire_rate_outside_target"] = (
            sum(asks_for_order(q) for q in outside) / len(outside) if outside else None
        )
        report["misses_inside_target"] = misses_inside
    return report


def beam_census(path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    by_type: dict[str, list[str]] = defaultdict(list)
    for record in pq.read_table(path).to_pylist():
        for kind, items in _parse_probing(record["probing_questions"]).items():
            by_type[kind].extend(str(item["question"]) for item in items)
    return _tally(by_type, "event_ordering")


def locomo_census(path: Path) -> dict[str, Any]:
    by_type: dict[str, list[str]] = defaultdict(list)
    for conversation in json.loads(path.read_text(encoding="utf-8")):
        for qa in conversation["qa"]:
            by_type[f"category_{qa.get('category')}"].append(str(qa["question"]))
    return _tally(by_type, None)


def lme_census(path: Path) -> dict[str, Any]:
    by_type: dict[str, list[str]] = defaultdict(list)
    for q in json.loads(path.read_bytes()):
        by_type[str(q["question_type"])].append(str(q["question"]))
    return _tally(by_type, None)


def scriptmem_census(directory: Path) -> dict[str, Any]:
    by_type: dict[str, list[str]] = defaultdict(list)
    for filename in SCRIPTMEM_FILES:
        for sample in json.loads((directory / filename).read_text(encoding="utf-8")):
            for qa in sample.get("qa", []):
                by_type[str(qa["qa_type"])].append(str(qa["question"]))
    return _tally(by_type, "ordering")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--beam", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--scriptmem", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "beam_100k": beam_census(args.beam),
        "scriptmem": scriptmem_census(args.scriptmem),
        "locomo": locomo_census(args.locomo),
        "longmemeval_s": lme_census(args.lme),
    }
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, section in report.items():
        line = f"{name}: fired {section['fired']}/{section['n']}"
        if "recall_on_target" in section:
            line += (
                f"; recall on {section['target_type']} {section['recall_on_target']:.3f}"
                f"; false fires {section['false_fire_rate_outside_target']:.3f}"
            )
        print(line)


if __name__ == "__main__":
    main()
