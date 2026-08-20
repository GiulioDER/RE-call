"""Build the preregistered stratified subset for the ATM answer-arm comparison.

Deterministic and answer-blind: the order comes from SHA256 of the question id, so it depends on
nothing but the official question file, and rerunning it anywhere reproduces the same 300 rows.

The strata are 120 `number`, 120 `open_end`, 60 `list_recall`, which deliberately OVERSAMPLES the
two deterministic types against their true 35.5 / 50.7 / 13.7 shares. They are free to score, so
buying precision there is cheap. The consequence is that a raw mean over this subset is NOT the
benchmark's QS, and `weights()` exists so every report reweights instead of quietly reporting a
different number under the same name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

STRATA = {"number": 120, "open_end": 120, "list_recall": 60}

#: The true share of each type in the 1,013 question split, counted from the official file. A
#: subset mean must be reweighted by these before it is called QS.
POPULATION = {"number": 360, "open_end": 514, "list_recall": 139}


def _order_key(question_id: str) -> str:
    return hashlib.sha256(question_id.encode("utf-8")).hexdigest()


def select(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Take the first N of each stratum in SHA256 order of the question id."""
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(str(row.get("qtype")), []).append(row)
    chosen: list[dict[str, Any]] = []
    for qtype, count in STRATA.items():
        available = sorted(by_type.get(qtype, []), key=lambda row: _order_key(str(row["id"])))
        if len(available) < count:
            raise ValueError(f"{qtype} has {len(available)} questions, need {count}")
        chosen.extend(available[:count])
    # Sorted by id so the file is stable regardless of stratum order.
    return sorted(chosen, key=lambda row: str(row["id"]))


def weights(counts: Counter[str] | dict[str, int]) -> dict[str, float]:
    """Per-question weights that turn a subset mean back into a population estimate.

    A question of type t stands for `POPULATION[t] / sampled[t]` questions of that type, divided by
    the population total. Reporting a subset mean without this is reporting a different metric.
    """
    total = sum(POPULATION.values())
    return {
        qtype: (POPULATION[qtype] / sampled) / total
        for qtype, sampled in dict(counts).items()
        if sampled and qtype in POPULATION
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = json.loads(args.qa_file.read_text(encoding="utf-8"))
    chosen = select(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" explicitly. Without it Windows writes CRLF, so the same selection produces a
    # different file digest on every platform, and a digest that depends on who wrote it verifies
    # nothing. Caught by generating this file on Windows and on the Linux host and comparing: the
    # selections were identical and the digests were not.
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(chosen, ensure_ascii=False, indent=1) + "\n")

    counts = Counter(str(row.get("qtype")) for row in chosen)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(json.dumps({
        "questions": len(chosen),
        "by_qtype": dict(counts),
        "weights": weights(counts),
        "subset_sha256": digest,
        "first_ids": [str(row["id"]) for row in chosen[:3]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
