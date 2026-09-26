"""Stage 0 of the E-2 pre-registration: both ordering gates on questions E-1 never read.

Pre-registration: docs/preregistrations/2026-09-25-aml-c9-revised-ordering-gate.md.
No model, no service. Two parts:

``count``  Runs E-1's ``asks_for_order`` and E-2's ``asks_for_order_v2`` over every held-out
           question and writes the counts. On BEAM 500K and 1M (labelled) it reports each gate's
           recall on ``event_ordering`` and false-fire rate elsewhere. On the four unlabelled
           sources it pools every question EITHER gate fires on, deduplicates by text, shuffles with
           ``random.Random(20260925)``, and writes two files: a judging file holding only an opaque
           id, the source and the question as displayed, and a separate key naming which gate
           fired. The judge reads the first; the key is read only after the judgements are
           committed. Counts that would reveal a gate per question are not printed.

``unblind``  Joins the committed judgements to the key and reports, per gate, the fires judged
             ordering and not ordering.

Questions are read with X-1's own loaders (``scripts/aml_x1_sources.py``), so the fields are the
ones X-1 settled; PersonaMem-v2 queries are taken before X-1's recall suffix. A CLBench question
is the final user turn of the task. Long questions are displayed to the judge as their first and
last 400 characters, since a document-length turn carries its ask at an end.

    python scripts/aml_e2_census.py count --beam500k 500K.parquet --beam1m 1M.parquet \\
        --x1-data ~/mm1-mm3/x1/data --out census.json --judge judge.jsonl --key key.json
    python scripts/aml_e2_census.py unblind --judged judged.jsonl --key key.json --out unblinded.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmarks.beam.dataset import _parse_probing  # noqa: E402
from recall_aml.order_gate import asks_for_order, asks_for_order_v2  # noqa: E402
import aml_x1_sources as x1  # noqa: E402

SEED = 20260925
DISPLAY_HEAD = DISPLAY_TAIL = 400


def beam_questions(path: Path) -> dict[str, list[str]]:
    import pyarrow.parquet as pq

    by_type: dict[str, list[str]] = defaultdict(list)
    for cell in pq.read_table(path, columns=["probing_questions"]).column("probing_questions").to_pylist():
        for kind, items in _parse_probing(cell).items():
            by_type[kind].extend(str(item["question"]) for item in items)
    return by_type


def beam_tally(by_type: dict[str, list[str]]) -> dict[str, Any]:
    target = by_type.get("event_ordering", [])
    others = [q for kind, qs in by_type.items() if kind != "event_ordering" for q in qs]
    report: dict[str, Any] = {"n": sum(len(v) for v in by_type.values()), "event_ordering": len(target),
                              "other": len(others)}
    for name, gate in (("e1", asks_for_order), ("e2", asks_for_order_v2)):
        report[name] = {
            "recall_on_event_ordering": sum(map(gate, target)) / len(target) if target else None,
            "fired_on_event_ordering": sum(map(gate, target)),
            "false_fire_rate_other": sum(map(gate, others)) / len(others) if others else None,
            "fired_on_other": sum(map(gate, others)),
            "by_type": {kind: sum(map(gate, qs)) for kind, qs in sorted(by_type.items())},
        }
    report["misses_on_event_ordering"] = {
        name: [q for q in target if not gate(q)] for name, gate in (("e1", asks_for_order), ("e2", asks_for_order_v2))
    }
    report["fires_on_other"] = {
        name: [{"type": kind, "question": q} for kind, qs in sorted(by_type.items()) if kind != "event_ordering"
               for q in qs if gate(q)]
        for name, gate in (("e1", asks_for_order), ("e2", asks_for_order_v2))
    }
    return report


def memlens_questions(data: Path) -> Iterator[str]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(data / x1.PINNED["memlens_32k"][0].local)
    for batch in parquet.iter_batches(batch_size=16, columns=["question"]):
        for item in batch.to_pylist():
            yield str(item["question"])


def mobilemem_questions(data: Path) -> Iterator[str]:
    for record in x1.iter_jsonl(data / "mobilemem_omni/filtered_questions.jsonl"):
        if record["language"] == "en":
            for q in record["questions"]:
                yield str(q["question"])


def personamem_questions(data: Path) -> Iterator[str]:
    for _, row in x1.personamem_rows(data):
        yield x1.personamem_query(row["user_query"])


def clbench_questions(data: Path) -> Iterator[str]:
    for item in x1.iter_jsonl(data / x1.PINNED["clbench"][0].local):
        users = [m for m in item["messages"] if m["role"] == "user"]
        if users:
            yield str(users[-1]["content"])


UNLABELLED = {
    "memlens_32k": memlens_questions,
    "mobilemem_omni_en": mobilemem_questions,
    "personamem_v2": personamem_questions,
    "clbench": clbench_questions,
}


def display(question: str) -> str:
    if len(question) <= DISPLAY_HEAD + DISPLAY_TAIL:
        return question
    return f"{question[:DISPLAY_HEAD]}\n[... {len(question) - DISPLAY_HEAD - DISPLAY_TAIL} characters ...]\n{question[-DISPLAY_TAIL:]}"


def count(args: argparse.Namespace) -> None:
    report: dict[str, Any] = {"preregistration": "docs/preregistrations/2026-09-25-aml-c9-revised-ordering-gate.md"}
    report["beam_500k"] = beam_tally(beam_questions(args.beam500k))
    report["beam_1m"] = beam_tally(beam_questions(args.beam1m))
    pooled: dict[str, dict[str, Any]] = {}
    report["unlabelled"] = {}
    for source, reader in UNLABELLED.items():
        n = e1 = e2 = 0
        for question in reader(args.x1_data):
            n += 1
            f1, f2 = asks_for_order(question), asks_for_order_v2(question)
            e1 += f1
            e2 += f2
            if f1 or f2:
                digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
                entry = pooled.setdefault(digest, {"source": source, "question": question, "e1": False, "e2": False})
                entry["e1"] |= f1
                entry["e2"] |= f2
        report["unlabelled"][source] = {"n": n, "e1_fired": e1, "e2_fired": e2}
    items = sorted(pooled.values(), key=lambda entry: hashlib.sha256(entry["question"].encode()).hexdigest())
    random.Random(SEED).shuffle(items)
    key = {}
    with args.judge.open("w", encoding="utf-8") as sink:
        for index, entry in enumerate(items):
            ident = f"j{index:04d}"
            sink.write(json.dumps({"id": ident, "source": entry["source"], "question": display(entry["question"]),
                                   "ordering": None}, ensure_ascii=False) + "\n")
            key[ident] = {"e1": entry["e1"], "e2": entry["e2"]}
    args.key.write_text(json.dumps(key, indent=1), encoding="utf-8")
    report["judge_items"] = len(items)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for name in ("beam_500k", "beam_1m"):
        section = report[name]
        print(name, {g: (section[g]["fired_on_event_ordering"], section["event_ordering"],
                         section[g]["fired_on_other"], section["other"]) for g in ("e1", "e2")})
    for source, section in report["unlabelled"].items():
        print(source, "n", section["n"], "fired: e1", section["e1_fired"], "e2", section["e2_fired"])
    print("judge items", len(items))


def unblind(args: argparse.Namespace) -> None:
    key = json.loads(args.key.read_text(encoding="utf-8"))
    judged = [json.loads(line) for line in args.judged.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [row["id"] for row in judged if row["ordering"] not in (True, False)]
    if missing:
        raise SystemExit(f"{len(missing)} items have no judgement, e.g. {missing[:3]}")
    result: dict[str, Any] = {}
    for gate in ("e1", "e2"):
        rows = [row for row in judged if key[row["id"]][gate]]
        by_source: dict[str, dict[str, int]] = defaultdict(lambda: {"ordering": 0, "not_ordering": 0})
        for row in rows:
            by_source[row["source"]]["ordering" if row["ordering"] else "not_ordering"] += 1
        ordering = sum(row["ordering"] for row in rows)
        result[gate] = {"fires": len(rows), "judged_ordering": ordering, "judged_not_ordering": len(rows) - ordering,
                        "share_ordering": ordering / len(rows) if rows else None, "by_source": dict(by_source)}
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("count")
    c.add_argument("--beam500k", type=Path, required=True)
    c.add_argument("--beam1m", type=Path, required=True)
    c.add_argument("--x1-data", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--judge", type=Path, required=True)
    c.add_argument("--key", type=Path, required=True)
    c.set_defaults(run=count)
    u = sub.add_parser("unblind")
    u.add_argument("--judged", type=Path, required=True)
    u.add_argument("--key", type=Path, required=True)
    u.add_argument("--out", type=Path, required=True)
    u.set_defaults(run=unblind)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
