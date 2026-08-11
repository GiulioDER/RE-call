"""The downstream trust query set: does retrieval prefer the successor over the stale document?

Each `(superseded, successor)` header edge is a natural `(stale_ids, successor_ids)` row. The
shipped set in `recall/eval/queries.json` has 6 trust rows, of which only 4 expect a successor,
and a Wilson interval on n=4 is uninterpretable.

A NEW file rather than an edit to `queries.json`: that file's ids address the synthetic memo
corpus under `recall/eval/corpus`, so appending PEP rows would build a query set no single corpus
can serve. The schema is identical.

Query text is the superseded PEP's `Title:`, taken mechanically. All 733 carry one, so this needs
no judgement and has no exceptions. It is a weaker probe than a real user question — a title is
what the document is called, not what someone would ask — and the artifact says so.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from benchmarks.labelling.truth_extraction.census import compute_census
from benchmarks.labelling.truth_extraction.peps_header import header_fields, split_header


def _fields(peps_dir: Path, stem: str) -> dict[str, str]:
    head, _ = split_header((peps_dir / f"{stem}.rst").read_text(
        encoding="utf-8", errors="replace"))
    return header_fields(head)


def build_queries(peps_dir: Path, *, n_abstain: int, seed: int) -> list[dict]:
    census = compute_census(peps_dir)
    rows: list[dict] = []

    for i, edge in enumerate(census.edges, 1):
        title = _fields(peps_dir, edge.superseded).get("Title", "")
        rows.append({
            "id": f"pt{i:02d}",
            "query": title.lower(),
            "trust": True,
            "expect": "successor",
            "stale_ids": [f"{edge.superseded}.rst:0"],
            "successor_ids": [f"{edge.successor}.rst:0"],
        })

    in_an_edge = {e.superseded for e in census.edges} | {e.successor for e in census.edges}
    abstain_pool = []
    for path in sorted(peps_dir.glob("pep-*.rst")):
        if path.stem in in_an_edge:
            continue
        fields = _fields(peps_dir, path.stem)
        if fields.get("Status") in {"Withdrawn", "Rejected"} and not fields.get("Superseded-By"):
            abstain_pool.append((path.stem, fields.get("Title", "")))

    random.Random(seed).shuffle(abstain_pool)
    for i, (stem, title) in enumerate(abstain_pool[:n_abstain], len(rows) + 1):
        rows.append({
            "id": f"pt{i:02d}",
            "query": title.lower(),
            "trust": True,
            "expect": "abstain",
            "stale_ids": [f"{stem}.rst:0"],
            "successor_ids": [],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peps-dir", type=Path, required=True)
    parser.add_argument("--n-abstain", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("recall/eval/peps_trust_queries.json"))
    args = parser.parse_args()

    rows = build_queries(args.peps_dir, n_abstain=args.n_abstain, seed=args.seed)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    successors = sum(1 for r in rows if r["expect"] == "successor")
    print(f"{args.out}\n  {len(rows)} queries ({successors} successor / "
          f"{len(rows) - successors} abstain)")


if __name__ == "__main__":
    main()
