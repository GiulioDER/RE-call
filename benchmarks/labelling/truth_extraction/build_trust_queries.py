"""The downstream trust query set: does retrieval prefer the successor over the stale document?

One row per superseded PEP, not one row per header edge. Five superseded PEPs have TWO
successors each (pep-0563, for one, is superseded by both pep-0649 and pep-0749), so a row per
edge would emit two rows with identical `query` text and identical `stale_ids`, differing only in
which single successor each expects. Identical query text retrieves identically, so at most one
row of each such pair could ever score correct, capping the successor arm's ceiling below 100%
without saying so. Grouping by the superseded PEP and collecting every successor into one row's
`successor_ids` removes that ceiling: a row is correct when retrieval prefers ANY of its
successors over the stale document.

The shipped set in `recall/eval/queries.json` has 6 trust rows, of which only 4 expect a successor,
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

    # Group by the superseded PEP: a PEP with two successors gets ONE row whose successor_ids
    # holds both, rather than two rows sharing identical query text and disjoint successor_ids.
    by_superseded: dict[str, list[str]] = {}
    for edge in census.edges:
        by_superseded.setdefault(edge.superseded, []).append(edge.successor)

    for i, stale in enumerate(sorted(by_superseded), 1):
        successors = sorted(by_superseded[stale])
        title = _fields(peps_dir, stale).get("Title", "")
        rows.append({
            "id": f"pt{i:02d}",
            "query": title.lower(),
            "trust": True,
            "expect": "successor",
            "stale_ids": [f"{stale}.rst:0"],
            "successor_ids": [f"{s}.rst:0" for s in successors],
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
