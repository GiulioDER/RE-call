"""Build a BLIND adjudication set over prose markers that no header confirms.

The 175 PEPs carrying a closure marker with no corresponding header edge are the PEPs analogue of
the 60-versus-2 gap on the private corpus, and they are where `fix.py`'s four measured false
positives lived. A negative label here is a human judgement, so it is made blind: the adjudicator
sees the evidence sentence and the candidate target and nothing about what surfaced them.

Blank is data. `score_beam_labels.read_verdict` reads an empty cell as *undecidable* and EXCLUDES
it, rather than counting it against whichever arm happened to be labelled. An adjudicator who
cannot tell should leave the cell empty.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from benchmarks.labelling.truth_extraction.census import compute_census
from benchmarks.labelling.truth_extraction.peps_header import pep_refs, sentences, split_header
from recall.lint import CLOSURE_MARKERS

#: Characters a spreadsheet executes as a formula rather than displaying. Same defence as
#: `build_beam_labelling._csv_safe`: these cells are third-party text, not author-written.
_FORMULA_LEAD = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(value: str) -> str:
    return "'" + value if value and value[0] in _FORMULA_LEAD else value


def build_rows(
    peps_dir: Path, *, seed: int, limit: int | None
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    census = compute_census(peps_dir)
    candidates: list[dict[str, str]] = []

    for stem in census.marker_without_header:
        _, body = split_header((peps_dir / f"{stem}.rst").read_text(
            encoding="utf-8", errors="replace"))
        for sentence in sentences(body):
            if not CLOSURE_MARKERS.search(sentence):
                continue
            refs = sorted(pep_refs(sentence) - {stem})
            # No target named in the sentence is not a negative — it is unprovable, and
            # `fix.py` reports that class rather than guessing at it. Excluded from adjudication.
            for target in refs:
                candidates.append({
                    "source_pep": stem,
                    "candidate_target": target,
                    "evidence_sentence": sentence.strip(),
                })

    random.Random(seed).shuffle(candidates)
    if limit:
        candidates = candidates[:limit]

    rows: list[dict[str, str]] = []
    key: dict[str, dict[str, str]] = {}
    for i, cand in enumerate(candidates, 1):
        key[str(i)] = dict(cand)
        rows.append({
            "item": str(i),
            "evidence_sentence": cand["evidence_sentence"],
            "candidate_target": cand["candidate_target"],
            "your_verdict_Y_or_N": "",
        })
    return rows, key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peps-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap items; applied AFTER the shuffle so the subset stays uniform")
    parser.add_argument(
        "--out", type=Path,
        default=Path("benchmarks/labelling/truth_extraction/adjudication"),
    )
    args = parser.parse_args()

    rows, key = build_rows(args.peps_dir, seed=args.seed, limit=args.limit)
    if not rows:
        raise SystemExit("no candidates selected")

    csv_path = args.out.with_suffix(".csv")
    key_path = args.out.parent / (args.out.name + "_key.json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" is required by the csv module (it does its own line-ending handling), and
    # lineterminator="\n" then stops it emitting CRLF on Windows. `.gitattributes` normalises to
    # LF on commit either way — `judge_labelling.csv` is committed at 0 CRLF despite
    # `build_beam_labelling.py` writing the default — but a working-tree file whose bytes depend
    # on the OS that wrote it is the thing the freeze discipline exists to prevent.
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["item", "evidence_sentence", "candidate_target", "your_verdict_Y_or_N"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows([{k: _csv_safe(v) for k, v in row.items()} for row in rows])
    with key_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(key, indent=1, sort_keys=True, ensure_ascii=False) + "\n")

    print(f"{len(rows)} items\n  {csv_path}\n  {key_path}   <- do NOT open until labelling is done")


if __name__ == "__main__":
    main()
