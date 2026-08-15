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
import io
import json
import os
import random
import tempfile
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
    if limit is not None:
        # `if limit:` read 0 as "no cap" and -1 as "drop the last candidate" via `[:-1]`, both
        # silently. `main` bounds the flag; this reads the argument as written.
        if limit < 1:
            raise ValueError(f"limit must be at least 1, not {limit}")
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


def pack_paths(out: Path) -> tuple[Path, Path]:
    """The two halves of a pack, derived from ONE stem.

    `out.with_suffix(".csv")` replaces the last dotted component while `out.name + "_key.json"`
    appends to the whole name, so the two disagreed on any `--out` containing a dot:
    `--out round2.2026-08-15` gave `round2.csv` beside `round2.2026-08-15_key.json`, halves that
    share no stem, so the obvious sibling lookup finds nothing. Worse, `--out adjudication.v2`
    resolved its CSV to `adjudication.csv`, the committed and human-labelled pack.
    """
    if not out.name:
        # `Path("").name` and `Path("C:/").name` are both empty, and `with_name` then raises a
        # bare `ValueError: WindowsPath('.') has an empty name` — after `build_rows` has already
        # walked the whole corpus. Refused with a sentence, and `main` checks it before that.
        raise SystemExit(f"--out {out} names no file")
    base = out.with_suffix("") if out.suffix in {".csv", ".json"} else out
    return base.with_name(base.name + ".csv"), base.with_name(base.name + "_key.json")


def write_pack(
    rows: list[dict[str, str]],
    key: dict[str, dict[str, str]],
    out: Path,
    *,
    force: bool = False,
) -> tuple[Path, Path]:
    """Write the blind CSV and its key. The ONLY place the pack is serialised.

    Extracted from `main` so a test can assert on the bytes an operator actually gets. It was
    inline, and the test written to cover the injection defence reimplemented this block in its
    own body: removing `_csv_safe` from the line below then left the whole suite green, because
    the assertion sat downstream of the test's copy rather than of this code. A guard that
    reimplements what it guards is testing itself.

    Two refusals, both learned the hard way on a file that cost a human an afternoon:

    **It will not overwrite an existing pack without `force`.** `main`'s default `--out` is the
    committed pack, and the row-count test tells an operator to set `RECALL_PEPS_DIR` and rebuild
    — that exact invocation took the labelled CSV from 5808 bytes to 108 and its 37 verdicts to
    0, exit code 0, no diagnostic. Git tracked it, so it was recoverable; that is luck, not a
    design.

    **The two files land together, or the pack on disk is the one that was already there.** Item
    numbers restart at 1 in every build, so a new CSV beside a previous run's key is not a broken
    pack, it is a pack whose un-blinding record attributes every sentence to the WRONG source
    PEP, and nothing about it looks wrong.

    ⚠️ The first version of this guarantee was false on the ONE path where it mattered. It wrote
    the CSV, then the key, and on failure UNLINKED the CSV. With `--force` over an existing pack
    the CSV it deleted was the human-labelled one `os.replace` had just overwritten, so a failed
    key write destroyed the verdicts outright and left the old key behind. It also used
    `except Exception`, so Ctrl-C during the key write skipped the rollback entirely and produced
    the exact wrong-PEP state the block existed to prevent. Both halves are now staged before
    either lands, the previous CSV's bytes are held for restore, and the handler catches
    `BaseException`.
    """
    if set(key) != {row["item"] for row in rows}:
        # The invariant every reader of this pack assumes, asserted where the pack is made
        # rather than only where it is read.
        raise ValueError("the key must cover exactly the CSV's items")

    csv_path, key_path = pack_paths(out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not force:
        existing = [str(p) for p in (csv_path, key_path) if p.exists()]
        if existing:
            raise SystemExit(
                f"{', '.join(existing)} already exists. Rebuilding replaces a pack that may "
                f"carry adjudicated verdicts; pass --force if that is what you mean"
            )

    # Rendered to strings FIRST, both of them, so a failure serialising the key cannot leave a
    # new CSV beside an old key. newline="" is required by the csv module (it does its own
    # line-ending handling), and lineterminator="\n" then stops it emitting CRLF on Windows.
    # `.gitattributes` normalises to LF on commit either way — `judge_labelling.csv` is committed
    # at 0 CRLF despite `build_beam_labelling.py` writing the default — but a working-tree file
    # whose bytes depend on the OS that wrote it is the thing the freeze discipline exists to
    # prevent.
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=["item", "evidence_sentence", "candidate_target", "your_verdict_Y_or_N"],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows([{k: _csv_safe(v) for k, v in row.items()} for row in rows])
    csv_text = buffer.getvalue()
    key_text = json.dumps(key, indent=1, sort_keys=True, ensure_ascii=False) + "\n"

    # BOTH halves staged before EITHER lands, so a failure writing the second cannot find the
    # first already in place. `os.replace` is atomic per file on both platforms; nothing portable
    # makes the pair atomic, so the residual window is the two renames, and the previous CSV is
    # held in memory to close even that.
    previous_csv = csv_path.read_bytes() if csv_path.exists() else None
    # Collected as they are created, so a failure staging the SECOND still cleans up the first.
    # Assigning both before the `try` leaked the CSV's scratch file whenever the key's staging
    # raised, because the cleanup block was never entered.
    staged: list[Path] = []
    try:
        staged.append(_stage(csv_path, csv_text))
        staged.append(_stage(key_path, key_text))
        os.replace(staged[0], csv_path)
        try:
            os.replace(staged[1], key_path)
        except BaseException as failure:
            # `BaseException`, not `Exception`: Ctrl-C here is the likeliest interruption of all,
            # and it used to skip this entirely.
            _restore(csv_path, previous_csv, failure)
            raise
    finally:
        for scratch in staged:
            scratch.unlink(missing_ok=True)
    return csv_path, key_path


def _stage(path: Path, text: str) -> Path:
    """Write `text` beside `path` under a unique name, ready to be renamed over it.

    Unique, because a fixed `<target>.tmp` is shared by two concurrent builds of the same `--out`
    and one run's rename then commits the other run's bytes — a CSV beside the wrong key, which
    is the state this whole function exists to prevent.
    """
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(handle)
    staged = Path(name)
    staged.write_text(text, encoding="utf-8", newline="")
    return staged


def _restore(csv_path: Path, previous: bytes | None, failure: BaseException) -> None:
    """Put the CSV back as it was, or say plainly that the pack on disk is now mismatched."""
    try:
        if previous is None:
            csv_path.unlink(missing_ok=True)
        else:
            csv_path.write_bytes(previous)
    except OSError as rollback_failed:
        # Windows locks a file the adjudicator has open in a spreadsheet, which is the normal
        # state during labelling, so this is reachable. Raising the rollback's error alone would
        # name the unlink and hide the write failure that caused it AND leave the operator not
        # knowing the two files no longer match.
        raise RuntimeError(
            f"{csv_path} could not be restored after the key write failed ({failure!r}), so the "
            f"CSV and its key on disk are now a MISMATCHED pack: the sheet un-blinds to the "
            f"wrong documents. Restore both from version control before labelling"
        ) from rollback_failed


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
    parser.add_argument(
        "--force", action="store_true",
        help="replace an existing pack. The default --out is the COMMITTED pack, so without "
             "this a rebuild refuses rather than discarding adjudicated verdicts",
    )
    args = parser.parse_args()

    # Both checked BEFORE the census walks 733 files, so a malformed argument costs a second
    # rather than a minute and does not surface as a bare `ValueError` from `with_name`.
    pack_paths(args.out)
    if args.limit is not None and args.limit < 1:
        raise SystemExit(f"--limit must be at least 1, not {args.limit}")

    rows, key = build_rows(args.peps_dir, seed=args.seed, limit=args.limit)
    if not rows:
        raise SystemExit("no candidates selected")

    csv_path, key_path = write_pack(rows, key, args.out, force=args.force)

    print(f"{len(rows)} items\n  {csv_path}\n  {key_path}   <- do NOT open until labelling is done")


if __name__ == "__main__":
    main()
