"""Every way `scan_truncation.py` could report a false clean, driven as a case matrix.

An audit tool whose failure mode is silence needs its own guard, and the guard has to be run, not
read. The FIRST version of `scan_truncation.py` printed "0 truncated" and exited 0 for: an empty
file, a whitespace-only file, a file of one good row behind four hundred unparseable ones, rows
carrying no `predictions` at all, a byte-order mark that made the first row (holding the only
over-ceiling answer) unparseable, a truncation sitting at `predictions[1]`, and a count one token
under the ceiling. It also crashed with `UnicodeEncodeError` when it DID find something, because
it echoed model output through `repr()` to a cp1252 console, losing the totals and every file
after it in argv. None of that was visible by reading it.

Each case below states the exit code it must produce. Run it after any edit to the scanner:

    python results/mtrag_generation/scripts/check_scan_truncation.py [--corpus <real .jsonl>]

`--corpus` adds the liveness check that matters: against a real file the scanner must exit 0 at
the true ceiling and 1 at `--ceiling 200`. A detector that never fires proves nothing by not
firing.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

SCRIPT = pathlib.Path(__file__).with_name("scan_truncation.py")


def row(text: str, task_id: str = "t<::>1") -> str:
    return json.dumps({"task_id": task_id, "predictions": [{"text": text}]})


def tokens(n: int) -> str:
    """A string of exactly `n` o200k_base tokens."""
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")
    ids = enc.encode("word " * (n * 2))[:n]
    out = enc.decode(ids)
    assert len(enc.encode(out)) == n, "the fixture must sit exactly on the boundary it names"
    return out


def run(*paths: object, ceiling: int = 512) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--ceiling", str(ceiling), *[str(p) for p in paths]],
        capture_output=True, text=True, encoding="cp1252", errors="replace",
    )
    return proc.returncode, proc.stdout + proc.stderr


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Self-check for scan_truncation.py.")
    ap.add_argument("--corpus", type=pathlib.Path, default=None,
                    help="a real .scoring.jsonl, for the liveness check")
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="scan_truncation_cases_") as tmp:
        here = pathlib.Path(tmp)

        def write(name: str, content: str, *, bom: bool = False) -> pathlib.Path:
            path = here / name
            path.write_text(("﻿" if bom else "") + content, encoding="utf-8", newline="\n")
            return path

        over, at, just_under = tokens(520), tokens(512), tokens(511)
        fine = tokens(40) + "."
        # The third field is the required exit code, the fourth the verdict that must appear.
        # ⛔ The exit code ALONE is not enough for several of these. A byte-order mark that hides
        # the over-ceiling first row also exits 1, as UNVERIFIED, so a case asserting only "not 0"
        # passes whether the row was read or dropped. Found by mutating the scanner back to plain
        # utf-8 and watching this matrix stay green.
        FOUND, NEAR, UNVERIFIED, CLEAN = (
            "TRUNCATION FOUND", "NEAR THE CEILING", "UNVERIFIED", "CLEAN")
        cases: list[tuple[str, list[object], int, str]] = [
            ("a clean file exits 0", [write("clean.jsonl", row(fine) + "\n")], 0, CLEAN),
            ("an over-ceiling answer is found", [write("over.jsonl", row(over) + "\n")], 1, FOUND),
            ("exactly at the ceiling is found", [write("at.jsonl", row(at) + "\n")], 1, FOUND),
            ("one token under is escalated",
             [write("under.jsonl", row(just_under) + "\n")], 1, NEAR),
            ("an absent file is not clean", [here / "nope.jsonl"], 1, UNVERIFIED),
            ("an empty file is not clean", [write("empty.jsonl", "")], 1, UNVERIFIED),
            ("a whitespace-only file is not clean",
             [write("blank.jsonl", "\n \n\n")], 1, UNVERIFIED),
            ("garbage lines are not clean",
             [write("garbage.jsonl", row(fine) + "\nnot json\n")], 1, UNVERIFIED),
            ("a non-dict row is not clean",
             [write("scalar.jsonl", "42\n" + row(fine) + "\n")], 1, UNVERIFIED),
            ("a row with no predictions is not clean",
             [write("nopred.jsonl", json.dumps({"task_id": "x"}) + "\n")], 1, UNVERIFIED),
            ("predictions: [] is not clean",
             [write("emptypred.jsonl", json.dumps({"task_id": "x", "predictions": []}) + "\n")],
             1, UNVERIFIED),
            ("text: null is not clean",
             [write("nulltext.jsonl",
                    json.dumps({"task_id": "x", "predictions": [{"text": None}]}) + "\n")],
             1, UNVERIFIED),
            ("a directory argument is not clean", [here], 1, UNVERIFIED),
            # Must be FOUND, not merely non-zero: the point is that the row was READ.
            ("a BOM must not hide the first row",
             [write("bom.jsonl", row(over) + "\n", bom=True)], 1, FOUND),
            ("truncation at predictions[1] is found",
             [write("second.jsonl", json.dumps(
                 {"task_id": "x", "predictions": [{"text": "short."}, {"text": over}]}) + "\n")],
             1, FOUND),
            ("a non-ascii tail does not crash the report",
             [write("unicode.jsonl", row(over[:-1] + "→") + "\n")], 1, FOUND),
            ("one bad file does not hide another's verdict",
             [write("bad.jsonl", "not json\n"), write("good.jsonl", row(fine) + "\n")],
             1, UNVERIFIED),
        ]

        failures: list[str] = []
        for name, paths, expected, verdict in cases:
            code, out = run(*paths)
            crashed = "Traceback" in out
            said = verdict in out
            ok = code == expected and said and not crashed
            print(f"[{'ok  ' if ok else 'FAIL'}] exit {code} (want {expected}) "
                  f"{'CRASHED ' if crashed else ''}"
                  f"{'' if said else f'NO {verdict!r} IN OUTPUT '}{name}")
            if not ok:
                failures.append(f"{name}: exit {code}, wanted {expected} + {verdict!r}"
                                f"\n{out[-400:]}")

        if args.corpus and args.corpus.exists():
            true_ceiling, _ = run(args.corpus, ceiling=512)
            lowered, _ = run(args.corpus, ceiling=200)
            print(f"[{'ok  ' if true_ceiling == 0 else 'FAIL'}] corpus at 512 -> {true_ceiling} (want 0)")
            print(f"[{'ok  ' if lowered == 1 else 'FAIL'}] corpus at 200 -> {lowered} (want 1, live)")
            if true_ceiling != 0:
                failures.append("real corpus at its true ceiling did not come back clean")
            if lowered != 1:
                failures.append("DETECTOR IS DEAD: lowering the ceiling found nothing")
        else:
            print("[warn] no --corpus given, so the detector's LIVENESS was not checked")

    print()
    if failures:
        print(f"{len(failures)} CASE(S) FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"all {len(cases)} cases behaved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
