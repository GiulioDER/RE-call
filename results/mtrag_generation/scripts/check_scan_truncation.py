"""Every way `scan_truncation.py` could report a false clean, driven as a case matrix.

An audit tool whose failure mode is silence needs its own guard, and the guard has to be run, not
read. Two rounds of audit found these, none of them visible by reading the scanner: it printed
"0 truncated" and exited 0 for an empty file, a whitespace-only file, one good row behind four
hundred unparseable ones, rows carrying no `predictions`, a byte-order mark that made the first
row (holding the only over-ceiling answer) unparseable, a truncation at `predictions[1]`, a count
one token under the ceiling, and a row whose readable prediction sat beside an unreadable one. It
also crashed with `UnicodeEncodeError` when it DID find something, because it echoed model output
through `repr()` to a cp1252 console, losing the totals and every file after it in argv.

Run it after any edit to the scanner:

    python results/mtrag_generation/scripts/check_scan_truncation.py [--corpus <real .jsonl>]

Each case asserts the per-file VERDICT, not merely a non-zero exit. Several inputs exit non-zero
either way: a byte-order mark that hides an over-ceiling first row exits 1 as UNVERIFIED, so a
case demanding only "not clean" passes whether the row was read or dropped. Verdicts are parsed
out of the scanner's own summary block and compared per file, because a substring test over the
whole output cannot tell which file a verdict belonged to.

`--corpus` adds the liveness check that matters: against a real file the scanner must exit 0 at
the true ceiling and report a finding at `--ceiling 200`. A detector that never fires proves
nothing by not firing. A `--corpus` path that does not exist is a FAILURE, not a skip.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

SCRIPT = pathlib.Path(__file__).with_name("scan_truncation.py")
FOUND, NEAR, UNVERIFIED, CLEAN = ("TRUNCATION FOUND", "NEAR THE CEILING", "UNVERIFIED", "CLEAN")

#: Forced on the child rather than inherited. The ascii()/repr() guard only bites on a console
#: that cannot encode the character, so on Linux, macOS, or Windows under PYTHONUTF8 an inherited
#: UTF-8 stdout makes that case inert and the matrix reports 17/17 while the crash is live.
CHILD_ENCODING = "cp1252"


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


def run(*paths: object, ceiling: int | None = 512,
        extra: list[str] | None = None) -> tuple[int, str, dict]:
    """Returns (exit code, output, {filename: verdict}) with the verdict block parsed.

    `--expect` is derived from the paths' own stems unless the caller supplies one, so each case
    still measures the thing it is named for rather than the coverage rule. Cases that test the
    coverage rule pass `extra` explicitly and that wins. `ceiling=None` omits `--ceiling`, which
    is the only way to exercise the scanner's DEFAULT: every other call passes it, so a drifting
    default was invisible to the whole matrix while the documented command relies on it.
    """
    extra = list(extra or [])
    args = [str(p) for p in paths]
    if not any(a == "--expect" for a in extra):
        stems = sorted({pathlib.Path(a).name.split(".")[0] for a in args
                        if not any(ch in a for ch in "*?[")})
        if stems:
            extra += ["--expect", ",".join(stems)]
    if not any(a == "--expect-rows" for a in extra):
        # An explicit disclaimer, not silence: `--expect-rows` is required, and a case that is not
        # about file size should say so rather than be exempted from saying anything. The cases
        # that DO test size pass a number, and the requirement itself is pinned by a direct
        # invocation that bypasses this helper.
        extra += ["--expect-rows", "unclaimed"]
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         *(["--ceiling", str(ceiling)] if ceiling is not None else []), *extra, *args],
        capture_output=True, text=True, encoding=CHILD_ENCODING, errors="replace",
        env={**os.environ, "PYTHONIOENCODING": CHILD_ENCODING},
    )
    out = proc.stdout + proc.stderr
    verdicts: dict[str, str] = {}
    if "verdicts:" in out:
        for line in out.split("verdicts:", 1)[1].splitlines():
            if not line.startswith("  "):
                continue
            # Split on the KNOWN label, not on the last space. `rpartition(" ")` mis-keyed any
            # filename containing a space, which on Windows is ordinary, and on the --corpus path
            # that produced the false report "DETECTOR IS DEAD" while the detector had fired.
            stripped = line.strip()
            for label in (FOUND, NEAR, UNVERIFIED, CLEAN):
                if stripped.startswith(label):
                    verdicts[stripped[len(label):].strip()] = label
                    break
    return proc.returncode, out, verdicts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Self-check for scan_truncation.py.")
    ap.add_argument("--corpus", type=pathlib.Path, default=None,
                    help="a real .scoring.jsonl, for the liveness check")
    args = ap.parse_args(argv)
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="scan_truncation_cases_") as tmp:
        here = pathlib.Path(tmp)

        def write(name: str, content: str, *, bom: bool = False) -> pathlib.Path:
            path = here / name
            path.write_text(("﻿" if bom else "") + content, encoding="utf-8", newline="\n")
            return path

        over, at, just_under = tokens(520), tokens(512), tokens(511)
        band_edge, outside_band = tokens(500), tokens(499)  # NEAR_BAND is 12: 500 in, 499 out
        fine = tokens(40) + "."
        many = "".join(row(fine, f"t{i}") + "\n" for i in range(300))

        cases: list[tuple[str, list[object], int, dict[str, str]]] = [
            ("a clean file exits 0", [write("clean.jsonl", row(fine) + "\n")], 0,
             {"clean.jsonl": CLEAN}),
            ("an over-ceiling answer is found", [write("over.jsonl", row(over) + "\n")], 1,
             {"over.jsonl": FOUND}),
            ("exactly at the ceiling is found", [write("at.jsonl", row(at) + "\n")], 1,
             {"at.jsonl": FOUND}),
            ("one token under is escalated", [write("under.jsonl", row(just_under) + "\n")], 1,
             {"under.jsonl": NEAR}),
            # Both edges of the near band, so its WIDTH is pinned and not just its existence.
            ("the far edge of the near band is escalated",
             [write("edge.jsonl", row(band_edge) + "\n")], 1, {"edge.jsonl": NEAR}),
            ("one token outside the band is clean",
             [write("outside.jsonl", row(outside_band) + "\n")], 0, {"outside.jsonl": CLEAN}),
            ("an absent file is not clean", [here / "nope.jsonl"], 1,
             {"nope.jsonl": UNVERIFIED}),
            ("an empty file is not clean", [write("empty.jsonl", "")], 1,
             {"empty.jsonl": UNVERIFIED}),
            ("a whitespace-only file is not clean", [write("blank.jsonl", "\n \n\n")], 1,
             {"blank.jsonl": UNVERIFIED}),
            ("garbage lines are not clean",
             [write("garbage.jsonl", row(fine) + "\nnot json\n")], 1,
             {"garbage.jsonl": UNVERIFIED}),
            ("a non-dict row is not clean",
             [write("scalar.jsonl", "42\n" + row(fine) + "\n")], 1, {"scalar.jsonl": UNVERIFIED}),
            ("a row with no predictions is not clean",
             [write("nopred.jsonl", json.dumps({"task_id": "x"}) + "\n")], 1,
             {"nopred.jsonl": UNVERIFIED}),
            ("predictions: [] is not clean",
             [write("emptypred.jsonl", json.dumps({"task_id": "x", "predictions": []}) + "\n")],
             1, {"emptypred.jsonl": UNVERIFIED}),
            ("text: null is not clean",
             [write("nulltext.jsonl",
                    json.dumps({"task_id": "x", "predictions": [{"text": None}]}) + "\n")],
             1, {"nulltext.jsonl": UNVERIFIED}),
            # One readable prediction must not vouch for an unreadable sibling.
            ("a good prediction beside an unreadable one is not clean",
             [write("sibling.jsonl", json.dumps(
                 {"task_id": "x", "predictions": [{"text": fine}, {"response": over}]}) + "\n")],
             1, {"sibling.jsonl": UNVERIFIED}),
            ("a nested prediction is not clean",
             [write("nested.jsonl", json.dumps(
                 {"task_id": "x", "predictions": [{"text": fine}, [{"text": over}]]}) + "\n")],
             1, {"nested.jsonl": UNVERIFIED}),
            # Multi-row and NOT in the last row, so the counter's ACCUMULATION is pinned. With a
            # one-row fixture, `unmeasurable = dropped` (assignment, not +=) passed every case
            # while reinstating the defect on any real file, where later clean rows reset it to 0.
            ("an unreadable prediction in row 0 of 4 is not clean",
             [write("early.jsonl", json.dumps(
                 {"task_id": "x", "predictions": [{"text": fine}, {"response": over}]}) + "\n"
                 + "".join(row(fine, f"r{i}") + "\n" for i in range(3)))],
             1, {"early.jsonl": UNVERIFIED}),
            ("an empty-string answer is not clean",
             [write("emptytext.jsonl", json.dumps(
                 {"task_id": "x", "predictions": [{"text": ""}]}) + "\n")],
             1, {"emptytext.jsonl": UNVERIFIED}),
            # Whichever value `json` keeps depends on ORDER, so both orderings must be refused.
            ("a duplicate key with the long value first is not clean",
             [write("dup1.jsonl",
                    '{"task_id": "x", "predictions": [{"text": %s, "text": %s}]}\n'
                    % (json.dumps(over), json.dumps(fine)))], 1, {"dup1.jsonl": UNVERIFIED}),
            ("a duplicate key with the long value last is not clean",
             [write("dup2.jsonl",
                    '{"task_id": "x", "predictions": [{"text": %s, "text": %s}]}\n'
                    % (json.dumps(fine), json.dumps(over)))], 1, {"dup2.jsonl": UNVERIFIED}),
            ("a directory argument is not clean", [here], 1, {here.name: UNVERIFIED}),
            ("a BOM must not hide the first row",
             [write("bom.jsonl", row(over) + "\n", bom=True)], 1, {"bom.jsonl": FOUND}),
            ("truncation at predictions[1] is found",
             [write("second.jsonl", json.dumps(
                 {"task_id": "x", "predictions": [{"text": "short."}, {"text": over}]}) + "\n")],
             1, {"second.jsonl": FOUND}),
            ("a non-ascii tail does not crash the report",
             [write("unicode.jsonl", row(over[:-1] + "→") + "\n")], 1,
             {"unicode.jsonl": FOUND}),
            # The whole file must be read: a scanner that stops early passes every one-line case.
            ("the LAST row of 300 is still measured",
             [write("long.jsonl", many + row(over, "last") + "\n")], 1, {"long.jsonl": FOUND}),
            ("300 rows with nothing wrong are clean", [write("long_clean.jsonl", many)], 0,
             {"long_clean.jsonl": CLEAN}),
            ("one good row behind 400 bad ones is not clean",
             [write("mostly_bad.jsonl", row(fine) + "\n" + "not json\n" * 400)], 1,
             {"mostly_bad.jsonl": UNVERIFIED}),
            # A finding must outrank an unread condition, or the report hides what it was run for.
            ("a finding is named even when the file is also unread",
             [write("both.jsonl", row(over) + "\nnot json\n")], 1, {"both.jsonl": FOUND}),
            ("each file gets its own verdict",
             [write("bad.jsonl", "not json\n"), write("good.jsonl", row(fine) + "\n")], 1,
             {"bad.jsonl": UNVERIFIED, "good.jsonl": CLEAN}),
            # The same pair in the OPPOSITE order. Without it, a scanner that only inspects
            # reports[0] passes every case while certifying a truncated second file.
            ("a finding in the SECOND file is not clean",
             [write("good2.jsonl", row(fine) + "\n"), write("bad2.jsonl", row(over) + "\n")], 1,
             {"good2.jsonl": CLEAN, "bad2.jsonl": FOUND}),
        ]

        for name, paths, expected, wanted in cases:
            code, out, verdicts = run(*paths)
            crashed = "Traceback" in out
            wrong = {f: (verdicts.get(f), v) for f, v in wanted.items() if verdicts.get(f) != v}
            ok = code == expected and not wrong and not crashed
            print(f"[{'ok  ' if ok else 'FAIL'}] exit {code} (want {expected}) "
                  f"{'CRASHED ' if crashed else ''}{'' if not wrong else f'{wrong} '}{name}")
            if not ok:
                failures.append(f"{name}: exit {code} (want {expected}), verdicts {wrong}"
                                f"\n{out[-400:]}")

        # The statistics RESULTS.md publishes come from code no verdict can see. The fixture is
        # 10/20/60 tokens so mean (30.0), p50 (20) and p95 (60) are pairwise DISTINCT: with 40/41/42
        # a mutation swapping two of them stays green.
        stats = write("stats.jsonl", "".join(row(tokens(n), f"s{n}") + "\n" for n in (10, 20, 60)))
        _, out, _ = run(stats)
        for fragment in ("rows 3  distinct task_ids 3  answers 3", "mean 30.0", "p50 20",
                         "p95 60", "max 60", "unreadable predictions 0"):
            ok = fragment in out
            print(f"[{'ok  ' if ok else 'FAIL'}] summary line reports {fragment!r}")
            if not ok:
                failures.append(f"summary line missing {fragment!r}\n{out[-400:]}")

        # The punctuation check is quoted in RESULTS.md ("three answers of 3,368 end without
        # terminal punctuation"), and no verdict depends on it, so it needs its own assertion.
        # The third row ends in "?" ON PURPOSE: with only a bare word and a full stop, shrinking
        # TERMINAL to ('.',) gives the same count, so the fixture could not see the mutation.
        punct = write("punct.jsonl",
                      row(tokens(10), "a") + "\n"
                      + row(tokens(10) + ".", "b") + "\n"
                      + row(tokens(10) + "?", "c") + "\n")
        _, out, _ = run(punct)
        ok = "no terminal punctuation: 1 of 3" in out
        print(f"[{'ok  ' if ok else 'FAIL'}] the punctuation count covers every terminal mark")
        if not ok:
            failures.append(f"punctuation count wrong\n{out[-400:]}")

        # A counter that accumulates needs a fixture where it must reach more than one.
        two_bad = write("twobad.jsonl", "".join(json.dumps(
            {"task_id": f"x{i}", "predictions": [{"text": fine}, {"response": over}]}) + "\n"
            for i in range(2)))
        _, out, _ = run(two_bad)
        ok = "unreadable predictions 2" in out
        print(f"[{'ok  ' if ok else 'FAIL'}] unreadable predictions accumulate across rows")
        if not ok:
            failures.append(f"the unreadable-prediction counter does not accumulate\n{out[-400:]}")

        # A wildcard must not quietly decide what CLEAN is a statement about. The unit is RUNS, and
        # a run restores several layers, so these fixtures mimic that shape: runA has two layers,
        # runB has one, and a pattern can match all of runA while runB is absent entirely.
        (here / "runA.predictions.jsonl").write_text(row(fine) + "\n", encoding="utf-8")
        (here / "runA.scoring.jsonl").write_text(row(fine) + "\n", encoding="utf-8")
        (here / "runB.predictions.jsonl").write_text(row(fine) + "\n", encoding="utf-8")
        both = str(here / "run?.predictions.jsonl")
        only_a = str(here / "runA.*.jsonl")
        for label, args_extra, want_code, target, must_say in [
            ("a pattern covering both runs is clean",
             ["--expect", "runA,runB"], 0, both, "runs covered: runA, runB"),
            ("a pattern without --expect cannot certify", [], 1, both, "--expect"),
            # ⛔ The round-4 defect, pinned. Two files, both CLEAN, both from ONE run: a COUNT of
            # two is satisfied while the other run is never named. Only a coverage check by name
            # can fail this, which is why --expect takes stems.
            ("two layers of ONE run cannot stand for two runs",
             ["--expect", "runA,runB"], 1, only_a, "no file was scanned for runB"),
            ("the run stems covered are always reported", [], 1, only_a, "runs covered: runA"),
            # `must_say` matters here: a pattern matching nothing exits 1 under the fallback (the
            # pattern is kept and reported ABSENT) AND under a scanner that drops the fallback and
            # scans nothing at all. Only the text tells those two apart, and the second is a
            # scanner that certifies an empty universe.
            ("a pattern matching nothing is reported ABSENT",
             ["--expect", "runA,runB"], 1, str(here / "nothing_*.jsonl"), "ABSENT"),
        ]:
            code, out, _ = run(target, extra=args_extra)
            ok = code == want_code and must_say in out
            print(f"[{'ok  ' if ok else 'FAIL'}] exit {code} (want {want_code}) "
                  f"{'' if must_say in out else f'NO {must_say!r} IN OUTPUT '}{label}")
            if not ok:
                failures.append(f"{label}: exit {code} (want {want_code}), "
                                f"{must_say!r} present: {must_say in out}\n{out[-400:]}")

        # The premise of the whole tool is the GENERATOR's encoding, and a default that drifts to
        # another BPE changes every count without changing any verdict.
        _, out, _ = run(stats)
        ok = "encoding o200k_base" in out
        print(f"[{'ok  ' if ok else 'FAIL'}] default encoding is o200k_base (gpt-4o)")
        if not ok:
            failures.append("the default encoding is no longer o200k_base")

        # The DEFAULT ceiling has the same property and had no assertion: every other call passes
        # --ceiling, so raising the default to 4096 left the whole matrix green while the command
        # RESULTS.md publishes, which passes no --ceiling, certified a truncated run CLEAN.
        code, out, verdicts = run(write("default.jsonl", row(at) + "\n"), ceiling=None)
        ok = (code == 1 and verdicts.get("default.jsonl") == FOUND
              and "ceiling 512 completion tokens" in out)
        print(f"[{'ok  ' if ok else 'FAIL'}] the default ceiling is 512 and finds a 512-token answer")
        if not ok:
            failures.append(f"default ceiling wrong: exit {code}, "
                            f"verdict {verdicts.get('default.jsonl')}\n{out[-300:]}")

        # Run coverage matches on BASENAME, so without a size claim a one-row file carrying the
        # right name certifies a whole run. The archive's manifests record `tasks: 842`.
        # 300 rows, not 3: at 3 the mutant `rows is not expected_rows` stays green, because small
        # ints are interned and `3 is 3` while `300 is not 300`. Every real file is far above the
        # cache, so a fixture inside it cannot see that mutation at all.
        sized = write("sized.jsonl", "".join(row(fine, f"n{i}") + "\n" for i in range(300)))
        oversized = write("oversized.jsonl", "".join(row(fine, f"n{i}") + "\n" for i in range(302)))
        dupids = write("dupids.jsonl", "".join(row(fine, "same") + "\n" for _ in range(300)))
        for label, target, rows_arg, want_code, want_verdict in [
            ("the right row count is clean", sized, "300", 0, CLEAN),
            ("a SHORT file cannot certify a run", sized, "842", 1, UNVERIFIED),
            # Both directions: with only short-vs-equal, weakening `!=` to `<` survives.
            ("an OVERSIZED file cannot certify a run", oversized, "300", 1, UNVERIFIED),
            # `tasks: 842` counts tasks; `rows` counts lines. Repeated ids reach the count while
            # fewer than 842 of the run's tasks are present.
            ("repeated task_ids cannot reach the count", dupids, "300", 1, UNVERIFIED),
            ("--expect-rows 0 claims nothing", sized, "0", 1, None),
            ("--expect-rows that is not a number is refused", sized, "many", 1, None),
        ]:
            name = pathlib.Path(str(target)).name
            code, out, verdicts = run(
                target, extra=["--expect", name.split(".")[0], "--expect-rows", rows_arg])
            ok = code == want_code and (want_verdict is None or verdicts.get(name) == want_verdict)
            print(f"[{'ok  ' if ok else 'FAIL'}] exit {code} (want {want_code}) {label}")
            if not ok:
                failures.append(f"{label}: exit {code} (want {want_code})\n{out[-300:]}")

        # A ceiling nobody could have sent must not certify anything. Driven at a real over-ceiling
        # answer, so the mutant that drops the bound is caught by a false CLEAN rather than by an
        # argument-parsing detail.
        at_512 = write("ceilingclaim.jsonl", row(at, "c0") + "\n")
        for label, ceiling_arg, want in [("absurdly high", "100000", 1), ("zero", "0", 1),
                                         ("the real one", "512", 1)]:
            code, out, _ = run(at_512, ceiling=None,
                               extra=["--expect", "ceilingclaim", "--expect-rows", "1",
                                      "--ceiling", ceiling_arg])
            ok = code == want
            print(f"[{'ok  ' if ok else 'FAIL'}] exit {code} (want {want}) a {label} ceiling cannot "
                  f"certify a 512-token answer")
            if not ok:
                failures.append(f"ceiling {ceiling_arg} gave exit {code}\n{out[-300:]}")

        # Two independent guards refuse `--expect-rows 0`: the parse-time rejection, and
        # `expected_rows is not None` in `unread`. Either alone makes the case above exit 1, so
        # that case pins the PAIR and neither member. Assert the parse-time complaint's own text,
        # so the two mechanisms have one assertion each.
        code, out, _ = run(sized, extra=["--expect", "sized", "--expect-rows", "0"])
        ok = "claims no rows" in out
        print(f"[{'ok  ' if ok else 'FAIL'}] a zero size claim is refused BY NAME, not as a mismatch")
        if not ok:
            failures.append(f"--expect-rows 0 was not refused at parse time\n{out[-300:]}")

        # The mechanism must be REQUIRED, not merely available: the previous round added it and
        # left it optional, so the false clean it was written to close stayed one omitted flag away.
        bare_rows = subprocess.run(
            [sys.executable, str(SCRIPT), "--expect", "sized", str(sized)],
            capture_output=True, text=True, encoding=CHILD_ENCODING, errors="replace",
            env={**os.environ, "PYTHONIOENCODING": CHILD_ENCODING},
        )
        ok = bare_rows.returncode == 1 and "how large these runs should be" in bare_rows.stdout
        print(f"[{'ok  ' if ok else 'FAIL'}] exit {bare_rows.returncode} (want 1) no --expect-rows "
              f"cannot certify")
        if not ok:
            failures.append(f"a run without --expect-rows certified a file\n{bare_rows.stdout[-300:]}")

        # ...but an EXPLICIT disclaimer is allowed, and says so in the report.
        code, out, verdicts = run(sized, extra=["--expect", "sized", "--expect-rows", "unclaimed"])
        ok = code == 0 and verdicts.get("sized.jsonl") == CLEAN and "rows expected unclaimed" in out
        print(f"[{'ok  ' if ok else 'FAIL'}] exit {code} (want 0) an explicit size disclaimer is "
              f"allowed and printed")
        if not ok:
            failures.append(f"the explicit size disclaimer did not work\n{out[-300:]}")

        # A degenerate --expect must not satisfy the requirement it appears to satisfy.
        for label, expect in [("--expect ,", ","), ("--expect empty", ""), ("--expect spaces", " ")]:
            code, out, _ = run(write("cov.jsonl", row(fine) + "\n"), extra=["--expect", expect])
            ok = code == 1 and "names no run" in out
            print(f"[{'ok  ' if ok else 'FAIL'}] exit {code} (want 1) {label} claims nothing")
            if not ok:
                failures.append(f"{label} was accepted as a coverage claim\n{out[-300:]}")

        # And no --expect at all cannot certify, on ANY input, not just an expanded wildcard:
        # under bash the shell expands first, so a gate conditioned on globbing is off by default.
        # Invoked directly rather than through run(), which now supplies --expect for every case.
        bare = subprocess.run(
            [sys.executable, str(SCRIPT), str(write("noexp.jsonl", row(fine) + "\n"))],
            capture_output=True, text=True, encoding=CHILD_ENCODING, errors="replace",
            env={**os.environ, "PYTHONIOENCODING": CHILD_ENCODING},
        )
        ok = bare.returncode == 1 and "which runs this was meant to cover" in bare.stdout
        print(f"[{'ok  ' if ok else 'FAIL'}] exit {bare.returncode} (want 1) a clean literal path "
              f"without --expect still cannot certify")
        if not ok:
            failures.append(f"a run without --expect certified a file anyway\n{bare.stdout[-300:]}")

    if args.corpus is None:
        print("[warn] no --corpus given, so the detector's LIVENESS was not checked")
    elif not args.corpus.exists():
        # A typo in the path must not silently skip the only check that proves it fires.
        print(f"[FAIL] --corpus {args.corpus} does not exist")
        failures.append(f"--corpus {args.corpus} does not exist, so liveness was never checked")
    else:
        true_ceiling, _, _ = run(args.corpus, ceiling=512)
        lowered, _, verdicts = run(args.corpus, ceiling=200)
        live = verdicts.get(args.corpus.name) == FOUND
        print(f"[{'ok  ' if true_ceiling == 0 else 'FAIL'}] corpus at 512 -> {true_ceiling} (want 0)")
        print(f"[{'ok  ' if live else 'FAIL'}] corpus at 200 -> {verdicts.get(args.corpus.name)} "
              f"(want {FOUND}, detector is live)")
        if true_ceiling != 0:
            failures.append("the real corpus did not come back clean at its true ceiling")
        if not live:
            failures.append("DETECTOR IS DEAD: lowering the ceiling did not produce a finding")

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("every case behaved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
