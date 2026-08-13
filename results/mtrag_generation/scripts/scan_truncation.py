"""Which archived MTRAG answers were cut off by the `--max-tokens` ceiling.

The six generation runs sent `--max-tokens 512` through a `generate_one` that never read
`finish_reason`, so a completion the ceiling stopped came back as an ordinary string, was written
to the submission and was judged as if the system had produced it. The code path raises now, but
the fix is forward-only and **the stop reason was never recorded**, so for the artifacts it has to
be recovered from the text.

Recovered by re-tokenising with the generator's own encoding (gpt-4o -> `o200k_base`). That is
close to exact rather than a proxy: a completion stopped by the ceiling carries EXACTLY 512
completion tokens, one the model ended carries fewer, and re-tokenising a string is deterministic
under the same BPE. The one edge case is the trailing whitespace token that `generate_one`'s
`.strip()` removes, which can move a count by one, so a row just under the ceiling is reported as
a finding rather than logged and passed.

⛔ **The only failure that matters here is reporting "clean" for rows nobody read.** Nothing that
went unread can end as CLEAN: an absent file, an empty one, a line that will not parse, a repeated
JSON key, a row with no `predictions`, a prediction this tool cannot read even when a readable one
sits beside it, or a scan that raised, all end as UNVERIFIED. A finding outranks that and ends as
TRUNCATION FOUND or NEAR THE CEILING, so those two are not clean either. Exit is 0 only when every
file came back CLEAN. Silence is not evidence, and this whole audit exists because a stop reason
nobody recorded was later read as a stop reason nobody needed.

A byte-order mark is the one case handled the other way round, deliberately: the file opens as
`utf-8-sig` so the mark is consumed and the first row is READ, rather than dropped as unparseable
and the file taken for a shorter one.

Reads any file whose rows carry `task_id` and `predictions[].text`: `.predictions.jsonl`,
`.scoring.jsonl` and `.scored.jsonl` all qualify. EVERY element of `predictions` is measured, not
just the first. ⚠️ The guarantee is over `predictions[].text` only: an answer stored under some
other key INSIDE a prediction object that also carries a readable `text` is not inspected, because
a prediction object's other keys are metadata in every file this has been pointed at.

    python results/mtrag_generation/scripts/scan_truncation.py <file> [<file> ...] [--ceiling N]

`--expect` is REQUIRED on every run, naming the run stems that must be covered:

    ... --expect taskc_benchmark_official,taskc_recall_official <restored>/taskc_*_official.*.jsonl

⚠️ Names, not a count. Completeness here is a property of RUNS, and each archived run restores five
`.jsonl` layers, so a count is both unsatisfiable on a correct restore and satisfiable by five
files of one run. It is required unconditionally rather than only when this script expands a
wildcard, because under bash the SHELL expands it first and a gate conditioned on that is switched
off by the operator's choice of shell. Nothing here can certify what nobody claimed. Every report
also ends with the stems it covered, so the universe is legible in the report itself.

A wildcard argument is expanded here as well, since PowerShell does not expand arguments and the
operator would otherwise be told a pattern is ABSENT.

Needs `tiktoken`. See `../runs/README.md` for restoring the payload pack these files come from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Answers ending in any of these look deliberately finished. A truncated one almost never does,
#: but a short refusal can end without one, so this only ever qualifies the token count.
TERMINAL = ('.', '!', '?', '"', "'", ')', ']', '`', ':', ';', '*')

#: How far below the ceiling still counts as a finding. Re-tokenising a mid-word cut is not
#: guaranteed to reproduce the sampled count, so the boundary is a band, not a line.
NEAR_BAND = 12


def _safe(value: object) -> str:
    """ASCII-safe rendering, because this prints to a Windows console under cp1252.

    `repr()` passes printable non-ASCII straight through, so echoing model output with `{x!r}`
    raises `UnicodeEncodeError`. That fires ONLY on the truncation-found branch, which means the
    unguarded version destroyed its own report exactly when it had something to report, and took
    the grand total and every later file down with it.
    """
    return ascii(value)


class FileReport:
    """One file's verdict. `CLEAN` requires that nothing was skipped, not merely that nothing
    was found: an unread row is unverified, and unverified is not clean."""

    def __init__(self, name: str, expected_rows: int | None = None) -> None:
        self.name = name
        self.expected_rows = expected_rows
        self.rows = 0
        self.unparseable = 0
        self.no_answer = 0
        self.unmeasurable = 0
        self.unterminated = 0
        self.error: str | None = None
        self.counts: list[int] = []
        self.at_ceiling: list[tuple[str, int, str]] = []
        self.near: list[tuple[str, int, str]] = []

    @property
    def unread(self) -> bool:
        # `unmeasurable` counts PREDICTIONS skipped, not rows. Without it, a row holding one
        # well-shaped prediction beside one this tool cannot read (`{"response": ...}`, a nested
        # list, a null text, a duplicate JSON key) counted as fully read, so a truncated answer in
        # the unreadable slot came back CLEAN. One good sibling must not vouch for another.
        # `expected_rows` closes the last false clean the audits found: run coverage matches on
        # BASENAME, so a one-row file named `taskc_recall_official.predictions.jsonl` certified
        # that run CLEAN while 841 answers were never measured. A name is not a size.
        return (
            bool(self.error)
            or self.unparseable > 0
            or self.no_answer > 0
            or self.unmeasurable > 0
            or self.rows == 0
            or (self.expected_rows is not None and self.rows != self.expected_rows)
        )

    @property
    def verdict(self) -> str:
        # A fixed four-word vocabulary: CLEAN, TRUNCATION FOUND, NEAR THE CEILING, UNVERIFIED.
        # Advice belongs in the block above, not inside the label, because the label is what
        # `check_scan_truncation.py` matches on and what a reader greps for.
        if self.at_ceiling:
            return "TRUNCATION FOUND"
        if self.near:
            return "NEAR THE CEILING"
        if self.unread:
            return "UNVERIFIED"
        return "CLEAN"

    def percentile(self, fraction: float) -> int:
        if not self.counts:
            return 0
        return sorted(self.counts)[min(int(len(self.counts) * fraction), len(self.counts) - 1)]


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    """Reject a repeated key instead of letting `json` keep the last one.

    `{"text": <520 tokens>, "text": <40 tokens>}` parses to the short value, so the long answer is
    physically in the file and invisible to the scan. Which value survives depends on the ORDER,
    which means such a row is not evidence either way: it is refused as unparseable.
    """
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate key in a JSON object")
    return dict(pairs)


def scan(path: Path, ceiling: int, encoding_name: str,
         expected_rows: int | None = None) -> FileReport:
    import tiktoken

    enc = tiktoken.get_encoding(encoding_name)
    report = FileReport(path.name, expected_rows)
    # utf-8-sig, not utf-8: a BOM is a normal product of PowerShell redirection on this platform,
    # and under plain utf-8 it makes the FIRST row unparseable. A dropped first row that happened
    # to hold the only over-ceiling answer is the precise shape of a false clean.
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line, object_pairs_hook=_no_duplicate_keys)
            except (json.JSONDecodeError, ValueError):
                # A file cut off mid-write is an interrupted archive, not a truncated answer. It
                # is counted, reported AND fails the file, because rows behind an unparseable line
                # were never measured.
                report.unparseable += 1
                continue
            if not isinstance(row, dict):
                report.unparseable += 1
                continue
            report.rows += 1
            task_id = _safe(row.get("task_id"))
            texts, dropped = _texts_of(row)
            report.unmeasurable += dropped
            if not texts:
                # Distinct from an empty answer: nothing here was measured, so nothing here is
                # evidence. Collapsing this to a 0-token row also dragged the mean and p95 down,
                # making the distribution look safer than the data supports.
                report.no_answer += 1
                continue
            for text in texts:
                n = len(enc.encode(text))
                report.counts.append(n)
                stripped = text.rstrip()
                if stripped and not stripped.endswith(TERMINAL):
                    report.unterminated += 1
                if n >= ceiling:
                    report.at_ceiling.append((task_id, n, _safe(text[-70:])))
                elif n >= ceiling - NEAR_BAND:
                    report.near.append((task_id, n, _safe(text[-70:])))
    return report


def _texts_of(row: dict) -> tuple[list[str], int]:
    """Every prediction string in a row, and how many predictions could NOT be read.

    The second number is the point. Returning only the readable ones let a row with one good
    prediction and one unreadable one look complete, which is how a truncated answer in the
    unreadable slot came back CLEAN. The caller escalates any non-zero count to UNVERIFIED.
    """
    predictions = row.get("predictions")
    if isinstance(predictions, dict):  # some derived files carry a single object
        predictions = [predictions]
    if not isinstance(predictions, list):
        return [], 0 if predictions is None else 1
    out: list[str] = []
    dropped = 0
    for prediction in predictions:
        if isinstance(prediction, dict):
            text = prediction.get("text")
        elif isinstance(prediction, str):
            text = prediction
        else:
            text = None
        if isinstance(text, str) and text != "":
            out.append(text)
        else:
            dropped += 1
    return out, dropped


def _expanded(paths: list[Path]) -> tuple[list[Path], bool]:
    """Expand any wildcard the shell left alone, and say whether one was used.

    ⛔ A wildcard cannot certify completeness, which is why the caller demands `--expect N`
    alongside one. With `taskc_*_official.*.jsonl` and only ONE of the two runs restored, an
    expanding scanner reports CLEAN and exits 0 while never naming the run it did not see: the
    pattern, not the operator, silently decided what CLEAN was a statement about. Before expansion
    existed the same command exited 1. Turning an explicit refusal into a quiet narrowing is the
    exact trade this whole audit exists to refuse.

    PowerShell does not glob arguments, so `taskc_*_official.*.jsonl` reaches this script as a
    literal. Without this the operator gets "ABSENT" for a pattern and reads it as "the files are
    missing" rather than "my shell did not expand this". A pattern matching nothing stays in the
    list so it is still reported ABSENT rather than silently disappearing.
    """
    out: list[Path] = []
    globbed = False
    for path in paths:
        if path.exists() or not any(ch in str(path) for ch in "*?["):
            out.append(path)
            continue
        matches = sorted(Path(path.anchor or ".").glob(
            str(path if not path.anchor else path.relative_to(path.anchor)).replace("\\", "/")
        ))
        globbed = True
        print(f"pattern {path} -> {len(matches)} file(s): "
              f"{', '.join(m.name for m in matches) or 'NONE'}")
        out.extend(matches or [path])
    return out, globbed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scan archived MTRAG answers for ceiling truncation.")
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--ceiling", type=int, default=512, help="the --max-tokens the run sent")
    ap.add_argument("--encoding", default="o200k_base", help="gpt-4o and gpt-4o-mini use o200k_base")
    ap.add_argument("--expect-rows", type=int, default=None, metavar="N",
                    help="rows each file must hold, from the run's manifest (`tasks`, 842 here); "
                         "without it a CLEAN says the named files were fully measured, not that "
                         "they are the complete runs")
    ap.add_argument("--expect", default=None, metavar="STEM,STEM",
                    help="run stems that must be covered, e.g. "
                         "taskc_benchmark_official,taskc_recall_official; required with a wildcard")
    args = ap.parse_args(argv)

    print(f"ceiling {args.ceiling} completion tokens, encoding {args.encoding}\n")
    reports: list[FileReport] = []
    files, globbed = _expanded(args.files)
    for path in files:
        if not path.exists():
            report = FileReport(path.name)
            report.error = "file absent"
            reports.append(report)
            print(f"{path.name}: ABSENT, so this run is UNVERIFIED rather than clean\n")
            continue
        try:
            report = scan(path, args.ceiling, args.encoding, args.expect_rows)
        except Exception as exc:  # noqa: BLE001 - one unreadable file must not hide the others
            report = FileReport(path.name)
            report.error = f"{type(exc).__name__}: {exc}"
            reports.append(report)
            print(f"{path.name}: SCAN FAILED ({_safe(report.error)}), so it is UNVERIFIED\n")
            continue
        reports.append(report)
        print(
            f"{report.name}\n"
            f"  rows {report.rows}"
            f"{'' if report.expected_rows is None else f' (expected {report.expected_rows})'}"
            f"  answers {len(report.counts)}  "
            f"unparseable {report.unparseable}  rows with no answer {report.no_answer}  "
            f"unreadable predictions {report.unmeasurable}\n"
            f"  answer tokens: mean "
            f"{round(sum(report.counts) / len(report.counts), 1) if report.counts else 0}  "
            f"p50 {report.percentile(0.5)}  p95 {report.percentile(0.95)}  "
            f"max {max(report.counts) if report.counts else 0}\n"
            f"  AT OR OVER THE CEILING: {len(report.at_ceiling)}   "
            f"within {NEAR_BAND} of it: {len(report.near)}\n"
            f"  no terminal punctuation: {report.unterminated} of {len(report.counts)}\n"
            f"  -> {report.verdict}"
        )
        for task_id, n, tail in (report.at_ceiling + report.near)[:10]:
            print(f"    !! {task_id}  {n} tokens  ...{tail}")
        if report.near and not report.at_ceiling:
            print(f"     read these by hand: re-tokenising a mid-word cut need not reproduce the "
                  f"sampled count, so within {NEAR_BAND} of the ceiling is not proof of a finish")
        print()

    print("verdicts:")
    for report in reports:
        print(f"  {report.verdict:38} {report.name}")

    # Always printed, so the report states its own universe in RUN terms. A reader can then see
    # which runs a CLEAN was about without reconstructing it from a file list, and a wildcard
    # expanded by the SHELL (which this script cannot detect at all) still shows its narrowing.
    covered = sorted({r.name.split(".")[0] for r in reports})
    print(f"runs covered: {', '.join(covered)}")

    complaints: list[str] = []
    if not reports:
        # Defence in depth, and unreachable while `_expanded` keeps a non-matching pattern in the
        # list: `all()` over nothing is True, so a set of arguments resolving to no files would
        # otherwise print an empty verdict block and certify it. The LIVE mechanism is that
        # fallback, which is what the case matrix pins (a non-matching pattern must print ABSENT,
        # not merely exit non-zero). This line only matters if that fallback is ever removed.
        complaints.append("no files were scanned at all")
    if args.expect is None:
        # Demanded ALWAYS, not only when this script did the globbing. Under bash the shell
        # expands the pattern before argv, so `globbed` is False and a gate conditioned on it is
        # switched off by the operator's choice of shell: a partial restore then exits 0 with a
        # block of nothing but CLEAN. A tool cannot certify what nobody claimed, so an exit of 0
        # requires the operator to say which runs this was supposed to cover.
        complaints.append(
            "--expect was not given, so nothing states which runs this was meant to cover")
    else:
        # Stems, not a count. A count is the wrong unit: each archived run restores five `.jsonl`
        # layers, so `--expect 2` against `taskc_*_official.*.jsonl` is unsatisfiable on a correct
        # restore (it matches ten), while two files from ONE run satisfy it perfectly. The
        # operator's natural repair, raising the number to whatever the pattern just printed, hands
        # the decision straight back to the wildcard. Only names can say "these runs".
        wanted = {stem.strip() for stem in args.expect.split(",") if stem.strip()}
        if not wanted:
            # `--expect ,` or `--expect ""` passed the "was it given" test and then filtered away
            # to nothing, so `wanted - covered` was empty and the check silently became a no-op:
            # the pre-fix scanner, reinstated by a degenerate value. Reachable in practice as
            # `--expect "$STEMS"` with the variable unset, which is how a replay script passes it.
            complaints.append("--expect names no run, so it claims nothing")
        missing = sorted(wanted - set(covered))
        if missing:
            complaints.append(f"no file was scanned for {', '.join(missing)}")
    if any(r.verdict != "CLEAN" for r in reports):
        complaints.append("not every file came back CLEAN")

    if complaints:
        print("\nNOT CHECKED: " + "; ".join(complaints) + ".")
        print("Nothing above may be reported as clean.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
