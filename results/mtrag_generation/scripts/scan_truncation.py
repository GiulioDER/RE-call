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

⛔ **The only failure that matters here is reporting "clean" for rows nobody read.** Every way a
file can be unreadable therefore ends as UNVERIFIED, never as CLEAN, and the exit code is 0 only
when every file was fully read and every row carried an answer: an absent file, an empty one, a
line that will not parse, a row with no `predictions`, an answer that is not a string, or a scan
that raised. Silence is not evidence, and this whole audit exists because a stop reason nobody
recorded was later read as a stop reason nobody needed.

Reads any file whose rows carry `task_id` and `predictions[].text`: `.predictions.jsonl`,
`.scoring.jsonl` and `.scored.jsonl` all qualify. EVERY prediction in a row is measured, not just
the first.

    python results/mtrag_generation/scripts/scan_truncation.py <file> [<file> ...] [--ceiling N]

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

    def __init__(self, name: str) -> None:
        self.name = name
        self.rows = 0
        self.unparseable = 0
        self.no_answer = 0
        self.unterminated = 0
        self.error: str | None = None
        self.counts: list[int] = []
        self.at_ceiling: list[tuple[str, int, str]] = []
        self.near: list[tuple[str, int, str]] = []

    @property
    def unread(self) -> bool:
        return bool(self.error) or self.unparseable > 0 or self.no_answer > 0 or self.rows == 0

    @property
    def verdict(self) -> str:
        if self.at_ceiling:
            return "TRUNCATION FOUND"
        if self.near:
            return "NEAR THE CEILING, read these by hand"
        if self.unread:
            return "UNVERIFIED"
        return "CLEAN"

    def percentile(self, fraction: float) -> int:
        if not self.counts:
            return 0
        return sorted(self.counts)[min(int(len(self.counts) * fraction), len(self.counts) - 1)]


def scan(path: Path, ceiling: int, encoding_name: str) -> FileReport:
    import tiktoken

    enc = tiktoken.get_encoding(encoding_name)
    report = FileReport(path.name)
    # utf-8-sig, not utf-8: a BOM is a normal product of PowerShell redirection on this platform,
    # and under plain utf-8 it makes the FIRST row unparseable. A dropped first row that happened
    # to hold the only over-ceiling answer is the precise shape of a false clean.
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
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
            texts = _texts_of(row)
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


def _texts_of(row: dict) -> list[str]:
    """Every prediction string in a row. Shapes that are not strings are dropped, so the row is
    reported as answerless rather than silently measured as zero tokens."""
    predictions = row.get("predictions")
    if isinstance(predictions, dict):  # some derived files carry a single object
        predictions = [predictions]
    if not isinstance(predictions, list):
        return []
    out = []
    for prediction in predictions:
        if isinstance(prediction, dict):
            text = prediction.get("text")
        elif isinstance(prediction, str):
            text = prediction
        else:
            continue
        if isinstance(text, str) and text != "":
            out.append(text)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scan archived MTRAG answers for ceiling truncation.")
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--ceiling", type=int, default=512, help="the --max-tokens the run sent")
    ap.add_argument("--encoding", default="o200k_base", help="gpt-4o and gpt-4o-mini use o200k_base")
    args = ap.parse_args(argv)

    print(f"ceiling {args.ceiling} completion tokens, encoding {args.encoding}\n")
    reports: list[FileReport] = []
    for path in args.files:
        if not path.exists():
            report = FileReport(path.name)
            report.error = "file absent"
            reports.append(report)
            print(f"{path.name}: ABSENT, so this run is UNVERIFIED rather than clean\n")
            continue
        try:
            report = scan(path, args.ceiling, args.encoding)
        except Exception as exc:  # noqa: BLE001 - one unreadable file must not hide the others
            report = FileReport(path.name)
            report.error = f"{type(exc).__name__}: {exc}"
            reports.append(report)
            print(f"{path.name}: SCAN FAILED ({_safe(report.error)}), so it is UNVERIFIED\n")
            continue
        reports.append(report)
        print(
            f"{report.name}\n"
            f"  rows {report.rows}  answers {len(report.counts)}  "
            f"unparseable {report.unparseable}  rows with no answer {report.no_answer}\n"
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
        print()

    print("verdicts:")
    for report in reports:
        print(f"  {report.verdict:38} {report.name}")
    clean = all(r.verdict == "CLEAN" for r in reports)
    if not clean:
        print("\nNot every file came back CLEAN. Nothing above may be reported as checked.")
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
