"""Which archived MTRAG answers were cut off by the `--max-tokens` ceiling.

The six generation runs sent `--max-tokens 512` through a `generate_one` that never read
`finish_reason`, so a completion the ceiling stopped came back as an ordinary string, was written
to the submission and was judged as if the system had produced it. The code path raises now, but
the fix is forward-only and **the stop reason was never recorded**, so for the artifacts it has to
be recovered from the text.

Recovered by re-tokenising with the generator's own encoding (gpt-4o -> `o200k_base`). That is
close to exact rather than a proxy: a completion stopped by the ceiling carries EXACTLY 512
completion tokens, one the model ended carries fewer, and re-tokenising a string is deterministic
under the same BPE. The single edge case is the trailing whitespace token that `generate_one`'s
`.strip()` removes, which moves a count by at most one.

A punctuation check runs alongside it. The two agreeing is worth little; the two DISAGREEING is
the case worth reading, so both are printed rather than combined into a verdict.

Reads any file whose rows carry `task_id` and `predictions[].text`: `.predictions.jsonl`,
`.scoring.jsonl` and `.scored.jsonl` all qualify. Partial files are tolerated and counted, because
a leftover intermediate is often what survives.

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


def scan(path: Path, ceiling: int, encoding_name: str) -> dict[str, object]:
    import tiktoken

    enc = tiktoken.get_encoding(encoding_name)
    counts: list[int] = []
    at_ceiling: list[tuple[str, int, str]] = []
    near: list[tuple[str, int, str]] = []
    unterminated = 0
    unparseable = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A file cut off mid-write is an interrupted archive, not a truncated answer.
                # Counted and reported rather than raised, so one bad tail cannot hide 800 good rows.
                unparseable += 1
                continue
            predictions = row.get("predictions") or []
            text = (predictions[0].get("text") if predictions else "") or ""
            n = len(enc.encode(text))
            counts.append(n)
            stripped = text.rstrip()
            if stripped and not stripped.endswith(TERMINAL):
                unterminated += 1
            if n >= ceiling:
                at_ceiling.append((str(row.get("task_id")), n, text[-70:]))
            elif n >= ceiling - 12:
                near.append((str(row.get("task_id")), n, text[-70:]))
    counts.sort()
    return {
        "rows": len(counts),
        "unparseable": unparseable,
        "mean": round(sum(counts) / len(counts), 1) if counts else 0,
        "p50": counts[len(counts) // 2] if counts else 0,
        "p95": counts[int(len(counts) * 0.95)] if counts else 0,
        "max": counts[-1] if counts else 0,
        "at_ceiling": at_ceiling,
        "near": near,
        "unterminated": unterminated,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--ceiling", type=int, default=512, help="the --max-tokens the run sent")
    ap.add_argument("--encoding", default="o200k_base", help="gpt-4o and gpt-4o-mini use o200k_base")
    args = ap.parse_args(argv)

    print(f"ceiling {args.ceiling} completion tokens, encoding {args.encoding}\n")
    total_at = 0
    missing = 0
    for path in args.files:
        if not path.exists():
            print(f"{path.name}: MISSING, so this run is UNVERIFIED rather than clean\n")
            missing += 1
            continue
        r = scan(path, args.ceiling, args.encoding)
        at = r["at_ceiling"]
        assert isinstance(at, list)
        total_at += len(at)
        print(
            f"{path.name}\n"
            f"  rows {r['rows']}  unparseable {r['unparseable']}\n"
            f"  answer tokens: mean {r['mean']}  p50 {r['p50']}  p95 {r['p95']}  max {r['max']}\n"
            f"  AT OR OVER THE CEILING: {len(at)}   within 12 of it: {len(r['near'])}\n"  # type: ignore[arg-type]
            f"  no terminal punctuation: {r['unterminated']} of {r['rows']}"
        )
        for task_id, n, tail in at[:10]:
            print(f"    !! {task_id}  {n} tokens  ...{tail!r}")
        print()

    print(f"rows at or over the ceiling across every file read: {total_at}")
    if missing:
        # ASCII deliberately: this prints to a Windows console under cp1252, where an emoji raises
        # UnicodeEncodeError and takes the warning down with it.
        print(f"WARNING: {missing} file(s) absent. Those runs were NOT checked; not clean, unknown.")
    # Non-zero for a truncation found OR a file absent, so this can gate a scoring step. Exiting 0
    # on an absent file would let "the run was never checked" pass as "the run is clean", which is
    # the same substitution of silence for evidence that left this audit to be reconstructed.
    return 1 if (total_at or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
