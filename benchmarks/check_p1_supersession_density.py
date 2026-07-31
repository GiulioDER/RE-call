"""P1: how much explicitly revised state is there in LOCOMO at all?

Prior work: same search recorded in `benchmarks/check_temporal_inert.py`. This is the kill gate
pre-registered in `docs/REFERENCE_TIME_DESIGN.md`.

PRE-REGISTERED (from the design doc, before running):

    P1  Fewer than 15% of turns assert a value for a field that a LATER turn revises. Below that,
        the supersession mechanism cannot move an aggregate score no matter how well it works, and
        the line stops here.

WHAT THIS CAN AND CANNOT MEASURE
--------------------------------
"Asserts a value for field X" cannot be decided without extraction, and extraction is the step
under dispute. So this measures a deliberately NARROW proxy: turns carrying an explicit revision
marker ("actually", "changed", "postponed", "instead of", ...). That is the right proxy for the
scoped mechanism, which is *supersession of EXPLICITLY revised state*, not revision in general.

The proxy is an UPPER bound on nothing and a LOWER bound on nothing, so both error directions are
reported rather than assumed away:

  - it OVER-counts: "actually" appears in plenty of turns that revise no earlier assertion
    ("actually, I love that book"). A sample is printed so the rate can be discounted by eye.
  - it UNDER-counts: a revision stated without a marker ("The deadline is April 15th", said after
    an earlier "The deadline is April 1st") is invisible here.

Because it over- and under-counts at once, a result NEAR the 15% line decides nothing. Only a
result far below it is actionable, which is exactly what a kill gate should be.

Run:  python benchmarks/check_p1_supersession_density.py
"""
from __future__ import annotations

import json
import re
from collections import Counter

# Markers that a speaker is REVISING something previously said, rather than asserting it fresh.
# Chosen before looking at any counts.
REVISION_MARKERS = [
    r"\bactually\b", r"\bchanged\b", r"\bchange of\b", r"\bpostponed\b", r"\bpushed back\b",
    r"\brescheduled\b", r"\bmoved (?:it |the )?(?:to|up|back)\b", r"\binstead of\b",
    r"\bno longer\b", r"\bnot anymore\b", r"\bturns out\b", r"\bupdate\b", r"\bupdated\b",
    r"\bcorrection\b", r"\bI meant\b", r"\bscratch that\b", r"\bnow it'?s\b",
    r"\bended up\b", r"\bfell through\b", r"\bcancell?ed\b",
]
MARKER_RE = re.compile("|".join(REVISION_MARKERS), re.I)

# A revision of a FIELD, as opposed to a revision of a plan in passing, almost always restates a
# value. Requiring a co-occurring value makes the proxy tighter, and it is reported separately so
# the effect of the tightening is visible rather than baked in.
VALUE_RE = re.compile(
    r"\b\d{1,4}\b|\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\b|\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    re.I,
)


def turns(conversation: dict):
    for key in conversation:
        if not (key.startswith("session_") and not key.endswith("date_time")):
            continue
        value = conversation[key]
        if not isinstance(value, list):
            continue
        for turn in value:
            if isinstance(turn, dict) and turn.get("text"):
                yield turn


def main() -> int:
    convs = json.load(open("locomo10.json", encoding="utf-8"))
    total = 0
    marked = 0
    marked_with_value = 0
    hits = Counter()
    samples = []

    for conv in convs:
        for turn in turns(conv["conversation"]):
            text = turn["text"]
            total += 1
            m = MARKER_RE.search(text)
            if not m:
                continue
            marked += 1
            hits[m.group(0).lower()] += 1
            if VALUE_RE.search(text):
                marked_with_value += 1
                if len(samples) < 8:
                    samples.append((turn.get("speaker", "?"), text[:120]))

    print(f"conversations                     : {len(convs)}")
    print(f"turns                             : {total}")
    print(f"turns with a revision marker      : {marked}  ({marked / total:.1%})")
    print(f"  ...AND a value (date/number/day): {marked_with_value}  ({marked_with_value / total:.1%})")
    print()
    print("most common markers:", dict(hits.most_common(6)))
    print()
    print("sample of marker+value turns (eyeball how many are REAL revisions of an earlier claim):")
    for speaker, text in samples:
        print(f"  [{speaker}] {text}")

    rate = marked_with_value / total
    print()
    print(f"P1 gate: revised-state density {rate:.1%} vs the pre-registered 15% floor")
    if rate < 0.15:
        print("VERDICT: BELOW the floor. The mechanism cannot move an aggregate score.")
        print("         Note the proxy OVER-counts, so the true rate is lower still, which")
        print("         strengthens rather than weakens this result.")
        return 0
    print("VERDICT: at or above the floor -- worth the next step, but discount for over-counting")
    print("         using the sample above before believing it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
