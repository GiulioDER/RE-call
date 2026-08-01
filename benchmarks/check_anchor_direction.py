"""Do question anchors fall BEFORE or AFTER the utterance date of their own gold evidence?

Prior work searched 2026-08-01. `docs_search(source_type="memory")` was UNAVAILABLE: the memory
corpus is served from VPS2, which is down, and the VPS3 mirror has no `docs_chunks` relation. Fell
back to grep and Read over `~/.claude/.../memory` and this repo's docs, which found the two
load-bearing items below. Recorded because an unavailable index is not the same as an empty one,
and a later reader must not mistake this for a search that came back clean.

Prior work: `docs/REFERENCE_TIME_DESIGN.md` rejects as-of retrieval keyed on utterance time,
because a filter that demotes any turn whose `valid_from` postdates the question's anchor deletes
retrospective testimony ("last month I got hurt", said in October, about September).

A reader proposed the mirror: anchor the query on EVENT time, treat an unresolvable "recently" as
an interval with an open lower bound and a hard upper bound at the utterance, and demote a turn
only when the anchor falls after that bound. The logic is sound. This script asks whether the DATA
lets it fire safely, which is a different question and the one that decides it.

It is the third member of the `check_temporal_*` family: `check_temporal_inert.py` showed the
temporal layer could not fire, `check_temporal_live.py` showed it fires end to end, and this shows
which DIRECTION of bound is safe on this corpus.

PRE-REGISTERED:

  P1  INVARIANT. Every gold-evidence id resolves to a session date. Any that does not is REPORTED,
      never silently dropped, because a dropped row shrinks the denominator and turns a count into
      a different count wearing the same label.
  P2  A majority of the dated questions anchor AFTER the utterance date of their own gold
      evidence, so an upper bound at the utterance demotes the only evidence there is.
  P3  CONTROL, and the one that matters. The anchor parser must read day precision. Re-running
      with day precision DISABLED must produce a DIFFERENT, lower count. If the two agree, the
      parser is coarsening every anchor to 1 January and the comparison is not measuring what its
      label says. This control is not hypothetical: the first run of this measurement reported
      3 of 9 because "April 10, 2022" degraded to 2022-01-01.

Needs `locomo10.json` in the working directory (gitignored, fetched on demand; see
`recall/eval/locomo.py`) and the vendored `benchmarks/audit_data/locomo_errors.json`.

Run:  python benchmarks/check_anchor_direction.py
"""
from __future__ import annotations

import json
import re
from datetime import date

MONTHS = {m: i for i, m in enumerate(
    "january february march april may june july august september october november december".split(), 1)}
MON = "|".join(MONTHS)

AUDIT = "benchmarks/audit_data/locomo_errors.json"
CORPUS = "locomo10.json"


def session_dates(conv: dict[str, object]) -> dict[str, date]:
    """Dia-id prefix -> session date, e.g. ``D1`` -> ``date(2023, 5, 8)``."""
    out: dict[str, date] = {}
    for key, val in conv.items():
        m = re.fullmatch(r"session_(\d+)_date_time", key)
        if not m or not isinstance(val, str):
            continue
        dm = re.search(rf"(\d{{1,2}})\s+({MON})[a-z]*,?\s+(\d{{4}})", val, re.I)
        if dm:
            out[f"D{m.group(1)}"] = date(int(dm.group(3)), MONTHS[dm.group(2).lower()], int(dm.group(1)))
            continue
        dm = re.search(rf"({MON})[a-z]*\s+(\d{{1,2}}),?\s+(\d{{4}})", val, re.I)
        if dm:
            out[f"D{m.group(1)}"] = date(int(dm.group(3)), MONTHS[dm.group(1).lower()], int(dm.group(2)))
    return out


def anchor_of(question: str, *, day_precision: bool = True) -> tuple[date | None, str]:
    """Explicit date named in the QUESTION, as ``(date, granularity)``.

    Order matters. Day-precision forms MUST be tried before the month-year and bare-year
    fallbacks, or "April 10, 2022" degrades to 2022-01-01 and the comparison silently changes
    meaning. `day_precision=False` reproduces that defect on purpose, for P3.
    """
    if day_precision:
        m = re.search(rf"({MON})[a-z]*\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})", question, re.I)
        if m:
            return date(int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2))), "day"
        m = re.search(rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+({MON})[a-z]*,?\s+(\d{{4}})", question, re.I)
        if m:
            return date(int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1))), "day"
    m = re.search(rf"({MON})[a-z]*\s+(\d{{4}})", question, re.I)
    if m:
        return date(int(m.group(2)), MONTHS[m.group(1).lower()], 1), "month"
    m = re.search(r"\b(19|20)\d{2}\b", question)
    if m:
        return date(int(m.group(0)), 1, 1), "year"
    return None, "none"


Row = tuple[str, str, date, str, date, date]


def measure(
    errors: list[dict], corpus: list[dict], *, day_precision: bool = True
) -> tuple[list[Row], list[tuple[str, str]]]:
    """Returns ``(rows, unresolved)``; a row is (qid, question, anchor, gran, lo, hi)."""
    rows: list[Row] = []
    unresolved: list[tuple[str, str]] = []
    for e in errors:
        if e["error_type"] != "TEMPORAL_ERROR":
            continue
        anchor, gran = anchor_of(e["question"], day_precision=day_precision)
        if anchor is None:
            continue
        m = re.match(r"locomo_(\d+)_", e["question_id"])
        if m is None:
            # Not silently skipped: an unparseable id means the audit set and this script
            # disagree about the id format, which would shrink the denominator invisibly.
            raise ValueError(f"cannot read a conversation index from {e['question_id']!r}")
        idx = int(m.group(1))
        dates = session_dates(corpus[idx]["conversation"])
        ev = e.get("correct_evidence") or e.get("cited_evidence") or []
        ev_dates = []
        for ref in ev:
            prefix = ref.split(":")[0]
            if prefix in dates:
                ev_dates.append(dates[prefix])
            else:
                unresolved.append((e["question_id"], ref))
        if not ev_dates:
            unresolved.append((e["question_id"], "NO EVIDENCE RESOLVED"))
            continue
        rows.append((e["question_id"], e["question"], anchor, gran, min(ev_dates), max(ev_dates)))
    return rows, unresolved


def after_count(rows: list[Row]) -> int:
    """Rows the reader's rule would demote: the anchor postdates every gold utterance."""
    return sum(1 for (_qid, _q, anchor, _gran, _lo, hi) in rows if anchor > hi)


def main() -> int:
    errors = json.load(open(AUDIT, encoding="utf-8"))
    corpus = json.load(open(CORPUS, encoding="utf-8"))

    temporal = [e for e in errors if e["error_type"] == "TEMPORAL_ERROR"]
    rows, unresolved = measure(errors, corpus)
    coarse_rows, _ = measure(errors, corpus, day_precision=False)

    print(f"TEMPORAL_ERROR rows: {len(temporal)}")
    print(f"...naming an explicit date in the question: {len(rows)}")
    print(f"unresolved gold-evidence ids: {len(unresolved)}"
          + (f"  {unresolved}" if unresolved else ""))
    print()

    for qid, question, anchor, gran, lo, hi in rows:
        span = str(lo) if lo == hi else f"{lo}..{hi}"
        mark = "ANCHOR AFTER (gold demoted)" if anchor > hi else "anchor before/at (kept)"
        print(f"{qid:<20} anchor={anchor} ({gran:<5}) gold_utterance={span:<24} {mark}")
        print(f"    {question[:104]}")

    after = after_count(rows)
    coarse_after = after_count(coarse_rows)
    grans = sorted({r[3] for r in rows})

    print()
    print(f"anchor AFTER gold utterance (a hard upper bound demotes the gold): {after}/{len(rows)}")
    print(f"same count with day precision DISABLED:                           "
          f"{coarse_after}/{len(coarse_rows)}")
    print(f"anchor granularities parsed: {grans}")

    p1 = not unresolved
    p2 = len(rows) > 0 and after * 2 > len(rows)
    p3 = "day" in grans and coarse_after != after

    print()
    for name, ok in (
        ("P1 INVARIANT every gold-evidence id resolved", p1),
        ("P2 a majority of dated anchors postdate their gold evidence", p2),
        ("P3 CONTROL day precision is read, and it changes the count", p3),
    ):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    print()
    if not p1:
        print("VERDICT: INVARIANT FAILED. The denominator is a subset; the rate means nothing.")
        return 1
    if not p3:
        print("VERDICT: CONTROL FAILED. Anchors are being coarsened, so P2 measures "
              "something other than what it says.")
        return 1
    print("VERDICT:", "a hard upper bound at the utterance is UNSAFE on this corpus"
          if p2 else "a hard upper bound at the utterance looks safe on this corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
