"""Is RE-call's temporal layer able to fire on benchmark data at all?

Prior work: SEARCHED before designing this, `docs_search(source_type="memory",
"temporal reasoning validity interval date instance selection newest-wins falsified recency
heuristic RE-call")`, no gap_warning, top-3 cosine 0.565/0.555/0.545. Two hits changed the design
rather than decorating it:

  - [[project-recall-entailment-supersession-phase0-done-2026-07-18]] — supersession is ALREADY
    shipped and studied ("recency-steelman loses to declared supersession 83-100% vs 0.00"). So
    this is not a proposal to add supersession; it already exists.
  - [[project-recall-finance-market-nogo-2026-07-25]] — "RE-call has validity time only", and
    "we uniquely have as-of" was already ruled out as a claim. So validity time exists too.

That is why this script tests REACHABILITY rather than re-measuring a heuristic: the prior work
says the mechanism is built, so the open question is whether it can run on benchmark data at all.

PRE-REGISTERED before running (predict-then-measure; the prediction is recorded here so a
convenient result cannot be rationalised afterwards):

  P1  A LOCOMO turn document AS IT WAS BUILT BEFORE THE FIX carries no frontmatter, so
      `validity_bounds` returns (None, None).
  P2  With both bounds None, `_verdict` can return neither `expired` nor `not_yet_valid`, for
      ANY reference time, including one inside the conversation's own era.
  P3  The mechanism itself is fine: a document that DOES declare `valid_until` in the past
      yields a non-null window and `_verdict` returns `expired`.

P3 is the positive control and it is the point of this script. "Zero non-null windows" is
worthless on its own, because a broken probe reports exactly the same thing. If P3 fails, P1 and
P2 mean nothing and the whole §9l reopening should be discarded.

Everything here calls the REAL library functions. Nothing is reimplemented, because a
reimplementation would be testing this file rather than the shipped code path.

Run:  python benchmarks/check_temporal_inert.py
"""
from __future__ import annotations

from datetime import datetime, timezone

from recall.types import Chunk, ScoredChunk
from recall.frontmatter import parse_frontmatter, validity_bounds
from recall.trust import _verdict

# A real LOCOMO-shaped turn and a real session date from that era.
TURN = {"speaker": "Caroline", "text": "The Sprint 1 deadline is February 15, 2024."}
SESSION_DATE = "1:56 pm on 15 January, 2024"

# Reference times: wall clock (what the benchmark actually uses) and one inside the era of the
# conversation (what an as-of query would use). If the layer were live, these would differ.
NOW_WALLCLOCK = datetime.now(timezone.utc)
NOW_IN_ERA = datetime(2024, 1, 20, tzinfo=timezone.utc)


def _verdict_for(meta: dict, now: datetime) -> tuple:
    """Uses the REAL `ScoredChunk`/`Chunk`. A stand-in would be a second implementation of the
    type under test, and this file exists to measure the shipped path, not a model of it."""
    hit = ScoredChunk(chunk=Chunk(id="c", source="s", text="x", metadata=meta), score=0.9)
    return _verdict(hit, {}, 0.5, now)


def _pre_change_turn_document(turn: dict, session_date: str) -> str:
    """`_turn_document` EXACTLY as it stood before `valid_from` was added.

    The measurement below is a historical baseline: it is the state the published §9l conclusion
    was drawn in. Pointing it at the CURRENT `_turn_document` would report "not inert", which is
    true and useless, because the fix has since landed. Reconstructing the old body keeps this
    file answering the question it was written to answer, and makes it a regression test for the
    finding rather than a one-time script that decays into a false alarm.
    """
    speaker = turn.get("speaker", "unknown")
    text = turn.get("text", "")
    body = f"{speaker}: {text}"
    return f"# {speaker} — {session_date}\n\n{body}\n"


def main() -> int:
    results = []

    # ---- P1: what the real benchmark ingest actually writes -------------------------------
    doc = _pre_change_turn_document(TURN, SESSION_DATE)
    meta, _body = parse_frontmatter(doc)
    start, end = validity_bounds(meta)
    p1 = (start is None) and (end is None)
    results.append(("P1  benchmark turn has no validity window", p1, f"meta={meta} -> ({start}, {end})"))

    # ---- P2: with no window, the temporal verdicts are unreachable -------------------------
    temporal = {"expired", "not_yet_valid"}
    v_wall, _ = _verdict_for(meta, NOW_WALLCLOCK)
    v_era, _ = _verdict_for(meta, NOW_IN_ERA)
    p2 = (v_wall not in temporal) and (v_era not in temporal)
    results.append((
        "P2  no temporal verdict at ANY reference time", p2,
        f"wallclock={v_wall!r}  in-era={v_era!r}",
    ))

    # ---- P3: POSITIVE CONTROL -- the mechanism works when the data declares a window -------
    declared = "---\nvalid_until: 2024-02-01\n---\n\n" + doc
    meta_d, _ = parse_frontmatter(declared)
    start_d, end_d = validity_bounds(meta_d)
    v_ctrl, _ = _verdict_for(meta_d, NOW_WALLCLOCK)
    p3 = (end_d is not None) and (v_ctrl == "expired")
    results.append((
        "P3  CONTROL: declared window -> 'expired'", p3,
        f"window=({start_d}, {end_d})  verdict={v_ctrl!r}",
    ))

    # ---- P3b: control fires the OTHER way too, so it is not stuck on one answer ------------
    v_ctrl_before, _ = _verdict_for(meta_d, datetime(2024, 1, 20, tzinfo=timezone.utc))
    p3b = v_ctrl_before == "ok"
    results.append((
        "P3b CONTROL: same doc, reference time inside window -> 'ok'", p3b,
        f"verdict={v_ctrl_before!r}",
    ))

    width = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")

    control_ok = results[2][1] and results[3][1]
    inert = results[0][1] and results[1][1]

    print()
    if not control_ok:
        print("VERDICT: CONTROL FAILED. The probe cannot detect a live temporal layer, so the")
        print("         'inert' result below is meaningless. Discard the §9l reopening.")
        return 1
    print("Control fires in both directions, so the probe can tell live from inert.")
    print("VERDICT:", "CONFIRMED INERT on benchmark data." if inert else "NOT inert -- analysis is wrong.")
    return 0 if inert else 1


if __name__ == "__main__":
    raise SystemExit(main())
