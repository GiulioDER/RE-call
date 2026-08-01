"""Do any turns postdate the reference instant their own scope would supply?

Prior work searched 2026-08-01. `docs_search(source_type="memory")` was UNAVAILABLE: the memory
corpus is served from VPS2, which is down, and the VPS3 mirror has no `docs_chunks` relation. Fell
back to grep and Read over this repo's docs, which found the load-bearing items below. Recorded
because an unavailable index is not the same as an empty one, and a later reader must not mistake
this for a search that came back clean.

Fourth member of the `check_temporal_*` family. `check_temporal_inert.py` showed the temporal layer
could not fire, `check_temporal_live.py` showed it fires end to end, `check_anchor_direction.py`
showed which DIRECTION of bound is safe, and this one pins the invariant that makes an as-of bound
INERT rather than merely safe.

WHY IT EXISTS, GIVEN THAT IT PASSES TRIVIALLY TODAY
---------------------------------------------------
`docs/REFERENCE_TIME_DESIGN.md` rejects as-of retrieval keyed on utterance time, because a bound
that demotes any turn said after the question's anchor deletes retrospective testimony. A reader
then proposed the repair: take the reference instant from the last utterance of the conversation in
scope, rather than from the question. That version is sound, and it is also inert. Of 5,882 turns
across the ten conversations, ZERO fall after their own conversation's end, and that is by
construction rather than by luck, because a turn cannot postdate the transcript it belongs to.

Inert is a property of THIS SHAPE of corpus, not of corpora generally. A closed transcript has a
last utterance. A live one does not, and a merged corpus has one boundary per source rather than
one for the whole. Either change makes the bound start demoting real evidence, and it would do so
QUIETLY: nothing in the retrieval path announces that a term which used to be a no-op has become
load-bearing. This script is that announcement.

It is deliberately NOT a ranking term, and the distinction is the point. The scoring path carries
no clock at all (`recall/retriever.py` composes a dense score and an optional reranker), and it
should stay that way, because a term pinned at zero still occupies a slot that invites someone to
re-weight it later. Removing it from the score and keeping it as an assertion buys the
observability without the slot: the count is computed and checked, and nothing can tune it.

PRE-REGISTERED
--------------
  P1  INVARIANT. Every turn resolves to an utterance date. Any that does not is REPORTED, never
      silently dropped, because a dropped turn shrinks the denominator and turns a count into a
      different count wearing the same label.
  P2  LIVE TRANSCRIPTS. Zero turns postdate the reference instant their scope DECLARED. The
      instant must come from a declared source, never from the turns being compared against it:
      derived as the maximum over a conversation's own turns it is `max(x) >= x`, which is
      arithmetic rather than an invariant and can never fail. A conversation with no declared
      instant is therefore reported VACUOUS, not PASS, because a check that cannot fail is not
      evidence that anything holds. Supply one with `--snapshot`, which is what a live scope has
      and a closed transcript does not.
  P3  CONTROL, and the one that decides whether any zero above means anything. Widening the scope
      to the whole corpus, which is exactly what merging haystacks does, MUST produce a non-zero
      count. If it does not, the predicate never fires at all and no reading here is worth
      anything. It is always taken over per-conversation instants, NEVER over a declared snapshot:
      a snapshot collapses every scope onto one instant, which makes the control non-empty exactly
      when P2 is, so it would agree with the thing it exists to check independently.

P3 is not hypothetical. The merged reading below is the same operation as §10's, which took
LongMemEval's per-question haystacks into one 19,195-session corpus and dropped hit@5 from 0.970 to
0.366. (0.719 is the intermediate Oracle arm over 940 sessions, not the per-question baseline.)

Needs `locomo10.json` in the working directory (gitignored, fetched on demand; see
`recall/eval/locomo.py`).

Run:  python benchmarks/check_utterance_postdating.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

if __package__ in (None, ""):
    # Standalone runs put `benchmarks/` on the path, not the repo root, so the shared parser below
    # would not import. Re-deriving it here instead would create a SECOND date parser free to
    # disagree with the first, which is the failure this import exists to avoid.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.check_anchor_direction import session_dates

CORPUS = "locomo10.json"

#: (conversation index, dia_id, utterance date)
Turn = tuple[int, str, date]


def corpus_turns(corpus: list[dict]) -> tuple[list[Turn], list[tuple[int, str, str]]]:
    """Every turn with a resolved utterance date, plus every turn without one.

    Returns ``(turns, unresolved)``, where an unresolved entry is
    ``(conversation index, dia_id, why)``. Unresolved turns are RETURNED rather than skipped: P1
    exists because a silent drop changes the denominator of both counts below without changing
    either label.
    """
    turns: list[Turn] = []
    unresolved: list[tuple[int, str, str]] = []
    for idx, entry in enumerate(corpus):
        if not isinstance(entry, dict):
            unresolved.append((idx, "", "corpus entry is not an object"))
            continue
        conv = entry.get("conversation")
        if not isinstance(conv, dict):
            unresolved.append((idx, "", "entry carries no conversation object"))
            continue
        dates = session_dates(conv)
        for key, val in conv.items():
            if key.endswith("date_time") or not key.startswith("session_"):
                continue
            if not isinstance(val, list):
                # NOT a silent skip. A session arriving as an object rather than a list is exactly
                # the ingest shape change this script exists to notice, and dropping it here would
                # shrink the denominator while the P1 line still printed zero.
                unresolved.append((idx, key, "session value is not a list of turns"))
                continue
            m = re.fullmatch(r"session_(\d+)", key)
            if m is None:
                unresolved.append((idx, key, "session key does not carry an index"))
                continue
            prefix = f"D{m.group(1)}"
            when = dates.get(prefix)
            for pos, turn in enumerate(val, 1):
                if not isinstance(turn, dict):
                    unresolved.append((idx, f"{prefix}:{pos}", "turn is not an object"))
                    continue
                dia = turn.get("dia_id")
                where = dia if isinstance(dia, str) and dia else f"{prefix}:{pos}"
                if when is None:
                    # The session exists and its turns are real; only the DATE is missing. Dropping
                    # them would understate both counts, so they are reported instead.
                    unresolved.append((idx, where, f"no parseable date for {prefix}"))
                    continue
                # TYPE before emptiness, and emptiness tested explicitly rather than by truthiness.
                # Ordered the other way, every FALSY non-string (`0`, `False`, `[]`, `{}`) was
                # reported as "absent or empty", which sends the reader hunting a missing field
                # that is in fact present with the wrong type. An ingest emitting a 0-based integer
                # index would have mislabelled exactly one row in the corpus, the one numbered 0.
                if dia is not None and not isinstance(dia, str):
                    unresolved.append((idx, where, f"dia_id is {type(dia).__name__}, not a string"))
                    continue
                if not dia:
                    # Reported under its OWN reason. Substituting a synthetic id built FROM the
                    # session prefix and then comparing it against that prefix is a tautology, so
                    # the disagreement check below could never see these.
                    unresolved.append((idx, where, "dia_id is absent or empty"))
                    continue
                said = dia.split(":")[0]
                if said != prefix:
                    # `check_anchor_direction` dates a reference by its dia_id PREFIX, this script
                    # by the session it is stored under. They agree on every turn in the corpus
                    # today, and if they ever stop the two members of the family would silently
                    # assign different dates to the same turn.
                    unresolved.append(
                        (idx, str(dia), f"dia_id prefix {said} disagrees with its session {prefix}")
                    )
                    continue
                turns.append((idx, str(dia), when))
    return turns, unresolved


def conversation_ends(
    turns: list[Turn], *, snapshot: date | None = None
) -> tuple[dict[int, date], bool]:
    """Conversation index -> the reference instant its scope supplies, and whether it was DECLARED.

    Two sources, and the difference between them is the whole point of P2.

    A `snapshot` is DECLARED: it is the instant the scope was built, which is what a live system
    has. Turns that arrive after it postdate it, so the comparison can fail and a zero means
    something.

    Without one the instant is DERIVED as the maximum over the conversation's own turns, and then
    `when > end` is `x > max(x)`, which is false for every turn no matter what the data says. That
    is not the invariant holding, it is arithmetic, and reporting it as a pass would be a guard
    that reads as protection and cannot fire.
    """
    if snapshot is not None:
        return {idx: snapshot for idx, _dia, _when in turns}, True
    ends: dict[int, date] = {}
    for idx, _dia, when in turns:
        if idx not in ends or when > ends[idx]:
            ends[idx] = when
    return ends, False


def postdating(turns: list[Turn], ends: dict[int, date], *, merged: bool) -> list[tuple[int, Turn]]:
    """Turns an as-of bound at the scope's reference instant would demote.

    ``merged=False`` is the closed-transcript reading: each conversation is its own scope, so a
    turn is only ever compared against the end of the transcript it belongs to. ``merged=True``
    widens the scope to the whole corpus, which is what merging haystacks does: a question asked in
    conversation *i* still takes its instant from conversation *i*, but now sees every turn in the
    corpus, including turns said after that instant in some other transcript.

    Returns ``(reference conversation index, turn)`` pairs.
    """
    out: list[tuple[int, Turn]] = []
    for ref_idx, end in sorted(ends.items()):
        for turn in turns:
            idx, _dia, when = turn
            if not merged and idx != ref_idx:
                continue
            if when > end:
                out.append((ref_idx, turn))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_utterance_postdating",
        description="Do any turns postdate the reference instant their own scope supplies?",
    )
    parser.add_argument(
        "--snapshot",
        type=date.fromisoformat,
        default=None,
        metavar="YYYY-MM-DD",
        help="the instant the scope DECLARED, as a live system records it. Without one the "
             "instant is derived from the turns themselves and P2 cannot fail.",
    )
    parser.add_argument("--corpus", default=CORPUS, help=f"path to the corpus (default {CORPUS})")
    try:
        # Hand-rolled parsing accepted `--snapshot X` but silently IGNORED `--snapshot=X`, running
        # the derived path and exiting 0 while the operator believed they had declared an instant.
        # A usage error also exited 1, which is the code for a broken invariant.
        args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:
        return 2 if exc.code else 0
    snapshot: date | None = args.snapshot

    path = Path(args.corpus)
    if not path.exists():
        print(f"MISSING: {path} is not in the working directory. It is gitignored and fetched "
              f"on demand; see recall/eval/locomo.py.")
        return 2
    # A corpus that exists but is not readable is an OPERATOR error, so it must not share exit
    # code 1 with "the invariant has broken". A mistyped --corpus that happens to land on another
    # JSON artifact would otherwise be indistinguishable from a real finding.
    try:
        corpus = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"MALFORMED: {path} is not JSON ({exc}).")
        return 2
    if not isinstance(corpus, list):
        print(f"MALFORMED: {path} is {type(corpus).__name__}, not a list of samples.")
        return 2

    turns, unresolved = corpus_turns(corpus)
    ends, declared = conversation_ends(turns, snapshot=snapshot)
    # The control is a property of the PREDICATE, so it is always taken over per-conversation
    # instants, never over `ends`. A declared snapshot collapses every scope onto one instant, and
    # then `merged` is non-empty exactly when `own` is, so the control could only ever agree with
    # the thing it is supposed to check independently: a declared run that genuinely HELD reported
    # "CONTROL FAILED" and exited 1, leaving the passing verdict unreachable in the one mode the
    # design mandates.
    derived_ends, _ = conversation_ends(turns)

    print(f"conversations: {len(ends)}")
    print(f"turns with a resolved utterance date: {len(turns)}")
    print(f"turns WITHOUT one (P1, reported not dropped): {len(unresolved)}")
    for idx, dia, why in unresolved[:20]:
        print(f"  conv {idx} {dia}: {why}")
    if len(unresolved) > 20:
        print(f"  ... and {len(unresolved) - 20} more")

    own = postdating(turns, ends, merged=False)
    merged = postdating(turns, derived_ends, merged=True)

    source = f"DECLARED (--snapshot {snapshot})" if declared else "DERIVED from the turns themselves"
    print()
    print(f"reference instant per conversation: {source}")
    print(f"P2  turns postdating the reference instant of their own scope: {len(own)}")
    for ref, (idx, dia, when) in own[:10]:
        print(f"      conv {ref} instant={ends[ref]} demotes conv {idx} {dia} said {when}")
    # NOT "same instants". The control is always taken over the per-conversation DERIVED instants,
    # which in declared mode are not the snapshot printed two lines above.
    print(f"P3  merged corpus, per-conversation derived instants, turns postdating them anywhere: "
          f"{len(merged)}")

    if len(turns) == 0:
        print("\nVERDICT: no turns resolved, so no count here means anything.")
        return 1
    if unresolved:
        print("\nVERDICT: P1 FAILED. Some turns carry no utterance date, so every count above is "
              "taken over a smaller corpus than its label claims.")
        return 1
    # ORDER IS LOAD-BEARING. `own` is checked FIRST because a non-empty `own` is itself a
    # demonstration that the predicate fired, which is all the control exists to establish. With
    # the control tested first, a single-scope corpus reported "the predicate never fires" while
    # the P2 line directly above it printed a non-zero count and named the demoted turn: merging
    # adds nothing when there is one transcript, so on exactly the live-transcript corpus this
    # script exists to announce, the announcement was unreachable.
    if own:
        print("\nVERDICT: THE INVARIANT HAS BROKEN. Turns postdate the reference instant their own "
              "scope declared, so this corpus is not made of closed transcripts. An as-of bound at "
              "the last utterance in scope is no longer inert on it: it would demote the evidence "
              "listed above. Re-argue the mechanism before relying on either behaviour.")
        return 1
    # An empty control does not decide anything on its own. What decides is whether P2 was a REAL
    # check, and that turns on where the instant came from, not on how many transcripts there are.
    #
    # The first attempt at this split gated on `len(derived_ends) < 2`, which fixed the
    # one-transcript case and left the identical defect at two: a declared run that genuinely held
    # over two same-day transcripts still reported "the predicate never fires" and exited 1, and
    # deleting one of those transcripts flipped the same run to exit 0. Adding a transcript that
    # cannot weaken P2 must not turn a pass into a failure.
    #
    # A live-ingest corpus is exactly where this bites: while every scope is still open, all the
    # derived ends are equal, so the merged reading is empty for ANY number of transcripts.
    if not merged:
        why = ("one transcript has nothing to merge with, so the control is empty by construction"
               if len(derived_ends) < 2 else
               "the transcripts carry no temporal spread, so merging their scopes demotes nothing")
        if declared:
            print(f"\nVERDICT: invariant holds against a declared instant, with the control "
                  f"UNINFORMATIVE. Nothing postdates the snapshot its scope recorded, and "
                  f"{why}, so the control says nothing either way. P2 is still a real check here, "
                  f"because the snapshot is an external bound rather than a maximum over the turns "
                  f"it is compared against. Re-run over transcripts that span different dates to "
                  f"exercise the control.")
            return 0
        print(f"\nVERDICT: NOTHING HERE IS EVIDENCE. The instant is derived, so P2 cannot fail, and "
              f"{why}, so the control cannot fire either. Pass --snapshot, or run over transcripts "
              f"that span different dates.")
        return 1

    share = len(merged) / (len(turns) * len(derived_ends))
    if not declared:
        print("\nVERDICT: P2 is VACUOUS on this corpus, and that is the honest reading. With no "
              "declared instant the bound is the maximum over each conversation's own turns, so "
              "`said > max(said)` is false by arithmetic and would read as a pass on ANY data. "
              "Pass --snapshot to compare against an instant a live scope actually recorded.\n"
              f"    The control still fires: the same bound over a merged corpus demotes "
              f"{len(merged)} turn-instant pairs ({share:.1%} of all pairs), so the predicate "
              f"works and it is the closed-transcript SHAPE, not the predicate, that makes an "
              f"as-of bound do nothing here.")
        return 0
    print(f"\nVERDICT: invariant holds against a declared instant. Nothing postdates the snapshot "
          f"its scope recorded, and the control confirms the predicate can fire: the same "
          f"predicate, over per-conversation derived instants rather than this snapshot, demotes "
          f"{len(merged)} turn-instant pairs ({share:.1%} of all pairs). Inert here, not inert in "
          f"general.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
