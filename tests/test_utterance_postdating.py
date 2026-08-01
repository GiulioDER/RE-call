"""Controls for `benchmarks/check_utterance_postdating.py`.

`locomo10.json` is gitignored and fetched on demand, so CI cannot run the script end to end. What
CI CAN run is the predicate, and the predicate is where the argument lives, so every rule the
script's verdicts depend on is pinned here against synthetic corpora.

The load-bearing test is `test_derived_instant_cannot_fire`. The script reports a DERIVED instant
as VACUOUS rather than as a pass, and that choice is only defensible if the tautology is real. It
is asserted here rather than described in a comment, because a comment claiming a check cannot fire
is exactly the kind of claim that rots into a check nobody re-examines.

Each control asserts a string unique to the rule it names. That is not decoration: a bug audit in
`tests/test_9l_temporal_fixture.py` found three controls passing because an EARLIER rule fired, so
they were green while the rule in their own name never ran.
"""
from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from datetime import date

import pytest

from benchmarks.check_utterance_postdating import (
    conversation_ends,
    corpus_turns,
    main,
    postdating,
)


def _conv(sessions: dict[str, tuple[str, list[str]]]) -> dict:
    """Build one LOCOMO-shaped conversation.

    `sessions` maps a session key to `(date string, [turn texts])`, matching the corpus layout:
    turns live under `session_N` and their utterance date under `session_N_date_time`.
    """
    conv: dict[str, object] = {}
    for key, (when, texts) in sessions.items():
        idx = key.split("_")[1]
        conv[key] = [
            {"speaker": "A", "dia_id": f"D{idx}:{i + 1}", "text": t} for i, t in enumerate(texts)
        ]
        conv[f"{key}_date_time"] = when
    return {"conversation": conv}


CLOSED = [
    _conv({
        "session_1": ("1:00 pm on 8 May, 2023", ["hello", "hi"]),
        "session_2": ("2:00 pm on 9 June, 2023", ["later"]),
    })
]


def test_every_turn_resolves_to_an_utterance_date() -> None:
    turns, unresolved = corpus_turns(CLOSED)
    assert unresolved == []
    assert [(i, d, w) for i, d, w in turns] == [
        (0, "D1:1", date(2023, 5, 8)),
        (0, "D1:2", date(2023, 5, 8)),
        (0, "D2:1", date(2023, 6, 9)),
    ]


def test_undated_session_is_reported_not_dropped() -> None:
    """P1. A dropped turn shrinks the denominator without changing either count's label."""
    corpus = [_conv({"session_1": ("no date here at all", ["hello", "hi"])})]
    turns, unresolved = corpus_turns(corpus)
    assert turns == []
    assert len(unresolved) == 2
    assert all("no parseable date for D1" in why for _idx, _dia, why in unresolved)


def test_derived_instant_cannot_fire() -> None:
    """The tautology the VACUOUS verdict rests on, asserted rather than described.

    With the instant derived as the maximum over a conversation's own turns, `said > max(said)` is
    false for every turn in every corpus. This is checked against a corpus built specifically to
    break it, with a late turn appended: if some data could make a derived instant fire, this is
    the shape that would, and it does not.
    """
    late = [
        _conv({
            "session_1": ("1:00 pm on 8 May, 2023", ["hello"]),
            "session_2": ("2:00 pm on 31 December, 2099", ["much later"]),
        })
    ]
    turns, _ = corpus_turns(late)
    ends, declared = conversation_ends(turns)
    assert declared is False
    assert ends == {0: date(2099, 12, 31)}
    assert postdating(turns, ends, merged=False) == []


def test_declared_instant_fires_on_a_live_transcript() -> None:
    """P2. A scope that recorded its instant BEFORE later turns arrived is the live-ingest case."""
    turns, _ = corpus_turns(CLOSED)
    ends, declared = conversation_ends(turns, snapshot=date(2023, 5, 20))
    assert declared is True
    hits = postdating(turns, ends, merged=False)
    assert [(ref, dia) for ref, (_idx, dia, _when) in hits] == [(0, "D2:1")]


def test_declared_instant_after_every_turn_holds() -> None:
    """The same declared path must also be able to PASS, or it only ever reports failure."""
    turns, _ = corpus_turns(CLOSED)
    ends, _ = conversation_ends(turns, snapshot=date(2024, 1, 1))
    assert postdating(turns, ends, merged=False) == []


def test_merging_widens_the_scope_and_starts_demoting() -> None:
    """P3. The control: one corpus, two transcripts, and each one's instant now sees the other."""
    merged_corpus = [
        _conv({"session_1": ("1:00 pm on 8 May, 2023", ["early"])}),
        _conv({"session_1": ("1:00 pm on 8 May, 2024", ["a year later"])}),
    ]
    turns, _ = corpus_turns(merged_corpus)
    ends, _ = conversation_ends(turns)

    assert postdating(turns, ends, merged=False) == []

    hits = postdating(turns, ends, merged=True)
    assert [(ref, idx, dia) for ref, (idx, dia, _when) in hits] == [(0, 1, "D1:1")]


@pytest.mark.parametrize("merged", [False, True])
def test_empty_corpus_yields_no_instants(merged: bool) -> None:
    turns, unresolved = corpus_turns([])
    ends, _ = conversation_ends(turns)
    assert (turns, unresolved, ends) == ([], [], {})
    assert postdating(turns, ends, merged=merged) == []


def test_non_list_session_and_non_dict_turn_are_reported() -> None:
    """P1 again, for the two shapes an ingest change actually produces.

    Both were silent `continue`s in the first version. The P1 line printed 0 while turns vanished
    from the denominator, the headline share moved, and the exit code stayed 0.
    """
    corpus = [{"conversation": {
        "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "ok"}, "a bare string", None],
        "session_1_date_time": "1:00 pm on 8 May, 2023",
        "session_2": {"turns": []},
        "session_2_date_time": "1:00 pm on 9 May, 2023",
    }}]
    turns, unresolved = corpus_turns(corpus)
    assert len(turns) == 1
    why = sorted(w for _idx, _dia, w in unresolved)
    assert why == ["session value is not a list of turns", "turn is not an object",
                   "turn is not an object"]


def test_non_dict_corpus_entry_is_reported() -> None:
    """A sample arriving as a bare value must not crash and must not vanish from the denominator."""
    turns, unresolved = corpus_turns(["a bare string", None, *CLOSED])
    assert len(turns) == 3
    assert [w for _idx, _dia, w in unresolved] == ["corpus entry is not an object"] * 2


def test_dia_id_disagreeing_with_its_session_is_reported() -> None:
    """The two family members would otherwise date the same turn differently, silently."""
    corpus = [_conv({"session_1": ("1:00 pm on 8 May, 2023", ["ok"])})]
    corpus[0]["conversation"]["session_1"][0]["dia_id"] = "D9:7"
    turns, unresolved = corpus_turns(corpus)
    assert turns == []
    assert [w for _idx, _dia, w in unresolved] == [
        "dia_id prefix D9 disagrees with its session D1"
    ]


# --- main(): the verdicts and exit codes, which no control touched until the audit -------------


def _run(tmp_path, corpus, *flags) -> tuple[int, str]:
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(corpus), encoding="utf-8")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["--corpus", str(path), *flags])
    return rc, buf.getvalue()


#: NON-SQUARE: 3 turns over 2 conversations, so `turns x instants` (6 pairs, 16.7%) is distinct from
#: `turns ** 2` (11.1%), `instants ** 2` (25.0%), `turns` (33.3%) and `instants` (50.0%). A square
#: fixture cannot tell any of those apart, which is how two denominator mutations shipped green.
OBLONG = [
    _conv({"session_1": ("1:00 pm on 8 May, 2023", ["early", "also early"])}),
    _conv({"session_1": ("1:00 pm on 8 May, 2024", ["a year later"])}),
]

#: Two transcripts a year apart, so the merged control fires and the derived path is vacuous.
TWO = [
    _conv({"session_1": ("1:00 pm on 8 May, 2023", ["early"])}),
    _conv({"session_1": ("1:00 pm on 8 May, 2024", ["a year later"])}),
]


def test_main_declared_instant_that_holds_can_actually_pass(tmp_path) -> None:
    """The defect that made this whole file worth auditing.

    With the control taken over the DECLARED instant, a snapshot after every turn made P2 and P3
    empty together, so the one mode the design mandates reported `CONTROL FAILED` and exited 1 on
    the very state the invariant asserts. The passing verdict was unreachable.
    """
    rc, out = _run(tmp_path, TWO, "--snapshot", "2099-01-01")
    assert "invariant holds against a declared instant" in out
    assert rc == 0


def test_main_declared_instant_that_breaks_fires(tmp_path) -> None:
    rc, out = _run(tmp_path, TWO, "--snapshot", "2023-06-01")
    assert "THE INVARIANT HAS BROKEN" in out
    assert rc == 1


def test_main_derived_instant_reports_vacuous_and_still_passes(tmp_path) -> None:
    rc, out = _run(tmp_path, TWO)
    assert "P2 is VACUOUS on this corpus" in out
    assert rc == 0


def test_main_derived_with_no_temporal_spread_is_evidence_of_nothing(tmp_path) -> None:
    """Two transcripts, but nothing to find: derived instant cannot fail, control cannot fire."""
    same_day = [
        _conv({"session_1": ("1:00 pm on 8 May, 2023", ["one"])}),
        _conv({"session_1": ("2:00 pm on 8 May, 2023", ["two"])}),
    ]
    rc, out = _run(tmp_path, same_day)
    assert "NOTHING HERE IS EVIDENCE" in out
    assert "no temporal spread" in out
    assert rc == 1


def test_main_declared_hold_with_an_empty_control_passes_at_any_transcript_count(tmp_path) -> None:
    """The defect the FIRST fix for this left behind, and the property that rules it out.

    Splitting on `len(derived_ends) < 2` repaired the one-transcript case and left the identical
    misreport at two: a declared run that genuinely held over two same-day transcripts still said
    `CONTROL FAILED ... the predicate never fires` and exited 1. What decides is whether P2 was a
    real check, which turns on where the instant came from, not on how many transcripts there are.

    The second assertion is the monotonicity that makes it checkable: adding a transcript that
    cannot weaken P2 must never turn a pass into a failure. It did, one flag value away from a run
    that demonstrated the predicate firing on the very same corpus.
    """
    one = [_conv({"session_1": ("1:00 pm on 8 May, 2023", ["one"])})]
    two = [*one, _conv({"session_1": ("2:00 pm on 8 May, 2023", ["two"])})]

    for corpus in (one, two):
        rc, out = _run(tmp_path, corpus, "--snapshot", "2099-01-01")
        assert "invariant holds against a declared instant" in out
        assert "control UNINFORMATIVE" in out
        assert rc == 0

    # ...and the same corpus DOES break under an earlier snapshot, so the control being empty was
    # never evidence that the predicate cannot fire.
    rc, out = _run(tmp_path, two, "--snapshot", "2023-05-01")
    assert "THE INVARIANT HAS BROKEN" in out
    assert rc == 1


def test_main_fails_p1_when_a_turn_carries_no_date(tmp_path) -> None:
    corpus = [*TWO, _conv({"session_1": ("no date at all", ["undated"])})]
    rc, out = _run(tmp_path, corpus)
    assert "P1 FAILED" in out
    assert rc == 1


@pytest.mark.parametrize("flags", [("--snapshot",), ("--snapshot", "notadate"), ("--bogus",)])
def test_main_usage_errors_are_not_reportable_as_a_broken_invariant(tmp_path, flags) -> None:
    """Exit 2, never 1. A typo sharing an exit code with a real finding is a false alarm."""
    rc, _out = _run(tmp_path, TWO, *flags)
    assert rc == 2


def test_main_accepts_the_equals_form(tmp_path) -> None:
    """`--snapshot=X` was silently ignored, running the derived path and exiting 0."""
    rc, out = _run(tmp_path, TWO, "--snapshot=2023-06-01")
    assert "DECLARED" in out
    assert rc == 1


def test_main_missing_corpus_is_its_own_exit_code(tmp_path) -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["--corpus", str(tmp_path / "absent.json")])
    assert "MISSING" in buf.getvalue()
    assert rc == 2


def test_main_single_scope_break_is_reported_as_a_break_not_a_control_failure(tmp_path) -> None:
    """The verdict ORDER, and the case the whole script exists for.

    One transcript means widening the scope adds nothing, so the merged control is empty by
    construction. With the control tested before P2, a live transcript whose turns postdate its
    declared instant reported `CONTROL FAILED ... the predicate never fires` on the very line after
    printing a non-zero P2 count and naming the demoted turn.
    """
    one = [_conv({
        "session_1": ("1:00 pm on 8 May, 2023", ["early"]),
        "session_2": ("1:00 pm on 9 June, 2023", ["arrived after the snapshot"]),
    })]
    rc, out = _run(tmp_path, one, "--snapshot", "2023-05-20")
    assert "THE INVARIANT HAS BROKEN" in out
    assert "CONTROL FAILED" not in out
    assert rc == 1


def test_main_reports_the_control_over_derived_instants_not_the_snapshot(tmp_path) -> None:
    """The label. In declared mode the P3 count is NOT taken over the snapshot printed above it."""
    rc, out = _run(tmp_path, TWO, "--snapshot", "2099-01-01")
    assert "per-conversation derived instants" in out
    assert "same instants" not in out
    assert rc == 0


def test_main_share_denominator_is_pairs_not_turns(tmp_path) -> None:
    """Pins the published percentage, on a NON-SQUARE fixture.

    The first version used `TWO`: 2 turns and 2 conversations, so `turns * instants`, `turns ** 2`
    and `instants ** 2` are all 4 and the assertion pinned the denominator's magnitude but not its
    composition. Two mutations that destroy the published 28.4% passed green under it.

    Three turns over two conversations separates them: the shipped denominator gives 6 pairs and
    16.7%, `turns ** 2` gives 11.1%, `instants ** 2` gives 25.0%.
    """
    _rc, out = _run(tmp_path, OBLONG)
    assert "1 turn-instant pairs (16.7% of all pairs)" in out


def test_main_malformed_corpus_is_a_usage_error_not_a_finding(tmp_path) -> None:
    """Exit 2. A mistyped --corpus landing on another JSON artifact must not read as a break."""
    bad = tmp_path / "bad.json"
    bad.write_text("not json at all", encoding="utf-8")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["--corpus", str(bad)])
    assert "MALFORMED" in buf.getvalue()
    assert rc == 2

    bad.write_text('{"conversations": []}', encoding="utf-8")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["--corpus", str(bad)])
    assert "MALFORMED" in buf.getvalue()
    assert rc == 2


def test_main_single_transcript_hold_names_construction_as_the_reason(tmp_path) -> None:
    """One transcript has nothing to merge with, so its control is empty whatever the data says."""
    one = [_conv({
        "session_1": ("1:00 pm on 8 May, 2023", ["early"]),
        "session_2": ("1:00 pm on 9 June, 2023", ["later"]),
    })]
    rc, out = _run(tmp_path, one, "--snapshot", "2099-01-01")
    assert "control UNINFORMATIVE" in out
    assert "nothing to merge with" in out
    assert rc == 0


def test_main_prints_the_counts_that_get_published(tmp_path) -> None:
    """The three headline lines carry every figure quoted in the design doc, and nothing pinned
    them: hard-coding the P1 counter to 0 left the suite green, which is round one's defect."""
    # NON-SQUARE. On the 2x2 `TWO` fixture both counts print 2, so swapping them is undetectable:
    # measured, that mutation left the suite green. Three turns over two conversations separates
    # them, and CI cannot fall back on the real corpus because locomo10.json is gitignored.
    _rc, out = _run(tmp_path, OBLONG)
    assert "conversations: 2" in out
    assert "turns with a resolved utterance date: 3" in out
    assert "turns WITHOUT one (P1, reported not dropped): 0" in out

    # BUG-003: the two counts that carry the published figures.
    assert "of their own scope: 0" in out
    assert "postdating them anywhere: 1" in out

    # A NON-ZERO reading too. Asserting only the zero case is satisfied by a counter hard-coded to
    # 0, which is precisely the defect P1 exists to catch: measured, that mutation left the suite
    # green while the line under test had stopped reporting anything.
    undated = [*TWO, _conv({"session_1": ("no date at all", ["one", "two"])})]
    _rc, out = _run(tmp_path, undated)
    assert "turns WITHOUT one (P1, reported not dropped): 2" in out


def test_main_detail_line_names_the_instant_that_caused_the_demotion(tmp_path) -> None:
    """It must be the declared snapshot, not the conversation's derived end."""
    _rc, out = _run(tmp_path, TWO, "--snapshot", "2023-06-01")
    assert "instant=2023-06-01" in out
    assert "instant=2024-05-08" not in out
    # NON-ZERO P2. Asserted only where the true count is zero, the line is satisfied by a counter
    # hard-coded to zero, which is the defect that had already slipped through once on the P1 line.
    assert "of their own scope: 1" in out


def test_sample_without_a_conversation_object_is_reported() -> None:
    turns, unresolved = corpus_turns([{"speaker": "A"}, *CLOSED])
    assert len(turns) == 3
    assert [w for _idx, _dia, w in unresolved] == ["entry carries no conversation object"]


def test_session_key_without_an_index_is_reported() -> None:
    corpus = [{"conversation": {
        "session_1a": [{"speaker": "A", "dia_id": "D1:1", "text": "ok"}],
        "session_1a_date_time": "1:00 pm on 8 May, 2023",
    }}]
    turns, unresolved = corpus_turns(corpus)
    assert turns == []
    assert [w for _idx, _dia, w in unresolved] == ["session key does not carry an index"]


def test_main_reads_the_real_command_line_when_argv_is_omitted(tmp_path, monkeypatch) -> None:
    """The installed entry point. No test called `main()` without an explicit argv, so a
    regression to round one's silently-ignored-flags defect would have shipped green."""
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(TWO), encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["check_utterance_postdating", "--corpus", str(path), "--snapshot", "2023-06-01"]
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main()
    assert "DECLARED (--snapshot 2023-06-01)" in buf.getvalue()
    assert rc == 1


def test_main_no_resolvable_turns_is_not_a_pass(tmp_path) -> None:
    """A corpus that parses but yields nothing must not exit 0; a CI caller reads 0 as a pass."""
    rc, out = _run(tmp_path, [])
    assert "no turns resolved" in out
    assert rc == 1

    rc, out = _run(tmp_path, [{"conversation": {"session_1": [],
                                                "session_1_date_time": "1:00 pm on 8 May, 2023"}}])
    assert "no turns resolved" in out
    assert rc == 1


def test_main_declared_verdict_carries_the_control_figures(tmp_path) -> None:
    """The passing verdict prints its OWN copy of the count and share, which nothing else pinned."""
    _rc, out = _run(tmp_path, OBLONG, "--snapshot", "2099-01-01")
    assert "invariant holds against a declared instant" in out
    assert "1 turn-instant pairs (16.7% of all pairs)" in out


def test_main_help_is_not_a_verdict(tmp_path) -> None:
    """`--help` must exit 0 and print no verdict a caller could mistake for a pass."""
    rc, out = _run(tmp_path, TWO, "--help")
    assert rc == 0
    assert "VERDICT" not in out


def test_absent_and_non_string_dia_ids_get_their_own_reasons(tmp_path) -> None:
    """Each reason must name the defect it found, or it sends the reader after the wrong thing."""
    corpus = [_conv({"session_1": ("1:00 pm on 8 May, 2023", ["a", "b", "c"])})]
    turns = corpus[0]["conversation"]["session_1"]
    turns[0]["dia_id"] = ""
    turns[1]["dia_id"] = 0
    resolved, unresolved = corpus_turns(corpus)
    assert [d for _i, d, _w in resolved] == ["D1:3"]
    assert sorted(w for _i, _d, w in unresolved) == [
        "dia_id is absent or empty",
        "dia_id is int, not a string",
    ]

    # A FALSY non-string, which is the whole point of testing type before emptiness. `12` is truthy
    # and lands identically under both orderings, so on its own it pins only "a type check exists".
    corpus[0]["conversation"]["session_1"][1]["dia_id"] = False
    _resolved, unresolved = corpus_turns(corpus)
    assert "dia_id is bool, not a string" in [w for _i, _d, w in unresolved]
