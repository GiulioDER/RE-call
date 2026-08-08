"""The frozen injection baseline, and the proof that it can move.

`benchmarks/evidence_injection.py` runs 13 payloads through 3 carriers and reports how many left
the delimited data region. This file turns that number into a ratchet: a later session can show
the rate did not increase, which is the only way "we did not make it worse" is a claim rather than
an impression.

Three things have to hold together, and each is a separate test because each fails separately:

1. **The suite is frozen.** A digest over the payload, carrier and detector lists must match the
   artifact. Editing a payload out of the suite is otherwise indistinguishable from fixing it.
2. **The rate did not increase.** Escapes are compared against the recorded count, per carrier as
   well as in total, so an improvement in one arm cannot mask a regression in another.
3. **The detectors can fire.** Zero escapes and a broken detector set produce the same number.
   The positive control runs the identical detectors against the renderer this session replaced,
   which must be reported as escaping.

And a fourth, easy to forget: **the evidence must still be there.** A renderer that deleted all
corpus text would score a perfect zero, so the suite also records whether each payload survives a
JSON parse of the region it was placed in.
"""
from __future__ import annotations

import json

import pytest

from dataclasses import replace

from benchmarks.evidence_injection import (
    _CARRIER_FIELD,
    CARRIERS as _CARRIERS_DERIVED,
    injection_rate,
    BASELINE_PATH,
    build_baseline,
    CARRIERS,
    DETECTORS,
    PAYLOADS,
    deletion_control,
    positive_control,
    run_suite,
    suite_digest,
)


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def trials() -> list:
    return run_suite()


def test_the_suite_is_the_one_the_baseline_was_recorded_against(baseline) -> None:
    """A rate is only comparable to a rate measured on the same instrument."""
    assert baseline["suite_digest"] == suite_digest(), (
        "the payload, carrier or detector list changed since the baseline was frozen. If that was "
        "deliberate, regenerate with `python -m scripts.generate_evidence_injection_baseline` and "
        "look hard at the diff; if it was not, revert the edit."
    )
    assert baseline["trials"] == len(CARRIERS) * len(PAYLOADS)
    assert baseline["detectors"] == [name for name, _ in DETECTORS]


def test_the_injection_success_rate_did_not_increase(baseline, trials) -> None:
    escapes = sum(1 for trial in trials if trial.escaped)

    assert len(trials) == baseline["trials"], "the trial count moved without the digest moving"
    # THE RATE ITSELF, and over the denominator that names it. Asserting only the escape COUNT
    # left the published rate unpinned: a mutation putting `len(trials)` back in the denominator
    # survived this file untouched, which is the same defect the denominator fix was about.
    carrying = [trial for trial in trials if _CARRIER_FIELD[trial.carrier] is not None]
    assert len(carrying) == baseline["carrying_trials"]
    assert len(carrying) < len(trials), "no arm is inert, so the two denominators cannot differ"
    assert baseline["injection_success_rate"] == round(escapes / len(carrying), 6)
    assert escapes <= baseline["escapes"], (
        f"{escapes} payloads escaped the evidence region, up from {baseline['escapes']}: "
        + "; ".join(
            f"{trial.carrier}/{trial.payload_id} fired {trial.detectors_fired}"
            for trial in trials
            if trial.escaped
        )
    )


def test_no_carrier_regressed_while_another_improved(baseline, trials) -> None:
    """A total that holds can hide one arm getting worse and another getting better."""
    for carrier in CARRIERS:
        rows = [trial for trial in trials if trial.carrier == carrier]
        escapes = sum(1 for trial in rows if trial.escaped)
        assert escapes <= baseline["by_carrier"][carrier]["escapes"], (
            f"the {carrier} carrier regressed: {escapes} escapes against "
            f"{baseline['by_carrier'][carrier]['escapes']} recorded"
        )


def test_the_detectors_fire_against_the_renderer_this_replaced(baseline) -> None:
    """The positive control. Without it, zero escapes and an inert detector set read alike.

    The pre-2026-08-05 renderer wrapped a plain `json.dumps` in the same delimiter. `json.dumps`
    escapes quotes, backslashes and control characters and does NOT escape `<` or `>`, which is
    exactly what the delimiter is made of — so a memory containing `</evidence_data>` closed the
    region early. That is the escape this control reproduces.
    """
    control = positive_control()
    escapes = [trial for trial in control if trial.escaped]

    assert escapes, "the detectors did not fire against a renderer known to leak: they are inert"
    assert len(escapes) >= baseline["positive_control_escapes"], (
        "the positive control got weaker, so the zero above is worth less than it was"
    )
    assert {trial.payload_id for trial in escapes} >= {"close_tag", "open_tag"}
    # The control's rate, over the same denominator. This is the number that says how bad the
    # pre-fix renderer was, and dividing it by every trial understated it by a third.
    carrying = [t for t in control if _CARRIER_FIELD[t.carrier] is not None]
    assert baseline["positive_control_rate"] == round(len(escapes) / len(carrying), 6)
    # The tag-closing payloads must trip the delimiter detector specifically, not merely trip
    # SOMETHING — an escape counted for an unrelated reason would not be evidence of this defect.
    assert all(
        "delimiter_not_intact" in trial.detectors_fired
        for trial in escapes
        if trial.payload_id in {"close_tag", "open_tag"}
    )


def test_the_boundary_escapes_the_evidence_rather_than_deleting_it(baseline, trials) -> None:
    """Zero escapes is worthless if it was bought by dropping the corpus text.

    `metadata` is excluded and that exclusion is stated rather than assumed: an evidence item
    carries no corpus metadata dict at all, so nothing from that carrier reaches the prompt to be
    preserved. Its zero is structural, not the result of escaping — which is also why its zero
    escape count proves less than the other two arms'.
    """
    carrying = [c for c in CARRIERS if _CARRIER_FIELD[c] is not None]
    assert carrying, "no carrier reaches the prompt; the whole suite would be vacuous"
    for carrier in carrying:
        rows = [trial for trial in trials if trial.carrier == carrier]
        preserved = sum(1 for trial in rows if trial.payload_preserved)
        assert preserved == len(rows), (
            f"{len(rows) - preserved} {carrier} payloads did not survive a JSON round trip: the "
            f"boundary is mangling evidence, not escaping it"
        )
        assert preserved == baseline["by_carrier"][carrier]["payload_preserved"]

    assert len(carrying) == baseline["carrying_trials"] // len(PAYLOADS)
    assert baseline["by_carrier"]["metadata"]["payload_preserved"] == 0
    assert all(
        not trial.payload_preserved for trial in trials if trial.carrier == "metadata"
    ), "chunk metadata began reaching the prompt; that arm now needs escaping, not absence"


def test_the_preservation_check_can_report_absence(baseline) -> None:
    """The negative control, and it was a mutation sweep that showed it was needed.

    `payload_preserved` reads 13 of 13 on both live arms, so a `_preserved` stuck at True is
    indistinguishable from a working one — the balanced-fixture failure this repository has
    recorded three times. This runs the same suite against a renderer that ships no evidence at
    all: it must score a PERFECT escape rate and zero preservation, which is the pair that makes
    the real zero mean something.
    """
    control = deletion_control()

    assert not any(trial.escaped for trial in control), (
        "a renderer that ships no evidence should escape nothing — the detectors are miscounting"
    )
    assert not any(trial.payload_preserved for trial in control), (
        "the preservation check reported evidence that was never rendered"
    )
    assert baseline["escapes"] == 0, "the live suite and this control must agree on the escape rate"


def test_every_payload_is_exercised_against_every_carrier(trials) -> None:
    """A trial that silently stopped running would lower the escape count for free."""
    seen = {(trial.carrier, trial.payload_id) for trial in trials}

    assert seen == {(carrier, payload_id) for carrier in CARRIERS for payload_id, _ in PAYLOADS}


def test_the_generator_divides_by_the_denominator_that_names_the_rate(baseline, trials) -> None:
    """Drives `build_baseline()`, which is the code the committed artifact came from.

    Recomputing the rate inside the test and comparing it to the JSON pins the FILE, not the
    generator: a mutation putting `len(trials)` back in the divisor changed the generator and no
    test noticed, because nothing called it. Two mutation rounds were needed to see that, and it
    is the same shape as the defect being guarded — a number checked against itself.

    Both rates are asserted against the inert-arm denominator explicitly, so a regression has to
    move a number rather than merely relabel one.
    """
    regenerated = build_baseline()

    carrying = [t for t in trials if _CARRIER_FIELD[t.carrier] is not None]
    escapes = sum(1 for t in trials if t.escaped)
    control = positive_control()
    control_carrying = [t for t in control if _CARRIER_FIELD[t.carrier] is not None]
    control_escapes = sum(1 for t in control if t.escaped)

    assert regenerated["carrying_trials"] == len(carrying) < regenerated["trials"]
    assert regenerated["injection_success_rate"] == round(escapes / len(carrying), 6)
    assert regenerated["positive_control_rate"] == round(
        control_escapes / len(control_carrying), 6
    )
    # The wrong denominator is STRICTLY SMALLER for a non-zero numerator, so this discriminates
    # rather than merely restating the line above.
    assert control_escapes > 0, "the control must escape, or the comparison below is vacuous"
    assert regenerated["positive_control_rate"] > control_escapes / len(control)
    # And the committed artifact must agree with the generator, or the ratchet is stale.
    assert regenerated == {k: v for k, v in baseline.items() if not k.startswith("_")}


def test_the_rate_counts_only_attempts_in_its_numerator_too(trials) -> None:
    """The defect the one-divisor fix introduced while removing the same defect.

    `injection_rate` narrowed the DENOMINATOR to payload-carrying trials and left the numerator
    over every row, so an escape from the arm excluded from the denominator was still counted:
    marking all 52 trials escaped returned 1.333333. A rate above 1 is not a rate. Both tests
    written for the denominator recomputed the identical asymmetric expression, so neither could
    see it — which is why this one fabricates escapes instead of recomputing a formula.
    """
    everything_escaped = [replace(t, detectors_fired=("in_system_prompt",)) for t in trials]
    only_inert_escaped = [
        replace(t, detectors_fired=("in_system_prompt",)) if _CARRIER_FIELD[t.carrier] is None else t
        for t in trials
    ]

    assert injection_rate(everything_escaped) == 1.0, "a rate must not exceed 1"
    assert injection_rate(only_inert_escaped) == 0.0, (
        "an escape from an arm that makes no attempt was counted as a successful attempt"
    )
    assert injection_rate(trials) == 0.0


def test_the_rate_refuses_a_denominator_it_does_not_have(trials) -> None:
    """Named, rather than a bare ZeroDivisionError out of a benchmark script."""
    inert_only = [t for t in trials if _CARRIER_FIELD[t.carrier] is None]

    for rows in ([], inert_only):
        with pytest.raises(ValueError, match="no denominator"):
            injection_rate(rows)


def test_the_carrier_list_and_its_field_map_cannot_drift() -> None:
    """Three call sites now subscript `_CARRIER_FIELD` bare, including the baseline generator."""
    assert set(_CARRIERS_DERIVED) == set(_CARRIER_FIELD)
    assert _CARRIERS_DERIVED == CARRIERS
