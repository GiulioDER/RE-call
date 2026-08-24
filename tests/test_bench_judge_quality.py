"""Offline tests for the constructed judge-quality probes.

No network, no database, no model: every judge here is a plain function. What is pinned is the
part that makes the measurement worth anything — that each probe's EXPECTED verdict really is
guaranteed by its construction:

* a reworded gold still states the same facts, and a transform that cannot preserve them is not
  applied;
* a swapped gold really is wrong, i.e. the pairs where it might be right are screened out and
  counted;
* both judges see the identical item list, so their columns are comparable;
* a maximally lenient judge and a maximally strict one are characterised correctly, which is what
  says the harness measures the judge rather than agreeing with it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.judge_quality import (
    REWORDED,
    SWAPPED,
    VERBATIM,
    build_probes,
    carrier_phrase,
    eligible_records,
    judge_quality_document,
    main,
    normalise,
    reformat_date,
    reorder_list,
    reword,
    select_sources,
    swap_reject_reason,
)

#: Benchmark-harness coverage, not product coverage; product CI can deselect with
#: `-m 'not benchharness'`.
pytestmark = pytest.mark.benchharness


def _record(
    question_id: str,
    question: str,
    gold: str,
    *,
    adversarial: bool = False,
    abstained: bool = False,
) -> dict[str, Any]:
    """One ``outcomes`` entry, in the shape ``benchmarks.run`` writes it."""
    return {
        "question_id": question_id,
        "category": "cat1",
        "is_adversarial": adversarial,
        "context": "retrieved context",
        "answer": "NO_ANSWER" if abstained else "the model's answer",
        "abstained": abstained,
        "correct": None if adversarial else True,
        "question": question,
        "gold": "" if adversarial else gold,
    }


def _doc(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "arm": "recall",
        "config": {"model": "openai/gpt-4o-mini", "judge_system_prompt": "grade it"},
        "outcomes": records,
    }


def _conversation() -> dict[str, Any]:
    """Four questions from one conversation, with four mutually unrelated gold answers."""
    return _doc(
        [
            _record("conv-1:0", "When did Melanie run the race?", "7 May 2023"),
            _record("conv-1:1", "What did Caroline buy?", "apples, oranges, pears"),
            _record("conv-1:2", "Who is Melanie's coach?", "Nadia"),
            _record("conv-1:3", "Where does Caroline work?", "a bookshop in Leeds"),
        ]
    )


class ScriptedJudge:
    """A judge whose verdict is decided by a substring of the prompt; remembers every call."""

    def __init__(self, yes_when: tuple[str, ...] = (), default: bool = False) -> None:
        self.yes_when = yes_when
        self.default = default
        self.calls: list[str] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append(user)
        if any(marker in user for marker in self.yes_when):
            return "YES"
        return "YES" if self.default else "NO"


def _always_yes(system: str, user: str) -> str:
    return "YES"


def _always_no(system: str, user: str) -> str:
    return "NO"


# --------------------------------------------------------------------------------------------
# Probe 2: the transforms preserve the facts, and are skipped when they cannot
# --------------------------------------------------------------------------------------------


def test_date_reformat_swaps_the_order_and_keeps_the_day() -> None:
    assert reformat_date("7 May 2023") == "May 7, 2023"
    assert reformat_date("July 14, 2023") == "14 July 2023"
    assert reformat_date("14th July, 2023") == "July 14, 2023"


def test_date_reformat_is_none_without_a_date() -> None:
    assert reformat_date("apples, oranges, pears") is None
    assert reformat_date("she was happy") is None


def test_list_reorder_is_a_permutation_of_the_same_items() -> None:
    out = reorder_list("apples, oranges, pears")

    assert out is not None
    assert out != "apples, oranges, pears"
    assert sorted(part.strip() for part in out.split(",")) == ["apples", "oranges", "pears"]


def test_list_reorder_keeps_a_trailing_full_stop_at_the_end() -> None:
    out = reorder_list("apples, oranges, pears.")

    assert out == "pears, apples, oranges."


def test_list_reorder_moves_a_closing_conjunction_to_the_new_last_item() -> None:
    """``and`` closes the enumeration, not the item — rotating it to the front would be nonsense."""
    assert reorder_list("the space, furniture, and decor") == "decor, the space, and furniture"
    assert reorder_list("tea, coffee, or juice") == "juice, tea, or coffee"


def test_list_reorder_refuses_a_bare_conjunction_item() -> None:
    assert reorder_list("apples, pears, and") is None


def test_list_reorder_refuses_a_gold_containing_a_date() -> None:
    """``May 8, 2023`` splits on its own comma; rotating it would invent a different date."""
    assert reorder_list("May 8, 2023") is None
    assert reorder_list("she ran on 8 May, 2023") is None


def test_list_reorder_refuses_a_sequenced_list() -> None:
    """An ordered narrative's order IS a fact, so it is not a list this transform may rotate."""
    assert reorder_list("she cooked dinner, then washed up") is None
    assert reorder_list("first the gym, second the shop") is None


def test_list_reorder_refuses_non_lists_and_clauses() -> None:
    assert reorder_list("Nadia") is None  # one item
    assert reorder_list("apples, , pears") is None  # empty item
    assert reorder_list("she went to the shop, and then she came home again") is None  # clauses


def test_carrier_phrase_embeds_the_gold_verbatim() -> None:
    assert carrier_phrase("Nadia") == "The answer is Nadia."
    assert carrier_phrase("She went home.") == "The answer is She went home."


def test_reword_prefers_the_date_transform_then_the_list_then_the_carrier() -> None:
    assert reword("7 May 2023") == ("date_reformat", "May 7, 2023")
    assert reword("apples, oranges, pears") == ("list_reorder", "pears, apples, oranges")
    assert reword("Nadia") == ("carrier_phrase", "The answer is Nadia.")


def test_reword_always_applies_to_a_non_empty_gold() -> None:
    """The carrier phrase is the fallback, so a non-empty gold never fails to produce an item."""
    for gold in ("Nadia", "May 8, 2023", "she cooked dinner, then washed up", "42"):
        assert reword(gold) is not None


def test_reworded_items_are_expected_to_be_accepted_and_record_their_transform() -> None:
    probes, report = build_probes(_conversation(), sample=10, seed=1)

    reworded = probes[REWORDED]
    assert len(reworded) == 4
    assert all(item.expected is True for item in reworded)
    assert report[REWORDED]["no_transform"] == 0
    assert report[REWORDED]["by_transform"] == {
        "carrier_phrase": 2,
        "date_reformat": 1,
        "list_reorder": 1,
    }


# --------------------------------------------------------------------------------------------
# Probe 3: the swap screen
# --------------------------------------------------------------------------------------------


def test_swap_screen_rejects_identical_and_contained_golds() -> None:
    source = {"question_id": "c:0", "question": "When did she run?", "gold": "7 May 2023"}
    same = {"question_id": "c:1", "question": "What day was the race?", "gold": "7 may 2023."}
    contained = {
        "question_id": "c:2",
        "question": "What happened?",
        "gold": "She ran the race on 7 May 2023",
    }
    echo = {"question_id": "c:3", "question": "when did she RUN?", "gold": "Tuesday"}
    blank = {"question_id": "c:4", "question": "What?", "gold": "  ,  "}
    fine = {"question_id": "c:5", "question": "Who coached her?", "gold": "Nadia"}

    assert swap_reject_reason(source, same) == "identical_gold"
    assert swap_reject_reason(source, contained) == "gold_containment"
    assert swap_reject_reason(contained, source) == "gold_containment"
    assert swap_reject_reason(source, echo) == "identical_question"
    assert swap_reject_reason(source, blank) == "empty_after_normalisation"
    assert swap_reject_reason(source, fine) is None


def test_containment_is_word_bounded_not_substring() -> None:
    """``May`` inside ``Maybe`` is not containment — discarding that pair would cost coverage."""
    source = {"question_id": "c:0", "question": "Which month?", "gold": "May"}
    mate = {"question_id": "c:1", "question": "Was she sure?", "gold": "Maybe"}

    assert normalise("Maybe.") == "maybe"
    assert swap_reject_reason(source, mate) is None


def test_swapped_items_use_another_question_from_the_same_conversation() -> None:
    probes, report = build_probes(_conversation(), sample=10, seed=1)

    swapped = probes[SWAPPED]
    assert len(swapped) == 4
    assert report[SWAPPED]["rejected_candidates"]["total"] == 0
    golds = {r["question_id"]: r["gold"] for r in eligible_records(_conversation())}
    for item in swapped:
        assert item.expected is False
        partner = item.construction.removeprefix("swap:")
        assert partner != item.question_id
        assert partner.startswith("conv-1:")
        assert item.prediction == golds[partner]
        assert item.prediction != item.gold


def test_swap_rejections_are_counted_by_reason() -> None:
    doc = _doc(
        [
            _record("conv-1:0", "When did she run?", "7 May 2023"),
            _record("conv-1:1", "What day was the race?", "7 May 2023."),
            _record("conv-1:2", "Who coached her?", "Nadia"),
        ]
    )

    probes, report = build_probes(doc, sample=10, seed=1)

    # conv-1:0 scans conv-1:1 first, which is its own gold wearing a full stop -> rejected, and
    # it falls through to conv-1:2. The screen cost one candidate and no item.
    rejected = report[SWAPPED]["rejected_candidates"]
    assert rejected["by_reason"] == {"identical_gold": 1}
    assert rejected["total"] == 1
    assert report[SWAPPED]["n_items"] == 3
    predictions = {item.question_id: item.prediction for item in probes[SWAPPED]}
    assert predictions["conv-1:0"] == "Nadia"


def test_a_source_with_no_usable_partner_yields_no_item() -> None:
    doc = _doc([_record("conv-1:0", "Who coached her?", "Nadia")])

    probes, report = build_probes(doc, sample=10, seed=1)

    assert probes[SWAPPED] == []
    assert report[SWAPPED]["rejected_candidates"]["by_reason"] == {"no_usable_partner": 1}
    assert len(probes[VERBATIM]) == 1  # the other probes are unaffected


# --------------------------------------------------------------------------------------------
# Source selection
# --------------------------------------------------------------------------------------------


def test_adversarial_and_goldless_records_are_not_probe_sources() -> None:
    doc = _doc(
        [
            _record("conv-1:0", "Who coached her?", "Nadia"),
            _record("conv-1:1", "What did Caroline realise?", "", adversarial=True),
            _record("conv-1:2", "", "orphan question"),
            _record("conv-1:3", "Where does she work?", "   "),
            _record("conv-1:4", "When did she run?", "7 May 2023", abstained=True),
        ]
    )

    records = eligible_records(doc)

    # The abstained record IS eligible: the probe is built from the gold, not from the answer.
    assert [r["question_id"] for r in records] == ["conv-1:0", "conv-1:4"]


def test_records_are_ordered_by_numeric_position_not_lexically() -> None:
    doc = _doc([_record(f"conv-1:{i}", f"q{i}?", f"g{i}") for i in (0, 9, 10, 2)])

    assert [r["question_id"] for r in eligible_records(doc)] == [
        "conv-1:0",
        "conv-1:2",
        "conv-1:9",
        "conv-1:10",
    ]


def test_the_same_seed_draws_the_same_sample() -> None:
    records = eligible_records(_doc([_record(f"c:{i}", f"q{i}?", f"gold {i}") for i in range(40)]))

    first = select_sources(records, sample=8, seed=7)
    again = select_sources(records, sample=8, seed=7)
    other = select_sources(records, sample=8, seed=8)

    assert [r["question_id"] for r in first] == [r["question_id"] for r in again]
    assert [r["question_id"] for r in first] != [r["question_id"] for r in other]
    assert len(first) == 8


def test_sample_larger_than_the_pool_uses_everything() -> None:
    records = eligible_records(_doc([_record(f"c:{i}", f"q{i}?", f"gold {i}") for i in range(3)]))

    assert len(select_sources(records, sample=100, seed=1)) == 3


def test_both_judges_are_shown_the_identical_items() -> None:
    """A per-model sample would make the two columns incomparable — pin that it is not one."""
    doc = _doc([_record(f"conv-1:{i}", f"question {i}?", f"gold {i}") for i in range(40)])

    payload = judge_quality_document(
        doc,
        {"lenient/judge": _always_yes, "strict/judge": _always_no},
        sample=9,
        seed=3,
        source="results/a.json",
    )

    for probe in (VERBATIM, REWORDED, SWAPPED):
        built = [item["item_id"] for item in payload["probes"][probe]]
        lenient = [v["item_id"] for v in payload["models"]["lenient/judge"]["verdicts"][probe]]
        strict = [v["item_id"] for v in payload["models"]["strict/judge"]["verdicts"][probe]]
        assert built == lenient == strict
        assert len(built) == 9


# --------------------------------------------------------------------------------------------
# Rates
# --------------------------------------------------------------------------------------------


def test_a_judge_that_accepts_everything_is_characterised_as_maximally_lenient() -> None:
    payload = judge_quality_document(
        _conversation(), {"yes/judge": _always_yes}, sample=10, seed=1, source="a.json"
    )

    result = payload["models"]["yes/judge"]
    assert result["verbatim_accept_rate"]["rate"] == 1.0
    assert result["reworded_accept_rate"]["rate"] == 1.0
    # The false-accept rate the audit measured: every wrong-but-on-topic answer waved through.
    assert result["swapped_accept_rate"]["rate"] == 1.0
    assert result["judge_errors"] == {VERBATIM: 0, REWORDED: 0, SWAPPED: 4}


def test_a_judge_that_rejects_everything_is_characterised_as_maximally_strict() -> None:
    payload = judge_quality_document(
        _conversation(), {"no/judge": _always_no}, sample=10, seed=1, source="a.json"
    )

    result = payload["models"]["no/judge"]
    assert result["verbatim_accept_rate"]["rate"] == 0.0
    assert result["reworded_accept_rate"]["rate"] == 0.0
    assert result["swapped_accept_rate"]["rate"] == 0.0
    # Rejecting the gold answer itself is the unambiguous error; the swap probe is all correct.
    assert result["judge_errors"] == {VERBATIM: 4, REWORDED: 4, SWAPPED: 0}


def test_accept_rate_carries_n_and_a_wilson_interval() -> None:
    """2 of 4 accepted -> 0.5 with the hand-checked Wilson 95% interval [0.15, 0.85]."""
    judge = ScriptedJudge(yes_when=("Gold answer: 7 May 2023", "Gold answer: Nadia"))

    payload = judge_quality_document(
        _conversation(), {"scripted/judge": judge}, sample=10, seed=1, source="a.json"
    )

    rate = payload["models"]["scripted/judge"]["verbatim_accept_rate"]
    assert rate["n"] == 4
    assert rate["rate"] == 0.5
    assert rate["ci95"] == [0.15, 0.85]


def test_an_empty_probe_reports_no_rate_rather_than_nan() -> None:
    """A one-question conversation builds no swap item; the cell must still be valid JSON."""
    doc = _doc([_record("conv-1:0", "Who coached her?", "Nadia")])

    payload = judge_quality_document(
        doc, {"yes/judge": _always_yes}, sample=10, seed=1, source="a.json"
    )

    rate = payload["models"]["yes/judge"]["swapped_accept_rate"]
    assert rate == {"n": 0, "rate": None, "ci95": [None, None]}
    json.dumps(payload)  # no bare NaN token


def test_the_judge_is_asked_about_the_source_question_and_its_own_gold() -> None:
    """The swap probe keeps the REAL gold in the prompt — the wrong answer is the prediction."""
    judge = ScriptedJudge()
    doc = _doc(
        [
            _record("conv-1:0", "Who coached her?", "Nadia"),
            _record("conv-1:1", "Where does she work?", "a bookshop in Leeds"),
        ]
    )

    judge_quality_document(doc, {"j": judge}, sample=10, seed=1, source="a.json")

    swap_prompts = [
        call
        for call in judge.calls
        if "Question: Who coached her?" in call and "Predicted answer: a bookshop in Leeds" in call
    ]
    assert len(swap_prompts) == 1
    assert "Gold answer: Nadia" in swap_prompts[0]


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def _write_artifact(path: Path) -> None:
    path.write_text(json.dumps(_conversation()), encoding="utf-8")


def test_cli_dry_run_builds_the_probes_without_calling_any_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    source = tmp_path / "a.json"
    _write_artifact(source)
    out = tmp_path / "q.json"

    assert main(["--artifact", str(source), "--out", str(out), "--dry-run"]) == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["models"] == {}
    assert payload["construction"]["eligible_records"] == 4
    assert len(payload["probes"][VERBATIM]) == 4


def test_cli_needs_models_unless_dry_running(tmp_path: Path, capsys: Any) -> None:
    source = tmp_path / "a.json"
    _write_artifact(source)

    with pytest.raises(SystemExit):
        main(["--artifact", str(source), "--out", str(tmp_path / "q.json")])
    assert "--models is required" in capsys.readouterr().err


def test_cli_refuses_to_clobber_an_existing_measurement(tmp_path: Path, capsys: Any) -> None:
    source = tmp_path / "a.json"
    _write_artifact(source)
    existing = tmp_path / "q.json"
    existing.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit):
        main(["--artifact", str(source), "--out", str(existing), "--dry-run"])
    assert "refusing to overwrite" in capsys.readouterr().err


def test_cli_validates_the_api_key_shape_before_spending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    source = tmp_path / "a.json"
    _write_artifact(source)
    monkeypatch.setenv("OPENROUTER_API_KEY", "...")

    with pytest.raises(SystemExit):
        main(
            [
                "--artifact",
                str(source),
                "--models",
                "openai/gpt-4o-mini",
                "--out",
                str(tmp_path / "q.json"),
            ]
        )
    assert "OpenRouter key" in capsys.readouterr().err
