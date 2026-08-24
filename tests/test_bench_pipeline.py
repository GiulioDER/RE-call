import json
from typing import Any

import pytest

from benchmarks.pipeline import (
    GEN_SYSTEM_PROMPT,
    NO_ANSWER,
    Outcome,
    aggregate,
    approx_tokens,
    context_size,
    generate_answer,
    is_abstention,
    judge_correct,
    run_question,
)
from benchmarks.evidence_tokens import PinnedReaderTokenizer

#: Benchmark-harness coverage, not product coverage; product CI can deselect with
#: `-m 'not benchharness'`.
pytestmark = pytest.mark.benchharness


def test_is_abstention_exact_token() -> None:
    assert is_abstention(NO_ANSWER) is True


def test_is_abstention_is_whitespace_and_case_tolerant() -> None:
    assert is_abstention("  no_answer\n") is True
    assert is_abstention("No_Answer") is True


@pytest.mark.parametrize(
    "answer",
    [
        "NO_ANSWER.",
        '"NO_ANSWER"',
        "'NO_ANSWER'",
        "**NO_ANSWER**",
        "`NO_ANSWER`",
        "(NO_ANSWER)",
        "  NO_ANSWER .  ",
        "NO_ANSWER!",
    ],
)
def test_is_abstention_tolerates_wrapping_punctuation(answer: str) -> None:
    """The token wearing the punctuation a chat model routinely adds is still an abstention.

    This is the headline number on BOTH arms, so an over-strict match does not fail safe: every
    `NO_ANSWER.` counted as a real answer deflates adversarial abstention and inflates nothing
    that would reveal the mistake.
    """
    assert is_abstention(answer) is True


def test_is_abstention_false_for_real_answer() -> None:
    assert is_abstention("The limit is 500 rps.") is False
    # a real answer that merely mentions the token is not an abstention
    assert is_abstention("There is no answer key labelled NO_ANSWER here, but it is 500.") is False


@pytest.mark.parametrize(
    "answer",
    [
        # the token plus real content is an ANSWER, however it is punctuated — loosening the match
        # far enough to swallow these would make the abstention column meaningless
        "NO_ANSWER for the second part, but the first is 500 rps.",
        "The code NO_ANSWER means nothing here.",
        "no_answer_key",
        "NO_ANSWERS",
        "",
    ],
)
def test_is_abstention_false_when_the_token_is_not_the_whole_answer(answer: str) -> None:
    assert is_abstention(answer) is False


def test_approx_tokens_counts_words_and_standalone_punctuation() -> None:
    assert approx_tokens("") == 0
    assert approx_tokens("hello world") == 2
    # punctuation is charged separately, as a BPE tokeniser broadly would
    assert approx_tokens("Alice: it is 500 rps.") == 7


def test_context_size_reports_mean_and_median_per_arm() -> None:
    """The retrieved-context volume must be a published number, not a thing a reader must infer.

    At the same k the arms do not retrieve comparable material (verbatim turns vs compressed
    facts), so this is the measurement that keeps an accuracy comparison honest.
    """
    outs = [
        Outcome("1", "cat1", False, "abcd", "a", abstained=False, correct=True),
        Outcome("2", "cat1", False, "ab", "a", abstained=False, correct=True),
        Outcome("3", "cat1", False, "abcdefghij", "a", abstained=False, correct=True),
    ]
    stats = context_size(outs)
    assert stats["n"] == 3
    assert stats["chars"]["median"] == 4.0
    assert stats["chars"]["mean"] == pytest.approx(16 / 3, abs=0.05)
    assert stats["tokens_approx"]["median"] == 1.0


def test_context_size_is_json_safe_when_empty() -> None:
    stats = context_size([])
    assert stats == {
        "n": 0,
        "chars": {"mean": None, "median": None},
        "tokens_approx": {"mean": None, "median": None},
    }
    assert "NaN" not in json.dumps(stats)


def test_generate_answer_passes_context_and_question_to_llm() -> None:
    seen: dict[str, str] = {}

    def completer(system: str, user: str) -> str:
        seen["system"], seen["user"] = system, user
        return "500 rps"

    out = generate_answer(completer, context="rate limit is 500 rps", question="how many rps?")
    assert out == "500 rps"
    assert "NO_ANSWER" in seen["system"]  # generator is told how to abstain
    assert "rate limit is 500 rps" in seen["user"]  # context is provided
    assert "how many rps?" in seen["user"]


def test_generate_answer_empty_context_still_calls_llm() -> None:
    # RE-call abstains by returning empty context; the generator must then emit NO_ANSWER itself
    def completer(system: str, user: str) -> str:
        return "NO_ANSWER"

    assert generate_answer(completer, context="", question="q") == "NO_ANSWER"


def test_generate_answer_wraps_context_in_memories_delimiters() -> None:
    seen: dict[str, str] = {}

    def completer(system: str, user: str) -> str:
        if "user" not in seen:
            seen["user"] = user
        return "500 rps"

    generate_answer(completer, context="rate limit is 500 rps", question="how many rps?")
    user = seen["user"]
    # `find`, not `index`: `index` RAISES when the delimiter is missing, so the `!= -1` assertion
    # below could never fail and never asserted anything. `find` returns -1 and the check is real.
    start = user.find("<memories>")
    end = user.find("</memories>")
    assert start != -1
    assert end > start
    # the untrusted context is inside the delimited block
    assert "rate limit is 500 rps" in user[start:end]


def test_generate_answer_question_is_outside_memories_block() -> None:
    seen: dict[str, str] = {}

    def completer(system: str, user: str) -> str:
        seen["user"] = user
        return "500 rps"

    generate_answer(completer, context="rate limit is 500 rps", question="how many rps?")
    user = seen["user"]
    end = user.index("</memories>")
    # the question must appear after the closing delimiter, not inside the untrusted block
    assert "how many rps?" in user[end:]
    assert "how many rps?" not in user[: user.index("<memories>")]


def test_gen_system_prompt_instructs_memories_are_data_not_instructions() -> None:
    lowered = GEN_SYSTEM_PROMPT.casefold()
    assert "<memories>" in lowered
    assert "instruction" in lowered


def test_judge_correct_parses_yes_no() -> None:
    def yes_completer(system: str, user: str) -> str:
        return "YES"

    def no_completer(system: str, user: str) -> str:
        return "no"

    def verbose_yes_completer(system: str, user: str) -> str:
        return "YES, they match."

    assert judge_correct(yes_completer, "q", "500", "500 rps") is True
    assert judge_correct(no_completer, "q", "500", "42") is False
    # judge must be robust to a verbose reply
    assert judge_correct(verbose_yes_completer, "q", "500", "500 rps") is True


def _q(
    qid: str, cat: str, adversarial: bool, question: str = "q", answer: str = "500"
) -> dict[str, Any]:
    return {
        "question_id": qid,
        "category": cat,
        "adversarial": adversarial,
        "question": question,
        "answer": answer,
    }


def test_run_question_answerable_correct() -> None:
    def retrieve(_q: str) -> str:
        return "rate limit is 500 rps"

    def completer(system: str, user: str) -> str:
        return "YES" if "Correct?" in user else "500 rps"

    out = run_question(retrieve, completer, _q("1", "cat1", False))
    assert out.is_adversarial is False
    assert out.abstained is False
    assert out.correct is True


def test_run_question_adversarial_abstains() -> None:
    def retrieve(_q: str) -> str:
        return ""  # RE-call abstained: empty context

    def completer(system: str, user: str) -> str:
        return "NO_ANSWER"  # generator abstains

    out = run_question(retrieve, completer, _q("2", "cat5", True))
    assert out.is_adversarial is True
    assert out.abstained is True
    assert out.correct is None  # correctness is undefined for adversarials


def test_run_question_answerable_abstain_is_incorrect_without_judging() -> None:
    def retrieve(_q: str) -> str:
        return "rate limit is 500 rps"

    def completer(system: str, user: str) -> str:
        if "Correct?" in user:
            raise AssertionError("judge must not be called when the answerable question abstained")
        return "NO_ANSWER"

    out = run_question(retrieve, completer, _q("3", "cat1", False))
    assert out.abstained is True
    assert out.correct is False


def test_run_question_applies_and_records_an_exact_evidence_budget() -> None:
    tokenizer = PinnedReaderTokenizer()
    seen: dict[str, str] = {}

    def retrieve(_q: str) -> str:
        return "alpha beta gamma delta epsilon zeta eta theta"

    def completer(system: str, user: str) -> str:
        if "user" not in seen:
            seen["user"] = user
        return "500 rps"

    budget = 12
    out = run_question(
        retrieve,
        completer,
        _q("4", "cat1", False),
        tokenizer=tokenizer,
        evidence_budget=budget,
    )
    evidence_start = seen["user"].index("<memories>")
    evidence_end = seen["user"].index("</memories>") + len("</memories>")
    assert tokenizer.count_tokens(seen["user"][evidence_start:evidence_end]) <= budget
    assert out.evidence_budget == budget


def test_run_question_records_the_selected_routing_mode() -> None:
    out = run_question(
        lambda _question: "context",
        lambda _system, _user: "500 rps",
        _q("5", "cat1", False),
        routing_mode_setting="active",
    )
    assert out.routing_mode == "active"


def test_aggregate_reports_both_columns() -> None:
    outs = [
        Outcome("1", "cat1", False, "c", "a", abstained=False, correct=True),
        Outcome("2", "cat1", False, "c", "a", abstained=False, correct=False),
        Outcome("3", "cat5", True, "", "NO_ANSWER", abstained=True, correct=None),
        Outcome("4", "cat5", True, "c", "wrong", abstained=False, correct=None),
        # an answerable question that abstained: must land in BOTH denominators below. Without
        # this fixture, a buggy aggregate() that dropped abstained-answerable questions from the
        # accuracy denominator (flattering the published accuracy number) would still pass.
        Outcome("5", "cat1", False, "c", "NO_ANSWER", abstained=True, correct=False),
    ]
    agg = aggregate(outs)
    assert agg["answerable_accuracy"]["n"] == 3
    assert agg["answerable_accuracy"]["rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert agg["adversarial_abstention"]["n"] == 2
    assert agg["adversarial_abstention"]["rate"] == 0.5
    assert agg["answerable_false_abstain"]["n"] == 3
    assert agg["answerable_false_abstain"]["rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert set(agg["by_category"]) == {"cat1", "cat5"}


def test_aggregate_empty_rate_blocks_are_json_safe() -> None:
    agg = aggregate([])
    for key in ("answerable_accuracy", "adversarial_abstention", "answerable_false_abstain"):
        block = agg[key]
        assert block["n"] == 0
        assert block["rate"] is None
        assert block["ci95"] == [None, None]
    dumped = json.dumps(agg)
    assert "NaN" not in dumped


def test_aggregate_json_dumps_empty_category_subblock_without_nan() -> None:
    # cat5 is adversarial-only on real LOCOMO, so its answerable_accuracy sub-block is built
    # from an empty list — the exact case Task 7's json.dumps must not choke on.
    outs = [
        Outcome("1", "cat5", True, "", "NO_ANSWER", abstained=True, correct=None),
    ]
    agg = aggregate(outs)
    cat5 = agg["by_category"]["cat5"]
    assert cat5["answerable_accuracy"]["n"] == 0
    assert cat5["answerable_accuracy"]["rate"] is None
    assert cat5["answerable_accuracy"]["ci95"] == [None, None]
    assert "NaN" not in json.dumps(agg)


def test_outcome_rejects_answerable_question_with_none_correct() -> None:
    with pytest.raises(ValueError):
        Outcome("1", "cat1", False, "c", "a", abstained=False, correct=None)


def test_outcome_rejects_adversarial_question_with_bool_correct() -> None:
    with pytest.raises(ValueError):
        Outcome("1", "cat5", True, "c", "a", abstained=False, correct=True)
