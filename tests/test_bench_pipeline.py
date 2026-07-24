from benchmarks.pipeline import (
    GEN_SYSTEM_PROMPT,
    NO_ANSWER,
    generate_answer,
    is_abstention,
    judge_correct,
)


def test_is_abstention_exact_token() -> None:
    assert is_abstention(NO_ANSWER) is True


def test_is_abstention_is_whitespace_and_case_tolerant() -> None:
    assert is_abstention("  no_answer\n") is True
    assert is_abstention("No_Answer") is True


def test_is_abstention_false_for_real_answer() -> None:
    assert is_abstention("The limit is 500 rps.") is False
    # a real answer that merely mentions the token is not an abstention
    assert is_abstention("There is no answer key labelled NO_ANSWER here, but it is 500.") is False


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
        seen["user"] = user
        return "500 rps"

    generate_answer(completer, context="rate limit is 500 rps", question="how many rps?")
    user = seen["user"]
    start = user.index("<memories>")
    end = user.index("</memories>")
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
