"""Environment resolution for the optional local answer provider.

`resolve_answer_provider` reads its whole configuration from the environment, so the parsing
contract is the surface an operator actually touches: empty means unset (the .env template ships
keys valueless), padded values read as their stripped value, a typo in a boolean raises rather
than silently reading False, and a malformed numeric raises an error naming the variable.
"""

from __future__ import annotations

import pytest

from recall.answer_provider import resolve_answer_provider


def _enabled(**extra: str) -> dict[str, str]:
    # The model is supplied here because enabling the provider REQUIRES an explicit model,
    # matching the expansion resolver; there is no default model any more.
    return {
        "RECALL_REASONING_ANSWER_ENABLED": "1",
        "RECALL_REASONING_ANSWER_MODEL": "qwen3:4b",
        **extra,
    }


def test_answer_provider_is_off_by_default() -> None:
    assert resolve_answer_provider({}) is None


def test_enabling_without_a_model_raises_naming_the_variable() -> None:
    """No silent default: the expansion resolver requires its model explicitly, and an answer
    model nobody chose is not a safer thing to fall back to."""
    with pytest.raises(ValueError, match="RECALL_REASONING_ANSWER_MODEL"):
        resolve_answer_provider({"RECALL_REASONING_ANSWER_ENABLED": "1"})


def test_a_whitespace_only_model_is_not_a_model() -> None:
    with pytest.raises(ValueError, match="RECALL_REASONING_ANSWER_MODEL"):
        resolve_answer_provider(
            {
                "RECALL_REASONING_ANSWER_ENABLED": "1",
                "RECALL_REASONING_ANSWER_MODEL": "   ",
            }
        )


def test_a_typo_in_the_thinking_flag_raises_instead_of_reading_false() -> None:
    """One sided leniency reads a typo as False while the sibling ENABLED flag raises; the two
    resolvers must agree, and a silent False on a typo is the worse failure."""
    with pytest.raises(ValueError, match="RECALL_REASONING_ANSWER_THINKING"):
        resolve_answer_provider(_enabled(RECALL_REASONING_ANSWER_THINKING="ture"))


def test_the_thinking_flag_tolerates_surrounding_whitespace() -> None:
    provider = resolve_answer_provider(_enabled(RECALL_REASONING_ANSWER_THINKING=" 1 "))
    assert provider is not None
    assert provider.thinking is True


def test_an_empty_timeout_falls_back_to_the_default() -> None:
    """Empty means unset, because .env templates ship keys valueless (recall/profiles.py)."""
    provider = resolve_answer_provider(_enabled(RECALL_REASONING_ANSWER_TIMEOUT=""))
    assert provider is not None
    assert provider.client.timeout == 60.0


def test_a_malformed_timeout_names_the_variable() -> None:
    with pytest.raises(ValueError, match="RECALL_REASONING_ANSWER_TIMEOUT"):
        resolve_answer_provider(_enabled(RECALL_REASONING_ANSWER_TIMEOUT="soon"))


def test_empty_max_tokens_falls_back_to_the_default() -> None:
    provider = resolve_answer_provider(_enabled(RECALL_REASONING_ANSWER_MAX_TOKENS="   "))
    assert provider is not None
    assert provider.max_tokens == 512


def test_malformed_max_tokens_names_the_variable() -> None:
    with pytest.raises(ValueError, match="RECALL_REASONING_ANSWER_MAX_TOKENS"):
        resolve_answer_provider(_enabled(RECALL_REASONING_ANSWER_MAX_TOKENS="many"))
