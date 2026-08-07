import pytest

from benchmarks.mtrag.late_interaction import (
    LATE_ARMS,
    LateArm,
    arm_record,
    holm_family,
)


def _by_name(name: str) -> LateArm:
    return next(a for a in LATE_ARMS if a.name == name)


def test_arms_are_frozen_before_any_score():
    """Arms are declared in code, as SPARSE_ARMS is, so they cannot be edited after seeing a
    number without it showing up in the diff."""
    assert isinstance(LATE_ARMS, tuple)
    assert [a.name for a in LATE_ARMS] == ["li_colbertv2", "li_answerai", "li_jina"]


def test_permissive_arms_are_deployable():
    assert _by_name("li_colbertv2").deployable is True
    assert _by_name("li_answerai").deployable is True


def test_jina_is_not_deployable():
    arm = _by_name("li_jina")
    assert arm.licence == "cc-by-nc-4.0"
    assert arm.deployable is False


def test_holm_family_accepts_deployable_arms():
    assert holm_family([_by_name("li_colbertv2"), _by_name("li_answerai")]) == (
        "li_colbertv2",
        "li_answerai",
    )


def test_holm_family_refuses_a_non_deployable_arm():
    """THE containment gate. The verdict that gates the follow-on project is computed from a
    family li_jina cannot mechanically enter. A refusal, not a docstring."""
    with pytest.raises(ValueError, match="li_jina"):
        holm_family([_by_name("li_colbertv2"), _by_name("li_jina")])


def test_holm_family_refuses_even_a_lone_non_deployable_arm():
    with pytest.raises(ValueError, match="li_jina"):
        holm_family([_by_name("li_jina")])


def test_arm_record_carries_the_taint():
    """Numbers get lifted out of these archives into later documents. A lifted number must arrive
    with its licence attached rather than as a bare float."""
    assert arm_record(_by_name("li_jina")) == {
        "arm": "li_jina",
        "checkpoint": "jinaai/jina-colbert-v2",
        "licence": "cc-by-nc-4.0",
        "deployable": False,
    }


def test_every_arm_checkpoint_is_registered():
    from recall.rerank import LATE_INTERACTION_MODELS

    for arm in LATE_ARMS:
        assert arm.checkpoint in LATE_INTERACTION_MODELS


def test_arm_with_an_unregistered_checkpoint_raises_on_licence():
    """LATE_ARMS is frozen, so this branch is unreachable today. It is tested because the failure
    it prevents is a future arm added without a matching registry entry, which would otherwise
    reach `deployable` and be answered from a licence that does not exist."""
    with pytest.raises(ValueError, match="unregistered checkpoint"):
        LateArm("li_future", "some/unrecorded").licence


def test_holm_family_raises_even_when_handed_a_single_use_iterator():
    """The gate must not degrade to silent omission on a type violation. Iterating the argument
    twice without materialising it would return an empty tuple here instead of raising."""
    arms = iter([_by_name("li_colbertv2"), _by_name("li_jina")])
    with pytest.raises(ValueError, match="li_jina"):
        holm_family(arms)
