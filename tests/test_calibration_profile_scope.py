"""Profile-scoped calibration: an EXTENSION of `load_for`, never a relaxation of it."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from recall.calibration import (
    PROFILE_FINGERPRINT_KEY,
    Calibration,
    load_for,
    load_for_profile,
    save,
    save_for_profile,
)
from recall.embeddings import EmbeddingProfile

PROFILE = EmbeddingProfile(
    profile_id="bge-small-symmetric-v1",
    model_name="BAAI/bge-small-en-v1.5",
    artifact_digest="a" * 64,
    dimension=384,
    query_mode="embed",
    passage_mode="embed",
)


def _calibration(embedder: str = PROFILE.profile_id) -> Calibration:
    return Calibration(
        embedder=embedder, threshold=0.42, scale=0.05, separability=0.99,
        n_answerable=30, n_unanswerable=30,
    )


def test_a_calibration_saved_for_a_profile_loads_back_for_that_profile(tmp_path) -> None:
    path = tmp_path / "calibration.json"
    save_for_profile(_calibration(), PROFILE, path)
    loaded = load_for_profile(PROFILE, path)
    assert loaded is not None and loaded.threshold == 0.42


def test_the_saved_file_records_the_complete_identity_not_only_the_profile_id(tmp_path) -> None:
    path = tmp_path / "calibration.json"
    save_for_profile(_calibration(), PROFILE, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload[PROFILE_FINGERPRINT_KEY] == PROFILE.fingerprint()
    assert payload["embedder"] == PROFILE.profile_id


@pytest.mark.parametrize(
    "field,value",
    [
        ("artifact_digest", "b" * 64),
        ("dimension", 768),
        ("query_mode", "query_embed"),
        ("passage_mode", "passage_embed"),
        ("normalization", "none"),
        ("instruction_version", "v2"),
        ("chunker_version", "chunk-text-v2"),
        ("context_version", "section-v1"),
    ],
)
def test_cross_profile_reuse_fails_closed_on_every_identity_field(tmp_path, field, value) -> None:
    """The profile IDs match. Everything else that moves the cosine regime does not, and each one
    is already key material in `EmbeddingProfile.fingerprint()`."""
    path = tmp_path / "calibration.json"
    save_for_profile(_calibration(), PROFILE, path)
    other = replace(PROFILE, **{field: value})
    assert other.profile_id == PROFILE.profile_id
    assert load_for_profile(other, path) is None
    # `load_for`, which filters on the profile ID alone, still returns it — which is exactly the
    # gap this function closes, and the reason it had to be a new function rather than an edit.
    assert load_for(other.profile_id, path) is not None


def test_a_file_with_no_fingerprint_fails_closed(tmp_path, caplog) -> None:
    """It cannot demonstrate which identity it belongs to, and serving it because it does not
    contradict us is cross-profile reuse with the evidence missing rather than absent.

    The log assertion is not decoration. Deleting the absent-fingerprint branch leaves the
    identity-equality check below it, which also returns None — so a test asserting only `is None`
    passes with the branch removed and cannot tell the two apart (measured: that mutation was the
    one survivor of this session's mutation sweep). The branch's actual contribution is the
    DIAGNOSIS: "this file records no fingerprint" is an actionable instruction to re-run
    `save_for_profile`, while "it was fitted under identity None" describes nothing that exists.
    """
    path = tmp_path / "calibration.json"
    save(_calibration(), path)
    assert load_for(PROFILE.profile_id, path) is not None
    with caplog.at_level("WARNING"):
        assert load_for_profile(PROFILE, path) is None
    assert any(
        "records no profile_fingerprint" in record.getMessage() for record in caplog.records
    ), [record.getMessage() for record in caplog.records]


def test_an_absent_file_is_still_just_uncalibrated(tmp_path) -> None:
    assert load_for_profile(PROFILE, tmp_path / "nothing.json") is None


def test_every_check_load_for_already_made_still_runs(tmp_path) -> None:
    """Extension, not replacement: an out-of-range threshold must not become loadable because it
    carries a matching fingerprint."""
    path = tmp_path / "calibration.json"
    save_for_profile(_calibration(), PROFILE, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["scale"] = 0.0  # `load_for` refuses a non-positive scale
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_for(PROFILE.profile_id, path) is None
    assert load_for_profile(PROFILE, path) is None


def test_saving_a_calibration_for_the_wrong_profile_is_refused(tmp_path) -> None:
    """The file would carry two disagreeing claims about what it applies to."""
    with pytest.raises(ValueError, match="the file would claim two different owners"):
        save_for_profile(_calibration("some-other-model"), PROFILE, tmp_path / "c.json")


def test_the_arm_config_keys_its_artifacts_on_the_full_fingerprint() -> None:
    """Two arms sharing a profile ID and differing in artifact digest must not share a ledger."""
    from recall.eval.promotion.run import ArmConfig

    def _arm(profile: EmbeddingProfile) -> ArmConfig:
        return ArmConfig(
            label="baseline",
            embedding_profile_id=profile.profile_id,
            retrieval_profile="fast",
            generation="g1",
            candidate_pool=20,
            embedding_fingerprint=profile.fingerprint(),
        )

    other = replace(PROFILE, artifact_digest="b" * 64)
    assert _arm(PROFILE).artifact_stem() != _arm(other).artifact_stem()
