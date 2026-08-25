"""A calibration belongs to one embedding profile and must be refused by any other.

The evaluation study behind `recall/calibration.py` is the reason: a cosine threshold does not
transfer across embedders, each model's cosines live in a different regime. Applying one profile's
threshold under another does not error; it silently abstains on real answers, or stops abstaining
at all, and every result afterwards carries a `calibrated=true` flag that is a lie.

These exercise the refusal itself, from both ends: the loader that must return None, and the
enterprise readiness check that must report a failure rather than a warning.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from recall.calibration import Calibration, load_for, save
from recall.embedding_registry import registered_profile
from recall.embeddings import EmbeddingProfile
from recall.readiness import check_enterprise_readiness

DIGEST = "9a443d711e063427f62cf559a38863122ee5ed107fdd7920de882fd66dbc919c"


class _Embedder:
    def __init__(self, profile: EmbeddingProfile) -> None:
        self.profile = profile

    @property
    def dim(self) -> int:
        return self.profile.dimension

    @property
    def name(self) -> str:
        return self.profile.profile_id

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dim for _ in texts]


class _Store:
    """The readiness facts a store reports, with no database behind them.

    A real `PgVectorStore` is the subject of `tests/test_store.py`; here the subject is the
    identity comparison, and binding it to a live table would only mean the branch is skipped
    wherever the DB is not up, which is exactly where these gates have never been shown to fire.
    """

    tenant = "acme"
    generation_id = None

    def __init__(self, dim: int) -> None:
        self._dim = dim

    def readiness_facts(self) -> dict[str, object]:
        return {
            "rls_enabled": True,
            "indexes_valid": True,
            "dimension": self._dim,
            "rows": 10,
            "rows_without_profile": 0,
        }

    def check_rls_effective(self) -> bool:
        return True

    def check_schema(self) -> None:
        """A healthy chunk-table ledger.

        `check_enterprise_readiness` now verifies BOTH schema ledgers, so it calls this. Without
        it the fake raised `AttributeError`, and every test in this file failed on a chunk-table
        schema complaint while claiming to be about profile identity. Returning cleanly is the
        right stand-in: the subject here is the identity comparison, and a fake that failed the
        schema check would mask the branch each test exists to exercise."""


def _identity(profile_id: str = "bge-small-asymmetric-v1") -> EmbeddingProfile:
    return registered_profile(profile_id).identity(artifact_digest=DIGEST)


def test_a_calibration_written_for_another_profile_is_not_loaded(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    save(Calibration(embedder="bge-small-symmetric-v1", threshold=0.42, scale=0.05), path)

    assert load_for("bge-small-symmetric-v1", path) is not None
    assert load_for("bge-small-asymmetric-v1", path) is None


def test_the_rejected_profile_cannot_borrow_the_active_profiles_calibration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calibration.json"
    save(Calibration(embedder="bge-small-asymmetric-v1", threshold=0.42), path)
    assert load_for("qwen3-embedding-0.6b-384-v1", path) is None


def test_a_calibration_file_with_no_embedder_field_is_refused(tmp_path: Path) -> None:
    """Absence is not a match. A file that names no profile must not be adopted by one."""
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"threshold": 0.42, "scale": 0.05}), encoding="utf-8")
    assert load_for("bge-small-asymmetric-v1", path) is None


def test_readiness_fails_when_the_calibration_identity_does_not_match() -> None:
    identity = _identity()
    result = check_enterprise_readiness(
        _Store(identity.dimension),  # type: ignore[arg-type]
        _Embedder(identity),  # type: ignore[arg-type]
        calibration=Calibration(embedder="bge-small-symmetric-v1", threshold=0.42),
    )
    assert not result.ready
    assert "calibration identity does not match the embedding profile" in result.failures


def test_readiness_passes_when_the_calibration_identity_matches() -> None:
    """The other half of the gate. A check only ever observed to fail is compatible with one
    that refuses everything."""
    identity = _identity()
    result = check_enterprise_readiness(
        _Store(identity.dimension),  # type: ignore[arg-type]
        _Embedder(identity),  # type: ignore[arg-type]
        calibration=Calibration(
            embedder="bge-small-asymmetric-v1",
            threshold=0.42,
            separability=1.0,
            n_answerable=40,
            n_unanswerable=40,
        ),
    )
    assert result.ready, result.failures
    assert not result.degraded, result.warnings


def test_readiness_fails_when_the_artifact_is_not_pinned() -> None:
    legacy = EmbeddingProfile(
        "bge-small-symmetric-v1", "BAAI/bge-small-en-v1.5", "legacy-unverified", 384,
        "embed", "embed",
    )
    result = check_enterprise_readiness(
        _Store(384),  # type: ignore[arg-type]
        _Embedder(legacy),  # type: ignore[arg-type]
        calibration=Calibration(embedder="bge-small-symmetric-v1", threshold=0.42),
    )
    assert "model artifact is not pinned by an immutable digest" in result.failures


def test_readiness_fails_when_the_profile_dimension_contradicts_the_runtime() -> None:
    """The registry declares 384. A runtime reporting anything else is a different artifact."""
    identity = _identity()

    class _Widened(_Embedder):
        @property
        def dim(self) -> int:
            return 512

    result = check_enterprise_readiness(
        _Store(512),  # type: ignore[arg-type]
        _Widened(identity),  # type: ignore[arg-type]
        calibration=Calibration(embedder="bge-small-asymmetric-v1", threshold=0.42),
    )
    assert "embedding profile dimension does not match the runtime embedder" in result.failures


@pytest.mark.parametrize(
    "profile_id",
    [
        "bge-small-symmetric-v1",
        "bge-small-asymmetric-v1",
        "bge-small-context-document-v1",
        "bge-small-context-section-v1",
        "bge-small-context-neighbor-v1",
        "bge-large-context-section-v1",
    ],
)
def test_every_active_profile_can_reach_a_ready_verdict(profile_id: str) -> None:
    """Each registered identity must be servable. A profile that can never be ready is a profile
    that only looks deployable."""
    identity = _identity(profile_id)
    result = check_enterprise_readiness(
        _Store(identity.dimension),  # type: ignore[arg-type]
        _Embedder(identity),  # type: ignore[arg-type]
        calibration=Calibration(
            embedder=profile_id, threshold=0.42, separability=1.0,
            n_answerable=40, n_unanswerable=40,
        ),
    )
    assert result.ready, result.failures
