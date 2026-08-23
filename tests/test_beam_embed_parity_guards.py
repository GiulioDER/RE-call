"""The parity gate must fire in both directions. Every test here pins a defect the audit found.

These are red→green proofs, written against the defects rather than around them. Each one is run
against the PRE-FIX code first; a test that was never red proves nothing, which is exactly how the
original suite for this module passed while the module's two guards were both vacuous.

Findings pinned:
  NUM-001 / BUG-004  the k clamp makes set_disagreement identically 0.0 once k reaches the document
                     count, so pure noise certifies as identical on any sample of <= n_queries + k
  STAKES-001 / CODE-005 / DOC-008 / BUG-010
                     the provider check is a DIFFERENCE test standing in for a VERIFICATION test:
                     an unknown provider is "different", so it passes
  NUM-003            an exact tie at the k boundary under an unstable sort rejects a build that is
                     identical to ~15 significant digits
  NUM-002 / NUM-004  k and n_queries are unvalidated: k=0 passes vacuously, n_queries=0 divides by zero
  BUG-012 / DAT-013  nothing binds the two emissions to the same input texts
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.beam import embed_parity as ep

#: Benchmark-harness coverage, not product coverage; product CI can deselect with
#: `-m 'not benchharness'`.
pytestmark = pytest.mark.benchharness

DIM = 16


def _emit(path: Path, vectors: np.ndarray, providers: list[str], **extra) -> Path:
    meta = {"model": "BAAI/bge-small-en-v1.5", "dim": int(vectors.shape[1]),
            "n": int(vectors.shape[0]), "providers": providers, **extra}
    np.savez_compressed(path, vectors=vectors, meta=json.dumps(meta))
    return path.with_suffix(".npz") if path.suffix != ".npz" else path


@pytest.fixture()
def tmp(tmp_path: Path) -> Path:
    return tmp_path


def test_unrelated_vectors_never_certify_as_identical(tmp: Path) -> None:
    """NUM-001: the whole point. Independent draws must not pass the ranking gate at ANY size."""
    rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(2)
    # Sizes where the corpus genuinely supports a top-45: anything smaller is now REFUSED
    # (see test_k_at_or_above_document_count_is_refused), which is the other half of this defect.
    for n_rows in (110, 150, 200):
        ref, cand = rng_a.random((n_rows, DIM)), rng_b.random((n_rows, DIM))
        r = ep.rank_disagreement(ref, cand, 32, 45)
        assert r["set_disagreement"] > 0.5, (
            f"n_rows={n_rows}: independent vector sets reported "
            f"set_disagreement={r['set_disagreement']}, k_effective={r['k']}"
        )


def test_k_at_or_above_document_count_is_refused(tmp: Path) -> None:
    """NUM-001: the degenerate regime must be an error, not a silent clamp."""
    rng = np.random.default_rng(0)
    with pytest.raises(SystemExit, match="(?i)document|corpus|k="):
        ep.rank_disagreement(rng.random((100, DIM)), rng.random((100, DIM)), 64, 45)


def test_non_positive_k_and_queries_are_refused(tmp: Path) -> None:
    """NUM-002 / NUM-004: k=0 passed vacuously; n_queries=0 raised ZeroDivisionError."""
    rng = np.random.default_rng(0)
    a, b = rng.random((200, DIM)), rng.random((200, DIM))
    with pytest.raises(SystemExit, match="(?i)k"):
        ep.rank_disagreement(a, b, 64, 0)
    with pytest.raises(SystemExit, match="(?i)quer"):
        ep.rank_disagreement(a, b, 0, 45)


def test_an_unknown_provider_is_refused_not_treated_as_different(tmp: Path) -> None:
    """STAKES-001: 'could not observe' must not read as 'observed a different runtime'."""
    v = np.random.default_rng(0).random((200, DIM))
    ref = _emit(tmp / "ref", v, ["CPUExecutionProvider"])
    cand = _emit(tmp / "cand", v, ["<session not reachable>"])
    with pytest.raises(SystemExit, match="(?i)did not record|unknown|not reachable"):
        ep.compare(ref, cand, min_cosine=0.9999, max_rank_disagreement=0.0,
                   n_queries=32, k=45, allow_same_provider=False)


def test_an_emission_predating_the_provenance_field_is_refused(tmp: Path) -> None:
    """The first fix refused known-BAD values, so an artifact with no `providers_source` passed.

    That is the same shape as the defect being repaired: refusing a marker is a negative check,
    and anything the producer never emits — or predates — slips through it. The guard now
    requires a positively recorded session.
    """
    v = np.random.default_rng(0).random((200, DIM))
    ref = _emit(tmp / "ref", v, ["CPUExecutionProvider"])            # no providers_source at all
    cand = _emit(tmp / "cand", v, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    with pytest.raises(SystemExit, match="(?i)session-recorded|unknown, not different"):
        ep.compare(ref, cand, min_cosine=0.9999, max_rank_disagreement=0.0,
                   n_queries=32, k=45, allow_same_provider=False)


def test_availability_masquerading_as_a_session_is_refused(tmp: Path) -> None:
    """STAKES-001: availability lists CUDA on a GPU box even after a silent CPU fallback."""
    v = np.random.default_rng(0).random((200, DIM))
    avail = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ref = _emit(tmp / "ref", v, ["CPUExecutionProvider"], available_providers=["CPUExecutionProvider"])
    cand = _emit(tmp / "cand", v, avail, available_providers=avail,
                 providers_requested=["CUDAExecutionProvider"], providers_source="availability")
    with pytest.raises(SystemExit, match="(?i)availability|did not record|session"):
        ep.compare(ref, cand, min_cosine=0.9999, max_rank_disagreement=0.0,
                   n_queries=32, k=45, allow_same_provider=False)


def test_a_genuinely_different_provider_still_passes(tmp: Path) -> None:
    """The guard must be able to PASS, or it trains you to override it."""
    v = np.random.default_rng(0).random((200, DIM))
    ref = _emit(tmp / "ref", v, ["CPUExecutionProvider"], providers_source="session")
    cand = _emit(tmp / "cand", v, ["CUDAExecutionProvider", "CPUExecutionProvider"],
                 providers_source="session")
    r = ep.compare(ref, cand, min_cosine=0.9999, max_rank_disagreement=0.0,
                   n_queries=32, k=45, allow_same_provider=False)
    assert r["verdict"] == "PARITY OK"


def test_an_exact_tie_does_not_reject_an_identical_build(tmp: Path) -> None:
    """NUM-003: unstable argsort + zero tolerance rejected builds identical to ~15 digits."""
    rejected = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        ref = rng.random((200, DIM))
        ref[115] = ref[114]                                   # exact tie
        cand = ref + np.random.default_rng(seed).normal(0, 1e-12, ref.shape)
        if ep.rank_disagreement(ref, cand, 64, 45)["set_disagreement"] > 0:
            rejected += 1
    assert rejected == 0, f"{rejected}/40 numerically identical builds rejected by tie instability"


def test_emissions_over_different_texts_are_refused(tmp: Path) -> None:
    """BUG-012 / DAT-013: nothing bound the two sides to the same sample."""
    rng = np.random.default_rng(0)
    ref = _emit(tmp / "ref", rng.random((200, DIM)), ["CPUExecutionProvider"],
                providers_source="session", texts_sha256="a" * 64)
    cand = _emit(tmp / "cand", rng.random((200, DIM)), ["CUDAExecutionProvider"],
                 providers_source="session", texts_sha256="b" * 64)
    with pytest.raises(SystemExit, match="(?i)text"):
        ep.compare(ref, cand, min_cosine=0.0, max_rank_disagreement=1.0,
                   n_queries=32, k=45, allow_same_provider=False)


def test_shape_mismatch_reports_itself_rather_than_a_numpy_error(tmp: Path) -> None:
    """BUG-012: arithmetic ran before the shape check, so numpy raised first."""
    rng = np.random.default_rng(0)
    ref = _emit(tmp / "ref", rng.random((200, DIM)), ["CPUExecutionProvider"],
                providers_source="session")
    cand = _emit(tmp / "cand", rng.random((200, DIM * 2)), ["CUDAExecutionProvider"],
                 providers_source="session")
    with pytest.raises(SystemExit, match="(?i)shape|dim"):
        ep.compare(ref, cand, min_cosine=0.9999, max_rank_disagreement=0.0,
                   n_queries=32, k=45, allow_same_provider=False)
