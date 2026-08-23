"""The BEAM arm must be able to give a reranker a pool bigger than the answerer's context.

`BeamRecallSystem` has accepted `candidate_k` since it was written. `benchmarks/beam/run.py` had
no flag for it, so every BEAM run that could actually be launched had `candidate_k == k`.

That is not a cosmetic gap. With pool == returned == answerer context, a cross-encoder reranks a
set that is shown in its entirety regardless: it can reorder the prompt, but it cannot change
WHICH memories reach the answerer. FINDINGS §11 measured the same mechanism from the other side —
the rerank gain decays with depth (+0.155 at k=1, +0.016 at k=20) precisely because "reordering
can only act where the truncation bites". At candidate_k == k nothing is truncated, so the
expected effect is nil.

Every published BEAM cell was therefore a no-rerank cell by construction, and the writeup never
said so. A flag the adapter accepts and the CLI cannot set fails silently: the run works, it just
measures a configuration nobody chose.
"""
from __future__ import annotations

import pytest

from benchmarks.beam.run import build_parser
from benchmarks.beam.systems import BeamRecallSystem

#: Benchmark-harness coverage, not product coverage; product CI can deselect with
#: `-m 'not benchharness'`.
pytestmark = pytest.mark.benchharness

DSN = "postgresql://user:pass@localhost/db"  # never connected to; construction only


def _system(**kwargs) -> BeamRecallSystem:
    return BeamRecallSystem(DSN, embedder_name="hashing", **kwargs)


def test_cli_exposes_candidate_k() -> None:
    args = build_parser().parse_args(["--data", "x.parquet", "--candidate-k", "600"])
    assert args.candidate_k == 600


def test_cli_defaults_candidate_k_to_none_so_the_adapter_falls_back_to_k() -> None:
    args = build_parser().parse_args(["--data", "x.parquet"])
    assert args.candidate_k is None


def test_default_pool_equals_k_which_is_the_inert_reranker_configuration() -> None:
    described = _system(k=45).describe()
    assert described["k"] == 45
    assert described["candidate_k"] == 45, (
        "pool == returned means a reranker only reorders the answerer's prompt"
    )


def test_a_wider_pool_lets_the_reranker_select_rather_than_reorder() -> None:
    described = _system(k=45, candidate_k=600).describe()
    assert described["k"] == 45
    assert described["candidate_k"] == 600


def test_a_pool_smaller_than_k_is_raised_to_k_not_silently_honoured() -> None:
    """Asking for fewer candidates than results is incoherent; k wins."""
    assert _system(k=45, candidate_k=10).describe()["candidate_k"] == 45


def test_the_configuration_is_recorded_in_the_artifact() -> None:
    """`describe()` is what lands in the results JSON — the only way a later reader can tell
    which configuration produced a cell. The BEAM writeup never stated `reranker: none`; the
    artifact did, which is how it was recoverable at all."""
    described = _system(k=45, candidate_k=600, reranker_name="none").describe()
    assert described["reranker"] == "none"
    assert described["candidate_k"] == 600


def test_the_shipped_local_reranker_is_reachable_without_a_cloud_call() -> None:
    """Until `local` existed the only resolvable reranker was `voyage:<model>`.

    A benchmark for a library that sells "your data never leaves your infrastructure" could not
    measure its own shipped cross-encoder without shipping every query and candidate to a third
    party — so the reranked configuration was unreachable and unmeasured for the whole BEAM arm.
    """
    from benchmarks.systems import resolve_reranker

    pytest.importorskip("sentence_transformers")
    assert resolve_reranker("none") is None
    assert type(resolve_reranker("local")).__name__ == "CrossEncoderReranker"


def test_an_unknown_reranker_names_the_local_route_in_its_error() -> None:
    from benchmarks.systems import resolve_reranker

    with pytest.raises(ValueError, match="local"):
        resolve_reranker("ms-marco")
