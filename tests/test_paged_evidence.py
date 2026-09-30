"""Paged evidence depth: 20 chunks for PDF and PPTX results when `k` is unset, 5 otherwise.

Measured basis (recall-lab PI-3, 2026-09-29): on 114 held-out long-PDF questions, 20 chunks beat
the MCP default of 5 by +0.132 [+0.061, +0.202]. The switch `RECALL_PAGED_EVIDENCE` ships `off`.

Three properties matter and each has a test that a plausible mutation turns red (the mutation and
the failing assertion are recorded in each docstring):

* the classifier votes on the first five hits with a strict-majority threshold, and a recorded
  extraction format is never overridden by a file suffix;
* a non-paged pool, cut at the standard depth before the trust gate, is exactly what a plain
  standard-depth retrieval returns, so memos are unchanged when the switch is on;
* the switch reaches the real service path, an explicit `k` is never widened, and `max_items`
  still only narrows.
"""
from __future__ import annotations

import contextlib
from datetime import datetime, timedelta, timezone

import pytest

from recall.paged_evidence import (
    decide_depth,
    paged_evidence_enabled,
    paged_evidence_mode,
)
from recall.profiles import (
    HOSTED_QUALITY_PROFILE,
    QUALITY_PROFILE,
    RetrievalProfile,
    resolve_retrieval_profile,
)
from recall.types import (
    Chunk,
    Provenance,
    RetrievalDiagnostics,
    RetrievalResult,
    ScoredChunk,
    StalenessReport,
    TrustedHit,
    TrustedResult,
    Validity,
)

_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)
_STALE = StalenessReport(False, None, None, timedelta(days=2))
_DIAG = RetrievalDiagnostics("bge-small-symmetric-v1", "legacy", "gen-7", 20, False, {})


def _scored(i: int, *, fmt: str | None, file: str) -> ScoredChunk:
    metadata: dict[str, object] = {"file": file}
    if fmt is not None:
        metadata["source_format"] = fmt
    return ScoredChunk(Chunk(f"c{i}", file, f"text {i}", metadata), 0.9 - i * 0.01, _AT)


def _pool(kinds: list[str]) -> list[ScoredChunk]:
    """`p` a PDF chunk with recorded format, `s` a PDF by suffix only, `m` a markdown chunk."""
    made = []
    for i, kind in enumerate(kinds):
        if kind == "p":
            made.append(_scored(i, fmt="pdf", file=f"doc{i}.pdf"))
        elif kind == "s":
            made.append(_scored(i, fmt=None, file=f"doc{i}.pdf"))
        else:
            made.append(_scored(i, fmt="md", file=f"note{i}.md"))
    return made


# --------------------------------------------------------------------------------------------
# 1. The classifier
# --------------------------------------------------------------------------------------------


def test_a_strict_majority_of_the_first_five_decides_paged() -> None:
    """3 of 5 paged widens to the paged depth; 2 of 5 keeps the standard depth.

    Red proof: `PAGED_VOTE_THRESHOLD = 2` in recall/paged_evidence.py makes the 2-of-5 pool widen
    (`assert 20 == 5` on the second assertion). Restored, green.
    """
    three = decide_depth(_pool(list("pppmm") + ["m"] * 15), standard_k=5, paged_k=20)
    two = decide_depth(_pool(list("ppmmm") + ["p"] * 15), standard_k=5, paged_k=20)
    assert (three.paged, three.depth, three.paged_in_window) == (True, 20, 3)
    assert two.depth == 5 and not two.paged


def test_only_the_first_five_hits_vote() -> None:
    """Fifteen PDF hits at ranks 6 to 20 cannot turn a memo answer into a paged one.

    Red proof: reading `hits[:PAGED_VOTE_WINDOW * 4]` instead of `hits[:PAGED_VOTE_WINDOW]` in
    `decide_depth` counts the tail and widens (`assert 20 == 5`). Restored, green.
    """
    decision = decide_depth(_pool(["m"] * 5 + ["p"] * 15), standard_k=5, paged_k=20)
    assert decision.depth == 5
    assert decision.paged_in_window == 0 and decision.signal == "none"


def test_the_suffix_counts_only_when_no_format_was_recorded() -> None:
    """Old rows fall back to the `.pdf` suffix; a recorded non-paged format is never overridden.

    Red proof: dropping the early `return ... "none"` for a recorded non-paged format in
    `_hit_signal` lets the `.pdf` suffix win for the docx-recorded chunks (`assert 20 == 5`).
    Restored, green.
    """
    by_suffix = decide_depth(_pool(list("sssmm")), standard_k=5, paged_k=20)
    assert (by_suffix.depth, by_suffix.signal) == (20, "suffix")

    recorded_docx = [
        ScoredChunk(Chunk(f"d{i}", f"r{i}.pdf", "t", {"file": f"r{i}.pdf", "source_format": "docx"}),
                    0.5, _AT)
        for i in range(5)
    ]
    assert decide_depth(recorded_docx, standard_k=5, paged_k=20).depth == 5


def test_pptx_and_converted_slide_formats_count_as_paged() -> None:
    pptx = [ScoredChunk(Chunk(f"s{i}", f"d{i}.pptx", "t", {"source_format": "pptx"}), 0.5, _AT)
            for i in range(3)]
    odp = [ScoredChunk(Chunk(f"o{i}", f"d{i}.odp", "t", {}), 0.5, _AT) for i in range(3)]
    assert decide_depth(pptx, standard_k=5, paged_k=20).paged
    assert decide_depth(odp, standard_k=5, paged_k=20).paged
    assert not decide_depth([], standard_k=5, paged_k=20).paged


def test_the_switch_parses_strictly_and_defaults_off() -> None:
    assert paged_evidence_mode(None) == "off" and paged_evidence_mode("") == "off"
    assert paged_evidence_mode(" ON ") == "on"
    assert not paged_evidence_enabled({})
    with pytest.raises(ValueError, match="RECALL_PAGED_EVIDENCE"):
        paged_evidence_mode("yes")
    with pytest.raises(ValueError):
        decide_depth([], standard_k=6, paged_k=5)


def test_startup_validation_refuses_a_malformed_switch() -> None:
    from recall_mcp.settings import _validate_runtime_options

    _validate_runtime_options({"RECALL_PAGED_EVIDENCE": "on"})
    with pytest.raises(ValueError, match="RECALL_PAGED_EVIDENCE"):
        _validate_runtime_options({"RECALL_PAGED_EVIDENCE": "20"})


# --------------------------------------------------------------------------------------------
# 2. The profiles
# --------------------------------------------------------------------------------------------


def test_the_local_profiles_page_at_20_and_hosted_keeps_its_own_depth() -> None:
    """The resolver must carry the field: it rebuilds each profile field by field.

    Red proof: removing `paged_returned_k=base.paged_returned_k` from `resolve_retrieval_profile`
    drops the field and the fast profile resolves `paged_k == 5` (`assert 5 == 20`). Restored,
    green.
    """
    for name in ("", "fast", "quality", "code"):
        profile = resolve_retrieval_profile({"RECALL_RETRIEVAL_PROFILE": name})
        assert (profile.returned_k, profile.paged_k) == (5, 20), name
    hosted = resolve_retrieval_profile({"RECALL_RETRIEVAL_PROFILE": "hosted-quality"})
    assert hosted.paged_k == hosted.returned_k == HOSTED_QUALITY_PROFILE.returned_k
    assert resolve_retrieval_profile({"RECALL_RETRIEVAL_PROFILE": "quality"}) == QUALITY_PROFILE
    with pytest.raises(ValueError, match="paged_returned_k"):
        RetrievalProfile("x", 20, 5, False, 100, paged_returned_k=4)


# --------------------------------------------------------------------------------------------
# 3. The retrieval boundary
# --------------------------------------------------------------------------------------------


def _trusted(hit: ScoredChunk) -> TrustedHit:
    return TrustedHit(
        chunk=hit.chunk,
        cosine=hit.score,
        confidence=0.9,
        verdict="ok",
        provenance=Provenance(source=hit.chunk.source, file=hit.chunk.source, ord=0, indexed_at=_AT),
        validity=Validity(valid_from=_AT, valid_until=None, superseded_by=None),
    )


class _FakeSearch:
    """`trusted_search` as the boundary sees it: ranked prefix, pre-trust hook, then verdicts."""

    def __init__(self, pool: list[ScoredChunk]) -> None:
        self.pool = pool
        self.ks: list[int] = []

    def __call__(self, store, embedder, query, *, k, pre_trust_transform=None, **_kw):  # type: ignore[no-untyped-def]
        self.ks.append(k)
        raw = RetrievalResult(query, self.pool[:k], False, _STALE, _DIAG)
        if pre_trust_transform is not None:
            raw = pre_trust_transform(raw)
        return TrustedResult(
            query=query,
            hits=[_trusted(h) for h in raw.hits],
            abstained=not raw.hits,
            reason="",
            gap_warning=False,
            staleness=_STALE,
            diagnostics=_DIAG,
            calibration_id="cal-fixture",
            calibration_status="certified",
        )


class _Store:
    generation_id = "gen-7"


class _Embedder:
    dim = 2
    name = "paged-fixture"

    def embed(self, texts):  # type: ignore[no-untyped-def]
        return [[1.0, 0.0] for _ in texts]


def _retrieve(pool: list[ScoredChunk], *, env: dict[str, str], k: int, paged: bool):
    from recall_mcp.retrieval import _retrieve_trusted

    fake = _FakeSearch(pool)
    retrieval = _retrieve_trusted(
        _Store(), _Embedder(), "q", None, k, None, None, env=env,
        reranker_builder=lambda *_a, **_k: None,
        admission_factory=lambda _profile: contextlib.nullcontext(),
        trusted_search_fn=fake,
        paged_depth=paged,
    )
    return retrieval, fake


def test_a_paged_pool_reaches_the_trust_gate_at_the_paged_depth() -> None:
    """Retrieve 20, keep 20: the decision widens both the pool and the effective k.

    Red proof: making `select_depth` cut to `standard_k` unconditionally returns 5 hits
    (`assert 5 == 20`). Restored, green.
    """
    retrieval, fake = _retrieve(_pool(["p"] * 25), env={}, k=5, paged=True)
    assert fake.ks == [20]
    assert len(retrieval.result.hits) == 20 and retrieval.effective_k == 20
    assert retrieval.paged_depth is not None and retrieval.paged_depth.paged


def test_a_memo_pool_is_exactly_what_a_standard_retrieval_returns() -> None:
    """With the switch on, a non-paged answer is unchanged: same hits, same order, same k.

    Red proof: cutting at `standard_k + 1` in `select_depth` hands the trust gate six hits, and
    the equality with the plain k=5 retrieval fails. Restored, green.
    """
    pool = _pool(["m"] * 25)
    widened, fake = _retrieve(pool, env={}, k=5, paged=True)
    plain, plain_fake = _retrieve(pool, env={}, k=5, paged=False)
    assert fake.ks == [20] and plain_fake.ks == [5]
    assert widened.result.hits == plain.result.hits
    assert widened.effective_k == plain.effective_k == 5
    assert plain.paged_depth is None


def test_an_explicit_k_keeps_the_profile_clamp() -> None:
    """Under the fast profile an explicit k=10 is still clamped to 5; only paged depth widens.

    Red proof: removing the `min(k, profile.returned_k)` clamp returns 10 hits for the explicit
    call (`assert 10 == 5`). Restored, green.
    """
    env = {"RECALL_RETRIEVAL_PROFILE": "fast"}
    explicit, _ = _retrieve(_pool(["p"] * 25), env=env, k=10, paged=False)
    widened, _ = _retrieve(_pool(["p"] * 25), env=env, k=5, paged=True)
    assert len(explicit.result.hits) == 5
    assert len(widened.result.hits) == 20


# --------------------------------------------------------------------------------------------
# 4. The service path
# --------------------------------------------------------------------------------------------


@pytest.fixture()
def service_pool(monkeypatch):
    """Route the real `service.evidence_memory` path through `_FakeSearch`, no database."""
    from recall_mcp import service

    holder: dict[str, _FakeSearch] = {}

    def _search(*args, **kwargs):  # type: ignore[no-untyped-def]
        return holder["fake"](*args, **kwargs)

    monkeypatch.setattr(service, "trusted_search", _search)
    monkeypatch.setattr(service, "_build_reranker", lambda *_a, **_k: None)
    return holder


def test_the_switch_reaches_the_real_evidence_path(service_pool) -> None:
    """On and k unset: 20 items for PDFs; off: 5; explicit k: never widened; max_items narrows.

    Red proof: dropping `paged_depth=paged_depth` from the `recall_mcp.service._retrieve_trusted`
    call (keeping the parameter) returns 5 items with the switch on (`assert 5 == 20`). This is
    the forwarding the first draft of this change missed. Restored, green.
    """
    from recall_mcp.service import evidence_memory

    on = {"RECALL_PAGED_EVIDENCE": "on"}
    service_pool["fake"] = _FakeSearch(_pool(["p"] * 25))
    assert len(evidence_memory(_Store(), _Embedder(), "q", env=on).items) == 20
    assert len(evidence_memory(_Store(), _Embedder(), "q", env={}).items) == 5
    assert len(evidence_memory(_Store(), _Embedder(), "q", k=5, env=on).items) == 5
    assert len(evidence_memory(_Store(), _Embedder(), "q", max_items=2, env=on).items) == 2

    service_pool["fake"] = _FakeSearch(_pool(["m"] * 25))
    assert len(evidence_memory(_Store(), _Embedder(), "q", env=on).items) == 5


def test_the_decision_is_in_the_explanation(service_pool) -> None:
    from recall_mcp.service import evidence_memory

    service_pool["fake"] = _FakeSearch(_pool(["p"] * 25))
    result = evidence_memory(
        _Store(), _Embedder(), "q", explain=True, env={"RECALL_PAGED_EVIDENCE": "on"}
    )
    depth = result.explanation["details"]["evidence_depth"]
    assert depth == {
        "paged": True, "depth": 20, "standard_k": 5, "paged_k": 20,
        "paged_in_window": 5, "window": 5, "signal": "metadata",
    }


def test_a_caller_supplied_retriever_sees_the_old_call_while_the_switch_is_off() -> None:
    """The keyword is passed only when paged depth applies, so existing doubles keep working."""
    from recall_mcp.retrieval import evidence_memory

    calls: list[tuple[tuple, dict]] = []

    def _spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((args, kwargs))
        raise RuntimeError("stop after the call is recorded")

    for env in ({}, {"RECALL_PAGED_EVIDENCE": "on"}):
        with pytest.raises(RuntimeError):
            evidence_memory(_Store(), _Embedder(), "q", env=env, _retrieve_trusted_fn=_spy)
    with pytest.raises(RuntimeError):
        evidence_memory(_Store(), _Embedder(), "q", k=5, env={"RECALL_PAGED_EVIDENCE": "on"},
                        _retrieve_trusted_fn=_spy)
    (off_args, off_kwargs), (on_args, on_kwargs), (explicit_args, explicit_kwargs) = calls
    assert off_kwargs == {} and off_args[4] == 5
    assert on_kwargs == {"paged_depth": True} and on_args[4] == 5
    assert explicit_kwargs == {} and explicit_args[4] == 5
