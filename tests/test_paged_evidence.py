"""Paged evidence depth: 20 chunks for PDF and PPTX results when `k` is unset, 5 otherwise.

Measured basis (recall-lab PI-3, 2026-09-29): on 114 held-out long-PDF questions, 20 chunks beat
the MCP default of 5 by +0.132 [+0.061, +0.202]. The switch `RECALL_PAGED_EVIDENCE` ships `off`.

Three properties matter and each has a test that a plausible mutation turns red (the mutation and
the failing assertion are recorded in each docstring):

* the classifier votes on the first five hits with a strict-majority threshold, and a recorded
  extraction format is never overridden by a file suffix;
* a non-paged pool, cut at the standard depth before the trust gate, is exactly what a plain
  standard-depth retrieval returns, so memos are unchanged when the switch is on;
* the switch reaches the evidence path (the `recall_mcp.service` compatibility wrapper in section
  4; the module path the MCP tool and the Agent SDK call in section 5), an explicit `k` is never
  widened, and `max_items` still only narrows.
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
    """Route the `service.evidence_memory` compatibility wrapper through `_FakeSearch`."""
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


# --------------------------------------------------------------------------------------------
# 5. Audit of #825 (2026-10-01). Each red proof ran on the testbench host against the named
#    mutation, failing at the named assertion, then green with the line restored.
# --------------------------------------------------------------------------------------------


def test_startup_and_request_parse_the_switch_identically() -> None:
    """Invariant: a value the server starts with is never refused per request (one parser).

    Red proofs: `paged_evidence_mode` reverted to `(value or "off").strip().casefold()` refuses a
    blank value; fails at the equality for `" "`. The startup call to `paged_evidence_mode`
    removed from `_validate_runtime_options` admits `"yes"` at startup; fails at the equality
    for `"yes"`.
    """
    from recall_mcp.settings import _validate_runtime_options

    def outcome(check) -> str:  # type: ignore[no-untyped-def]
        try:
            check()
        except ValueError:
            return "refused"
        return "accepted"

    expected = {"": "accepted", " ": "accepted", "\t": "accepted", "ON": "accepted",
                " off ": "accepted", "yes": "refused", "20": "refused"}
    for value, wanted in expected.items():
        startup = outcome(lambda: _validate_runtime_options({"RECALL_PAGED_EVIDENCE": value}))
        request = outcome(lambda: paged_evidence_mode(value))
        assert (startup, request) == (wanted, wanted), value


def test_a_profile_that_does_not_opt_in_keeps_its_depth() -> None:
    """Invariant: hosted-quality (`paged_returned_k` None) retrieves and serves 5 for a PDF pool.

    Red proof: dropping `and profile.paged_returned_k is not None` from the paged branch of
    `_retrieve_trusted` widens it to its returned count; fails at `fake.ks == [5]` with [12].
    """
    env = {"RECALL_RETRIEVAL_PROFILE": "hosted-quality"}
    retrieval, fake = _retrieve(_pool(["p"] * 25), env=env, k=5, paged=True)
    assert fake.ks == [5]
    assert len(retrieval.result.hits) == 5 and retrieval.effective_k == 5
    assert retrieval.paged_depth is None


class _LedgerSearch(_FakeSearch):
    """`_FakeSearch` that witnesses its decision the way `trusted_search` does."""

    def __call__(self, store, embedder, query, *, k, pre_trust_transform=None, ledger=None, **kw):  # type: ignore[no-untyped-def]
        result = super().__call__(
            store, embedder, query, k=k, pre_trust_transform=pre_trust_transform, **kw
        )
        if ledger is not None:
            ledger.record_decision(result, k=k)
        return result


class _AuditStore(_Store):
    def __init__(self) -> None:
        self.events: list[dict] = []

    def append_audit_event(self, event_type, payload, *, generation_id=None, actor=None):  # type: ignore[no-untyped-def]
        self.events.append(payload)
        return "event"


@pytest.mark.parametrize(("kinds", "served"), [(["m"] * 25, 5), (["p"] * 25, 20)])
def test_the_ledger_records_the_depth_served(kinds, served) -> None:
    """Invariant: the decision ledger records what the request returned, not what it retrieved.

    Red proof: never wrapping the ledger (`if ledger is not None and widened and False`) records
    the retrieval depth; fails the memo row at `== served` with 20.
    """
    from recall_mcp.retrieval import _retrieve_trusted

    store = _AuditStore()
    fake = _LedgerSearch(_pool(kinds))
    _retrieve_trusted(
        store, _Embedder(), "q", None, 5, None, None, env={"RECALL_DECISION_LEDGER": "1"},  # type: ignore[arg-type]
        reranker_builder=lambda *_a, **_k: None,
        admission_factory=lambda _profile: contextlib.nullcontext(),
        trusted_search_fn=fake,
        paged_depth=True,
    )
    assert fake.ks == [20]
    assert len(store.events) == 1, "precondition: the decision was witnessed"
    assert store.events[0]["k"] == served


def test_an_unreadable_served_depth_never_fails_the_witness() -> None:
    """Invariant: the ledger is a witness and never raises; if the served depth cannot be read it
    records the retrieved `k` instead.

    Red proof: `_ServedDepthLedger._k` returning `self._served_k()` without its try; the error
    escapes and fails at the outcome equality.
    """
    from recall_mcp.retrieval import _ServedDepthLedger

    def unreadable() -> int:
        raise RuntimeError("depth unavailable")

    store = _AuditStore()
    ledger = _ServedDepthLedger(store, actor="mcp-service", served_k=unreadable)
    result = _FakeSearch(_pool(["m"] * 5))(store, _Embedder(), "q", k=5)
    try:
        outcome: object = ledger.record_decision(result, k=20)
    except Exception as error:  # the property under test is that nothing escapes
        outcome = error
    assert outcome == "event"
    assert store.events[0]["k"] == 20


def test_a_refusal_before_the_decision_records_the_standard_depth() -> None:
    """Invariant: a refusal on a widened request records the standard depth, not the retrieval
    depth (architect gate on the audit: the refusal override was untested).

    Red proof: `_ServedDepthLedger.record_refusal` removed (the base method records the `k` it is
    handed); fails at `== 5` with 20.
    """
    from recall.trust_policy import TrustFailureCode, TrustRefusal
    from recall_mcp.retrieval import _retrieve_trusted

    seen_k: list[int] = []

    def refusing(store, embedder, query, *, k, ledger=None, **_kw):  # type: ignore[no-untyped-def]
        seen_k.append(k)
        refusal = TrustRefusal(
            code=TrustFailureCode.INDEX_NOT_READY, calibration_status="missing",
            tenant_id="t", generation_id="gen-7",
        )
        ledger.record_refusal(refusal, query=query, k=k)
        raise refusal

    store = _AuditStore()
    with pytest.raises(TrustRefusal):
        _retrieve_trusted(
            store, _Embedder(), "q", None, 5, None, None,  # type: ignore[arg-type]
            env={"RECALL_DECISION_LEDGER": "1"},
            reranker_builder=lambda *_a, **_k: None,
            admission_factory=lambda _profile: contextlib.nullcontext(),
            trusted_search_fn=refusing,
            paged_depth=True,
        )
    assert seen_k == [20], "precondition: retrieved wide"
    assert len(store.events) == 1, "precondition: the refusal was witnessed"
    assert store.events[0]["k"] == 5


class _Stop(Exception):
    pass


def _tool_evidence_k(monkeypatch, **arguments):  # type: ignore[no-untyped-def]
    """The `k` the registered `recall_evidence` tool hands `evidence_memory`."""
    import asyncio
    from types import SimpleNamespace

    pytest.importorskip("mcp")
    import recall_mcp.server as server
    from recall.federation import FederationConfig
    from recall_mcp.federation_adapter import FederationExecution
    from recall_mcp.settings import Settings

    seen: dict[str, object] = {}

    def capture(**kwargs):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        raise _Stop

    class _Plan:
        def as_dict(self) -> dict:
            return {}

    monkeypatch.setattr(server, "_retrieval_plan_for", lambda *_a, **_k: _Plan())
    monkeypatch.setattr(
        server,
        "prepare_federated_execution",
        lambda plan, *, current_store, current_embedder, **_k: FederationExecution(
            current_store, current_embedder, None, {}
        ),
    )
    monkeypatch.setattr(server, "evidence_memory", capture)
    tools = {tool.name: tool for tool in server.build_server()._tool_manager.list_tools()}
    context = SimpleNamespace(request_context=SimpleNamespace(lifespan_context={
        "store": SimpleNamespace(tenant="default"), "embedder": object(), "calibration": None,
        "federation_config": FederationConfig(mode="off"), "settings": Settings.from_env({}),
    }))
    with pytest.raises(_Stop):
        asyncio.run(tools["recall_evidence"].fn(ctx=context, query="q", **arguments))
    return seen["k"]


def test_the_mcp_tool_forwards_an_unset_k_as_none(monkeypatch) -> None:
    """Invariant: the registered `recall_evidence` tool hands `evidence_memory` an unset `k` as
    None, or the switch never sees one; an explicit `k` is forwarded as given.

    Red proofs: `k: int = 5` restored in the tool's signature, and `k=5 if k is None else k` in
    its body (architect gate on the audit: a signature check missed the second); each fails at
    `is None` with 5.
    """
    assert _tool_evidence_k(monkeypatch) is None
    assert _tool_evidence_k(monkeypatch, k=7) == 7


def test_the_switch_reaches_the_retrieval_module_path() -> None:
    """The path the MCP server and the Agent SDK call: `recall_mcp.retrieval.evidence_memory`
    over the real `recall_mcp.retrieval._retrieve_trusted`, with only `trusted_search` faked.

    Red proof: `paged = False` in `recall_mcp.retrieval.evidence_memory`; fails at the first
    `== 20` with 5.
    """
    import functools

    from recall_mcp import retrieval

    def run(kinds: list[str], **kwargs):  # type: ignore[no-untyped-def]
        retrieve = functools.partial(
            retrieval._retrieve_trusted,
            reranker_builder=lambda *_a, **_k: None,
            admission_factory=lambda _profile: contextlib.nullcontext(),
            trusted_search_fn=_FakeSearch(_pool(kinds)),
        )
        return retrieval.evidence_memory(
            _Store(), _Embedder(), "q", _retrieve_trusted_fn=retrieve, **kwargs  # type: ignore[arg-type]
        )

    on = {"RECALL_PAGED_EVIDENCE": "on"}
    assert len(run(["p"] * 25, env=on).items) == 20
    assert len(run(["p"] * 25, env=on, k=3).items) == 3
    assert len(run(["m"] * 25, env=on).items) == 5
    assert len(run(["p"] * 25, env={}).items) == 5


def test_every_depth_decision_is_counted() -> None:
    """Invariant: the decision is observable on ordinary traffic, not only under `explain`, so
    the firing rate the default switch waits on can be measured.

    Red proof: the `METRICS.increment` in `select_depth` removed; fails at the paged delta.
    """
    from recall.observability import METRICS

    paged_key = "recall_evidence_depth_total{paged=true,profile=legacy,signal=metadata}"
    memo_key = "recall_evidence_depth_total{paged=false,profile=legacy,signal=none}"
    before = METRICS.snapshot()["counters"]
    _retrieve(_pool(["p"] * 25), env={}, k=5, paged=True)
    _retrieve(_pool(["m"] * 25), env={}, k=5, paged=True)
    after = METRICS.snapshot()["counters"]
    assert after.get(paged_key, 0) - before.get(paged_key, 0) == 1
    assert after.get(memo_key, 0) - before.get(memo_key, 0) == 1


def test_the_paged_depth_is_bounded_by_the_search_maximum(monkeypatch) -> None:
    """Invariant: a profile's paged depth never retrieves past `MAX_SEARCH_K`.

    Red proof: `wide_k = max(standard_k, profile.paged_k)` without the `MAX_SEARCH_K` bound;
    fails at `fake.ks == [50]` with [60].
    """
    from recall_mcp import retrieval

    wide = RetrievalProfile("legacy", 100, 5, False, 100, paged_returned_k=60)
    monkeypatch.setattr(retrieval, "resolve_retrieval_profile", lambda _env: wide)
    _, fake = _retrieve(_pool(["p"] * 70), env={}, k=5, paged=True)
    assert fake.ks == [50]


def test_the_file_metadata_and_every_extracted_slide_suffix_count() -> None:
    """Invariant: the suffix fallback reads `metadata["file"]` as well as the source, covers every
    suffix the extractor records as pptx, and ignores a URI's query or fragment.

    Red proofs: dropping `metadata.get("file")` from `_hit_signal`'s names fails the first
    assertion; `PAGED_SUFFIXES` without `.pptm`, `.potx` and `.potm` fails the second; the suffix
    read without `_path_part` fails the third.
    """

    def chunk(i: int, source: str, file: str | None = None) -> ScoredChunk:
        metadata = {} if file is None else {"file": file}
        return ScoredChunk(Chunk(f"x{i}", source, "t", metadata), 0.5, _AT)

    by_file = [chunk(i, f"blob-{i}", file=f"doc{i}.pdf") for i in range(3)]
    assert decide_depth(by_file, standard_k=5, paged_k=20).paged
    slides = [chunk(0, "a.PPTM"), chunk(1, "b.potx"), chunk(2, "c.potm")]
    assert decide_depth(slides, standard_k=5, paged_k=20).paged
    uris = [chunk(0, "s3://b/x.pdf?versionId=1"), chunk(1, "y.pdf#page=3"), chunk(2, "z.pptx?a=1#b")]
    assert decide_depth(uris, standard_k=5, paged_k=20).paged


def test_a_window_mixing_recorded_and_suffix_hits_reports_the_suffix() -> None:
    """Invariant: `signal` is `metadata` only when every paged hit in the window was recorded.

    Red proof: `all(...)` replaced by `any(...)` in `decide_depth`; fails at the equality with
    "metadata".
    """
    decision = decide_depth(_pool(list("ppsmm")), standard_k=5, paged_k=20)
    assert (decision.paged, decision.signal) == (True, "suffix")


class _FixedStore:
    """Two legs with different orders, so fusion and reranking have something to decide."""

    def query_dense(self, vector, k, source=None, scope=None):  # type: ignore[no-untyped-def]
        return [
            ScoredChunk(Chunk(f"c{i}", f"c{i}.md", f"t{i}", {}), 0.9 - i * 0.01, _AT) for i in range(k)
        ]

    def query_sparse(self, query, k, source=None, vec=None, scope=None):  # type: ignore[no-untyped-def]
        return [ScoredChunk(Chunk(f"c{i}", f"c{i}.md", f"t{i}", {}), 0.5, _AT) for i in range(k)][::-1]

    def newest_indexed_at(self):  # type: ignore[no-untyped-def]
        return _AT


class _PromotingReranker:
    """Promotes the candidates fusion ranked lowest: what a cross-encoder does to a buried hit."""

    def rerank(self, query, hits):  # type: ignore[no-untyped-def]
        return list(reversed(hits))


def test_the_real_retriever_ranks_independently_of_k() -> None:
    """Invariant behind the memo guarantee: `HybridRetriever.search` at k=20, cut to 5, is exactly
    the k=5 search. Every other test here fakes `trusted_search` with a prefix slice, which makes
    this true by construction (audit of #825).

    Red proof: reranking `hits[:k]` instead of the whole fused pool in `HybridRetriever.search`
    (the mistake its own comment warns against); fails at the id equality.
    """
    from recall.retriever import HybridRetriever

    retriever = HybridRetriever(
        _FixedStore(), _Embedder(), _PromotingReranker(), candidate_k=20  # type: ignore[arg-type]
    )
    five = retriever.search("q", k=5)
    twenty = retriever.search("q", k=20)
    assert len(twenty.hits) == 20, "precondition: the wide search returns the wide pool"
    assert [h.chunk.id for h in twenty.hits[:5]] == [h.chunk.id for h in five.hits]
    assert [h.score for h in twenty.hits[:5]] == [h.score for h in five.hits]
