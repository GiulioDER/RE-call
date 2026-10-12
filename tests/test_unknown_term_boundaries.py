"""Regressions for PR905's request, authorization and diagnostic boundaries.

Red receipts against e9ce4a5ac on a Linux test host: the supplied-env nodes fail 'the explicit
environment was ignored'; the scoped authorization node fails 'denied vocabulary
changed the authorized response'; trace and metric nodes fail their named gate/count
assertions; serialized-state nodes fail 'the public response lost gate diagnostics'.
The shadow nodes fail their selected_count assertion. The schema node fails because
the new control has no canonical EnvironmentSpec. Controls share these failing nodes.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from recall.calibration import Calibration
from recall.lite import LiteStore
from recall.observability import METRICS
from recall.scope import Scope
from recall.security_policy import AccessContext, SourceRule, SourceSecurityPolicy
from recall.trust_policy import TrustPolicy
from recall.types import Chunk
from recall_mcp import reasoning_diagnostics, service
from recall_mcp.models import EvidenceResult, SearchResult
from recall_mcp.retrieval import evidence_memory, search_memory
from recall_mcp.settings import ENVIRONMENT_SCHEMA
from tests.conftest import dev_search
from tests.test_source_conditioning import _artifact, _item, _trusted_result


class TinyEmbedder:
    """Fixed query vectors isolate trust decisions without model or network work."""
    name = "unknown-term-boundary-offline"
    dim = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _store(tmp_path: Path, metadata: dict[str, str] | None = None) -> LiteStore:
    store = LiteStore(tmp_path / "tiny.db", dim=2)
    store.upsert(
        [Chunk("public-hit", "public/note.md", "The public service rotates the blue piece.",
               metadata or {"file": "public/note.md"})],
        [[1.0, 0.0]],
    )
    return store


@pytest.mark.parametrize("process,explicit,expected", [("1", "0", "disabled"), ("0", "1", "checked")])
def test_explicit_environment_controls_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                          process: str, explicit: str, expected: str) -> None:
    monkeypatch.setenv("RECALL_UNKNOWN_TERM_GATE", process)
    result = dev_search(_store(tmp_path), TinyEmbedder(), "What rotates in QQ01?", k=1,
                        env={"RECALL_UNKNOWN_TERM_GATE": explicit})
    assert result.diagnostics.unknown_term_check == expected, "the explicit environment was ignored"
    assert result.hits[0].verdict == ("ok" if expected == "disabled" else "unknown_term")
    assert result.abstained is (expected == "checked")


def test_denied_vocabulary_is_not_observable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SEC-001 P1. Both authorized-hit controls pass before the intended red assertion.

    Explicit folder scope isolates the new vocabulary side channel from the older legacy
    source-prefix-only filtering problem. The fix must never perform a global vocabulary
    lookup for this constrained request, and unscoped capability remains covered below.
    """
    policy = SourceSecurityPolicy((SourceRule("public", classification="public"),))
    store = _store(tmp_path, {"file": "public/note.md", "security_policy_digest": policy.digest})
    context = AccessContext("reader", store.tenant, clearance="public")
    lookups: list[tuple[str, ...]] = []
    original = store.unknown_terms

    def lookup(tokens: list[str]) -> list[str]:
        lookups.append(tuple(tokens))
        return original(tokens)

    monkeypatch.setattr(store, "unknown_terms", lookup)
    kwargs = dict(k=1, scope=Scope(folder="public"), security_policy=policy, access_context=context, env={})
    before = dev_search(store, TinyEmbedder(), "What rotates in QK4817?", **kwargs)
    assert [h.chunk.id for h in before.hits] == ["public-hit"], "authorized fixture retrieved nothing"
    store.upsert(
        [Chunk("denied-hit", "restricted/hidden.md", "QK4817",
               {"file": "restricted/hidden.md", "security_policy_digest": policy.digest})],
        [[1.0, 0.0]],
    )
    after = dev_search(store, TinyEmbedder(), "What rotates in QK4817?", **kwargs)
    assert [h.chunk.id for h in after.hits] == ["public-hit"], "retrieval authorization itself failed"

    def visible(result):  # noqa: ANN001, ANN202
        return (result.reason, result.abstained, [h.verdict for h in result.hits],
                result.diagnostics.unknown_term_check, result.diagnostics.unknown_terms)

    assert visible(before) == visible(after), "denied vocabulary changed the authorized response"
    assert before.diagnostics.unknown_term_check == "unavailable"
    assert before.diagnostics.unknown_terms == () and lookups == []


def test_candidate_trace_obeys_unknown_term_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_path)
    lookups: list[tuple[str, ...]] = []
    original = store.unknown_terms

    def lookup(tokens: list[str]) -> list[str]:
        lookups.append(tuple(tokens))
        return original(tokens)

    monkeypatch.setattr(store, "unknown_terms", lookup)
    captured = []
    served = dev_search(store, TinyEmbedder(), "What rotates in QQ01?", k=1, env={},
                        candidate_trace_callback=lambda raw, trusted, cal: captured.append((raw, trusted)))
    assert served.abstained and served.hits[0].verdict == "unknown_term"
    assert len(captured) == 1
    raw, trace = captured[0]
    assert trace.abstained and all(h.verdict != "ok" for h in trace.hits), "the candidate trace bypassed the gate"
    assert trace.diagnostics.unknown_terms == served.diagnostics.unknown_terms
    assert {h.chunk.id for h in raw.result.hits} == {h.chunk.id for h in trace.hits}
    assert {h.chunk.id: h.score for h in raw.result.hits} == {h.chunk.id: h.cosine for h in trace.hits}
    assert len(lookups) == 1


def test_served_unknown_term_decision_is_counted(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for query, metadata in [
        ("What rotates in QQ01?", {"file": "public/note.md"}),
        ("What rotates?", {"file": "public/note.md"}),
        ("What rotates in QQ01?", {"file": "public/note.md", "valid_until": "2000-01-01"}),
    ]:
        store.upsert([Chunk("public-hit", "public/note.md", "The public service rotates.", metadata)], [[1.0, 0.0]])
        before = METRICS.snapshot()["counters"]
        result = dev_search(store, TinyEmbedder(), query, k=1, env={})
        after = METRICS.snapshot()["counters"]

        def delta(key: str) -> int:
            return after.get(key, 0) - before.get(key, 0)

        assert delta("recall_searches_total") == 1
        assert delta("recall_abstentions_total") == int(result.abstained), "the served abstention was not counted"
        for verdict in ("unknown_term", "ok", "expired"):
            assert delta(f"recall_verdicts_total{{verdict={verdict}}}") == sum(h.verdict == verdict for h in result.hits)


@pytest.mark.parametrize("state", ["checked", "disabled", "unavailable"])
def test_public_responses_keep_gate_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    store = _store(tmp_path)
    if state == "unavailable":
        monkeypatch.setattr(store, "unknown_terms", None)
    env = {"RECALL_UNKNOWN_TERM_GATE": "0"} if state == "disabled" else {}
    kwargs = dict(k=1, env=env, policy=TrustPolicy.development(),
                  calibration=Calibration(embedder="test-development", threshold=0.5))
    search = search_memory(store, TinyEmbedder(), "What rotates in QQ01?", **kwargs)
    evidence = evidence_memory(store, TinyEmbedder(), "What rotates in QQ01?",
                               register_evidence_cards_fn=lambda *a, **kw: None, **kwargs)
    for response in (search, evidence):
        payload = response.model_dump(mode="json")
        assert payload.get("unknown_term_check") == state, "the public response lost gate diagnostics"
        assert payload["unknown_terms"] == (["QQ01"] if state == "checked" else [])
        legacy = {k: v for k, v in payload.items() if k not in {"unknown_term_check", "unknown_terms"}}
        model = SearchResult if isinstance(response, SearchResult) else EvidenceResult
        restored = model.model_validate(legacy)
        assert restored.unknown_term_check == "not_run" and restored.unknown_terms == []


@pytest.mark.parametrize("policy", ["alpha008", "guarded_spare_slot"])
def test_shadow_cannot_rescue_a_query_rejected_for_unknown_terms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                              policy: str) -> None:
    artifact_path = tmp_path / "source-model.json"
    artifact_path.write_text(_artifact().to_json(), encoding="utf-8")
    monkeypatch.setattr(reasoning_diagnostics, "embedding_profile_id", lambda embedder: "voyage-context-4-v1")
    original = _trusted_result()
    baseline = replace(original, abstained=True, hits=[replace(original.hits[0], verdict="unknown_term")],
                       diagnostics=replace(original.diagnostics, unknown_term_check="checked", unknown_terms=("QQ01",)))
    pool = [_item("candidate", "public/source.md", 0.495, 1, "low_confidence")]
    kwargs = dict(artifact_path=str(artifact_path),
                  leg_audit={"dense": [{"chunk_id": "candidate", "rank": 1}], "sparse": []},
                  pool_audit={"candidate_k": 20, "threshold": 0.5, "items": pool},
                  baseline=baseline, embedder=TinyEmbedder(), profile=service.FAST_PROFILE, policy=policy)
    payload = service._source_conditioning_shadow_payload(**kwargs)
    assert payload["selected_count"] == 0, "the shadow rescued an unknown-term-rejected query"
    assert payload["would_abstain"] is True and payload["added_count"] == 0
    assert pool[0]["verdict"] == "low_confidence" and pool[0]["cosine"] == 0.495
    from recall.source_conditioning import SourceConditioningArtifactError

    with pytest.raises(SourceConditioningArtifactError, match="pipeline_fingerprint"):
        service._source_conditioning_shadow_payload(**{**kwargs, "baseline": replace(baseline, pipeline_fingerprint="wrong")})


def test_gate_control_is_in_runtime_reference() -> None:
    specs = {spec.name: spec for spec in ENVIRONMENT_SCHEMA}
    assert "RECALL_UNKNOWN_TERM_GATE" in specs, "the gate control lacks a canonical environment spec"
    assert specs["RECALL_UNKNOWN_TERM_GATE"].default == "1"
    root = Path(__file__).resolve().parents[1]
    for path in (".env.example", "docs/ENVIRONMENT_GENERATED.md", "docs/ENVIRONMENT.md", "docs/API.md"):
        assert "RECALL_UNKNOWN_TERM_GATE" in (root / path).read_text(encoding="utf-8")
