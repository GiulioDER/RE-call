"""Native link expansion keeps ACL filtering before text hydration and Voyage."""
from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from recall.security_policy import AccessContext, SourceRule, SourceSecurityPolicy
from recall.trust_gate import evaluate
from recall.types import Chunk, RetrievalResult, ScoredChunk, StalenessReport
from recall_mcp import jira_native
from recall_mcp.retrieval import _retrieve_trusted, evidence_memory, search_memory

BASE = "https://jira.example.org"


def test_native_jira_authorizes_before_voyage_and_preserves_primary_evidence(tmp_path, monkeypatch):
    import json

    policy = SourceSecurityPolicy([
        SourceRule("seed"), SourceRule("shared"),
        SourceRule("owner", principals=frozenset({"owner"})),
    ])

    def issue(key, path, text):
        return Chunk(key, f"{BASE}/browse/{key}", text,
                     {"file": path, "security_policy_digest": policy.digest})

    seed = issue("OPS-41", "seed", "OPS-41 Seed title\nTest")
    shared = issue("OPS-42", "shared", "Allowed text")
    owner_only = issue("OPS-43", "owner", "Restricted text")
    now = datetime.now(UTC)
    baseline = evaluate(
        RetrievalResult("test evidence", [ScoredChunk(seed, 1)], False,
                        StalenessReport(False, now, timedelta(0), timedelta(days=2))),
        {}, None, now,
    )
    baseline = replace(baseline, tenant_id="test", generation_id="pinned",
                       calibration_id="certified", calibration_status="certified")
    graph = {"site": BASE, "tenant_id": "test", "generation_id": "pinned",
             "links": {"OPS-41": ["OPS-42", "OPS-43", "OPS-9999"], "OPS-42": ["OPS-44"]}}
    export = tmp_path / "native.json"
    export.write_text(json.dumps(graph), encoding="utf-8")
    configured = {"VOYAGE_API_KEY": "synthetic", "RECALL_JIRA_NATIVE_LINKS_FILE": str(export),
                  "RECALL_JIRA_LINK_FOLLOW": "on"}

    class Store:
        tenant = "test"
        generation_id = "pinned"

        def __init__(self):
            self.hydrated = []

        def iter_scoped_chunks(self, scope, *, sources, include_text):
            assert not include_text and scope.security_policy_digest == policy.digest
            assert set(sources) == {shared.source, owner_only.source, f"{BASE}/browse/OPS-9999"}
            return (replace(chunk, text="") for chunk in (shared, owner_only))

        def chunk_by_id(self, chunk_id):
            self.hydrated.append(chunk_id)
            return {shared.id: shared, owner_only.id: owner_only}[chunk_id]

        def supersession_all(self):
            return {}, frozenset(), {}

    store = Store()
    requests = []

    def rank(query, docs, key):
        requests.append(docs)
        return list(reversed(range(len(docs)))), 10

    monkeypatch.setattr(jira_native, "rank_jira_documents", rank)

    def run(env=configured, principal="reader", query="test evidence", source=None, *, egress=True):
        return _retrieve_trusted(
            store, object(), query, source, 5, None, None,
            security_policy=policy, access_context=AccessContext(principal, "test", egress_allowed=egress),
            env=env, reranker_builder=lambda *a, **kw: None,
            admission_factory=lambda profile: nullcontext(),
            trusted_search_fn=lambda *a, **kw: baseline,
        )

    assert run({}).native_jira_items == ()
    assert run(query="test OPS-9999").native_jira_items == ()
    assert run(source=seed.source).native_jira_items == ()
    assert run(egress=False).native_jira_items == ()
    assert not store.hydrated and not requests
    retrieved = run()
    assert [hit.chunk.id for hit in retrieved.native_jira_items] == [shared.id]
    assert store.hydrated == [shared.id]
    assert requests == [[seed.text, "Allowed text\nLinked from retrieved issue: Seed title"]]

    owner = run(principal="owner")
    assert [hit.chunk.id for hit in owner.native_jira_items] == [owner_only.id, shared.id]
    search = search_memory(store, object(), "test evidence", env={},
                           _retrieve_trusted_fn=lambda *a, **kw: retrieved)
    evidence = evidence_memory(store, object(), "test evidence", env={},
                               _retrieve_trusted_fn=lambda *a, **kw: retrieved,
                               register_evidence_cards_fn=lambda *a, **kw: None)
    assert [hit.chunk_id for hit in search.hits] == [seed.id]
    assert [item.chunk_id for item in evidence.items] == [seed.id]
    assert [item.chunk_id for item in search.related_items] == [shared.id]
    assert [item.chunk_id for item in evidence.related_items] == [shared.id]

    def unavailable(*args):
        raise ValueError("provider failed")

    monkeypatch.setattr(jira_native, "rank_jira_documents", unavailable)
    failed = run()
    assert failed.result is baseline and failed.native_jira_items == ()
    export.write_text(json.dumps({**graph, "generation_id": "retired"}), encoding="utf-8")
    count = len(store.hydrated)
    assert run().native_jira_items == ()
    assert len(store.hydrated) == count
