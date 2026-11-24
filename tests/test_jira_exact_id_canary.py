"""Single-connector Jira canary on a real, certified local index."""
from contextlib import closing
from pathlib import Path

from pytest import MonkeyPatch

from recall.lite.calibration import CalibrationStatus, auto_calibrate
from recall.security_policy import AccessContext, SourceRule, SourceSecurityPolicy
from recall.types import Chunk
from recall_mcp import retrieval
from tests.test_lite_calibration import ContentWordEmbedder, _store

BASE = "https://jira.example.org"
TOPIC = "rotor pressure troubleshooting guide"


def test_exact_jira_connector_permissions_and_fallback(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Red proof: base without routing never attempts the authorized canonical Jira source."""
    allowed = f"{BASE}/browse/OPS-41"
    hidden = f"{BASE}/browse/OPS-42"
    expired = f"{BASE}/browse/OPS-43"
    missing = f"{BASE}/browse/OPS-44"
    guide = "runbooks/rotor.md"
    policy = SourceSecurityPolicy((
        SourceRule(allowed, classification="public"),
        SourceRule(hidden, principals=frozenset({"owner"}), classification="public"),
        SourceRule(expired, classification="public"),
        SourceRule(guide, classification="public"),
    ))
    embedder = ContentWordEmbedder()
    env = {"RECALL_JIRA_EXACT_ID_BASE_URL": BASE, "RECALL_JIRA_LINK_FOLLOW": "off"}
    sources = [
        Chunk("allowed", allowed, f"{TOPIC} OPS-41", {}),
        Chunk("hidden", hidden, f"{TOPIC} OPS-42 restricted-marker", {}),
        Chunk("expired", expired, f"{TOPIC} OPS-43", {"valid_until": "2000-01-01"}),
        Chunk("guide", guide, TOPIC, {}),
    ]
    for chunk in sources:
        chunk.metadata.update(file=chunk.source, security_policy_digest=policy.digest)
    scopes = []
    original = retrieval.trusted_search

    def observed(*args, **kwargs):
        scopes.append(kwargs.get("source"))
        return original(*args, **kwargs)

    monkeypatch.setattr(retrieval, "trusted_search", observed)
    with closing(_store(tmp_path, 48)) as store:
        store.upsert(sources, embedder.embed([chunk.text for chunk in sources]))
        fit = auto_calibrate(store, embedder)
        assert fit.status is CalibrationStatus.CERTIFIED, fit.reason

        for tool, field in ((retrieval.search_memory, "hits"), (retrieval.evidence_memory, "items")):
            def run(key, principal="reader", *, route=True, source=None):
                scopes.clear()
                response = tool(
                    store, embedder, f"{TOPIC} {key}", source=source, k=5, env=env if route else {},
                    security_policy=policy, access_context=AccessContext(principal, store.tenant, clearance="public"),
                )
                return tuple(scopes), [item.chunk_id for item in getattr(response, field)], response

            attempted, hits, result = run("OPS-41")
            assert attempted == (allowed,), "authorized query was not routed to its exact source"
            assert hits == ["allowed"] and result.trust_state == "trusted"

            attempted, hits, _ = run("OPS-42", principal="owner")
            assert attempted == (hidden,) and hits == ["hidden"]

            for key, target in (("OPS-42", hidden), ("OPS-43", expired), ("OPS-44", missing)):
                _, baseline, _ = run(key, route=False)
                attempted, hits, response = run(key)
                assert attempted == (target, None), "exact failure did not fall back in the same request"
                assert hits == baseline and "guide" in hits
                assert "hidden" not in hits and "restricted-marker" not in response.model_dump_json()

            attempted, hits, _ = run("OPS-41", source=guide)
            assert attempted == (guide,) and hits == ["guide"]
