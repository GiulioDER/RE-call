from __future__ import annotations

import pytest

from recall.scope import Scope
from recall.security_policy import (
    AccessContext,
    SourceRule,
    SourceSecurityPolicy,
    access_context_from_environment,
)
from recall.embeddings import HashingEmbedder
from recall.index import Indexer
from recall.extraction import ExtractedBlock
from recall.generations import _secure_generation_text


class _Store:
    def __init__(self) -> None:
        self.known: dict[str, str] = {}
        self.chunks = []

    def source_content_hashes(self):
        return dict(self.known)

    def replace_sources(self, sources, chunks, embeddings):
        self.chunks.extend(chunks)
        return len(chunks)

    def analyze_if_stale(self, modified):
        return True

    def delete_sources(self, sources):
        return 0


def _policy() -> SourceSecurityPolicy:
    return SourceSecurityPolicy(
        (
            SourceRule(
                "finance",
                classification="confidential",
                principals=frozenset({"alice"}),
                redactions=("email", "secret"),
            ),
            SourceRule("public", classification="public"),
        )
    )


def test_policy_denies_unmatched_sources_and_wrong_principals() -> None:
    policy = _policy()
    context = AccessContext("bob", "tenant", clearance="restricted")
    assert not policy.decide("finance/budget.md", context).allowed
    assert not policy.decide("private/plan.md", context).allowed


def test_policy_rejects_duplicate_normalized_prefixes() -> None:
    with pytest.raises(ValueError, match="duplicate prefixes"):
        SourceSecurityPolicy((SourceRule("docs"), SourceRule("/docs/")))


def test_policy_digest_is_independent_of_rule_and_redaction_order() -> None:
    first = SourceRule(
        "docs",
        redactions=("phone", "email"),
        principals=frozenset({"alice", "bob"}),
        purposes=frozenset({"retrieval", "indexing"}),
    )
    second = SourceRule(
        "private",
        redactions=("secret",),
        principals=frozenset({"admin"}),
        purposes=frozenset({"erasure"}),
    )
    assert SourceSecurityPolicy((first, second)).digest == SourceSecurityPolicy(
        (
            SourceRule(
                "private",
                redactions=("secret",),
                principals=frozenset({"admin"}),
                purposes=frozenset({"erasure"}),
            ),
            SourceRule(
                "docs",
                redactions=("email", "phone"),
                principals=frozenset({"bob", "alice"}),
                purposes=frozenset({"indexing", "retrieval"}),
            ),
        )
    ).digest


def test_security_environment_rejects_malformed_egress_boolean() -> None:
    with pytest.raises(ValueError, match="RECALL_EGRESS_ALLOWED"):
        access_context_from_environment("tenant", {"RECALL_EGRESS_ALLOWED": "enabled"})


def test_redaction_happens_as_a_policy_operation() -> None:
    policy = _policy()
    context = AccessContext("alice", "tenant", clearance="confidential")
    text, decision = policy.redact(
        "finance/budget.md", "email alice@example.com api_key=secret-value", context
    )
    assert "alice@example.com" not in text
    assert "secret-value" not in text
    assert decision.policy_digest == policy.digest


def test_authorized_scope_is_a_hard_sql_scope() -> None:
    policy = _policy()
    context = AccessContext("alice", "tenant", clearance="confidential")
    scope = policy.constrain_scope(Scope(folder="finance"), context)
    sql, params = scope.predicate("c")
    assert "scope_source_prefix_0" in sql
    assert "finance" in params["scope_source_prefix_0"]
    assert "AND" in sql
    assert params["scope_security_policy_digest"] == policy.digest
    assert " ~ %(scope_source_prefix_0)s" in sql
    assert sql.count("COALESCE(NULLIF") == len(scope.source_prefixes or ())


def test_redaction_refuses_unknown_rules_and_denied_access() -> None:
    policy = SourceSecurityPolicy((SourceRule("private", redactions=("unknown",)),))
    context = AccessContext("alice", "tenant")
    with pytest.raises(ValueError, match="unknown redaction"):
        policy.redact("private/a.md", "x", context)
    with pytest.raises(PermissionError):
        policy.redact("other/a.md", "x", context)


def test_indexer_redacts_before_embedding_and_persists_policy_identity(tmp_path) -> None:
    root = tmp_path / "corpus"
    (root / "finance").mkdir(parents=True)
    source = root / "finance" / "budget.md"
    source.write_text("contact alice@example.com api_key=secret-value", encoding="utf-8")
    policy = _policy()
    store = _Store()
    Indexer(
        store,
        HashingEmbedder(dim=64),
        security_policy=policy,
        security_context=AccessContext("alice", "tenant", clearance="confidential"),
    ).index_path(root)
    assert store.chunks
    assert "alice@example.com" not in store.chunks[0].text
    assert "secret-value" not in store.chunks[0].text
    assert store.chunks[0].metadata["security_policy_digest"] == policy.digest


def test_generation_redacts_raw_and_extracted_views_before_chunking() -> None:
    policy = _policy()
    context = AccessContext("alice", "tenant", clearance="confidential")
    text, blocks = _secure_generation_text(
        "finance/budget.csv",
        "contact alice@example.com api_key=secret-value",
        (ExtractedBlock("owner alice@example.com", "table", {}),),
        policy,
        context,
    )
    assert "alice@example.com" not in text
    assert "secret-value" not in text
    assert "alice@example.com" not in blocks[0].text


def test_generation_security_refuses_a_denied_source_before_reading() -> None:
    policy = _policy()
    context = AccessContext("bob", "tenant", clearance="restricted")
    with pytest.raises(PermissionError):
        _secure_generation_text("finance/budget.csv", "secret", (), policy, context)
