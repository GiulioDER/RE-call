"""A retained record keeps its canonical policy subject during withdrawal."""
from pathlib import Path

import pytest

from recall.lite import LiteStore
from recall.security_policy import AccessContext, SourceRule, SourceSecurityPolicy
from recall.types import Chunk
from recall_mcp.lifecycle import forget_memory, withdraw_memory


def test_allowed_source_cannot_withdraw_a_denied_retained_record(tmp_path: Path) -> None:
    """FIX-003 / SEC-001, public record-only withdrawal.

    Baseline: PR987 head ab1a48ba427599bd174e9d99b5280182f37b2aea.
    Node: test_withdrawal_authorization_cca.py::test_allowed_source_cannot_withdraw_a_denied_retained_record.
    Expected RED: DID NOT RAISE PermissionError, because the allowed absolute spelling bypasses
    the retained record's denied canonical file. The original head authorizes before expansion.
    Original-head DID NOT RAISE and repaired GREEN are recorded with the audit.
    """
    purpose = frozenset({"erasure"})
    policy = SourceSecurityPolicy((
        SourceRule("corpus", purposes=purpose),
        SourceRule("public", purposes=purpose),
        SourceRule("private", purposes=purpose, principals=frozenset({"admin"})),
    ))
    predecessor = Chunk("older", "/corpus/public/a.md", "the earlier claim", {"file": "public/a.md"})
    successor = Chunk("newer", "/corpus/private/b.md", "the later claim", {
        "file": "private/b.md", "supersedes": "public/a.md",
    })
    with LiteStore(tmp_path / "memory.db", dim=8) as store:
        store.upsert([predecessor, successor], [[1.0] + [0.0] * 7] * 2)
        admin = AccessContext("admin", store.tenant, purpose="erasure")
        caller = AccessContext("caller", store.tenant, purpose="erasure")
        forget_memory(store, [successor.source], security_policy=policy, security_context=admin)
        assert store.supersession_graph().successor("public/a.md") == "private/b.md"
        with pytest.raises(PermissionError, match="private/b.md"):
            withdraw_memory(store, [successor.source], security_policy=policy, security_context=caller)
        assert store.supersession_graph().successor("public/a.md") == "private/b.md"
