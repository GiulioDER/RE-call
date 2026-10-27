"""Public promotion regressions for PR987 BUG-002, with real isolated PostgreSQL state.

Only the audit INSERT is intercepted. Builds, validation, publication, independent reader
connections and transaction rollback remain real. Original-head RED and repaired GREEN are recorded with the audit. Execution used an owned
test database; no live database is an allowed target.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import psycopg
import pytest

from recall.generation_store import GenerationStore
from recall.generations import GenerationManager
from recall.lineage import GenerationState, IndexManifestV1
from recall.manifest import S3ObjectReader
from tests.conftest import TEST_DSN
from tests.test_absent_memo_facts import _corpus
from tests.test_generations import _Embedder, _pipeline, manager as manager


_A = b"# A\n\nThe earlier decision.\n"
_B = b"---\nsupersedes: a.md\n---\n# B\n\nThe replacement decision.\n"
_RECORD_TYPE = "supersession_forgotten"


def _ready(manager: GenerationManager, memos: dict[str, bytes], version: str) -> str:
    manifest, reader = _corpus(manager.tenant_id, memos, version)
    generation = manager.create(cast(IndexManifestV1, manifest), _pipeline("model-a"))
    manager.build(
        generation.generation_id, cast(S3ObjectReader, reader), _Embedder(1), lambda text: [text]
    )
    manager.validate(generation.generation_id)
    return generation.generation_id


def _pair(manager: GenerationManager) -> tuple[str, str]:
    outgoing = _ready(manager, {"a.md": _A, "b.md": _B}, "outgoing")
    manager.promote(outgoing, unsafe_development=True)
    incoming = _ready(manager, {"a.md": _A}, "incoming")
    with GenerationStore(TEST_DSN, 64, tenant=manager.tenant_id) as reader:
        assert reader.supersession_graph().successor("a.md") == "b.md", "invalid initial chain"
    assert manager.active_generation_id() == outgoing
    assert manager.get(incoming).state is GenerationState.READY
    return outgoing, incoming


def _omitted_append(query: object, params: Any, tenant: str) -> bool:
    """Recognize the real retained-record write without depending on parameter positions."""
    return (
        isinstance(query, str)
        and query.lstrip().upper().startswith("INSERT INTO RECALL_AUDIT_EVENTS")
        and isinstance(params, (list, tuple))
        and _RECORD_TYPE in params
        and tenant in params
    )


class _AppendRefused(RuntimeError):
    pass


def test_omitted_record_append_failure_preserves_previous_active_generation(
    manager: GenerationManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused ghost append must roll back publication and preserve the prior active state.

    Node: tests/test_generation_retirement_cca.py::
    test_omitted_record_append_failure_preserves_previous_active_generation.
    Baseline: ab1a48ba427599bd174e9d99b5280182f37b2aea (PR987).
    Target: GenerationManager.promote and its omitted-record audit INSERT. Expected baseline
    RED: "a failed retirement append published the incoming generation" after the old helper
    swallows the injected failure. Original-head RED and repaired GREEN were observed; see the audit receipts.
    """
    outgoing, incoming = _pair(manager)
    real_execute = psycopg.Connection.execute
    refused: list[str] = []

    def execute(conn: Any, query: Any, params: Any = None, *args: Any, **kwargs: Any) -> Any:
        if _omitted_append(query, params, manager.tenant_id):
            refused.append(incoming)
            raise _AppendRefused("injected omitted-record audit INSERT refusal")
        return real_execute(conn, query, params, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(psycopg.Connection, "execute", execute)
        try:
            manager.promote(incoming, unsafe_development=True)
        except _AppendRefused:
            # An atomic implementation may propagate the refusal; durable state decides safety.
            pass

    assert refused == [incoming], "the test did not reach the omitted-record audit INSERT"
    assert manager.active_generation_id() == outgoing, (
        "a failed retirement append published the incoming generation"
    )
    assert manager.get(outgoing).state is GenerationState.ACTIVE
    assert manager.get(incoming).state is GenerationState.READY
    assert manager.get(incoming).activated_at is None
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (manager.tenant_id,))
        committed = conn.execute(
            "SELECT count(*) FROM recall_audit_events WHERE tenant_id = %s "
            "AND generation_id = %s AND event_type = 'generation_promoted_unsafe_development'",
            (manager.tenant_id, incoming),
        ).fetchone()
    assert committed == (0,), "the failed promotion left a committed publication event"
    with GenerationStore(TEST_DSN, 64, tenant=manager.tenant_id) as reader:
        assert reader.supersession_graph().successor("a.md") == "b.md"

    # The failed attempt must also leave the READY build usable for a successful retry.
    manager.promote(incoming, unsafe_development=True)
    assert manager.active_generation_id() == incoming
    with GenerationStore(TEST_DSN, 64, tenant=manager.tenant_id) as reader:
        assert reader.supersession_graph().successor("a.md") == "b.md"


def test_publication_boundary_never_exposes_a_before_its_ghost_is_durable(
    manager: GenerationManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Independent readers must see either outgoing B or incoming A plus B's durable ghost.

    Node: tests/test_generation_retirement_cca.py::
    test_publication_boundary_never_exposes_a_before_its_ghost_is_durable.
    Baseline: ab1a48ba427599bd174e9d99b5280182f37b2aea (PR987).
    Target: GenerationManager.promote's publication transaction. Reader observations occur
    synchronously on separate real connections before/after the actual audit INSERT and at
    the stated-facts callback, which marks the post-commit boundary. No scheduling or timing
    assumption is needed. Expected baseline RED: "a publication-boundary reader saw A current"
    at the callback before the old implementation appends B's ghost. Original-head RED and repaired GREEN
    were observed; see the audit receipts.
    """
    outgoing, incoming = _pair(manager)
    real_execute = psycopg.Connection.execute
    real_stated: Callable[[str], int] = manager._record_stated_facts_after_activation
    observations: list[tuple[str, str, str | None, bool]] = []
    source = f"s3://approved/corpora/{manager.tenant_id}/b.md"

    def observe(stage: str) -> None:
        active = manager.active_generation_id()
        with GenerationStore(TEST_DSN, 64, tenant=manager.tenant_id) as reader:
            successor = reader.supersession_graph().successor("a.md")
            durable = reader.recorded_sources_for_identifiers(["b.md"]) == {"b.md": [source]}
        observations.append((stage, active, successor, durable))

    def execute(conn: Any, query: Any, params: Any = None, *args: Any, **kwargs: Any) -> Any:
        omitted = _omitted_append(query, params, manager.tenant_id)
        if omitted:
            observe("before-append")
        result = real_execute(conn, query, params, *args, **kwargs)
        if omitted:
            observe("after-uncommitted-append")
        return result

    def after_activation(generation_id: str) -> int:
        assert generation_id == incoming
        observe("post-commit")
        return real_stated(generation_id)

    with monkeypatch.context() as patch:
        patch.setattr(psycopg.Connection, "execute", execute)
        patch.setattr(manager, "_record_stated_facts_after_activation", after_activation)
        manager.promote(incoming, unsafe_development=True)

    assert sorted(stage for stage, *_ in observations) == [
        "after-uncommitted-append", "before-append", "post-commit"
    ], "the test did not observe every retirement/publication boundary"
    assert all(successor == "b.md" for _, _, successor, _ in observations), (
        f"a publication-boundary reader saw A current: {observations}"
    )
    for stage, active, _, durable in observations:
        if stage == "post-commit":
            assert active == incoming and durable, "publication preceded durable retirement"
        else:
            assert active == outgoing and not durable, "uncommitted publication leaked to a reader"
    assert manager.active_generation_id() == incoming
    with GenerationStore(TEST_DSN, 64, tenant=manager.tenant_id) as reader:
        assert reader.supersession_graph().successor("a.md") == "b.md"
        assert reader.recorded_sources_for_identifiers(["b.md"]) == {"b.md": [source]}
