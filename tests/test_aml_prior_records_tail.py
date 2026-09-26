"""An Add reads only the prior records its compile sends, and the compile still sees them all.

``PgHostedRepository.prior_records`` used to read and validate every row a session had ever
written, on every Add, while the compile sends only the last ``PRIOR_RECORDS_SENT`` (24) valid
ones. It now pages backwards from the newest row. Its answer must be exactly the last 24 of the
full read, including when invalid rows, raw rows, other sessions and ``indexed_at`` ties sit
between them, and the compile's ``supersedes`` check must still accept any valid record of the
session, which the full read allowed.

Red proof, 2026-09-26, each by mutating the named production line with this file unchanged,
watching the named assertion fail, then restoring it:

* ``test_the_paged_tail_is_the_last_24_of_the_full_read``: setting ``cursor = ""`` in
  ``PgVectorStore.compiled_chunks_for_source_newest_first`` (``recall/store.py``), so every page
  after the first repeated it, failed ``assert tail == full[-PRIOR_RECORDS_SENT:]`` for the graph
  and the primary scopes.
* the same test: deleting ``newest_first.reverse()`` in ``PgHostedRepository.prior_records``
  (``recall_aml/storage.py``) failed the same assertion (newest first instead of oldest first).
* the same test: ordering ties by ``id ASC`` instead of ``id DESC`` in the store query failed the
  same assertion, because the rows of each batch share one ``indexed_at``.
* ``test_a_supersedes_reference_older_than_the_tail_is_still_accepted``: making
  ``_supersedable`` in ``recall_aml/compiler.py`` return ``{item.id for item in prior}`` without
  asking ``supersedable_ids`` failed ``assert records[0].supersedes == [old_id]`` (``[] == [...]``).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from recall.pool import SharedPool
from recall.store import PgVectorStore
from recall.types import Chunk
from recall_aml.compiler import (
    PRIOR_RECORDS_SENT,
    OpenAICompiler,
    PriorRecords,
    StoredCodingRecord,
)
from recall_aml.identity import graph_tenant
from recall_aml.models import CodingMemoryRecord, Message
from recall_aml.storage import PgHostedRepository
from tests.conftest import TEST_DSN, requires_db

SOURCE = "aml://session/tail"
VECTOR = [1.0, 0.0, 0.0]


class _Embedder:
    dim = 3
    name = "unused"

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("prior records never embed")


def _record(n: int) -> dict[str, Any]:
    quote = f"moved step {n} into the queue"
    return CodingMemoryRecord.model_validate(
        {
            "kind": "procedure",
            "action": quote,
            "evidence_spans": [{"message_ordinal": 0, "start": 0, "end": len(quote), "quote": quote}],
            "evidence_quotes": [quote],
            "source_session_id": "tail",
        }
    ).model_dump(mode="json")


def _compiled(chunk_id: str, payload: object, source: str = SOURCE) -> Chunk:
    return Chunk(
        chunk_id, source, f"text of {chunk_id}", {"record_type": "compiled", "coding_record": payload}
    )


def _batches() -> list[list[Chunk]]:
    """Several transactions (so several ``indexed_at`` values), oldest first.

    The newest batch holds more invalid rows than the first page, so a correct read must page
    back past them, and its rows share one ``indexed_at``, so ties are ordered by id.
    """
    batches: list[list[Chunk]] = []
    for batch in range(4):
        rows: list[Chunk] = []
        for n in range(15):
            rows.append(_compiled(f"mem_{batch}_{n:02d}", _record(batch * 100 + n)))
        rows.append(Chunk(f"raw_{batch}", SOURCE, "a raw window", {"record_type": "raw"}))
        rows.append(_compiled(f"mem_{batch}_bad", {"kind": "procedure"}))  # fails validation
        rows.append(_compiled(f"mem_{batch}_str", "not a dict"))
        rows.append(_compiled(f"mem_{batch}_other", _record(batch), source="aml://session/other"))
        batches.append(rows)
    newest = [_compiled(f"mem_z_bad_{n:02d}", {"kind": "procedure"}) for n in range(40)]
    newest += [_compiled(f"mem_z_{n:02d}", _record(900 + n)) for n in (3, 1, 2)]
    batches.append(newest)
    return batches


@pytest.fixture
def base_store(make_store: Any) -> Any:
    """A serving-shaped store (shared pool, so ``for_tenant`` works) on a throwaway table."""
    fixture_store = make_store(3)
    pool = SharedPool(TEST_DSN, min_size=1, max_size=4)
    store = PgVectorStore(
        TEST_DSN,
        3,
        table=fixture_store.table,
        tenant="aml_service_readiness",
        shared_pool=pool,
        owns_pool=True,
    )
    try:
        yield store
    finally:
        store.close()


@requires_db
@pytest.mark.parametrize("graph_sidecar", [True, False])
def test_the_paged_tail_is_the_last_24_of_the_full_read(base_store: Any, graph_sidecar: bool) -> None:
    base = base_store
    repository = PgHostedRepository(base, _Embedder())
    tenant = "aml_tail"
    store = base.for_tenant(graph_tenant(tenant) if graph_sidecar else tenant)
    for rows in _batches():
        store.upsert(rows, [VECTOR] * len(rows))

    full = repository.all_prior_records(tenant, SOURCE, graph_sidecar=graph_sidecar)
    tail = repository.prior_records(tenant, SOURCE, graph_sidecar=graph_sidecar)

    assert len(full) == 4 * 15 + 3
    assert tail == full[-PRIOR_RECORDS_SENT:]
    assert isinstance(tail, PriorRecords)
    assert tail.supersedable_ids() == {record.id for record in full}


@requires_db
def test_a_short_session_returns_every_valid_record(base_store: Any) -> None:
    """Nonbehavioural control: under 24 valid records the tail is the whole read."""
    base = base_store
    repository = PgHostedRepository(base, _Embedder())
    store = base.for_tenant("aml_short")
    rows = _batches()[0]
    store.upsert(rows, [VECTOR] * len(rows))

    full = repository.all_prior_records("aml_short", SOURCE)

    assert len(full) == 15
    assert repository.prior_records("aml_short", SOURCE) == full


def test_a_supersedes_reference_older_than_the_tail_is_still_accepted() -> None:
    old_id = "mem_" + "ab" * 16
    content = f"this replaces {old_id}: the queue now runs on postgres"

    def create(**request: Any) -> Any:
        data = json.loads(
            request["messages"][1]["content"].removeprefix("<stored_data>").removesuffix("</stored_data>")
        )
        answer = {
            "records": [
                {
                    "kind": "architectural decision",
                    "action": content,
                    "evidence_anchor_ids": [data["anchors"][0]["id"]],
                    "source_session_id": "s",
                    "supersedes": [old_id],
                }
            ]
        }
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(answer)))]
        )

    compiler = OpenAICompiler(
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
        sleep=lambda _: None,
    )
    latest = [
        StoredCodingRecord(f"mem_{n:02d}", CodingMemoryRecord.model_validate(_record(n)))
        for n in range(PRIOR_RECORDS_SENT)
    ]
    prior = PriorRecords(latest, lambda: {old_id} | {item.id for item in latest})

    records = compiler.compile_anchored_v3([Message(role="user", content=content)], "s", prior)

    assert records[0].supersedes == [old_id]
