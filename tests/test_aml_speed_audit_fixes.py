"""Fixes from the CCA audit of #789 (run cca789): the Add prefetch, pool sizing, /version, SQL copies.

Red proof, 2026-09-27. Each test was run against the code before its fix and failed in its named
assertion; the fix was then applied and the same test passed:

* ``test_version_reports_the_digest_of_the_prompt_the_output_mode_sends`` (FIX-005): calling
  ``anchor_prompt_digest()`` without the mode in ``recall_aml/app.py`` (the pre-fix line) failed the
  digest equality with the full prompt's digest.
* ``test_the_pool_serves_every_executor_thread_beside_the_pinned_lock_connections`` (FIX-006): the
  pre-fix ``max(36, add + search + 2)`` in ``recall_aml/__main__.py`` failed ``36 >= 72 + 16`` and
  ``36 >= 30 + 8``.
* ``test_prefetched_vectors_are_released_once_their_persist_has_used_them`` (FIX-007): deleting
  ``setattr(prefetch, field, None)`` in ``recall_aml.service._take_vectors`` failed
  ``[(False, False)] == [(True, True)]``.
* ``test_a_failed_add_stops_its_prefetch_after_the_call_in_flight`` (FIX-008): replacing
  ``prefetch_stop.set()`` in ``HostedService.add`` with ``pass`` failed ``3 == 1``.
* ``test_prefetched_vectors_are_refused_when_the_add_builds_other_raw_windows`` (FIX-009): the
  guard existed before the audit, so it was proved by mutation: with ``if raw_chunks !=
  prefetch.raw_chunks:`` in ``HostedService._prefetched`` replaced by ``if False:``, it failed the
  writes equality (the edited window stored the unedited text's vector).
* ``test_the_cached_supersession_read_is_the_store_query_verbatim`` (FIX-010): adding ``AND
  metadata->>'record_type' = 'compiled'`` to ``PgVectorStore.explicit_superseded_chunk_ids`` failed
  the containment assertion.
* ``test_the_cached_row_read_is_iter_chunks_with_the_version_columns_added`` (FIX-010): changing
  ``PgVectorStore.iter_chunks`` to ``ORDER BY source, id`` failed the SQL equality.

* ``test_compiled_rows_name_the_output_mode_that_produced_them`` (ENV-002, owner decision after the
  audit): before the change in ``HostedService._compile_and_persist`` it failed
  ``{'anchor-v3'} == {'anchor-v3-lean'}`` and the same for ``select``.

Every file was restored byte for byte and each test passed again.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import dataclasses
import hashlib
from pathlib import Path
import re
import threading
from types import SimpleNamespace
from typing import Any, Iterator

import pytest
from starlette.testclient import TestClient

from recall.store import PgVectorStore
from recall_aml.app import create_app, executor_workers
from recall_aml.compiler import ANCHOR_COMPILER_SELECT_SYSTEM_PROMPT, ANCHOR_COMPILER_SYSTEM_PROMPT
from recall_aml.config import HostedSettings
import recall_aml.search_cache as search_cache
from recall_aml.service import HostedService
from recall_aml.variants import variant
from tests.test_aml_add_concurrency import (
    C9,
    GroundedCompiler,
    RecordingRepository,
    add_request,
    c9_service,
)
from tests.test_aml_hosted import FakeEmbedder


def _sql(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@pytest.mark.parametrize("mode", ["lean", "select"])
def test_compiled_rows_name_the_output_mode_that_produced_them(mode: str) -> None:
    """ENV-002 (owner decision 2026-09-27): a lean or select corpus is told apart from a full one.

    C9 serves ``full`` and keeps ``anchor-v3``, which ``tests/test_aml_hosted.py`` pins.
    """
    behavior = dataclasses.replace(variant(C9), anchor_compile_output=mode)
    repository = RecordingRepository(prefetch=True)
    service = HostedService(
        repository,
        GroundedCompiler(),
        object(),  # type: ignore[arg-type]
        behavior=behavior,
        multimodal_embedder=object(),  # type: ignore[arg-type]
        specialist_retrievers={behavior.context_embedding_profile: object()},  # type: ignore[dict-item]
    )

    asyncio.run(service.add(add_request()))

    profiles = {
        metadata["compiler_profile"]
        for _tenant, rows, _vectors in repository.writes
        for _id, _source, _text, metadata in rows
        if metadata.get("record_type") == "compiled"
    }
    assert profiles == {f"anchor-v3-{mode}"}


def test_version_reports_the_digest_of_the_prompt_the_output_mode_sends() -> None:
    """FIX-005 (ENV-001): a lean or select deployment sends another prompt than ``full``."""
    behavior = dataclasses.replace(variant(C9), anchor_compile_output="select")
    service = HostedService(
        RecordingRepository(),
        GroundedCompiler(),
        object(),  # type: ignore[arg-type]
        behavior=behavior,
        multimodal_embedder=object(),  # type: ignore[arg-type]
        specialist_retrievers={behavior.context_embedding_profile: object()},  # type: ignore[dict-item]
    )
    settings = HostedSettings("postgresql://unused", "secret", "abc123")
    version = TestClient(create_app(settings, service)).get("/version").json()

    assert version["anchor_compile_output"] == "select"
    assert (
        version["anchor_compiler_prompt_digest"]
        == hashlib.sha256(ANCHOR_COMPILER_SELECT_SYSTEM_PROMPT.encode()).hexdigest()
    )
    assert (
        version["anchor_compiler_prompt_digest"]
        != hashlib.sha256(ANCHOR_COMPILER_SYSTEM_PROMPT.encode()).hexdigest()
    )


@pytest.mark.parametrize("adds,searches", [(16, 16), (8, 3)])
def test_the_pool_serves_every_executor_thread_beside_the_pinned_lock_connections(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, adds: int, searches: int
) -> None:
    """FIX-006 (BUG-001+DEPLOY-004+PERF-006): each admitted Add pins a pooled connection for its
    tenant lock for its whole life, and any executor thread may borrow one more, so a pool smaller
    than both together queues on ``PoolTimeout`` under full admission."""
    import recall_aml.__main__ as hosted_main

    seen: dict[str, Any] = {}

    class Stop(Exception):
        pass

    def pool(*_args: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        raise Stop

    monkeypatch.setattr(hosted_main, "_resolve_hosted_embedders", lambda *_: (FakeEmbedder(), {}))
    monkeypatch.setattr(hosted_main, "SharedPool", pool)
    settings = HostedSettings(
        "postgresql://unused",
        "secret",
        "abc123",
        variant_name="A0_raw",
        voyage_api_key="voyage-key",
        embedding_lock_path=tmp_path / "embed.lock",
        add_concurrency=adds,
        search_concurrency=searches,
    )

    with pytest.raises(Stop):
        hosted_main.build_app(settings)

    assert seen["max_size"] >= executor_workers(settings) + settings.add_concurrency


def test_prefetched_vectors_are_released_once_their_persist_has_used_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIX-007 (PERF-001): the raw and primary-view vectors are not held to the end of the Add."""
    repository = RecordingRepository(prefetch=True)
    service = c9_service(repository, GroundedCompiler())
    captured: list[Any] = []
    at_last_persist: list[tuple[bool, bool]] = []
    original_prefetched = HostedService._prefetched

    async def capturing(task: Any, chunks: Any) -> Any:
        prefetch = await original_prefetched(task, chunks)
        captured.append(prefetch)
        return prefetch

    monkeypatch.setattr(HostedService, "_prefetched", staticmethod(capturing))
    original_views = repository.persist_atomic_views

    def persist_views(tenant: str, profile: Any, chunks: Any, vectors: Any = None) -> None:
        if profile is not None:  # the specialist scope, the Add's last embedded write
            prefetch = captured[0]
            at_last_persist.append(
                (prefetch.raw_vectors is None, prefetch.primary_view_vectors is None)
            )
        original_views(tenant, profile, chunks, vectors)

    repository.persist_atomic_views = persist_views  # type: ignore[method-assign]

    asyncio.run(service.add(add_request()))

    assert captured and captured[0] is not None
    assert at_last_persist == [(True, True)]


def test_a_failed_add_stops_its_prefetch_after_the_call_in_flight() -> None:
    """FIX-008 (BUG-003+PERF-002): an abandoned prefetch makes no further provider call."""
    release = threading.Event()
    in_flight = threading.Event()
    repository = RecordingRepository(prefetch=True)
    original = repository.embed_texts
    calls: list[Any] = []

    def blocked(*args: Any) -> Any:
        calls.append(args[0])
        if len(calls) == 1:
            in_flight.set()
            release.wait(3.0)
        return original(*args)

    def broken(*_args: Any, **_kwargs: Any) -> Any:
        # Fail only once the prefetch's first call is in flight, the case the fix is about;
        # failing earlier would let it stop before any call, which proves nothing here.
        in_flight.wait(3.0)
        raise RuntimeError("prior records unavailable")

    repository.embed_texts = blocked  # type: ignore[method-assign]
    repository.prior_records = broken  # type: ignore[method-assign]
    service = c9_service(repository, GroundedCompiler())

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="prior records unavailable"):
            await service.add(add_request())
        release.set()

    try:
        # ``asyncio.run`` joins the executor's threads on close, so the prefetch has finished.
        asyncio.run(scenario())
    finally:
        release.set()

    assert len(calls) == 1


def test_prefetched_vectors_are_refused_when_the_add_builds_other_raw_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIX-009 (BUG-002): the equality in ``_prefetched`` is all that ties vectors to rows."""

    def diverging(service: HostedService) -> None:
        build = service._build_chunks

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            chunks = build(*args, **kwargs)
            if "compiler_profile" not in kwargs:  # the prefetch's own build
                return chunks
            first = next(i for i, c in enumerate(chunks) if c.metadata.get("record_type") == "raw")
            chunks = list(chunks)
            chunks[first] = dataclasses.replace(chunks[first], text=chunks[first].text + " edited")
            return chunks

        monkeypatch.setattr(service, "_build_chunks", wrapped)

    inline = RecordingRepository(prefetch=False)
    prefetched = RecordingRepository(prefetch=True)
    inline_service = c9_service(inline, GroundedCompiler())
    prefetched_service = c9_service(prefetched, GroundedCompiler())
    diverging(inline_service)
    diverging(prefetched_service)

    asyncio.run(inline_service.add(add_request()))
    asyncio.run(prefetched_service.add(add_request()))

    assert prefetched.writes == inline.writes


class _Capture:
    """A connection, cursor and store stand-in that records the SQL a read executes."""

    def __init__(self) -> None:
        self.sql: list[str] = []

    def execute(self, sql: str, _params: Any = None) -> "_Capture":
        self.sql.append(sql)
        return self

    def fetchall(self) -> list[Any]:
        return []

    def __iter__(self) -> Iterator[Any]:
        return iter(())

    @contextmanager
    def transaction(self) -> Iterator[None]:
        yield

    @contextmanager
    def cursor(self, name: str | None = None) -> Iterator["_Capture"]:
        yield self

    def store(self) -> Any:
        capture = self

        @contextmanager
        def borrowed() -> Iterator[_Capture]:
            yield capture

        return SimpleNamespace(
            _table="t",
            _tenant="u",
            _dsn="d",
            _borrowed=borrowed,
            _with_retry=lambda fn: fn(capture),
        )


def test_the_cached_supersession_read_is_the_store_query_verbatim() -> None:
    """FIX-010 (CODE-001): an edit to the store's query must fail here, not diverge silently."""
    capture = _Capture()
    PgVectorStore.explicit_superseded_chunk_ids(capture.store())  # type: ignore[arg-type]
    (store_sql,) = capture.sql

    # The cache names the column (``AS e``) for its outer ``array_agg``; nothing else may differ.
    cached = _sql(search_cache._SUPERSESSION_SQL.format(table="t")).replace(" AS e ", " ")
    assert _sql(store_sql).replace("%s", "%(tenant)s") in cached


def test_the_cached_row_read_is_iter_chunks_with_the_version_columns_added() -> None:
    """FIX-010 (CODE-001): same rows, same order, as the reference BM25 path reads them."""
    reference = _Capture()
    list(PgVectorStore.iter_chunks(reference.store()))  # type: ignore[arg-type]
    cached = _Capture()
    search_cache._read_all(cached.store())  # type: ignore[arg-type]

    (reference_sql,) = reference.sql
    (cached_sql,) = cached.sql
    assert _sql(cached_sql).replace(", xmin::text::bigint, indexed_at", "") == _sql(reference_sql)
