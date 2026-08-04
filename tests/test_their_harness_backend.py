"""Contract tests for `recall_interop.RecallBackend`.

These pin the adapter against the shape `mem0ai/memory-benchmarks` actually consumes, because a
mismatch there does not raise — it degrades silently into a worse benchmark score that looks like
a retrieval result. Each test names the number it protects.

No API calls anywhere. Two embedders are used, and which one is deliberate:

- ``hashing`` (`backend`) for tests about SHAPE — the dict keys, the error paths, tenancy, close.
  It needs no model at all.
- ``fastembed`` (`sem_backend`, the local bge-small the real runs use) for tests about RETRIEVAL.
  `hashing` embeddings carry no semantics, so their cosines land near 0.3-0.4 and RE-call's
  confidence threshold correctly ABSTAINS on almost every query — a real behaviour that makes
  `hashing` useless for asserting anything about what comes back. The model is local and cached;
  no network and no spend either way.

Only a local pgvector database is required.
"""

from __future__ import annotations

import asyncio
import uuid

import psycopg
import pytest

from recall_interop import RecallBackend
from recall_interop.memory_benchmarks import (
    _created_at_of,
    _messages_to_document,
    _resolve_epoch,
    resolve_embedder,
)
from tests.conftest import TEST_DSN, requires_db, requires_fastembed

#: Two LOCOMO session dates as their runner would supply them (`locomo_date_to_epoch`).
MAY_2023 = 1683554160     # 2023-05-08 13:56:00Z
AUG_2023 = 1691500560     # 2023-08-08 13:56:00Z


@pytest.fixture
def table():
    """A uuid-named benchmark table, dropped afterwards."""
    name = "tb_" + uuid.uuid4().hex[:8]
    yield name
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {name}")


def backend(table: str, tmp_path, embedder_name: str = "hashing", **kw) -> RecallBackend:
    return RecallBackend(
        TEST_DSN,
        embedder_name=embedder_name,
        table=table,
        workspace_root=tmp_path / "ws",
        # Their runners pass these to `Mem0Client`; the adapter must swallow them so the swap
        # needs no call-site edit.
        rpm=200,
        host="http://localhost:8888",
        max_retries=5,
        **kw,
    )


def sem_backend(table: str, tmp_path, **kw) -> RecallBackend:
    """The backend as the real runs configure it: local bge-small, so cosines mean something."""
    return backend(table, tmp_path, embedder_name="fastembed", **kw)


def turn(text: str) -> list[dict[str, str]]:
    """One of their LOCOMO ingestion chunks (CHUNK_SIZE=1)."""
    return [{"role": "user", "content": text}]


# ===========================================================================
# Pure unit tests — no database
# ===========================================================================


def test_document_is_content_verbatim_with_no_role_injected():
    """The document RE-call indexes must be what Mem0's extractor is handed, and nothing more.

    Writing a synthetic ``user:``/``assistant:`` line would give RE-call a lexical token Mem0
    never sees, quietly turning the comparison into "who got fed more".
    """
    doc = _messages_to_document(
        [{"role": "user", "content": "Caroline: I ran a 10k."},
         {"role": "assistant", "content": "Melanie: Congrats!"}]
    )
    assert doc == "Caroline: I ran a 10k.\n\nMelanie: Congrats!"
    assert "user:" not in doc and "assistant:" not in doc


def test_empty_chunk_yields_empty_document():
    assert _messages_to_document([{"role": "user", "content": "   "}]) == ""
    assert _messages_to_document([]) == ""


def test_resolve_epoch_matches_their_two_spellings():
    assert _resolve_epoch(MAY_2023, None) == MAY_2023
    assert _resolve_epoch(None, "2023-05-08") == 1683504000
    assert _resolve_epoch(None, "not-a-date") is None
    assert _resolve_epoch(None, None) is None
    # `timestamp` wins, as it does in `Mem0Client._add_oss`.
    assert _resolve_epoch(MAY_2023, "2020-01-01") == MAY_2023


class _FakeHit:
    def __init__(self, file_key):
        self.chunk = type("C", (), {"metadata": {"file": file_key}, "source": file_key})()
        self.provenance = type("P", (), {"indexed_at": None})()


def test_created_at_decodes_the_filename_stamp_on_both_path_separators():
    """`metadata["file"]` is posix on Linux and can arrive with backslashes on Windows."""
    assert _created_at_of(_FakeHit(f"m000001__ts{MAY_2023}.md")) == "2023-05-08T13:56:00"
    assert _created_at_of(_FakeHit(f"sub\\m000001__ts{MAY_2023}.md")) == "2023-05-08T13:56:00"


def test_missing_timestamp_is_absent_not_epoch_zero():
    """A `NONE` stamp must not decode to 1970: their answerer PRINTS the date and reasons about it,
    so a fabricated one is worse than a missing one."""
    assert _created_at_of(_FakeHit("m000001__tsNONE.md")) is None
    assert _created_at_of(_FakeHit("m000001.md")) is None


def test_paid_embedders_are_refused():
    """This arm is $0 by construction. A paid route must fail loudly, not be selectable."""
    with pytest.raises(ValueError, match="unknown embedder"):
        resolve_embedder("voyage")
    with pytest.raises(ValueError, match="unknown embedder"):
        resolve_embedder("openai")


# ===========================================================================
# Contract tests against their seam — local pgvector only
# ===========================================================================


@requires_fastembed
@requires_db
def test_search_returns_memory_and_created_at(table, tmp_path):
    """Their answerer reads exactly these two keys (`benchmarks/locomo/prompts.py`). A result
    without `created_at` is printed as "(unknown date)" and every temporal question is lost."""

    async def scenario():
        async with sem_backend(table, tmp_path) as be:
            await be.add(turn("Caroline: I adopted a dog named Rex."), "u1", timestamp=MAY_2023)
            return await be.search("dog", "u1", top_k=5)

    hits = asyncio.run(scenario())
    assert hits, "nothing retrieved — the ingest path is broken"
    assert hits[0]["memory"] == "Caroline: I adopted a dog named Rex."
    assert hits[0]["created_at"] == "2023-05-08T13:56:00"


@requires_db
def test_add_returns_their_results_shape(table, tmp_path):
    async def scenario():
        async with backend(table, tmp_path) as be:
            return (
                await be.add(turn("Caroline: hello"), "u1", timestamp=MAY_2023),
                await be.add([{"role": "user", "content": "  "}], "u1", timestamp=MAY_2023),
            )

    ok, empty = asyncio.run(scenario())
    assert isinstance(ok, dict) and len(ok["results"]) == 1
    assert ok["results"][0]["event"] == "ADD"
    assert empty == {"results": []}


@requires_fastembed
@requires_db
def test_add_without_timestamp_omits_created_at(table, tmp_path):
    async def scenario():
        async with sem_backend(table, tmp_path) as be:
            await be.add(turn("Caroline: undated note about kayaking."), "u1")
            return await be.search("kayaking", "u1", top_k=5)

    hits = asyncio.run(scenario())
    assert hits
    assert "created_at" not in hits[0]


@requires_fastembed
@requires_db
def test_top_k_is_respected_and_the_candidate_pool_is_widened_to_match(table, tmp_path):
    """The number this protects is the top_200 cell.

    RE-call's default per-leg pool is `DEFAULT_CANDIDATE_K = 20`, so the fused pool holds at most
    ~40 distinct chunks. Left at the default, a request for `top_k=200` returns ~40 and RE-call
    loses at their budget for a reason that has nothing to do with retrieval quality.
    """
    from recall.retriever import DEFAULT_CANDIDATE_K

    n_docs = 3 * DEFAULT_CANDIDATE_K  # 60 > the ~40 an un-widened fused pool can hold

    async def scenario():
        async with sem_backend(table, tmp_path) as be:
            for i in range(n_docs):
                await be.add(turn(f"Caroline: memory number {i} about hiking."), "u1",
                             timestamp=MAY_2023 + i)
            return (
                await be.search("hiking", "u1", top_k=n_docs),
                await be.search("hiking", "u1", top_k=5),
            )

    wide, narrow = asyncio.run(scenario())
    assert len(narrow) <= 5
    assert len(wide) > 2 * DEFAULT_CANDIDATE_K, (
        f"only {len(wide)} of {n_docs} returned at top_k={n_docs} — the candidate pool was not "
        "widened, so every 200-budget cell would be measured on a ~40-memory context"
    )


@requires_fastembed
@requires_db
def test_score_preserves_recalls_ranking_under_their_sort(table, tmp_path):
    """Their harness re-sorts by `score` descending and then slices `[:cutoff]`, so whatever is in
    `score` IS the ranking. RE-call ranks by RRF fusion while each hit carries its true dense
    cosine — handing them the cosine would re-rank RE-call by its dense leg alone."""

    async def scenario():
        async with sem_backend(table, tmp_path) as be:
            for i in range(12):
                await be.add(turn(f"Caroline: note {i} on lake tahoe kayaking trips."), "u1",
                             timestamp=MAY_2023 + i)
            return await be.search("kayaking at the lake", "u1", top_k=12)

    hits = asyncio.run(scenario())
    assert len(hits) >= 2
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True) and len(set(scores)) == len(scores), (
        "score is not strictly decreasing, so their sort would reorder RE-call's hits"
    )
    resorted = sorted(hits, key=lambda h: h["score"], reverse=True)
    assert [h["id"] for h in resorted] == [h["id"] for h in hits]
    # The real numbers are still reported, so the artifact is auditable.
    assert set(hits[0]["score_debug"]) == {"recall_rank", "cosine", "confidence", "verdict"}


@requires_db
def test_unknown_user_returns_empty_not_an_error(table, tmp_path):
    """Their runner searches whatever `user_id` its ingest produced; a raise here would abort a
    whole conversation instead of scoring it as a miss."""

    async def scenario():
        async with backend(table, tmp_path) as be:
            return await be.search("anything", "never-ingested", top_k=10)

    assert asyncio.run(scenario()) == []


@requires_fastembed
@requires_db
def test_abstention_propagates_as_an_empty_result_list(table, tmp_path, monkeypatch):
    """RE-call abstaining must reach their answerer as "(No relevant memories found)".

    On LOCOMO categories 1-4 this can only ever cost points, and that is the point: the run
    measures the library that ships, not a benchmark-only variant with the guard removed.

    Both directions are asserted on the SAME corpus and query. Asserting only the abstaining
    direction would pass just as well if `search` were broken and returned nothing at all — the
    control run proves the query does retrieve, so the empty list can only come from the
    abstention branch.
    """
    import recall.trust as trust_mod

    real = trust_mod.trusted_search

    def abstaining(*a, **kw):
        result = real(*a, **kw)
        return type(result)(
            query=result.query, hits=result.hits, abstained=True, reason="forced",
            gap_warning=result.gap_warning,
            staleness=result.staleness,
            calibration_id=result.calibration_id,
            calibration_status=result.calibration_status,
            tenant_id=result.tenant_id,
            generation_id=result.generation_id,
            pipeline_fingerprint=result.pipeline_fingerprint,
            corpus_fingerprint=result.corpus_fingerprint,
            query_set_digest=result.query_set_digest,
        )

    async def scenario():
        async with sem_backend(table, tmp_path) as be:
            await be.add(turn("Caroline: I adopted a dog named Rex."), "u1", timestamp=MAY_2023)
            control = await be.search("dog", "u1", top_k=5)
            # The backend reaches the trust layer through `recall.eval._research_trust`, which
            # binds `trusted_search` at import time. Patching `recall.trust` alone would leave
            # the real function in place and make the assertion below vacuous.
            import recall.eval._research_trust as research_mod

            monkeypatch.setattr(trust_mod, "trusted_search", abstaining)
            monkeypatch.setattr(research_mod, "trusted_search", abstaining)
            return control, await be.search("dog", "u1", top_k=5)

    control, abstained = asyncio.run(scenario())
    assert control, "control retrieval returned nothing — the assertion below would be vacuous"
    assert abstained == []


@requires_db
def test_rerank_is_refused_rather_than_ignored(table, tmp_path):
    """RE-call HAS a reranker seam, so silently dropping the flag would publish a run labelled
    "reranked" that was not."""

    async def scenario():
        async with backend(table, tmp_path) as be:
            with pytest.raises(ValueError, match="rerank=True is not supported"):
                await be.search("q", "u1", top_k=5, rerank=True)
            with pytest.raises(ValueError):
                await be.search("q", "u1", top_k=0)

    asyncio.run(scenario())


@requires_fastembed
@requires_db
def test_a_second_run_measures_the_same_corpus_not_a_doubled_one(table, tmp_path):
    """Two runs against the same table+user must not accumulate.

    A doubled corpus fills top-k with duplicate copies of the same turns, so the effective context
    SHRINKS with each rerun and the published number becomes a function of how many times the
    harness happened to be run.
    """

    async def one_run():
        async with sem_backend(table, tmp_path) as be:
            for i in range(4):
                await be.add(turn(f"Caroline: fact {i} about pottery class."), "u1",
                             timestamp=MAY_2023 + i)
            return await be.search("pottery", "u1", top_k=50)

    first = asyncio.run(one_run())
    second = asyncio.run(one_run())
    assert len(first) == len(second) == 4
    assert sorted(h["memory"] for h in first) == sorted(h["memory"] for h in second)


@requires_fastembed
@requires_db
def test_users_are_isolated_from_each_other(table, tmp_path):
    """Their harness gives each conversation its own `user_id`. One conversation answering
    another's questions would inflate accuracy with no error and no visible symptom."""

    async def scenario():
        async with sem_backend(table, tmp_path) as be:
            await be.add(turn("Caroline: I adopted a dog named Rex."), "u1", timestamp=MAY_2023)
            await be.add(turn("Melanie: I bought a kayak."), "u2", timestamp=AUG_2023)
            return await be.search("dog named Rex", "u2", top_k=10)

    hits = asyncio.run(scenario())
    assert all("Rex" not in h["memory"] for h in hits)


@requires_db
def test_delete_user_and_close_are_both_idempotent(table, tmp_path):
    async def scenario():
        be = backend(table, tmp_path)
        await be.add(turn("Caroline: transient note."), "u1", timestamp=MAY_2023)
        first = await be.delete_user("u1")
        second = await be.delete_user("u1")
        await be.close()
        await be.close()
        with pytest.raises(RuntimeError, match="closed"):
            await be.search("q", "u1", top_k=5)
        return first, second

    assert asyncio.run(scenario()) == (True, False)


@requires_db
def test_describe_reports_the_configuration_and_no_dsn(table, tmp_path):
    """The artifact has to say which embedder, which pool and which version produced a number —
    and must not carry the DSN, which may embed a password."""

    async def scenario():
        async with backend(table, tmp_path) as be:
            await be.add(turn("Caroline: hello"), "u1", timestamp=MAY_2023)
            return be.describe()

    d = asyncio.run(scenario())
    assert d["system"] == "recall"
    assert d["embedder"]["name"] == "hashing"
    assert d["users"] == ["u1"]
    assert d["abstention"] == "honoured (empty result list)"
    assert TEST_DSN not in repr(d)
