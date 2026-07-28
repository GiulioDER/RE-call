"""Both LOCOMO readers must refuse to score a tenant that holds zero rows — proven through their
REAL entry points (`recall.eval.locomo_abstention.run`, `recall.eval.locomo_entailment_sweep.run`),
not just through `check_tenants_populated` in isolation. `tests/test_tenant_guard.py` covers that
function's own logic without a database; this file proves the two readers actually CALL it, at the
right point, scoped to the right tenants — the gap a prior review on this branch flagged: a test
that only calls the helper directly proves the helper works, not that the runner uses it.

Every table used here is a fresh `t_<uuid8>` scratch table, dropped in a fixture `finally`. NONE
of these tests ever open, write to, or drop `locomo_chunks` — that table is being read right now
by a long-running entailment sweep, and touching it would corrupt that measurement. The "populated
tenant" fixtures below index through the REAL builder (`recall.eval.locomo.run`, imported
read-only, never modified) rather than hand-crafting rows, so the scratch corpus a reader sees
here has the same shape a real corpus would.

Both readers unconditionally load a cross-encoder immediately after the point the guard now
occupies (`QnliEntailmentJudge()` in `locomo_abstention.run`; `build_scorer()` — called from
inside `gather_scores` — in `locomo_entailment_sweep.run`). Driving either reader all the way to
a real, completed report would mean downloading real model weights in a unit test. Where a test
needs to prove execution proceeded PAST the guard on a healthy corpus, it monkeypatches that next
step to raise a private marker (`_ReachedHeavyWork`) instead — a fast, network-free, unambiguous
signal that distinguishes "the guard did not fire" from "the guard fired" without paying for a
real model load. See `sdd/tenant-guard-report.md` for the fuller rationale.
"""
from __future__ import annotations

import json
import uuid

import psycopg
import pytest

import recall.eval.locomo_abstention as locomo_abstention
import recall.eval.locomo_entailment_sweep as locomo_entailment_sweep
from recall.embeddings import HashingEmbedder
from recall.eval.locomo import index_conversation
from recall.eval.locomo import run as build_corpus
from recall.eval.tenant_guard import check_tenants_populated
from recall.types import Chunk

from tests.conftest import TEST_DSN, requires_db

DIM = 64


def _conversation(sample_id: str) -> dict:
    """One minimal, valid LOCOMO conversation record — real enough for the actual BUILDER
    (`recall.eval.locomo.run`) to index it, and for a reader to partition its `qa`."""
    return {
        "sample_id": sample_id,
        "conversation": {
            "speaker_a": "Ann",
            "speaker_b": "Bo",
            "session_1_date_time": "1:00 pm on 1 January, 2024",
            "session_1": [
                {"speaker": "Ann", "dia_id": "D1:1", "text": f"{sample_id}: I got a new bike."},
                {"speaker": "Bo", "dia_id": "D1:2", "text": f"{sample_id}: I baked bread."},
            ],
        },
        "qa": [
            {"question": f"What did Ann get, in {sample_id}?", "category": 1, "evidence": ["D1:1"]},
            {"question": f"What did Bo actually realize, in {sample_id}?", "category": 5},
        ],
    }


@pytest.fixture
def scratch_table():
    """A uuid-named table, dropped afterwards. NEVER `locomo_chunks` — see the module docstring:
    a background job is reading that table right now."""
    table = "t_tguard_" + uuid.uuid4().hex[:8]
    yield table
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {table}")


class _ReachedHeavyWork(Exception):
    """Marker: a reader's run() proceeded PAST the tenant guard into real scoring work.

    Raised by the stubs below in place of the real next step. RuntimeError(UNBUILT-TENANT) means
    the guard fired; this means it did not — the guard is not what stopped (or didn't stop) the
    run, which is exactly the distinction the "no false positive" requirement needs to observe
    through the real entry point without downloading a real cross-encoder.
    """


@pytest.fixture
def stub_abstention_judge(monkeypatch):
    """Replace QnliEntailmentJudge — `locomo_abstention.run`'s own next step after the guard —
    with a tripwire, so driving that function never loads (or downloads) a real cross-encoder."""
    import recall.entailment as entailment_module

    def _boom(*_a, **_kw):
        raise _ReachedHeavyWork(
            "QnliEntailmentJudge() was constructed — the tenant guard did not fire"
        )

    monkeypatch.setattr(entailment_module, "QnliEntailmentJudge", _boom)


@pytest.fixture
def stub_entailment_scorer(monkeypatch):
    """Replace build_scorer — `locomo_entailment_sweep.run`'s own next step after the guard,
    called from inside `gather_scores` for the first judge — with a tripwire."""

    def _boom(model_id):
        raise _ReachedHeavyWork(
            f"build_scorer({model_id!r}) was called — the tenant guard did not fire"
        )

    monkeypatch.setattr(locomo_entailment_sweep, "build_scorer", _boom)


# --- 1. Detection, through the real runner: a build that died partway through must be caught ---


@requires_db
def test_locomo_abstention_run_raises_on_a_partial_corpus(
    tmp_path, monkeypatch, scratch_table, stub_abstention_judge
):
    """Reproduces the incident from the task: a build died after 1 of N conversations.

    'populated0' is genuinely indexed by the real builder; 'gap1'/'gap2'/'gap3' are listed in
    the SAME data file `locomo_abstention.run` reads but were never indexed — their tenants are
    simply absent from the table, exactly like a build that died partway through.
    """
    conversations = [
        _conversation("populated0"), _conversation("gap1"),
        _conversation("gap2"), _conversation("gap3"),
    ]
    data_path = tmp_path / "partial.json"
    data_path.write_text(json.dumps(conversations), encoding="utf-8")

    # The REAL builder indexes ONLY the first conversation.
    build_corpus(
        data_path, dsn=TEST_DSN, embedder_name="hashing", k=5,
        limit=1, keep_corpus=None, table=scratch_table,
    )

    monkeypatch.setattr(locomo_abstention, "TABLE", scratch_table)
    with pytest.raises(RuntimeError, match="UNBUILT-TENANT") as exc:
        locomo_abstention.run(
            data_path, dsn=TEST_DSN, embedder_name="hashing", k=5,
            answerable_sample=0, limit=None, seed=0,
        )
    message = str(exc.value)
    assert "1/4" in message, f"expected '1/4 present' in: {message!r}"
    for gap in ("locomo-gap1", "locomo-gap2", "locomo-gap3"):
        assert gap in message, f"{gap} must be named in: {message!r}"


@requires_db
def test_locomo_entailment_sweep_run_raises_on_a_partial_corpus(
    tmp_path, monkeypatch, scratch_table, stub_entailment_scorer
):
    conversations = [
        _conversation("populated0"), _conversation("gap1"), _conversation("gap2"),
    ]
    data_path = tmp_path / "partial_sweep.json"
    data_path.write_text(json.dumps(conversations), encoding="utf-8")

    build_corpus(
        data_path, dsn=TEST_DSN, embedder_name="hashing", k=5,
        limit=1, keep_corpus=None, table=scratch_table,
    )

    monkeypatch.setattr(locomo_entailment_sweep, "TABLE", scratch_table)
    with pytest.raises(RuntimeError, match="UNBUILT-TENANT") as exc:
        locomo_entailment_sweep.run(
            data_path, dsn=TEST_DSN, embedder_name="hashing", k=5,
            answerable_sample=0, limit=None, seed=0,
            judges=(("fake", "fake-model-id"),),
        )
    message = str(exc.value)
    assert "1/3" in message, f"expected '1/3 present' in: {message!r}"
    assert "locomo-gap1" in message and "locomo-gap2" in message


# --- 2. No false positive: a complete corpus must pass the guard untouched -----------------


@requires_db
def test_locomo_abstention_run_does_not_block_a_complete_corpus(
    tmp_path, monkeypatch, scratch_table, stub_abstention_judge
):
    conversations = [_conversation("healthyA"), _conversation("healthyB")]
    data_path = tmp_path / "complete.json"
    data_path.write_text(json.dumps(conversations), encoding="utf-8")

    build_corpus(
        data_path, dsn=TEST_DSN, embedder_name="hashing", k=5,
        limit=None, keep_corpus=None, table=scratch_table,
    )

    monkeypatch.setattr(locomo_abstention, "TABLE", scratch_table)
    # Proceeding to the tripwire (rather than raising UNBUILT-TENANT, rather than returning
    # cleanly) is the proof: the guard looked at a fully-populated corpus and did not block it.
    with pytest.raises(_ReachedHeavyWork):
        locomo_abstention.run(
            data_path, dsn=TEST_DSN, embedder_name="hashing", k=5,
            answerable_sample=0, limit=None, seed=0,
        )


@requires_db
def test_locomo_entailment_sweep_run_does_not_block_a_complete_corpus(
    tmp_path, monkeypatch, scratch_table, stub_entailment_scorer
):
    conversations = [_conversation("healthyA"), _conversation("healthyB")]
    data_path = tmp_path / "complete_sweep.json"
    data_path.write_text(json.dumps(conversations), encoding="utf-8")

    build_corpus(
        data_path, dsn=TEST_DSN, embedder_name="hashing", k=5,
        limit=None, keep_corpus=None, table=scratch_table,
    )

    monkeypatch.setattr(locomo_entailment_sweep, "TABLE", scratch_table)
    with pytest.raises(_ReachedHeavyWork):
        locomo_entailment_sweep.run(
            data_path, dsn=TEST_DSN, embedder_name="hashing", k=5,
            answerable_sample=0, limit=None, seed=0,
            judges=(("fake", "fake-model-id"),),
        )


# --- 3. Scoping: --conversations/limit must bound what the guard checks, both directions ---


@requires_db
def test_locomo_abstention_run_scopes_the_guard_to_conversations_flag(
    tmp_path, monkeypatch, scratch_table, stub_abstention_judge
):
    """A `--conversations 2` run (limit=2) must check ONLY the 2 tenants it iterates — it must
    not fire just because a THIRD, out-of-scope conversation happens to be unbuilt. The same
    dataset with limit=None (all 3) DOES fire, on exactly the tenant the limited run correctly
    ignored — proving the guard's scope tracks what the run actually iterates, not a fixed
    dataset size or a hardcoded roster.
    """
    conversations = [
        _conversation("scopedA"), _conversation("scopedB"), _conversation("scopedC"),
    ]
    data_path = tmp_path / "scoped.json"
    data_path.write_text(json.dumps(conversations), encoding="utf-8")

    # Only the first two are ever indexed — "scopedC" is permanently absent from this table.
    build_corpus(
        data_path, dsn=TEST_DSN, embedder_name="hashing", k=5,
        limit=2, keep_corpus=None, table=scratch_table,
    )
    monkeypatch.setattr(locomo_abstention, "TABLE", scratch_table)

    # limit=2: only scopedA/scopedB are iterated, both populated -> passes the guard.
    with pytest.raises(_ReachedHeavyWork):
        locomo_abstention.run(
            data_path, dsn=TEST_DSN, embedder_name="hashing", k=5,
            answerable_sample=0, limit=2, seed=0,
        )

    # limit=None: now scopedC is iterated too, and it is empty -> the guard fires on it alone.
    with pytest.raises(RuntimeError, match="UNBUILT-TENANT") as exc:
        locomo_abstention.run(
            data_path, dsn=TEST_DSN, embedder_name="hashing", k=5,
            answerable_sample=0, limit=None, seed=0,
        )
    message = str(exc.value)
    assert "locomo-scopedC" in message
    assert "2/3" in message


@requires_db
def test_locomo_entailment_sweep_run_scopes_the_guard_to_conversations_flag(
    tmp_path, monkeypatch, scratch_table, stub_entailment_scorer
):
    conversations = [
        _conversation("scopedA"), _conversation("scopedB"), _conversation("scopedC"),
    ]
    data_path = tmp_path / "scoped_sweep.json"
    data_path.write_text(json.dumps(conversations), encoding="utf-8")

    build_corpus(
        data_path, dsn=TEST_DSN, embedder_name="hashing", k=5,
        limit=2, keep_corpus=None, table=scratch_table,
    )
    monkeypatch.setattr(locomo_entailment_sweep, "TABLE", scratch_table)

    with pytest.raises(_ReachedHeavyWork):
        locomo_entailment_sweep.run(
            data_path, dsn=TEST_DSN, embedder_name="hashing", k=5,
            answerable_sample=0, limit=2, seed=0,
            judges=(("fake", "fake-model-id"),),
        )

    with pytest.raises(RuntimeError, match="UNBUILT-TENANT") as exc:
        locomo_entailment_sweep.run(
            data_path, dsn=TEST_DSN, embedder_name="hashing", k=5,
            answerable_sample=0, limit=None, seed=0,
            judges=(("fake", "fake-model-id"),),
        )
    message = str(exc.value)
    assert "locomo-scopedC" in message
    assert "2/3" in message


# --- 4. Discrimination, live: all three locomo.py-family guards fired for real -------------


@requires_db
def test_the_three_locomo_guards_are_mutually_distinguishable(make_store, tmp_path, monkeypatch):
    """Read `recall/eval/locomo.py` (`index_conversation`'s pre-check, ~line 244, message
    "...already holds N chunk(s)..."; its post-condition, ~line 266, message "...table
    CONCURRENTLY...") — neither contains UNBUILT-TENANT, and this guard's message contains
    neither of theirs. Verified here by triggering all three guards for real, not by re-typing a
    second copy of their wording to compare against (which could drift from the source silently).
    """
    embedder = HashingEmbedder(dim=DIM)
    conversation = {
        "speaker_a": "Ann", "speaker_b": "Bo",
        "session_1_date_time": "1:00 pm on 1 January, 2024",
        "session_1": [{"speaker": "Ann", "dia_id": "D1:1", "text": "hello"}],
    }

    # 1) The PRE-check: index into a tenant that already holds a row.
    pre_store = make_store(DIM)
    pre_store.upsert(
        [Chunk(id="existing", source="existing.md", text="pre-existing row", metadata={})],
        [[0.1] * DIM],
    )
    with pytest.raises(RuntimeError, match="already holds") as pre_exc:
        index_conversation(pre_store, embedder, conversation, corpus_dir=tmp_path / "pre")
    assert "UNBUILT-TENANT" not in str(pre_exc.value)

    # 2) The POST-condition: a writer racing in during indexing (same technique as
    #    tests/test_locomo_corpus_postcondition.py::test_a_concurrent_writer_fails_the_run...).
    post_store = make_store(DIM)
    from recall.index import Indexer

    real_index_path = Indexer.index_path

    def racing_index_path(self, *args, **kwargs):
        stats = real_index_path(self, *args, **kwargs)
        post_store.upsert(
            [Chunk(id="intruder", source="intruder.md", text="race", metadata={})],
            [[0.1] * DIM],
        )
        return stats

    monkeypatch.setattr(Indexer, "index_path", racing_index_path)
    with pytest.raises(RuntimeError, match="CONCURRENTLY") as post_exc:
        index_conversation(post_store, embedder, conversation, corpus_dir=tmp_path / "post")
    assert "UNBUILT-TENANT" not in str(post_exc.value)

    # 3) And the converse direction: THIS guard's message contains neither of theirs.
    with pytest.raises(RuntimeError, match="UNBUILT-TENANT") as this_exc:
        check_tenants_populated({"t": 0}, table="t")
    assert "already holds" not in str(this_exc.value)
    assert "CONCURRENTLY" not in str(this_exc.value)
