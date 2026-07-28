"""`run()`'s provenance block, exercised through the real function — not re-derived by hand.

`recall/eval/locomo.py::run()` builds a result report whose provenance fields (`corpus_rows`,
`table`, `tenants`, `git_sha` — see `recall/eval/provenance.py`) exist so a result JSON records the
corpus it was measured against. Two recent fixes touched exactly that code: `corpus_rows` now sums
a hard `res["corpus_rows"]` subscript instead of `res.get("corpus_rows", 0)`, and `tenants` is
captured inside the indexing loop instead of re-derived afterwards from `conversations`. Neither
fix had a test that called `run()` itself: `tests/test_eval_provenance.py` only exercises
`provenance_block()` in isolation (pure function, no DB, no indexing), and
`tests/test_locomo_corpus_postcondition.py` only exercises `run_conversation()`, never `run()`'s
own aggregation step. This file closes that gap: it runs the real `run()` against a real
(throwaway) Postgres table and checks the provenance block it returns against what actually landed
in that table — not against a second copy of the same formula `run()` uses internally.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import psycopg

from recall.eval.locomo import run
from tests.conftest import TEST_DSN, requires_db

#: One conversation, two turns, one (answerable) question — just enough for `run()` to index a
#: real corpus and score a real retrieval, without pulling in the adversarial/abstention arm this
#: file has no interest in. This is `run()`'s own input shape: `conversations = json.loads(...)`
#: is a LIST of conversation records, each carrying `sample_id` (used to name the tenant),
#: `conversation` (the LOCOMO session/turn structure `write_conversation_corpus` walks) and `qa`
#: (the questions) — NOT the flat `_CONVERSATION`/`_QA` shape `test_locomo_corpus_postcondition.py`
#: passes straight to `run_conversation`, which skips this outer envelope entirely.
_SAMPLE_ID = "provtest1"
_LOCOMO_DOC = [
    {
        "sample_id": _SAMPLE_ID,
        "conversation": {
            "speaker_a": "Caroline",
            "speaker_b": "Melanie",
            "session_1_date_time": "1:00 pm on 8 May, 2023",
            "session_1": [
                {"speaker": "Caroline", "dia_id": "D1:1", "text": "I finally adopted a greyhound."},
                {"speaker": "Melanie", "dia_id": "D1:2", "text": "I signed up for a pottery class."},
            ],
        },
        "qa": [
            {"question": "What did Caroline adopt?", "category": 1, "evidence": ["D1:1"]},
        ],
    },
]


@requires_db
def test_run_reports_the_provenance_of_the_corpus_it_actually_indexed(tmp_path: Path) -> None:
    """`run()`'s provenance block must describe the run that just happened, not a config echo."""
    # Unique per test run (not "locomo_chunks", the real benchmark table, and not a fixed name
    # shared with any other test): a uuid suffix means this cannot collide.
    table = "t_" + uuid.uuid4().hex[:8]
    data_path = tmp_path / "locomo_fixture.json"
    data_path.write_text(json.dumps(_LOCOMO_DOC), encoding="utf-8")

    try:
        report = run(
            data_path,
            dsn=TEST_DSN,
            # _make_embedder("hashing") -> HashingEmbedder(dim=64): local and deterministic, no
            # model download. A test that pulls fastembed's model on first run is a test nobody
            # runs.
            embedder_name="hashing",
            k=5,
            limit=None,
            keep_corpus=None,
            table=table,
        )

        # corpus_rows: positive, and consistent with what the run itself reported per conversation.
        assert isinstance(report["corpus_rows"], int)
        assert report["corpus_rows"] > 0, "the run reported no corpus at all"
        assert report["corpus_rows"] == sum(
            c["corpus_rows"] for c in report["per_conversation"]
        ), "top-level corpus_rows must equal the sum of what each conversation reported"

        # tenants: checked against what is ACTUALLY sitting in the table — queried independently
        # of whatever formula `run()` used internally — not merely re-derived the same way `run()`
        # derives it, which would agree with a stale second copy of the formula just as readily as
        # with the real one.
        with psycopg.connect(TEST_DSN) as conn:
            rows = conn.execute(f"SELECT DISTINCT tenant_id FROM {table}").fetchall()
        actually_indexed = sorted(r[0] for r in rows)
        assert report["tenants"] == actually_indexed, (
            "reported tenants must match the tenant_id(s) actually present in the table"
        )
        assert report["tenants"] == [f"locomo-{_SAMPLE_ID}"]

        assert report["table"] == table

        assert "git_sha" in report
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
