"""`scripts/aml_w0_r1_vote_evidence.py`: the Task Solve evidence file, tested offline.

Each test states the invariant and the mutation it was watched to fail on (the red proof).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aml_w0_embedding_compare as w0  # noqa: E402
import aml_w0_r1_offline as off  # noqa: E402
import aml_w0_r1_vote_evidence as ev  # noqa: E402

SESSIONS = ["x", "g", "g", "g", "y", "x"]


def _row(sessions: list[str]) -> dict[str, object]:
    return {"served_facts": {"kept": [{"session_id": s, "content": f"window {i} of {s}"} for i, s in enumerate(sessions)]}}


def test_the_vote_arm_shows_the_offline_l2_first_order_and_the_others_served() -> None:
    """Invariant: c9_pv_replay is exactly the order scored offline as ``L2+first`` (lexical weight 2,
    then the session vote's ``first`` layout); c9_a_replay and c9_p_replay keep the served order.

    Red proof: changing `arm_order` to pass the served order (``list(range(len(items)))``) as the
    vote's base instead of the lexical order makes it differ from the L2+first reference and fails
    the PV assertion (the query below gives the lexical step something to move).
    """
    rows = {"A": {"t": _row(SESSIONS)}, "P": {"t": _row(SESSIONS)}}
    rows["P"]["t"]["served_facts"]["kept"][5]["content"] = "parser refactor parser refactor"
    questions = {"t": w0.Question("t", "coding", "u", "refactor the parser", frozenset({"g"}))}
    out = ev.build(rows, questions, 6)[0]
    items = rows["P"]["t"]["served_facts"]["kept"]
    reference = off.session_vote_order(off.lexical_order("refactor the parser", [i["content"] for i in items], 2.0), SESSIONS, "first")
    assert [w["text"] for w in out["arms"]["c9_pv_replay"]] == [items[i]["content"] for i in reference]
    assert [w["text"] for w in out["arms"]["c9_p_replay"]] == [i["content"] for i in items]
    assert [w["text"] for w in out["arms"]["c9_a_replay"]] == [i["content"] for i in rows["A"]["t"]["served_facts"]["kept"]]


def test_each_window_carries_its_rank_session_and_text_hash_and_the_task_its_prompt_hash() -> None:
    """Invariant: windows are cut to k with ranks 1..k, each text_sha256 is the sha256 of its utf-8
    text, and query_sha256 is the sha256 of the task prompt (the replay rejects any mismatch).

    Red proof: changing `windows` to hash ``session_id`` instead of the text fails the hash assertion.
    """
    rows = {"A": {"t": _row(SESSIONS)}, "P": {"t": _row(SESSIONS)}}
    questions = {"t": w0.Question("t", "coding", "u", "fix the bug", frozenset({"g"}))}
    out = ev.build(rows, questions, 3)[0]
    assert out["query_sha256"] == hashlib.sha256(b"fix the bug").hexdigest()
    a = out["arms"]["c9_a_replay"]
    assert [w["rank"] for w in a] == [1, 2, 3]
    assert [w["session_id"] for w in a] == ["x", "g", "g"]
    assert all(w["text_sha256"] == hashlib.sha256(w["text"].encode("utf-8")).hexdigest() for w in a)
