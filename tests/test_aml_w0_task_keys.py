"""`scripts/aml_w0_task_keys.py`: the parts of session task keys stage 1 that decide a number.

Each test states the invariant and the mutation it was watched to fail on (the red proof). Nothing
here calls a model, embeds, or reaches a network.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aml_w0_r1_offline as off  # noqa: E402
import aml_w0_task_keys as tk  # noqa: E402

SESSIONS = ["x", "g", "g", "g", "y", "x"]


def test_weight_zero_is_exactly_the_session_vote_first_layout() -> None:
    """Invariant: with w 0 the key leg is inert, so the layout equals the vote's ``first`` layout.

    Red proof: changing `keyed_order` to keep each session's LAST window (``best[sessions[index]] =
    index``) instead of its best one gives a different head and fails the assertion. (Dropping the
    vote term is invisible at w 0; the w 1 assertion of the weight test catches that.)
    """
    scores = {"y": 0.9, "x": 0.5, "g": 0.1}
    assert tk.keyed_order(list(range(6)), SESSIONS, scores, 0.0) == off.session_vote_order(list(range(6)), SESSIONS, "first")


def test_key_only_ranks_sessions_by_key_score_and_unscored_sessions_last() -> None:
    """Invariant: Konly leads with the best-scored session's best window; a session with no key
    score goes last among the head; the rest follow in base order.

    Red proof: changing the default for a missing score from ``-math.inf`` to ``math.inf`` puts the
    unscored session x first and fails the assertion.
    """
    scores = {"y": 0.9, "g": 0.1}
    assert tk.keyed_order(list(range(6)), SESSIONS, scores, None) == [4, 1, 0, 2, 3, 5]


def test_the_key_weight_moves_a_session_only_when_large_enough() -> None:
    """Invariant: g leads the vote (rank 1) and y trails (rank 3); with y first by key, w 1 keeps
    g first (1/61 + 1/62 against 1/63 + 1/61) and w 2 puts y first (1/63 + 2/61 against 1/61 + 2/62).

    Red proof: changing ``w / (RRF_K + key_rank[s])`` to ``1 / (RRF_K + key_rank[s])`` makes w 2
    behave like w 1 and fails the second assertion; dropping the vote term from the fused score
    puts y first at w 1 and fails the first.
    """
    scores = {"y": 0.9, "g": 0.5, "x": 0.1}
    assert tk.keyed_order(list(range(6)), SESSIONS, scores, 1.0)[0] == 1
    assert tk.keyed_order(list(range(6)), SESSIONS, scores, 2.0)[0] == 4


def test_generated_keys_need_three_to_five_tasks_and_the_files_line() -> None:
    """Invariant: a valid reply gives each task plus the files line as keys; two tasks, six tasks or a
    missing files line is refused.

    Red proof: changing ``3 <= len(tasks) <= 5`` to ``1 <= len(tasks) <= 5`` accepts two tasks and
    fails the first ``pytest.raises``.
    """
    ok = json.dumps({"tasks": ["a", "b", "c"], "files_and_outcome": "repo x, file y, fixed"})
    assert tk.validate_keys(ok) == ["a", "b", "c", "repo x, file y, fixed"]
    with pytest.raises(ValueError):
        tk.validate_keys(json.dumps({"tasks": ["a", "b"], "files_and_outcome": "f"}))
    with pytest.raises(ValueError):
        tk.validate_keys(json.dumps({"tasks": ["a"] * 6, "files_and_outcome": "f"}))
    with pytest.raises(ValueError):
        tk.validate_keys(json.dumps({"tasks": ["a", "b", "c"]}))


def test_key_score_is_the_maximum_cosine_over_a_sessions_keys() -> None:
    """Invariant: a session scores its best-matching key, not the mean or the first.

    Red proof: changing ``max(...)`` to ``min(...)`` in `key_scores` scores s1 by its orthogonal key
    (0.0) and fails the assertion.
    """
    out = tk.key_scores({"t": [1.0, 0.0]}, {"s1": [[0.0, 1.0], [1.0, 0.0]], "s2": [[1.0, 1.0]]})
    assert out["t"]["s1"] == pytest.approx(1.0)
    assert out["t"]["s2"] == pytest.approx(2 ** -0.5)


def test_user_keys_take_only_the_users_messages() -> None:
    """Invariant: key source U is the session's user messages only, never the agent's.

    Red proof: dropping ``m.get("role") == "user" and`` from `user_keys` includes the assistant
    message and fails the assertion.
    """
    messages = [{"role": "user", "content": "fix the parser"}, {"role": "assistant", "content": "done"}, {"role": "user", "content": " "}]
    assert tk.user_keys(messages) == ["fix the parser"]


def test_chunked_embedding_keeps_order_and_retries_a_rate_limited_chunk() -> None:
    """Invariant: vectors come back in input order across chunks, and a chunk whose first call
    fails is retried after a pause rather than dropped or reordered.

    Red proof: changing ``out.extend(...)`` to ``out[:0] = [...]`` (prepending each chunk) reverses
    the chunk order and fails the order assertion.
    """
    calls: list[list[str]] = []
    pauses: list[float] = []

    def fake(part: list[str]) -> list[list[float]]:
        calls.append(part)
        if len(calls) == 2:
            raise RuntimeError("429 Model busy")
        return [[float(t)] for t in part]

    out = tk.embed_chunked(fake, [str(i) for i in range(5)], chunk=2, pause_s=1.0, sleep=pauses.append)
    assert out == [[0.0], [1.0], [2.0], [3.0], [4.0]]
    assert pauses == [1.0]


def test_the_verdicts_read_each_rule_from_its_own_contrast() -> None:
    """Invariant: help reads G(P) vs P above 0, add reads G(P) vs L2+first(P) above 0, close reads
    G(P) vs A above -0.05.

    Red proof: changing keys_add_to_vote to read ``LOO-G(P)-vs-P`` makes it True (0.10) and fails the
    assertion.
    """
    v = tk.verdicts({
        "LOO-G(P)-vs-P": {"ci95_low": 0.10},
        "LOO-G(P)-vs-L2first(P)": {"ci95_low": -0.02},
        "LOO-G(P)-vs-A": {"ci95_low": -0.04},
    })
    assert v == {"keys_help_proxy": True, "keys_add_to_vote": False, "keys_close_gap": True}
