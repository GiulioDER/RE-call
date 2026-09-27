"""Input and scoring rules of the C9 Coding window check.

Pre-registration: docs/preregistrations/2026-09-24-aml-c9-coding-window-check.md. Red proof,
2026-09-24, each by mutating ``scripts/aml_c9_coding_window_check.py``:

* ``test_event_messages_rejoin_to_the_flattened_session``: dropping ``tool_result`` from the
  per-event fields (iterating ``_TEXT_FIELDS[:2]``) lost the tool output and failed the equality.
* ``test_event_messages_carry_role_and_unix_milliseconds``: returning seconds
  (``int(parsed.timestamp())``) gave 1785488400 and failed the timestamp assertion.
* ``test_first_relevant_rank_is_one_based``: ``enumerate(sessions, start=0)`` returned 1 for the
  second item and failed.

CD-1 additions, red proof 2026-09-27 against the named line, with ``PYTHONDONTWRITEBYTECODE=1``:

* ``test_a_condition_corpus_must_be_an_unchanged_subset``: dropping the ``changed`` comparison
  (``changed = []``) let a session with a different hash through, failing at ``pytest.raises``.
* ``test_returned_cosines_follow_the_returned_order``: returning the dense order
  (``[score for _, score in dense][: len(ids)]``) failed the list equality.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aml_c7_qualification import _render_transcript  # noqa: E402
import pytest  # noqa: E402

from aml_c9_coding_window_check import (  # noqa: E402
    condition_sessions,
    event_messages,
    first_relevant_rank,
    returned_cosines,
)

EVENTS = [
    {"role": "user", "content": "Why is the **July** total short?", "ts": "2026-07-31T09:00:00Z"},
    {
        "role": "assistant",
        "content": "",
        "tool_name": "Read",
        "tool_input": "{\"file_path\": \"incidents/gap.md\"}",
        "tool_result": "delta — 11,415 vs 10,201",
        "ts": "2026-07-31T09:00:40Z",
    },
    {"role": "assistant", "content": "", "ts": "2026-07-31T09:00:41Z"},
    {"role": "assistant", "content": "The consolidation step drops duplicates.", "ts": "2026-07-31T09:01:00Z"},
]


def _write(tmp_path: Path) -> Path:
    path = tmp_path / "p01.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in EVENTS) + "\n", encoding="utf-8")
    return path


def test_event_messages_rejoin_to_the_flattened_session(tmp_path: Path) -> None:
    path = _write(tmp_path)
    messages = event_messages(path)
    assert " ".join(m["content"] for m in messages) == _render_transcript(path)
    assert len(messages) == 3


def test_event_messages_carry_role_and_unix_milliseconds(tmp_path: Path) -> None:
    messages = event_messages(_write(tmp_path))
    assert [m["role"] for m in messages] == ["user", "assistant", "assistant"]
    assert messages[0]["timestamp"] == 1_785_488_400_000
    assert messages[1]["timestamp"] == 1_785_488_440_000


def test_first_relevant_rank_is_one_based() -> None:
    gold = frozenset({"sessions/t/p01.jsonl"})
    assert first_relevant_rank(["sessions/x/p01.jsonl", "sessions/t/p01.jsonl"], gold) == 2
    assert first_relevant_rank(["sessions/x/p01.jsonl"], gold) is None


def test_a_condition_corpus_must_be_an_unchanged_subset() -> None:
    base = {"sessions/a/p01.jsonl": "h1", "sessions/b/p01.jsonl": "h2", "distractors/d001.jsonl": "h3"}
    kept = condition_sessions(base, {"sessions": {"sessions/b/p01.jsonl": "h2", "distractors/d001.jsonl": "h3"}})
    assert kept == ["distractors/d001.jsonl", "sessions/b/p01.jsonl"]
    with pytest.raises(SystemExit, match="not a subset"):
        condition_sessions(base, {"sessions": {"sessions/b/p01.jsonl": "changed"}})
    with pytest.raises(SystemExit, match="not a subset"):
        condition_sessions(base, {"sessions": {"sessions/new/p01.jsonl": "h9"}})


def test_returned_cosines_follow_the_returned_order() -> None:
    dense = [("w1", 0.9), ("w2", 0.8), ("w3", 0.4)]
    assert returned_cosines(["w3", "g1", "w1"], dense) == [0.4, None, 0.9]
