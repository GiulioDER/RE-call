"""Offline invariants of the gpt-4o-mini reasoning arms for C8 atomic rescue.

Red proofs (2026-09-22, each mutation applied to ``scripts/c8_atomizer_reasoning.py`` and
reverted):

* ``test_only_grounded_bounded_facts_survive`` failed when ``ground_facts`` stopped marking an
  ungrounded quote as rejected.
* ``test_rejected_and_duplicate_facts_never_become_views`` failed when ``llm_views`` skipped the
  ``rejected`` check.
* ``test_atom_requests_are_pinned_and_strict`` failed when the atoms payload dropped
  ``response_format``.
* ``test_decomposition_is_truncated_to_three_subqueries`` failed when the ``[:MAX_SUBQUERIES]``
  slice was removed.
* ``test_a_refused_gate_leaves_the_c8_ranking_identical_to_off`` failed when ``gated_replay``
  fused the unmodified dense list even after an admitted rescue (``ranked`` replaced by
  ``dense``), through its admitted-case assertion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from recall.atomic_rescue import clear_atomic_rescue_artifact_cache  # noqa: E402
from scripts import c8_atomizer_reasoning as reasoning  # noqa: E402
from scripts import c8_atomizer_reference as ref  # noqa: E402


WINDOW = "we ran pytest, then the ledger.py migration failed on version 0017 again after the retry."


def test_only_grounded_bounded_facts_survive() -> None:
    facts = [
        {"statement": "The ledger.py migration failed on version 0017.", "quote": "ledger.py migration failed on version 0017"},
        {"statement": "The migration of ledger.py broke.", "quote": "the migration of ledger.py broke"},
        {"statement": " ".join(["word"] * 31), "quote": "we ran pytest"},
        *({"statement": f"extra {index}", "quote": "we ran pytest"} for index in range(10)),
    ]
    grounded = reasoning.ground_facts(WINDOW, facts)
    assert len(grounded) == reasoning.MAX_FACTS
    assert [fact["rejected"] for fact in grounded[:3]] == [None, "ungrounded", "statement_length"]
    start, end = grounded[0]["span"]
    assert " ".join(WINDOW.split()[start:end]) == "ledger.py migration failed on version 0017"


def test_rejected_and_duplicate_facts_never_become_views(tmp_path: Path) -> None:
    windows = ref.build_windows({"sessions/t0/s.jsonl": " ".join(f"w{index}" for index in range(300))})
    atoms = tmp_path / "atoms.jsonl"
    rows = [
        {
            "chunk_id": windows[1].chunk.id,
            "session": windows[1].session,
            "segment": 1,
            "malformed": False,
            "facts": [
                {"statement": "kept one", "quote": "q", "span": [0, 3], "rejected": None},
                {"statement": "kept one", "quote": "q", "span": [0, 3], "rejected": None},
                {"statement": "dropped", "quote": "q", "span": None, "rejected": "ungrounded"},
            ],
        }
    ]
    atoms.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    texts, metadata = reasoning.llm_views(windows, atoms)
    assert texts == ["kept one"]
    assert metadata == [
        {"chunk_id": windows[1].chunk.id, "source": windows[1].session, "parent_ordinal": 1, "view_ordinal": 0}
    ]


def _response(content: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": reasoning.MODEL,
        "provider": "OpenAI",
        "usage": {"cost": 0.0001},
        "choices": [{"message": {"content": json.dumps(content)}}],
    }


def test_atom_requests_are_pinned_and_strict(tmp_path: Path) -> None:
    windows = ref.build_windows({"sessions/t0/s.jsonl": WINDOW})
    seen: list[dict[str, Any]] = []

    def call(payload: dict[str, Any]) -> dict[str, Any]:
        seen.append(payload)
        return _response({"facts": [{"statement": "pytest ran first.", "quote": "we ran pytest"}]})

    summary = reasoning.generate_atoms(windows, tmp_path / "atoms.jsonl", call=call, workers=1)
    assert summary["facts_grounded"] == 1 and summary["grounding_rate"] == 1.0
    payload = seen[0]
    assert payload["model"] == "openai/gpt-4o-mini-2024-07-18"
    assert payload["provider"] == {"order": ["openai"], "allow_fallbacks": False}
    assert payload.get("response_format", {}).get("json_schema", {}).get("strict") is True
    assert json.loads(payload["messages"][1]["content"]) == {"WINDOW": windows[0].chunk.text}


def test_decomposition_is_truncated_to_three_subqueries(tmp_path: Path) -> None:
    def call(payload: dict[str, Any]) -> dict[str, Any]:
        return _response({"subqueries": ["a b", " ", "c d", "e f", "g h"]})

    summary = reasoning.generate_decompositions([("q1", "question")], tmp_path / "d.jsonl", call=call, workers=1)
    row = json.loads((tmp_path / "d.jsonl").read_text().splitlines()[0])
    assert row["subqueries"] == ["a b", "c d", "e f"]
    assert summary["subqueries_mean"] == 3


def test_budget_refuses_further_calls(tmp_path: Path) -> None:
    windows = ref.build_windows({f"sessions/t{i}/s.jsonl": WINDOW for i in range(5)})
    calls = 0

    def call(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _response({"facts": []})

    with pytest.raises(RuntimeError, match="budget exhausted"):
        reasoning.generate_atoms(windows, tmp_path / "a.jsonl", call=call, budget_usd=0.00025, workers=1)
    assert calls == 3


def _unit(values: list[float]) -> Any:
    array = np.asarray(values, dtype=np.float32)
    return array / np.linalg.norm(array)


def test_a_refused_gate_leaves_the_c8_ranking_identical_to_off(tmp_path: Path) -> None:
    clear_atomic_rescue_artifact_cache()
    windows = ref.build_windows({f"sessions/t{i}/s.jsonl": " ".join(f"s{i}w{j}" for j in range(100)) for i in range(8)})
    count = len(windows)
    dim = count + 1
    matrix = np.stack([_unit([1.0 if c == r else 0.0 for c in range(dim)]) for r in range(count)])
    query = _unit([1.0 - 0.05 * c if c < count else 0.0 for c in range(dim)])
    target = windows[-1]
    view_rows = [{"chunk_id": target.chunk.id, "source": target.session, "parent_ordinal": 0, "view_ordinal": 0}]

    def artifact(name: str, vector: Any) -> Any:
        return reasoning.write_artifact(name, ["v"], view_rows, np.stack([vector]), windows, tmp_path)

    weak = _unit([0.05] + [0.0] * (count - 2) + [0.05, 1.0])
    strong = _unit([1.0] + [0.0] * (count - 2) + [1.0, 0.0])
    off = ref.replay(ref.Query("q", "probe", "", frozenset(), frozenset()), query, matrix, windows, [], None)
    refused = reasoning.gated_replay(query, [], matrix, windows, [], artifact("weak", weak))
    assert refused.rescued is None and refused.fallback is False
    assert refused.ranked == off.ranked
    admitted = reasoning.gated_replay(query, [], matrix, windows, [], artifact("strong", strong))
    assert admitted.rescued == target.chunk.id
    assert admitted.ranked.index(target.chunk.id) == 5
