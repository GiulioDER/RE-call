from __future__ import annotations

import pytest

from scripts.run_query_construction_batch import _assert_generation, _server_command


PINNED = "gen_clean"


def test_pinned_generation_accepts_construction_and_retrieval_shapes():
    _assert_generation({"generation": {"generation_id": PINNED}}, PINNED)
    _assert_generation({"generation_id": PINNED}, PINNED)
    _assert_generation({"retrieval": {"generation_id": PINNED}}, PINNED)


def test_pinned_generation_rejects_missing_or_different_binding():
    with pytest.raises(RuntimeError, match="generation mismatch"):
        _assert_generation({"generation": {"generation_id": "gen_other"}}, PINNED)
    with pytest.raises(RuntimeError, match="generation mismatch"):
        _assert_generation({"status": "complete"}, PINNED)


def test_server_command_only_sets_a_pin_when_requested():
    _, unpinned = _server_command("memory", "voyage:voyage-4", "/srv/memory", "fast", None)
    _, pinned = _server_command("memory", "voyage:voyage-4", "/srv/memory", "fast", PINNED)

    assert "RECALL_BENCHMARK_PIN=1" in unpinned[-1]
    assert "RECALL_PINNED_GENERATION_ID=" not in unpinned[-1]
    assert "RECALL_PINNED_GENERATION_ID=gen_clean" in pinned[-1]
