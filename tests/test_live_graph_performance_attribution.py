"""Tests for the live graph performance attribution artifact."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from scripts import run_live_graph_performance_attribution as runner


def _payload(generation_id: str) -> str:
    return json.dumps(
        {
            "generation_id": generation_id,
            "outcome": "abstained",
            "refusal_reason": "no_answer_provider",
            "trust_state": "trusted",
            "trusted_evidence": {"items": []},
            "diagnostics": {
                "performance": {
                    "spans_ms": {"graph_readiness_check_ms": 4.2},
                    "counters": {"projection_cache_misses": 1},
                    "values": {"graph_readiness": "ready"},
                }
            },
        }
    )


class _FakeClient:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def close(self) -> None:
        pass


def test_attribution_runner_persists_warmup_rows(tmp_path: Path, monkeypatch) -> None:
    """Node warmup-spans-01: the baseline discarded warmup spans entirely.

    The target symbol is ``main`` in ``scripts/run_live_graph_performance_attribution.py``.
    The failure reason is that removing the warmup append leaves no cold graph denominator in
    the artifact, even though the recorded rows still look valid.
    """
    query_path = tmp_path / "queries.json"
    query_path.write_text(
        json.dumps([{"query": "q", "answerable": True, "relevant_ids": []}]),
        encoding="utf-8",
    )
    output_path = tmp_path / "result.json"
    generation_id = "gen-test"

    monkeypatch.setattr(runner, "TTYMCP", _FakeClient)
    monkeypatch.setattr(runner, "_command", lambda *_args: [])
    monkeypatch.setattr(runner, "_initialize", lambda _client: 2)
    monkeypatch.setattr(
        runner,
        "_call_query",
        lambda *_args, **_kwargs: (_payload(generation_id), 12.5),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_live_graph_performance_attribution.py",
            "--query-set",
            str(query_path),
            "--output",
            str(output_path),
            "--generation-id",
            generation_id,
            "--passes",
            "1",
            "--warmup-passes",
            "1",
        ],
    )

    runner.main()

    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(artifact["rows"]) == 2
    assert len(artifact["warmup_rows"]) == 2
    assert {row["phase"] for row in artifact["rows"]} == {"recorded"}
    assert {row["phase"] for row in artifact["warmup_rows"]} == {"warmup"}
    assert artifact["warmup_rows"][0]["server_performance"]["spans_ms"] == {
        "graph_readiness_check_ms": 4.2
    }


def test_attribution_runner_rejects_stale_warmup_generation() -> None:
    """Node warmup-spans-02: a warmup row must obey the same generation pin as scored rows.

    Mutation proof: removing the generation check from ``_attribution_row`` would record a
    stale cold span and make this assertion fail. The failure reason is cross-generation latency
    attribution contaminating the preregistered cold versus warm comparison.
    """
    try:
        runner._attribution_row(
            _payload("gen-stale"),
            1.0,
            query_index=0,
            query={"query": "q"},
            arm="one_hop",
            phase="warmup",
            pass_index=1,
            generation_id="gen-current",
        )
    except RuntimeError as exc:
        assert "pinned generation mismatch" in str(exc)
    else:
        raise AssertionError("Node warmup-spans-02: stale warmup generation was accepted")
