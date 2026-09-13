from __future__ import annotations

from scripts.run_live_graph_candidate_mode_comparison import ARM_CONFIGS, _arm_order
from scripts.run_live_graph_candidate_headroom import _summarize
from scripts.run_live_tty_graph_precision import _command


def test_graph_candidate_mode_runner_rotates_every_first_mover() -> None:
    """Every arm receives one first position in each complete query block.

    Invariant: arm order rotates by query index while every query still runs every arm exactly
    once. Red proof node ``graph-candidate-arm-order-01`` mutates ``_arm_order`` to return the fixed
    declaration order. The first mover assertion then reports only `off` instead of every arm.
    """
    names = tuple(config["arm"] for config in ARM_CONFIGS)
    orders = [_arm_order(index) for index in range(len(names))]

    assert {order[0] for order in orders} == set(names)
    assert all(set(order) == set(names) and len(order) == len(names) for order in orders)


def test_tty_command_forwards_candidate_mode_and_tail_margin() -> None:
    """The isolated VPS2 process receives both selector controls.

    Red proof node ``graph-candidate-command-01`` removes the candidate mode assignment from the
    remote command. The environment assertion then fails while command construction still
    succeeds.
    """
    command = _command(
        "memory",
        "voyage:voyage-4",
        "/srv/memory",
        "fast",
        "combined",
        "none",
        20260912,
        32,
        0.10,
        "generation-one",
        candidate_mode="linked_tail",
        tail_replacement_margin="0.05",
    )
    remote = command[-1]

    assert "RECALL_GRAPH_FIRST_CANDIDATE_MODE=linked_tail" in remote
    assert "RECALL_GRAPH_TAIL_REPLACEMENT_MARGIN=0.05" in remote


def test_tty_command_can_launch_from_an_isolated_remote_checkout(monkeypatch) -> None:
    """A live experiment must execute the committed candidate code, not serving code.

    Red proof node ``graph-candidate-code-root-01`` runs against the hard coded serving root. The
    expected isolated checkout is absent from the remote command even though command construction
    succeeds.
    """
    code_root = "/home/sentiment/recall-repos/measure-linked-tail-ae1e3543"
    monkeypatch.setenv("RECALL_BENCHMARK_REMOTE_CODE_ROOT", code_root)

    command = _command(
        "memory",
        "voyage:voyage-4",
        "/home/sentiment/recall-repos/memory",
        "fast",
        "combined",
        "none",
        20260912,
        32,
        0.10,
    )
    remote = command[-1]

    assert f"cd {code_root}" in remote
    assert f"PYTHONPATH={code_root}" in remote


def test_tty_command_enables_candidate_audit_only_when_requested() -> None:
    """The private runner explicitly opts into candidate identity diagnostics.

    Invariant: the audit environment flag is absent by default and present only for the headroom
    runner. Red proof node ``graph-candidate-audit-command-01`` accepts the new argument but omits
    its environment assignment. Command construction succeeds and this assertion fails.
    """
    ordinary = _command(
        "memory", "voyage:voyage-4", "/srv/memory", "fast", "combined", "none", 1, 32, 0.10
    )[-1]
    audited = _command(
        "memory",
        "voyage:voyage-4",
        "/srv/memory",
        "fast",
        "combined",
        "none",
        1,
        32,
        0.10,
        "generation-one",
        candidate_mode="linked_tail",
        tail_replacement_margin="0.05",
        benchmark_graph_audit=True,
    )[-1]

    assert "RECALL_BENCHMARK_GRAPH_AUDIT" not in ordinary
    assert "RECALL_BENCHMARK_GRAPH_AUDIT=1" in audited


def test_headroom_summary_separates_available_connected_and_promoted_gold() -> None:
    """The audit keeps retrieval headroom separate from graph connectivity.

    Invariant: gold at raw rank 11 counts as tail headroom, then as connected and promoted only
    when the linked candidate trace says so. Red proof node ``graph-headroom-summary-01`` changes
    the tail lower bound from rank 9 to rank 12. The summary then loses query zero from
    ``tail_headroom_queries`` while the fabricated payload remains valid.
    """
    raw_hits = [
        {
            "chunk_id": f"c{rank}",
            "source": "recall/gold.md" if rank == 11 else f"recall/other-{rank}.md",
            "ordinal": 0,
            "rank": rank,
            "cosine": 1.0 - rank * 0.01,
        }
        for rank in range(1, 21)
    ]
    linked = [
        {
            "chunk_id": "c11",
            "source": "recall/gold.md",
            "ordinal": 0,
            "raw_rank": 11,
            "raw_cosine": 0.89,
            "selected_pre_trust": True,
            "relation_types": ["depends_on"],
        }
    ]

    def payload(*, include_gold: bool, linked_candidates: list[dict[str, object]]) -> str:
        items = [{"source": "recall/gold.md", "ordinal": 0}] if include_gold else []
        return __import__("json").dumps(
            {
                "trusted_evidence": {"items": items},
                "diagnostics": {
                    "performance": {
                        "values": {
                            "graph_benchmark_audit": {
                                "raw_hits": raw_hits,
                                "linked_candidates": linked_candidates,
                            }
                        }
                    }
                },
            }
        )

    query = {"query": "q", "answerable": True, "relevant_ids": ["gold.md:0"]}
    rows = [
        {
            "query_index": 0,
            "query": query,
            "arm": "linked_tail_true",
            "payload": payload(include_gold=True, linked_candidates=linked),
        },
        {
            "query_index": 0,
            "query": query,
            "arm": "linked_tail_removed",
            "payload": payload(include_gold=False, linked_candidates=[]),
        },
    ]

    summary = _summarize(rows)

    assert summary["tail_headroom_queries"] == [0]
    assert summary["connected_gold_queries"] == [0]
    assert summary["gold_promotion_queries"] == [0]
    assert summary["gold_recall_gain_queries"] == [0]
    assert summary["complete_gold_rescue_queries"] == [0]
    assert summary["connected_gold_relation_types"] == {"depends_on": 1}
