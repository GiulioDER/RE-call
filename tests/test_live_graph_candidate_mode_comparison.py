from __future__ import annotations

from scripts.run_live_graph_candidate_mode_comparison import ARM_CONFIGS, _arm_order
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
