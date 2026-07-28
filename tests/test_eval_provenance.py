"""The block every result artifact embeds so it can be checked later.

No result JSON in results/ records the corpus it was measured against. That makes the
2026-07-27 doubled-corpus failure undetectable retroactively on every published number,
including postfix_pool20.json and postfix_abstention.json.

Pure functions only — this must be testable without a database, because it is the part that
has to work on every runner.
"""
from __future__ import annotations

from recall.eval.provenance import provenance_block


def test_block_carries_the_row_count_and_where_it_came_from():
    b = provenance_block(corpus_rows=5882, table="locomo_chunks", tenants=["locomo-conv-26"])
    assert b["corpus_rows"] == 5882
    assert b["table"] == "locomo_chunks"
    assert b["tenants"] == ["locomo-conv-26"]


def test_git_sha_is_present_so_a_result_names_the_tree_that_made_it():
    """Absent a repo it must degrade to None, never to a wrong or invented sha."""
    b = provenance_block(corpus_rows=1, table="t", tenants=[])
    assert "git_sha" in b
    assert b["git_sha"] is None or isinstance(b["git_sha"], str)


def test_tenants_are_sorted_so_two_runs_of_one_config_produce_equal_blocks():
    """Dict ordering must not make a diff of two identical runs look like a change."""
    a = provenance_block(corpus_rows=2, table="t", tenants=["b", "a"])
    b = provenance_block(corpus_rows=2, table="t", tenants=["a", "b"])
    assert a["tenants"] == b["tenants"] == ["a", "b"]


def test_a_zero_row_corpus_is_representable_not_swallowed():
    """0 is a real, alarming value. It must not be dropped as falsy."""
    b = provenance_block(corpus_rows=0, table="t", tenants=[])
    assert b["corpus_rows"] == 0


def test_every_locomo_runner_embeds_the_provenance_block():
    """A runner that writes a result JSON must embed the block.

    Checked by import rather than by running the benchmark: each run costs tens of minutes, so
    a test that ran one would never be run. This catches the realistic regression — a new
    runner, or a refactor that drops the call — which is exactly how the count failed to reach
    #103's artifacts in the first place.
    """
    import inspect

    from recall.eval import locomo, locomo_abstention, locomo_entailment_sweep

    for mod in (locomo, locomo_abstention, locomo_entailment_sweep):
        src = inspect.getsource(mod)
        assert "provenance_block(" in src, (
            f"{mod.__name__} writes a result artifact but does not embed provenance_block — "
            f"its output could not be told apart from a doubled-corpus run"
        )
