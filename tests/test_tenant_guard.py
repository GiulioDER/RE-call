"""`check_tenants_populated` (recall/eval/tenant_guard.py) — pure, no database required.

This is the READER-side twin of the two guards already pinned by
`tests/test_locomo_corpus_postcondition.py`: those catch a corpus that came out too BIG (a
sequential re-index, or a concurrent writer). This one catches the opposite — a tenant a reader
is about to score against that was never built at all, or whose build died partway through and
left it holding zero rows. See `recall/eval/tenant_guard.py`'s module docstring for the incident
that motivated it (a build that died after 1 of 10 conversations, silently).

Deliberately DB-free: the function takes a plain `{tenant: row_count}` mapping, so every test
here constructs one by hand instead of touching Postgres. The end-to-end proof that the two LOCOMO
readers actually WIRE this in — not just that the function works in isolation — lives in
`tests/test_locomo_reader_tenant_guard.py`, which drives `locomo_abstention.run()` and
`locomo_entailment_sweep.run()` themselves against a real (scratch-table) corpus.
"""
from __future__ import annotations

import pytest

from recall.eval.tenant_guard import check_tenants_populated

#: Substrings from the two EXISTING guards in recall/eval/locomo.py (read there directly —
#: index_conversation's pre-check and post-condition, ~lines 244 and 267). check_tenants_populated
#: must never emit either, and neither may ever emit ITS token ("UNBUILT-TENANT") — otherwise a
#: test (or an operator) cannot tell which of the three guards actually fired.
PRECHECK_TOKEN = "already holds"
POSTCONDITION_TOKEN = "CONCURRENTLY"
THIS_GUARD_TOKEN = "UNBUILT-TENANT"


def test_all_populated_tenants_pass_untouched() -> None:
    """The healthy path: every tenant this run would iterate has rows. Must not raise."""
    counts = {"locomo-1": 500, "locomo-2": 612, "locomo-3": 480}
    check_tenants_populated(counts, table="locomo_chunks")  # no exception


def test_no_tenants_at_all_does_not_raise() -> None:
    """An empty mapping (e.g. `--conversations 0`) has nothing to refuse — that is not this
    guard's problem, and it must not invent a failure out of nothing to check."""
    check_tenants_populated({}, table="locomo_chunks")


def test_single_empty_tenant_raises_with_the_unique_token() -> None:
    counts = {"locomo-1": 500, "locomo-2": 0, "locomo-3": 480}
    with pytest.raises(RuntimeError, match=THIS_GUARD_TOKEN) as exc:
        check_tenants_populated(counts, table="locomo_chunks")
    assert "locomo-2" in str(exc.value)


def test_collects_every_empty_tenant_not_just_the_first() -> None:
    """Failing on the first empty tenant would hide how much of the corpus is actually missing —
    the exact failure mode the task describes ("1 of 10 present" is the fact that matters). All
    three empty tenants below must be NAMED in the one error raised, not just tenant 'b'."""
    counts = {"a": 100, "b": 0, "c": 0, "d": 50, "e": 0}
    with pytest.raises(RuntimeError) as exc:
        check_tenants_populated(counts, table="locomo_chunks")
    message = str(exc.value)
    assert "b" in message and "c" in message and "e" in message, (
        f"all three empty tenants must be named together; got: {message!r}"
    )
    # And the two POPULATED tenants must NOT be reported as missing.
    assert "'a'" not in message
    assert "'d'" not in message


def test_reports_present_over_total_the_1_of_10_case() -> None:
    """The task's own motivating scenario: a build died after 1 of 10 conversations. The message
    must say 1/10, not just list ten tenant names for an operator to count by hand."""
    counts = {f"locomo-conv{i}": (419 if i == 0 else 0) for i in range(10)}
    with pytest.raises(RuntimeError) as exc:
        check_tenants_populated(counts, table="locomo_chunks")
    message = str(exc.value)
    assert "1/10" in message, f"expected a '1/10 present' style count in: {message!r}"


def test_message_names_the_table() -> None:
    counts = {"locomo-1": 0}
    with pytest.raises(RuntimeError, match="scratch_table_xyz"):
        check_tenants_populated(counts, table="scratch_table_xyz")


def test_message_tells_the_operator_how_to_fix_it() -> None:
    """Not just 'something is wrong' — the rebuild command, naming the actual builder module."""
    with pytest.raises(RuntimeError) as exc:
        check_tenants_populated({"locomo-1": 0}, table="locomo_chunks")
    message = str(exc.value)
    assert "recall.eval.locomo" in message
    assert "python -m recall.eval.locomo" in message


def test_only_the_given_mapping_is_ever_consulted() -> None:
    """Documents the scoping contract directly: this function has no concept of a full dataset,
    a `--conversations` flag, or a hardcoded roster size. A caller that scoped `counts` down to
    3 tenants (e.g. a `--conversations 3` run) gets checked against exactly those 3 — nothing
    about a LARGER dataset those 3 came from can make this call raise or not raise differently.
    """
    limited_run = {"locomo-conv0": 500, "locomo-conv1": 612, "locomo-conv2": 480}
    check_tenants_populated(limited_run, table="locomo_chunks")  # no exception: all 3 populated

    # The same 3, now standing in for a scenario where conversations OUTSIDE this run's slice
    # (conv3..conv9) are empty in the underlying table — but this call never even sees them.
    still_limited_run = dict(limited_run)
    check_tenants_populated(still_limited_run, table="locomo_chunks")  # still no exception


# --- Discrimination: prove this guard's token cannot be confused for the other two guards' ---


def test_this_guards_message_does_not_contain_the_precheck_token() -> None:
    with pytest.raises(RuntimeError) as exc:
        check_tenants_populated({"locomo-1": 0}, table="locomo_chunks")
    assert PRECHECK_TOKEN not in str(exc.value)


def test_this_guards_message_does_not_contain_the_postcondition_token() -> None:
    with pytest.raises(RuntimeError) as exc:
        check_tenants_populated({"locomo-1": 0}, table="locomo_chunks")
    assert POSTCONDITION_TOKEN not in str(exc.value)


def test_unique_token_is_actually_present_when_it_fires() -> None:
    # Belt-and-suspenders alongside the pytest.raises(match=...) tests above: spelled out as an
    # explicit `in` check so a future refactor that renames the token cannot pass by accident
    # (a loosened `match=` regex would; `in` on the literal string will not).
    with pytest.raises(RuntimeError) as exc:
        check_tenants_populated({"x": 0}, table="t")
    assert THIS_GUARD_TOKEN in str(exc.value)
