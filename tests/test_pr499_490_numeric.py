"""Properties behind NUM-001 and NUM-002, proved red on addaa974d remotely.

The audit's CCA property artifacts contain the original falsifying examples.
These tests retain those relations with Hypothesis alone, without requiring the
auditor package in the product's test environment.
"""

import sys
from types import SimpleNamespace
from unittest.mock import patch

from hypothesis import given, settings, strategies as st

from benchmarks import operational
from benchmarks.evidence_tokens import PinnedReaderTokenizer, truncate_evidence_context


def _reported_rss(value, platform):
    resource = SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _: SimpleNamespace(ru_maxrss=value))
    with (
        patch.dict(sys.modules, {"resource": resource}),
        patch.object(sys, "platform", platform),
        patch.object(operational.gc, "collect", lambda: 0),
    ):
        return operational.measure_staged_indexing(lambda: None, lambda: None, lambda: None, lambda: None).rss_bytes


@settings(derandomize=True, max_examples=20, deadline=None)
@given(kib=st.integers(9_999_998, 10_000_001))
def test_linux_rss_units_do_not_change_with_magnitude(kib):
    """NUM-001: each Linux KiB always contributes 1024 reported bytes."""
    assert _reported_rss(kib, "linux") == kib * 1024
    assert _reported_rss(kib + 1, "linux") > _reported_rss(kib, "linux")


@settings(derandomize=True, max_examples=20, deadline=None)
@given(byte_count=st.integers(1, 256))
def test_apple_rss_already_uses_bytes(byte_count):
    """NUM-001: small Apple measurements must not be multiplied by 1024."""
    assert _reported_rss(byte_count, "darwin") == byte_count


@settings(derandomize=True, max_examples=40, deadline=None)
@given(budget=st.integers(1, 127))
def test_every_accepted_budget_bounds_rendered_evidence(budget):
    """NUM-002: an impossible wrapper budget is refused, never exceeded."""
    tokenizer = PinnedReaderTokenizer()
    try:
        context = truncate_evidence_context("A fixed nonempty evidence context.", budget, tokenizer)
    except ValueError:
        assert tokenizer.count_tokens("<memories>\n\n</memories>") > budget
    else:
        assert tokenizer.count_tokens("<memories>\n" + context + "\n</memories>") <= budget
