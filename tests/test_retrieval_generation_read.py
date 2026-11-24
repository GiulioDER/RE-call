"""A search names the generation it read, reads it once, and counts a failure to find one.

Since `GenerationStore.generation_id` became the real generation (S1, #852), reading it is a query
for the active pointer unless a snapshot pins it. `_retrieve_trusted` read it before the snapshot
`trusted_search` takes, so:

- a promotion landing between the two reads made one response report `index_generation` from the
  old generation while its hits came from the new one;
- every search paid one extra active-pointer query;
- a tenant with no active generation raised before the `try`, so `recall_retrieval_failed_total`
  did not count it.

The store double below moves its active pointer after the first lookup, which is what a promotion
mid-request does, and the search double enters the store's snapshot exactly as `trusted_search`
does.

Red proof, recorded 2026-10-03 against `recall_mcp/retrieval.py` at `8ab38b4e` (the read at line
211, before the `try`), both tests run on a Linux test host:

- `test_a_search_names_the_generation_it_read_and_reads_it_once` fails at
  `assert retrieval.result["index_generation"] == retrieval.result["searched"]`
  (`'gen-old' == 'gen-new'`);
- `test_no_active_generation_is_counted_as_a_failed_search` fails at the counter assertion
  (`None == 1`): the error escaped before the `try` that counts it.

With `_searched_generation` both pass.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from recall.generation_types import NoActiveGeneration
from recall.observability import METRICS
from recall.security_policy import AccessContext, SourceSecurityPolicy
from recall_mcp.retrieval import _retrieve_trusted


class _PromotingStore:
    """A generation store whose active pointer moves after its first lookup."""

    def __init__(self, *, has_active: bool = True) -> None:
        self.lookups = 0
        self._pinned: str | None = None
        self._has_active = has_active

    def _active(self) -> str:
        if not self._has_active:
            raise NoActiveGeneration("tenant 't' has no active generation")
        self.lookups += 1
        return "gen-old" if self.lookups == 1 else "gen-new"

    @property
    def generation_id(self) -> str:
        return self._pinned or self._active()

    @contextmanager
    def snapshot(self) -> Iterator[str]:
        if self._pinned is not None:
            yield self._pinned
            return
        self._pinned = self._active()
        try:
            yield self._pinned
        finally:
            self._pinned = None


def _search(store, embedder, query, *, index_generation, **_kwargs):
    with store.snapshot() as searched:
        return {"index_generation": index_generation, "searched": searched}


def _run_search(store: _PromotingStore):
    return _retrieve_trusted(
        store,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        "what is the rate limit",
        None,
        5,
        None,
        None,
        env={},
        reranker_builder=lambda *_args, **_kwargs: None,
        admission_factory=lambda _profile: nullcontext(),
        trusted_search_fn=_search,
    )


def test_a_search_names_the_generation_it_read_and_reads_it_once() -> None:
    store = _PromotingStore()
    retrieval = _run_search(store)
    assert retrieval.result["index_generation"] == retrieval.result["searched"]
    assert store.lookups == 1, f"{store.lookups} active-pointer lookups for one search"


def test_no_active_generation_is_counted_as_a_failed_search() -> None:
    METRICS.reset()
    with pytest.raises(NoActiveGeneration):
        _run_search(_PromotingStore(has_active=False))
    counters = METRICS.snapshot()["counters"]
    assert counters.get("recall_retrieval_failed_total{profile=legacy}") == 1


@pytest.mark.parametrize(
    ("query", "configured", "source", "outcome", "expected"),
    [
        ("Fix ops-42", True, None, "ok", ("exact",)),
        ("OPS-42 and OPS-42", True, None, "ok", ("exact",)),
        ("Fix OPS-42", True, None, "file-export", ("exact",)),
        ("Fix OPS-42", True, None, "s3-export", ("exact",)),
        ("Fix OPS-42", False, None, "ok", (None,)),
        ("Fix OPS-42 or OPS-43", True, None, "ok", (None,)),
        ("Fix OPS-42", True, "caller-selected", "ok", ("caller-selected",)),
        ("Review https://elsewhere.example/OPS-42", True, None, "ok", (None,)),
        ("Fix OPS-42", True, None, "missing", ("exact", None)),
        ("Fix OPS-42", True, None, "stale", ("exact", None)),
        ("Fix OPS-42", True, None, "uncertified", ("exact", None)),
        ("Fix OPS-42", True, None, "wrong-source", ("exact", None)),
        ("Fix OPS-42", True, None, "denied", ("exact", None)),
        ("Fix OPS-42", True, None, "foreign-tenant", (None,)),
        ("Fix OPS-42", True, None, "no-policy", (None,)),
    ],
)
def test_exact_jira_routing_keeps_the_trust_and_generation_boundary(
    query, configured, source, outcome, expected,
) -> None:
    """Removing the route in _retrieve_trusted fails the exact-source call assertions."""
    store = _PromotingStore()
    store.tenant = "jira"
    access = AccessContext("member", "other" if outcome == "foreign-tenant" else store.tenant)
    policy = None if outcome == "no-policy" else SourceSecurityPolicy(())
    base = "https://jira.example.org/jira"
    prefix = {"file-export": "file:///index/jira", "s3-export": "s3://staging-bucket/jira"}.get(outcome)
    exact = f"{prefix}/OPS-42.md" if prefix else f"{base}/browse/OPS-42"
    env = {"RECALL_JIRA_EXACT_ID_BASE_URL": base} if configured else {}
    if prefix:
        env["RECALL_JIRA_EXACT_ID_SOURCE_PREFIX"] = prefix
    calls = []

    def guarded_search(_store, _embedder, _query, *, source, **kwargs):
        assert kwargs["index_generation"] == store.generation_id == "gen-old"
        assert kwargs["access_context"] is access
        assert kwargs["security_policy"] is policy
        calls.append("exact" if source == exact else source)
        if source != exact:
            return SimpleNamespace(label="ordinary", abstained=True)
        return SimpleNamespace(
            label="exact",
            abstained=outcome == "missing",
            calibrated=outcome != "uncertified",
            staleness=SimpleNamespace(stale=outcome == "stale"),
            hits=[] if outcome == "denied" else [SimpleNamespace(verdict="ok", chunk=SimpleNamespace(
                source=exact if outcome != "wrong-source" else "other",
            ))],
        )

    result = _retrieve_trusted(
        store, object(), query, source, 5, None, None,
        security_policy=policy, access_context=access,
        env=env,
        reranker_builder=lambda *_args, **_kwargs: None,
        admission_factory=lambda _profile: nullcontext(),
        trusted_search_fn=guarded_search,
    )
    assert tuple(calls) == expected
    assert result.result.label == ("exact" if expected == ("exact",) else "ordinary")
    assert store.lookups == 1
