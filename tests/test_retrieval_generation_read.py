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

import pytest

from recall.generation_types import NoActiveGeneration
from recall.observability import METRICS
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
