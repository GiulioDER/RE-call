"""One time budget per Search, so a slow provider costs a stage, never the answer past the cut.

A Search waits on up to four provider calls in a row: the query embedding, the Voyage rerank, the
graph sidecar's embedding and the image leg's embedding. Each had only its own client timeout (60 s
for a text embedding, with up to three attempts and an honoured ``Retry-After``; 180 s for an image
query), so one slow provider took a Search past the ~100 s cut in front of the service, and the
client lost the answer while the server went on working.

`HostedService.search` opens a budget of `SEARCH_BUDGET_SECONDS`. Each stage then takes what is left:

* the query embedding (`DeadlineEmbedder`) waits at most the remaining budget less
  `RESPONSE_RESERVE_SECONDS`, in its own small thread pool, and raises `SearchDeadlineExceeded`
  when that runs out; the service answers 503 with a short ``Retry-After``, which the client resends;
* the rerank waits at most its own timeout or what is left (less the reserve), in its own small
  pool, and is not tried below a few seconds; a miss serves the fused order, as on any rerank
  failure;
* the image legs wait at most what is left (less the reserve, capped per leg); a query with text is
  then served its text ranking, a query without text fails as before.

Outside a Search (an Add, a readiness probe) no budget is open, and every call runs exactly as
before. The budget is a `ContextVar`, so it follows the Search into the worker threads
`asyncio.to_thread` starts (they run in a copy of the caller's context) and never leaks to another
request.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from typing import Any, Callable, Iterator, TypeVar

#: The whole Search, under the ~100 s origin cut with room for the response write.
SEARCH_BUDGET_SECONDS = 90.0
#: Kept back from every provider stage for the database reads, rendering and the response.
RESPONSE_RESERVE_SECONDS = 10.0
#: Threads that may wait on query embeddings at once. A call abandoned at the deadline keeps its
#: thread until the provider answers or its own client timeout ends it; the pool keeps those
#: threads apart from the shared executor every Add and Search step runs in.
QUERY_EMBED_THREADS = 16

_deadline: ContextVar[float | None] = ContextVar("recall_aml_search_deadline", default=None)
_query_pool = ThreadPoolExecutor(
    max_workers=QUERY_EMBED_THREADS, thread_name_prefix="recall-search-embed"
)
#: The rerank waits here (`bounded_call`), apart from the query embeddings.
_stage_pool = ThreadPoolExecutor(
    max_workers=QUERY_EMBED_THREADS, thread_name_prefix="recall-search-stage"
)
_T = TypeVar("_T")


class SearchDeadlineExceeded(RuntimeError):
    """A Search ran out of its time budget before it could rank anything."""


@contextmanager
def search_budget(seconds: float | None = None) -> Iterator[None]:
    """Open a budget for the Search running in this context (`SEARCH_BUDGET_SECONDS` by default)."""
    budget = SEARCH_BUDGET_SECONDS if seconds is None else seconds
    token = _deadline.set(time.monotonic() + budget)
    try:
        yield
    finally:
        _deadline.reset(token)


def remaining() -> float | None:
    """Seconds left in this Search's budget, or None outside a Search."""
    deadline = _deadline.get()
    return None if deadline is None else deadline - time.monotonic()


def stage_budget(cap: float | None = None) -> float | None:
    """What a provider stage may wait now: the budget left less the reserve, at most ``cap``.

    None outside a Search (no limit beyond the stage's own). Never negative.
    """
    left = remaining()
    if left is None:
        return cap
    budget = max(0.0, left - RESPONSE_RESERVE_SECONDS)
    return budget if cap is None else min(cap, budget)


def backoff_before(deadline: float, next_attempt_seconds: float) -> Callable[[float], None]:
    """A retry backoff for a stage that must end by ``deadline`` (a `time.monotonic` value).

    It sleeps only when the backoff and one more attempt of ``next_attempt_seconds`` still fit
    before the deadline, and raises otherwise, so a provider's long ``Retry-After`` ends the stage
    at once instead of holding the thread past it and resending after the Search has answered.
    """

    def sleep(seconds: float) -> None:
        if time.monotonic() + seconds + next_attempt_seconds > deadline:
            raise TimeoutError("the stage ends before the retry backoff and one more attempt do")
        time.sleep(seconds)

    return sleep


def bounded_call(fn: Callable[[], _T], wait: float) -> _T:
    """Run ``fn`` in its own small pool and give up after ``wait`` seconds (TimeoutError).

    For a stage whose client cannot bound its own wall time (connect retries multiply a per-request
    timeout). A call still queued at the deadline is cancelled; a running one ends on its own.
    """
    future = _stage_pool.submit(copy_context().run, fn)
    try:
        return future.result(timeout=wait)
    except FutureTimeout as exc:
        future.cancel()
        raise TimeoutError(f"the stage did not answer within {wait:.1f} s") from exc


class DeadlineEmbedder:
    """Bound an embedder's query embedding by the open Search budget; delegate everything else.

    Passages (an Add) never go through the budget. A query outside a Search runs inline, as before.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def embed_query(self, text: str) -> list[float]:
        from recall.embeddings import embed_query

        wait = stage_budget()
        if wait is None:
            return embed_query(self._inner, text)
        if wait <= 0:
            raise SearchDeadlineExceeded("no time left in the Search budget for the query embedding")
        future = _query_pool.submit(copy_context().run, embed_query, self._inner, text)
        try:
            return future.result(timeout=wait)
        except FutureTimeout as exc:
            raise SearchDeadlineExceeded(
                f"the query embedding did not answer within {wait:.1f} s of the Search budget"
            ) from exc
