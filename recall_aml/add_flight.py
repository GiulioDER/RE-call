"""Answer a slow Add before the tunnel does, and let its retry join the work already running.

AML reaches the official service through a Cloudflare tunnel, and Cloudflare answers 524 to any
request whose origin has not sent a response within about 100 seconds. AML retries an Add 524,
but a 524 carries no ``Retry-After``, so AML waits its 60 second default first. A client that
goes away does not cancel a Starlette handler, so the Add itself kept its place in the Add queue,
ran, and recorded its receipt. The retry then queued for an Add slot of its own, behind every
compile already waiting, and when it reached one it either read that receipt or waited on the
tenant lock behind the original, holding the slot idle all the while. The Textual Full of
2026-09-25 to 27 resent about 3,340 Adds, each answered from its receipt; that fits this cause,
but the tunnel's own log was not kept, so it is not confirmed.

With a response budget set (``HostedSettings.add_response_budget_seconds``), the app instead:

* runs each Add as a *flight* keyed by tenant, request id and payload fingerprint, so a resend
  of the same request joins the work already in progress instead of queueing a second copy;
* answers 503 with ``Retry-After: 1`` when the budget runs out first, well inside Cloudflare's
  cut, while the flight carries on; AML honours a ``Retry-After`` up to 60 seconds and retries an
  Add 503 up to 32 times with the same request, so the resend comes back a second later and
  waits on the same flight;
* answers a resend of an Add that already finished from its receipt, without taking a slot.

Nothing here changes what an Add stores or how it is computed: the flight runs the same
``HostedService.add`` under the same Add semaphore, and the service still checks the receipt
under the tenant lock before doing any work.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import logging
from typing import Generic, TypeVar

from recall_aml.identity import canonical_digest, tenant_for
from recall_aml.models import AddRequest


log = logging.getLogger("recall_aml")
#: What a pending Add answers in ``Retry-After``. AML reads a missing header as 60 seconds, and
#: the resend only waits on the flight again, so the shortest valid value wastes the least.
PENDING_RETRY_AFTER_SECONDS = 1

ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class AddIdentity:
    """What makes two Add requests the same request: AML resends the same id and payload."""

    tenant: str
    request_digest: str
    fingerprint: str


def add_identity(request: AddRequest) -> AddIdentity:
    """The flight key, and the fingerprint ``HostedService.add`` checks the receipt against.

    Serialises the whole request, which for a Coding session of several MiB is too slow for the
    event loop; call it in a worker thread.
    """
    return AddIdentity(
        tenant=tenant_for(request.user_id),
        request_digest=canonical_digest(request.request_id),
        fingerprint=canonical_digest(request.model_dump(mode="json")),
    )


class AddFlights(Generic[ResultT]):
    """At most one running Add per identity, awaited by every request that carries it.

    A flight is forgotten the moment it finishes, whether it succeeded or failed: a success has
    left its receipt, which answers any later resend, and a failure stored nothing, so a later
    resend must do the work afresh exactly as it would have without this class.
    """

    def __init__(self) -> None:
        self._running: dict[AddIdentity, asyncio.Task[ResultT]] = {}

    def __len__(self) -> int:
        return len(self._running)

    def running(self, identity: AddIdentity) -> asyncio.Task[ResultT] | None:
        return self._running.get(identity)

    def join_or_start(
        self, identity: AddIdentity, start: Callable[[], Awaitable[ResultT]]
    ) -> tuple[asyncio.Task[ResultT], bool]:
        """The running flight for ``identity`` and True, or a new one started now and False.

        No await happens between the lookup and the insert, so two requests on one event loop
        can never both start a flight for the same identity.
        """
        task = self._running.get(identity)
        if task is not None:
            return task, True

        async def run() -> ResultT:
            return await start()

        task = asyncio.get_running_loop().create_task(run())
        self._running[identity] = task
        task.add_done_callback(lambda done: self._finished(identity, done))
        return task, False

    def _finished(self, identity: AddIdentity, task: asyncio.Task[ResultT]) -> None:
        if self._running.get(identity) is task:
            del self._running[identity]
        if task.cancelled():
            return
        # Retrieving it here also stops asyncio reporting "exception was never retrieved" for a
        # flight whose requests had all been answered pending by the time it failed.
        error = task.exception()
        if error is not None:
            log.warning(
                "hosted_add_flight_failed",
                extra={
                    "tenant_digest": identity.tenant.removeprefix("aml_")[:16],
                    "request_digest": identity.request_digest[:16],
                    "error_class": type(error).__name__,
                },
            )


async def within(task: asyncio.Task[ResultT], seconds: float) -> ResultT | None:
    """Wait up to ``seconds`` for ``task`` without ever cancelling it.

    Its result when it finished (re-raising its exception if it failed), None when the time ran
    out first, so the task's own result must never be None. ``asyncio.wait`` leaves a pending task
    alone on timeout, and if the waiting request is itself cancelled the flight still runs to the
    end for its resend.
    """
    if seconds > 0:
        await asyncio.wait({task}, timeout=seconds)
    return task.result() if task.done() else None
