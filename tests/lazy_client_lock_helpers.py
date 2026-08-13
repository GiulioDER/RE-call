"""Instrumentation for a lazily built, lock-guarded SDK client slot.

Two classes in this suite build a third-party client lazily on first use, and `benchmarks.beam.run`
hands ONE instance of each to a `ThreadPoolExecutor(max_workers=8)`: `benchmarks.llm.OpenRouterLLM`
(via the `answerer`/`judge_llm` closures) and `benchmarks.voyage_rerank.VoyageReranker` (via the
single `system` the `_score` closure retrieves through). Both had the same defect, an unguarded
``if self._client is None: self._client = Client(...)``, and both take the same fix. This module
holds the instrumentation once so the two guards cannot drift apart.

Why the guard is not just "assert the lock was held". Five shapes were tried on the
`OpenRouterLLM` side, each forced by review, and four of the five were GREEN against the mutant
that replaced them. They are recorded here because the sequence is the lesson: an assertion about a
lock keeps looking sufficient while testing something strictly weaker than mutual exclusion.

1. The lock is held while the client is CONSTRUCTED. Green under broken double-checked locking,
   where the `is None` test is hoisted into an unlocked fast path: the lock genuinely is held
   during the build, so a build-time assertion sees nothing wrong. 9 runs out of 9.
2. `Lock.locked()` at the read as well. That answers "is anyone holding this object", not "does the
   caller hold the lock that excludes the others". Two mutants keep it true: a mishandled
   ``acquire(blocking=False)`` whose loser falls through into the check-then-set anyway, and
   rebinding the lock attribute per call so it excludes nobody. Hence `OwnerRecordingLock`.
3. Ownership and identity on both halves. Pins each half to A critical section, not to the SAME
   one, so releasing the lock between the check and the build passed it. That is the likeliest
   shape to be written on purpose, because holding a lock across tens of milliseconds of client
   construction looks like exactly the thing to avoid. Hence the acquisition counter.
4. All of the above is spotless SLOT discipline and blind to a client BUILT outside the lock and
   merely published inside it with a re-check. Every thread pays for a client and all but one
   discards theirs, which IS the leak. Hence `built_under_the_lock`, which watches the build.
5. Ownership alone does not close the non-blocking acquire on ONE thread either: a lone thread wins
   a non-blocking acquire and is the rightful owner. What closes it single-threaded is asserting
   that the acquire BLOCKS, which is a genuine requirement of such a guard, since its whole purpose
   is that a thread which loses waits rather than proceeds.

The two instruments are complementary and neither subsumes the other. Build ownership catches the
speculative build; the slot assertions catch the non-blocking acquire and the double-check, which
build ownership cannot see because the lock really is held while the client is constructed. Step 2
above REPLACED the build-time assertion instead of keeping it, and that is exactly what opened
step 4, so both are kept here.

What this cannot prove is stated rather than papered over, and there are two such statements.

A single-threaded test shows the guard is present, blocking, owned by its caller, always the same
object, unbroken between the halves, and covering the construction. It cannot show that mutual
exclusion actually excludes. Only a threaded test measures that, which is why every caller of this
module pairs the two.

And a test only sees a REBUILD whose trigger it reaches. The publish count closes any rebuild
inside a test's horizon, and the write-once tests make several hundred calls so that a "recycle
every N calls" rotation cannot hide behind a horizon of one; both are measured. A rebuild on a
WALL-CLOCK trigger, "this client is an hour old, rotate it", is not closed, and cannot be without
a clock seam in the production classes, which neither has and neither should grow for this. It is
the same defect if it lands: a run long enough to cross the threshold builds and drops a client at
every crossing. Left as a known gap rather than an unstated one.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any


class OwnerRecordingLock:
    """A lock that records which thread holds it and how many times it has been taken.

    `threading.Lock` exposes neither, and `locked()` alone is not the question: see shapes 2, 3
    and 5 in this module's docstring. `owner` closes the mishandled non-blocking acquire under
    contention, `acquisitions` closes the release-between-the-halves shape, and refusing a
    non-blocking acquire outright closes the first of those on a single thread as well.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.owner: int | None = None
        #: Bumped on every successful acquire, so a caller can tell "inside a critical section"
        #: from "inside THE SAME critical section".
        self.acquisitions = 0

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        assert blocking, (
            "the client guard acquired its lock non-blocking; a thread that fails to take it then "
            "falls through into the check-then-set, which is the defect this lock exists for"
        )
        acquired = self._lock.acquire(blocking, timeout)
        if acquired:
            self.owner = threading.get_ident()
            self.acquisitions += 1
        return acquired

    def release(self) -> None:
        # Cleared BEFORE the release: after it, another thread may already own the lock, and
        # clearing then would erase its claim.
        self.owner = None
        self._lock.release()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *_exc: object) -> None:
        self.release()


def built_under_the_lock(holder: list[Any], *, lock_attr: str = "_client_lock") -> Callable[[], None]:
    """A hook for a fake SDK constructor, asserting the CALLER owns the lock while it runs.

    `holder` is a one-slot list rather than a direct closure, because the fake SDK has to be
    installed before the instance it will be asked about exists.
    """

    def _check() -> None:
        # NOTE for the fake that calls this: record the construction BEFORE calling the hook. A
        # fake that appends afterwards makes the count and this assertion one signal rather than
        # two, and production code that wraps its build in `try: ... except Exception:` then
        # erases both at once.
        #
        # An empty holder means the fake fired before the instance was registered, which can only
        # be a build during `__init__`. Reported as itself: folding it into the ownership
        # assertion below would blame a speculative build for what is really an eager one.
        assert holder, (
            "the client was constructed during __init__, before the guard was installed. If this "
            "is an eager build, see the calling test module on why the client stays lazy."
        )
        lock = holder[0].__dict__.get(lock_attr)
        assert lock is not None and lock.owner == threading.get_ident(), (
            "the client was CONSTRUCTED without this thread holding the lock. Building "
            "speculatively outside it and publishing inside leaves the slot discipline spotless "
            "while every thread still pays for a client and all but one discard theirs, which is "
            "the leak this guard exists for"
        )

    return _check


def asserting_lazy_client(
    base: type,
    *,
    slot: str = "_client",
    lock_attr: str = "_client_lock",
    publishes_at_init: bool = False,
) -> Any:
    """A subclass of `base` whose `slot` refuses to be touched unguarded while still unset.

    The real value moves to a different `__dict__` key, so the property is the only way in or out.
    Both halves of the check-then-set are covered: the READ that finds it unset, and the assignment
    that fills it. A read AFTER it is set passes unguarded, deliberately, since that is how a
    caller uses the client and it is sound while the slot is write-once. The setter enforces that
    premise rather than leaving it as a comment.

    `publishes_at_init` opts a class into publishing a real client before the guard is installed.
    `VoyageReranker(client=…)` does exactly that: a documented offline seam, single-threaded, with
    no other thread holding a reference yet. It defaults to FALSE so that a class without such a
    seam, `OpenRouterLLM`, keeps reporting a construction-time publish as the defect it is. Making
    the allowance unconditional would have handed the LLM side a seam it does not have and left
    `built_under_the_lock` as the only net under it.
    """
    assert not slot.startswith("__"), (
        f"slot {slot!r} would be name-mangled by `base`, and a key in a `type()` dict is not, so "
        "the property would be installed under a name nothing touches and instrument nothing"
    )
    storage = f"{slot}__storage"
    seen_key = f"{slot}__unset_seen_in"
    published_key = f"{slot}__published_in"
    at_init_key = f"{lock_attr}__at_init"

    def _standing_down(self: Any) -> bool:
        """True while the guard is not yet installed AND this class has a seam that needs it.

        Both halves of the property consult this, not just the setter. Gating only the setter left
        a construction-time READ of the unset slot allowed for every class, including one with no
        seam at all, which is the kind of asymmetry that drifts into a hole.
        """
        return publishes_at_init and at_init_key not in self.__dict__

    def _assert_guarded(self: Any, action: str) -> OwnerRecordingLock:
        lock = self.__dict__.get(lock_attr)
        at_init = self.__dict__.get(at_init_key)
        # Both absences mean one thing, the guard not being installed yet, and must say so rather
        # than being reported as a different-lock failure.
        assert lock is not None and at_init is not None, (
            f"the client was {action} during __init__, before the guard was installed. If this is "
            "an eager build, see the calling test module on why the client stays lazy; if the "
            "class has a documented construction-time seam, pass publishes_at_init=True."
        )
        assert lock is at_init, (
            f"the client was {action} under a DIFFERENT lock object than the one installed at "
            "construction, which excludes nobody"
        )
        assert lock.owner == threading.get_ident(), (
            f"the client was {action} without this thread holding the lock"
        )
        return lock

    def _get(self: Any) -> Any:
        value = self.__dict__.get(storage)
        if value is None and not _standing_down(self):
            # Half of a check-then-set: two threads can both pass this read. Which acquisition it
            # happened in is recorded so the setter can insist the other half is the same one.
            lock = _assert_guarded(self, "read while still unset")
            self.__dict__[seen_key] = lock.acquisitions
        return value

    def _set(self: Any, value: Any) -> None:
        if value is None:
            # Initialising the slot. A reset of an ALREADY SET slot is a different thing: it breaks
            # the write-once premise that makes an unlocked read of the built client sound, and
            # leaves one thread dereferencing None while another rebuilds. Reached by "drop the
            # client so the next call rebuilds it", which is how a poisoned connection pool is
            # usually handled and is a leak here: one client per failed request.
            assert self.__dict__.get(storage) is None, (
                "the client was reset to None; callers read the built client without the lock, "
                "and that is only sound while the slot is write-once"
            )
        elif _standing_down(self):
            pass  # the documented construction-time seam this class opted into
        else:
            lock = _assert_guarded(self, "published")
            # FIRST, ahead of both clauses below, because both presume this is a first publish and
            # would otherwise answer for it with the wrong cause. A rebuild that publishes over a
            # live client without resetting leaves `seen` stale from the first build, so the
            # same-acquisition comparison fails and blames a released lock that was in fact held
            # throughout. It also makes the canary's `published == 1` unreachable as a failure:
            # publish #2 either follows a reset, which the write-once assertion catches, or leaves
            # `seen` stale, which that comparison catches, and neither says "rebuilt".
            assert self.__dict__.get(published_key, 0) == 0, (
                "the client was REBUILT: a second publish over a slot that already held one. "
                "Nothing was reset, so the write-once assertion cannot see it, and every call "
                "after the first pays for a client and drops the last"
            )
            seen = self.__dict__.get(seen_key)
            # Split from the comparison below rather than folded into it. No recorded read means
            # no read ever found the slot unset, which is a different regression with a different
            # cause: someone concluded the lock made the `is None` check redundant.
            assert seen is not None, (
                "the client was published without any read having found the slot unset: the "
                "check-then-set has no check, so every call builds a client and drops the last"
            )
            assert seen == lock.acquisitions, (
                "the lock was RELEASED between the check and the set. Each half was guarded, but "
                "not by the same critical section, so two threads can interleave between them and "
                "both build a client"
            )
            # A COUNT, not the acquisition number, and recorded for two jobs. It lets the canary
            # prove THIS branch ran: `storage` below is written on every path and `seen_key` by
            # the getter, so neither shows whether the published half is still alive, and it is
            # the only deterministic detector for the released-between-halves mutation, which the
            # barrier test cannot reproduce because the window it holds open is inside the lock
            # and serialises the other workers past their own check. Counting rather than stamping
            # also makes a REBUILD visible whatever triggered it: the write-once assertion above
            # only fires on the reset it is given, so a rebuild on a trigger no test reaches, an
            # age or call-count rotation, slips past it while publishing a second time here.
            self.__dict__[published_key] = self.__dict__.get(published_key, 0) + 1
        self.__dict__[storage] = value

    def _init(self: Any, *args: Any, **kwargs: Any) -> None:
        base.__init__(self, *args, **kwargs)
        # Installed AFTER construction, in the spirit of `_LockAssertingDict` in
        # `test_bench_llm_usage_lock.py`. Until this runs, the slot assertions stand down and
        # `built_under_the_lock` is what covers construction.
        self.__dict__[lock_attr] = OwnerRecordingLock()
        self.__dict__[at_init_key] = self.__dict__[lock_attr]

    def _assert_was_live(self: Any, *, lazily_built: bool = True) -> None:
        """Canary: prove the property above actually intercepted the build that just happened.

        Without this, the whole slot half can go inert and every test still passes. Measured, not
        feared: replacing both branch conditions in `_set` with `if False:` makes the property a
        passthrough and leaves all five tests across both callers green, because they only ever
        observe the client, which a passthrough returns just as happily. The same goes for the
        realistic drift, renaming the production slot while leaving the lock alone: the instrument
        would then be watching an attribute nothing uses, and say nothing about it.

        Each key is written by exactly one branch of the property, so each is absent the moment
        that branch stops running. `lazily_built=False` is for an instance whose client came in
        through a construction-time seam: no read ever found the slot unset there and no publish
        went through the guarded branch, so demanding either would be a false red, and the message
        would accuse the production guard of something the test itself arranged.
        """
        assert self.__dict__.get(storage) is not None, (
            f"the instrumented {slot!r} property never stored a client, so it is not intercepting "
            "and every assertion in this module is inert"
        )
        if not lazily_built:
            return
        assert self.__dict__.get(seen_key) is not None, (
            f"the instrumented {slot!r} property never recorded an unset read, so the "
            "check-then-set assertions never ran"
        )
        published = self.__dict__.get(published_key)
        assert published is not None, (
            f"the instrumented {slot!r} property never took its guarded publish branch, so the "
            "ownership, identity and same-acquisition assertions never ran"
        )
        # Belt to the setter's braces. The setter now catches a rebuild at the moment it happens,
        # with a traceback pointing at the rebuilding call rather than at the end of a test, so
        # this is not expected to be the assertion that fires. It is kept because it costs one
        # comparison and it is the half that survives if the setter's branch ever goes inert.
        assert published == 1, (
            f"{slot!r} was published {published} times: the client was REBUILT. Whatever triggered "
            "it, every thread or call after the first pays for a client and drops the last, which "
            "is the leak this guard exists for"
        )

    return type(
        f"Asserting{base.__name__}",
        (base,),
        {slot: property(_get, _set), "__init__": _init, "assert_guard_was_live": _assert_was_live},
    )
