"""A raiser that KNOWS its failure will repeat must be able to say so, and be believed.

`recall.embeddings._is_transient` classifies by heuristic, and its last resort is substring
matching the exception's rendered text against markers that include the literal ``"429"``. That
text is written for a human, so any wording that happens to contain a marker is read as retryable
and `retry_with_backoff` resends the whole payload.

`CompletionTruncated` is the case this was found on, and it documented itself as "deliberately NOT
transient" while nothing enforced it. The message interpolates the ceiling, so the property held
only for ceilings spelled without a marker. Measured against `_is_transient` before this change:

    max_tokens=16384   False   correct, and the shipped default
    max_tokens=4290    True    retried
    max_tokens=429     True    retried
    max_tokens=1429    True    retried
    max_tokens=42900   True    retried

⚠️ That table is about `_is_transient` ALONE, and an earlier version of this docstring turned it
into an end-to-end cost claim it does not support. No bill was ever paid: `OpenRouterLLM.complete`
passes `is_transient=_classify`, and `benchmarks/mtrag/generation.py` re-raises this type out of
its own loop, so BOTH callers that raise it already billed ONE request rather than four. The 4x is what a
caller using the DEFAULT classifier would pay, which is the hazard being closed here, not a loss
already taken. `DEFAULT_MAX_TOKENS`'s own docstring invites the edit that lands on such a ceiling
("Raise it if that ever happens"), and the next caller of `retry_with_backoff` has no reason to
know it needs a bespoke classifier.

⚠️ **FIXED AT THE CLASSIFIER, NOT IN THE WORDING.** Rewording the message would only relocate the
coincidence, and would leave the same fragility for every other caller of `retry_with_backoff` —
`recall/embeddings.py` itself (two sites) and `recall/truth_extraction/_openai_engine.py`, none of
which pass a custom classifier. `benchmarks/llm.py` already protected ITS path with a
`PERMANENT_ERRORS` tuple consulted by a local `_classify`, which is why this went unnoticed: the
one caller that had a fix was the one everybody looked at.

🔑 The marker is honoured AHEAD OF EVERY HEURISTIC, including the numeric one. An explicit
declaration by the raiser outranks a guess, because the raiser knows something neither the status
nor the text can express: that repeating the call reproduces the failure at full price.
"""

from __future__ import annotations

import pytest

from benchmarks.llm import CompletionTruncated, EmptyCompletion, NoCompletionChoices
from recall.embeddings import NonTransientError, _is_transient, retry_with_backoff


class _Declared(NonTransientError, RuntimeError):
    """A caller's own error, declaring itself permanent while its text screams rate limit."""


class _Undeclared(RuntimeError):
    """The same wording without the marker, so the fallback can be shown still working."""


def test_the_marker_is_believed_over_a_message_that_reads_like_a_rate_limit() -> None:
    """The whole point. `"429"` is a substring of any number containing it."""
    assert not _is_transient(_Declared("failed after 429 tokens"))
    assert _is_transient(_Undeclared("failed after 429 tokens")), (
        "guards the guard: the text fallback still works for anything that has NOT declared itself"
    )


def test_the_marker_outranks_even_a_numeric_status() -> None:
    """⚠️ Ahead of the numeric branch, not merely ahead of the text.

    A numeric status is otherwise decisive here, and rightly so. But a status describes what the
    TRANSPORT saw, while the marker describes what the RAISER knows, and only the raiser can know
    that resending reproduces the failure. A wrapper that carries an upstream 500 alongside its own
    "do not retry" verdict must be believed about its own verdict.
    """

    class _DeclaredWithStatus(NonTransientError, RuntimeError):
        status_code = 500

    assert not _is_transient(_DeclaredWithStatus("upstream exploded, but resending cannot help"))


@pytest.mark.parametrize("ceiling", [16384, 4290, 429, 1429, 42900, 512])
def test_truncation_is_permanent_whatever_the_ceiling_spells(ceiling: int) -> None:
    """The measured defect, closed for every spelling rather than for the lucky ones.

    4290, 429, 1429 and 42900 were all classified transient before this change; 16384 and 512 were
    correct by accident. A property that holds only for some values of a constant the docstring
    invites you to change is not a property.
    """
    exc = CompletionTruncated(
        f"completion hit max_tokens={ceiling} and was cut off. Raise --max-tokens."
    )

    assert not _is_transient(exc)


def test_truncation_is_still_a_runtime_error() -> None:
    """`RuntimeError` stays in the bases, so every existing `except RuntimeError` and every
    `pytest.raises(RuntimeError)` written against this type is unaffected. The marker is added
    ALONGSIDE, not instead."""
    assert issubclass(CompletionTruncated, RuntimeError)
    assert issubclass(CompletionTruncated, NonTransientError)


def test_an_empty_completion_is_permanent_by_declaration_too() -> None:
    """Same argument, same fix. A content filter fires on the PROMPT, which is byte-identical on
    every attempt, so the three extra calls buy the same refusal at the same price.

    `benchmarks/llm.py` already got this right for its own path via `PERMANENT_ERRORS`; the marker
    is what makes it true for a caller using the DEFAULT classifier.

    ⛔ The message here CONTAINS a marker on purpose, and the first version of this arm did not.
    With an ordinary message ("no usable text (finish_reason=content_filter)") the verdict is
    already False from the text fallback alone, so removing the declaration left this test green:
    mutation-proven inert. The declaration is only load-bearing where the text disagrees with it,
    which is the entire reason the marker exists, so that is what has to be asserted. Whether
    today's wording happens to contain a marker is not the point — the marker's job is to make the
    property independent of a wording nobody is watching.
    """
    assert not _is_transient(EmptyCompletion("no usable text; upstream said timeout"))
    assert not _is_transient(EmptyCompletion("no usable text after 429 blocks"))
    assert issubclass(EmptyCompletion, RuntimeError)


def test_no_completion_choices_is_deliberately_left_transient() -> None:
    """⛔ The one that must NOT carry the marker, so the fix cannot be applied by reflex to every
    error in the module. An empty `choices` array is a fault on the provider's side of the wire,
    not a property of the request, so a re-route can serve the same request correctly."""
    assert not issubclass(NoCompletionChoices, NonTransientError)


def test_retry_with_backoff_stops_at_one_attempt_for_a_declared_error() -> None:
    """End to end, because the classifier's verdict is only worth what the retry loop does with it.
    This is what turns four billed requests into one."""
    calls: list[int] = []
    slept: list[float] = []

    def _fn() -> str:
        calls.append(1)
        raise CompletionTruncated("completion hit max_tokens=4290 and was cut off.")

    with pytest.raises(CompletionTruncated):
        retry_with_backoff(_fn, attempts=4, sleep=slept.append)

    assert len(calls) == 1, "a failure guaranteed to repeat must not burn the retry budget"
    assert slept == [], "and must not pay the backoff either"


def test_retry_with_backoff_still_retries_an_undeclared_rate_limit() -> None:
    """Liveness for the arm above: same loop, same fixture, four attempts. Without this, a
    classifier that returned False for everything would satisfy the `len(calls) == 1` assertion."""
    calls: list[int] = []

    def _fn() -> str:
        calls.append(1)
        raise _Undeclared("429 slow down")

    with pytest.raises(_Undeclared):
        retry_with_backoff(_fn, attempts=4, sleep=lambda _: None)

    assert len(calls) == 4


def test_the_marker_check_does_not_beat_the_error_it_is_classifying() -> None:
    """⛔ THE REGRESSION THIS COMMIT INTRODUCED, in the one function built around not doing it.

    `isinstance(exc, NonTransientError)` falls back to reading `exc.__class__` whenever the fast
    type check misses — which is EVERY exception that is not a marker, i.e. all of them today —
    and `__class__` is free to raise. `_is_transient` is called from inside `retry_with_backoff`'s
    `except Exception`, so a raise there does not misclassify: it REPLACES the provider's error.

    Master returns False for these objects. This commit made them raise, and it did so as the
    FIRST statement of a function whose every other line (`_probe`, the guarded `str(exc)`, the
    nested `type(exc).__name__` fallback) exists precisely to prevent that. `tests/
    test_embeddings_retry_classifier.py` already pins the same property for all three of those.

    `issubclass(type(exc), ...)` is the fix: `type()` cannot be intercepted, and
    `type.__subclasscheck__` on a plain metaclass runs no user code.
    """

    class _HostileClass(Exception):
        @property
        def __class__(self) -> type:  # type: ignore[override]
            raise ValueError("hostile __class__")

    class _HostileGetattr(Exception):
        def __getattribute__(self, name: str) -> object:
            if name == "__class__":
                raise RuntimeError("hostile __getattribute__")
            return object.__getattribute__(self, name)

    # Neither may escape, and each must still get a real verdict from the evidence that remains.
    assert _is_transient(_HostileClass.__new__(_HostileClass)) is False
    assert isinstance(_is_transient(_HostileGetattr.__new__(_HostileGetattr)), bool)


def _all_marker_subclasses() -> list[type]:
    """Every marker subclass, TRANSITIVELY. `__subclasses__()` is direct-only, so a subclass of a
    subclass is invisible to it — and `class TruncatedAtCeiling(CompletionTruncated)` carrying a
    429 is exactly the hazard the walk below exists to catch. The direct-only version reported
    green on that, while the class was permanently unretryable."""
    seen: list[type] = []
    work = list(NonTransientError.__subclasses__())
    while work:
        cls = work.pop()
        # ⛔ IDENTITY, not `in`. `cls not in seen` is `==`, and a metaclass overriding `__eq__`
        # makes two distinct subclasses compare equal, so the walk silently drops one of them —
        # along with whatever status it was carrying. Measured: two such classes, one returned.
        if not any(c is cls for c in seen):
            seen.append(cls)
            work.extend(cls.__subclasses__())
    return seen


def test_the_never_raise_property_rests_on_a_plain_metaclass() -> None:
    """`issubclass(type(exc), NonTransientError)` is total ONLY because `NonTransientError` has a
    plain metaclass: CPython then takes the `PyType_CheckExact` fast path and walks `tp_mro` in C,
    consulting nothing on the exception's side. Give the MARKER a metaclass with a
    `__subclasscheck__` and that runs Python again, so the guard can raise once more.

    Nothing pinned that premise: switching the marker to `metaclass=abc.ABCMeta` left every test
    green, and `ABCMeta.__subclasscheck__` runs a `__subclasshook__` that is free to raise.
    """
    assert type(NonTransientError) is type, (
        "the never-raise property of `_is_transient`'s marker check holds only for a plain "
        "metaclass; a `__subclasscheck__` on this class would run Python code again"
    )


def test_no_marker_subclass_smuggles_in_a_retryable_status() -> None:
    """The precedence puts the marker ahead of the numeric status, which is right today because no
    marker subclass carries one. That is an assumption, not a guarantee, and nothing was watching
    it: the day someone mixes the marker into a wrapper that carries an upstream 429, that wrapper
    becomes permanently unretryable and SILENTLY.

    This walks the actual subclasses rather than naming them, so a new one is covered on the day
    it is written.

    ⚠️ SHIPPED MODULES ONLY. `__subclasses__()` reports whatever is currently imported, which
    includes test doubles — `test_the_marker_outranks_even_a_numeric_status` in this very file
    defines a marker subclass carrying `status_code = 500` ON PURPOSE. Walking unfiltered made
    this arm pass alone and fail in a full-suite run, i.e. order-dependent on which test had
    already defined its double. The filter is what makes it a statement about the codebase rather
    than about import order.
    """
    shipped = [
        cls for cls in _all_marker_subclasses()
        if cls.__module__.split(".")[0] in {"recall", "recall_mcp", "benchmarks"}
    ]
    # ⛔ Membership, not mere non-emptiness. `assert shipped` fires only when the filter matches
    # NOTHING, so a partial loss — a subclass moved to another package, or one whose module this
    # file stops importing — stayed green on the strength of the two that remained.
    assert {c.__name__ for c in shipped} >= {"CompletionTruncated", "EmptyCompletion"}, (
        f"the walk lost a known marker subclass; it found {sorted(c.__name__ for c in shipped)}"
    )

    for cls in shipped:
        for spelling in ("status_code", "status", "http_status"):
            assert getattr(cls, spelling, None) is None, (
                f"{cls.__name__} declares itself non-transient AND carries {spelling} on the "
                f"CLASS; the marker would override a status that may well be retryable"
            )
        # ⛔ And on an INSTANCE. `_probe` reads the instance, and every SDK sets its status in
        # `__init__`, so that is where a real wrapper would carry one — invisible to a class-level
        # read. Adding `self.status_code = 429` to `EmptyCompletion.__init__` was green before.
        probe = None
        for args in (("probe",), ("probe", "upstream"), ()):
            try:
                probe = cls(*args)
                break
            except Exception:  # noqa: BLE001 - try the next shape
                continue
        # ⛔ NOT `continue`. The hazard this test names is "someone mixes the marker into a wrapper
        # carrying an upstream 429", and a wrapper is the subclass most likely to take a second
        # constructor argument — so a bare skip abandoned exactly the shape it exists to catch.
        # Measured: a two-argument subclass setting `status_code = 429` was green.
        assert probe is not None, (
            f"{cls.__name__} could not be constructed by this walk, so its instances are "
            f"UNCHECKED; teach the walk its signature rather than skipping it"
        )
        for spelling in ("status_code", "status", "http_status"):
            assert getattr(probe, spelling, None) is None, (
                f"{cls.__name__} instances carry {spelling} behind the marker"
            )


def test_a_hostile_status_VALUE_does_not_beat_the_error_either() -> None:
    """⛔ The same argument ONE INDIRECTION IN, and the door left open when the marker check was
    closed. `_probe` guards READING `status_code`; the value it hands back is still arbitrary
    provider data, and `isinstance(status, int)` reads ITS `__class__`.

    Measured before the fix: all three classifiers raised on this object, including
    `_is_transient`, the one the previous commit had declared total.

    `issubclass(type(status), int)` also keeps int-SUBCLASS semantics, so an `IntEnum` status still
    classifies, which `type(status) is int` would have silently dropped.
    """
    from benchmarks.llm import _classify, is_terminal

    class _HostileValue:
        @property
        def __class__(self) -> type:  # type: ignore[override]
            raise ValueError("hostile status __class__")

    class _CarriesIt(Exception):
        @property
        def status_code(self) -> object:
            return _HostileValue()

    exc = _CarriesIt("boom")

    assert _is_transient(exc) is False
    assert is_terminal(exc) is False
    assert _classify(exc) is False


def test_an_int_subclass_status_still_classifies() -> None:
    """Guards the guard: the fix must not become `type(status) is int`, which would stop reading a
    perfectly good `IntEnum` status that providers and wrappers do use."""
    import enum

    class _Code(enum.IntEnum):
        RATE_LIMITED = 429
        UNAUTHORIZED = 401

    assert _is_transient(type("S", (Exception,), {"status_code": _Code.RATE_LIMITED})("x")) is True
    assert _is_transient(type("S", (Exception,), {"status_code": _Code.UNAUTHORIZED})("x")) is False
