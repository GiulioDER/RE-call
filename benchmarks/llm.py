from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Protocol

from recall._chat_content import assistant_text
from recall.embeddings import (
    NonTransientError,
    _is_transient,
    _probe,
    retry_with_backoff,
)
from recall.provider_metadata import ProviderMetadata

#: The injected-LLM seam: (system_prompt, user_prompt) -> completion text. Everything downstream
#: depends on this, not on any SDK, so the pipeline is testable with a plain function.
Completer = Callable[[str, str], str]

#: Output ceiling per request. Sized from measurement, not taste: the longest answerer completion
#: observed across the BEAM and pilot runs is 4,214 tokens (a multi-session plan, gpt-5 reasoning
#: tokens included), against a mean near 850. 16,384 is ~3.9x that observed maximum and still 4x
#: below the model maximum that omitting this reserves.
#:
#: Erring high is deliberate. The failure this constant prevents costs a refused request; the
#: failure it could CAUSE — truncating an answer that is then scored — corrupts a result, so the
#: headroom is bought on the side where being wrong is cheap.
#:
#: It is a CEILING, not a target — nothing is truncated silently, because `CompletionTruncated`
#: fires when a response stops for `length`. Raise it if that ever happens; do not remove it.
DEFAULT_MAX_TOKENS = 16384


class CompletionTruncated(NonTransientError, RuntimeError):
    """A completion stopped because it hit `max_tokens` rather than finishing.

    Raised rather than returned, because the alternative is worse than the error: a truncated
    answer is a plausible-looking string that a judge — human or model — scores as if the system
    had produced it. That is a measurement error introduced by our own configuration, and it is
    indistinguishable from a genuine failure once it lands in a results artifact.

    Deliberately NOT transient: `retry_with_backoff` re-raises immediately instead of paying for
    the same over-long request three more times. The fix is a bigger ceiling, not another attempt.

    That is declared by membership of `PERMANENT_ERRORS` below, not left to the wording here.
    Classifying on the message alone made the property depend on how the ceiling is SPELLED:
    `_is_transient` falls back to substring markers that include the literal "429", and this
    message interpolates `max_tokens`, so 16384 read as permanent (correct) while 4290, 429 and
    1429 all read as rate limits and bought four attempts at a failure guaranteed to repeat.
    `DEFAULT_MAX_TOKENS`'s own docstring invites the edit that lands on such a ceiling.
    """


class EmptyCompletion(NonTransientError, RuntimeError):
    """A completion came back carrying no text at all.

    Distinct from `CompletionTruncated` on purpose, and the distinction is the operator's next
    action. Truncation means the completion hit OUR ceiling, so the fix is a bigger one. Empty text
    is what an OpenAI-compatible provider returns when it filtered the completion
    (`finish_reason == "content_filter"`) or emitted a tool call instead of prose (`"tool_calls"`),
    and no value of `DEFAULT_MAX_TOKENS` changes either. Folding the two together would print
    "Raise max_tokens" for a cause that constant cannot reach.

    Raised rather than returned, because the returned value was `""`. `complete` is the `Completer`
    seam every consumer downstream reads as the system's answer — the BEAM answerer, the judge —
    and an empty answer is indistinguishable, in a results artifact, from a system that had nothing
    to say. Nothing past this point looks at the string again.

    Classified permanent. A filter fires on the PROMPT, and the prompt is byte-identical on every
    attempt, so the three extra calls buy the same refusal at the same price. Refusals also land on
    particular questions rather than on whole runs, so nothing upstream caps them.

    ⛔ The raw `finish_reason` is carried as an ATTRIBUTE and kept out of the message. It is chosen
    by the provider, and `benchmarks/beam/run.py:_is_terminal` substring-matches the rendered
    exception against markers including "402" and "401", where a hit aborts the whole run instead
    of dropping one question. Classifying by type below removes that hazard from `_is_transient`
    and would have left it untouched one layer up, which is the same mistake in a different place.
    """

    def __init__(self, message: str, *, finish_reason: object = None) -> None:
        super().__init__(message)
        #: Kept off `args`, so it never reaches `str(self)` and no substring classifier can read
        #: it. Anything that genuinely wants the provider's word reads this instead.
        self.finish_reason = finish_reason


class NoCompletionChoices(RuntimeError):
    """A 200 whose `choices` list is empty or absent, so there is no completion to read at all.

    "or absent": `not choices` takes this branch for `None` too, which is what the SDK surfaces
    when the body omits the field entirely. That is OpenRouter's documented failure shape, so it is
    the realistic case rather than the exotic one, and the raise site says so; this docstring used
    to name only the empty one.

    `benchmarks/mtrag/generation.py` raises this same type for the same shape, and widens it once
    more: a first choice carrying no `message.content` FIELD is the same class of malformed body.

    OpenRouter answers this way when the upstream it routed to faults. `resp.choices[0]` used to be
    indexed blind here, so this surfaced as `IndexError: list index out of range`: a message that
    names a Python operation rather than the provider, and sends whoever reads it hunting for a bug
    in the harness.

    Deliberately TRANSIENT, unlike `EmptyCompletion` above, and it has to say so out loud — see
    `TRANSIENT_ERRORS`. This is a fault on the provider's side of the wire rather than a property
    of the request, so a second attempt can be served by a healthy upstream. Being wrong in this
    direction costs at most `max_attempts` calls on one question; being wrong the other way fails a
    call that would have succeeded.
    """


#: Our own failures, classified by TYPE at the point they are raised rather than left to
#: `_is_transient`'s heuristics. Both directions are stated because neither default is right here.
#:
#: ⛔ Type, not message, and not name. `_is_transient`'s last resort is substring-matching the
#: exception's rendered text against markers including "429", "timeout" and "unavailable". That
#: text is written for a human: `CompletionTruncated` interpolates the ceiling, and
#: `EmptyCompletion` interpolates `finish_reason`, which is a string the PROVIDER chooses. Leaving
#: the classification to phrasing means a provider can flip our retry policy from the wire.
#: Both members ALSO inherit `recall.embeddings.NonTransientError`, which is what makes them
#: permanent for a caller using the DEFAULT classifier. This tuple is not thereby redundant: it
#: is the half of `_classify` that pairs with `TRANSIENT_ERRORS`, which has no marker equivalent
#: because `_is_transient`'s default for an unrecognised error is already "not transient".
PERMANENT_ERRORS: tuple[type[Exception], ...] = (CompletionTruncated, EmptyCompletion)

#: ⚠️ Stated explicitly because the default here is the OPPOSITE of the one in
#: `benchmarks/mtrag/generation.py`, which retries anything not named permanent. `_is_transient`
#: returns False for an exception carrying no numeric status and no marker word, which is exactly
#: what `NoCompletionChoices` is, so omitting it would silently make it fail fast.
TRANSIENT_ERRORS: tuple[type[Exception], ...] = (NoCompletionChoices,)

#: Distinguishes "the field is not there" from "the field is there and null". `getattr(..., None)`
#: cannot, and the two are priced differently: a missing `message` is a MALFORMED BODY (the
#: provider's fault, transient) while a `content` that arrived null is an EMPTY ANSWER
#: (permanent). Shared in spirit with `benchmarks/mtrag/generation.py`, which reads the same
#: field the same way.
_NO_CONTENT_FIELD = object()

#: The values the OpenAI wire format actually defines for `finish_reason`. Anything else is text
#: the provider chose, and only these are safe to render into an exception message: see
#: `EmptyCompletion`, and `benchmarks/beam/run.py:_is_terminal`, which substring-matches that
#: message against markers where a hit aborts an entire run.
KNOWN_FINISH_REASONS = frozenset(
    {"stop", "length", "content_filter", "tool_calls", "function_call"}
)


class RunAborted(RuntimeError):
    """A condition that will fail every remaining item identically, so the run stops.

    Lives here rather than in `benchmarks/beam/run.py`, where it was defined, because it is the
    other half of `is_terminal` below and four drivers now need both. `benchmarks.beam.run`
    re-exports the name, so `except RunAborted` written against either module catches the same
    class.
    """


#: Statuses that mean every REMAINING request fails the same way: a revoked key, an empty
#: balance. Not 429 and not 5xx, which are what the retry layer exists to absorb.
TERMINAL_STATUS_CODES = frozenset({401, 402})

#: Phrases that name the ACCOUNT explicitly, checked BEFORE the status and regardless of it.
#:
#: ⛔ This is not symmetric with `_AMBIGUOUS_MARKERS` below, and the asymmetry is the point.
#: OpenAI signals exhausted credit as **HTTP 429 with `code: insufficient_quota`** — a status that
#: is stated, and that means "slow down" for every other 429. Putting these behind the numeric
#: branch made the marker dead code for its own canonical carrier: `is_terminal` answered False,
#: `_is_transient` answered True, and an empty account bought four retries per question and was
#: then dropped one question at a time. That is precisely the "tidy `n: 599`, exit 0" artifact the
#: quarantine exists to prevent, reached through the guard meant to prevent it.
#:
#: These two phrases are safe ahead of the status because neither can appear in an unrelated
#: message the way a bare "401" can appear in any number containing it.
_ACCOUNT_MARKERS = ("insufficient_quota", "requires more credits")

#: Last resort ONLY, for a transport that carried no numeric status (`openai.APIConnectionError`
#: carries none). Bare digits, so they stay BEHIND the status: these are the ones that produced the
#: false positives on our own messages.
_AMBIGUOUS_MARKERS = ("401", "402")


def is_terminal(exc: Exception) -> bool:
    """Does this failure mean the whole run is over, rather than one item having failed?

    Takes `Exception`, not `BaseException`. Every call site is inside an `except Exception as exc:`
    block, so that is the real domain, and it is what `_probe` and `_is_transient` accept. A
    `KeyboardInterrupt` is not a provider verdict, and asking this about one is a type error worth
    having rather than a question with a meaningless answer.

    The difference is worth a run's paid work in both directions. Treating a terminal error as one
    bad item is how an empty OpenRouter balance became "a tidy `n: 599` summary and exit 0" with
    101 questions quietly dropped, which `benchmarks/beam/run.py:_run_pool` calls infrastructure
    wearing a verdict's clothes. Treating one bad item as terminal throws away everything already
    bought.

    PRECEDENCE, and it is the whole function:

      1. **Our own failures are never terminal.** They describe one completion, not the account.
         This is not a nicety: the substring version returned True for `max_tokens=1402` and for
         any provider-chosen `finish_reason` containing "402", so a truncation at an unlucky
         ceiling, or a string chosen by the provider, aborted an entire run.
      2. **An unambiguous ACCOUNT phrase wins, whatever the status says.** `insufficient_quota`
         ships as a 429, so anything that let the status decide first made this unreachable for
         its own canonical carrier. See `_ACCOUNT_MARKERS`.
      3. **Then a numeric status is decisive.** When the transport states the status, that is the
         answer and the digit markers below are not consulted.
      4. **Bare digits last**, for errors carrying no status at all, where the text is the only
         evidence there is.

    Steps 3 and 4 are the precedence `recall.embeddings._is_transient` settled on, for the same
    reason: a digit is a substring of any number containing it, and messages are written for
    humans. Step 2 is the exception that proves it — a phrase that can only mean one thing is
    evidence a status cannot override, because the status is about the REQUEST and the phrase is
    about the ACCOUNT.
    """
    if isinstance(exc, PERMANENT_ERRORS + TRANSIENT_ERRORS):
        return False
    # Guarded for the same reason `_probe` exists, and it is the OTHER door: formatting an
    # arbitrary exception runs ITS `__str__`, which is free to raise — an undecoded response body
    # is a realistic way for that to happen. Unguarded, this raised out of a function every caller
    # invokes from inside `except Exception as exc:`, so it replaced the provider's error and took
    # the quarantine with it. `_is_transient` guards the same line; closing only the attribute door
    # left this one open with the identical failure.
    #
    # The fallback is EMPTY, not `type(exc).__name__` as `_is_transient` uses. Its markers are
    # words; two of mine are bare digits, so a class named `Error402` would abort a whole run on
    # its own name. Empty text means "no evidence from the TEXT", not "no evidence at all": the
    # status is still read below, so an unreadable 402 is still terminal. What it does soften is
    # the account phrases, which is why an `insufficient_quota` whose body never decoded reads as
    # one bad item — quarantined, and visible through whichever accounting its driver keeps.
    # This function is called from FOUR modules, and only the first has a consecutive-failure
    # floor: `CONSECUTIVE_FAILURE_LIMIT` in `benchmarks/run.py`; `coverage` in beam's `_run_pool`;
    # the `failed` record in judge_quality; the `rejudge_failed` record in rejudge.
    #
    # ⛔ NOT mtrag, which two earlier versions of this comment claimed. `benchmarks/mtrag/`
    # defines its own `CONSECUTIVE_FAILURE_LIMIT` and its own `PERMANENT_ERROR_NAMES`, and never
    # calls this function at all, so its counter can never see a verdict from here.
    #
    # `str.lower(...)` unbound, not `.lower()`: `str(exc)` may hand back a str SUBCLASS, and the
    # marker scans below sit OUTSIDE this guard, so a hostile `__contains__` would run there
    # instead. The unbound method returns an exact `str` for any subclass, which is the immunity
    # `_is_transient` gets for free by building its text from an f-string.
    try:
        text = str.lower(str(exc))
    except Exception:  # noqa: BLE001 - a classifier must never beat the error it is classifying
        text = ""
    if any(marker in text for marker in _ACCOUNT_MARKERS):
        return True
    # All THREE spellings, because the SDKs do not agree and `recall.embeddings._is_transient`
    # already reads all three: `status_code` (openai), `status`, and `http_status` (voyageai).
    # Reading only the first two left a Voyage 401 falling through to the digit markers, so a
    # revoked key whose message did not happen to contain "401" was not terminal — and retrieval
    # embeds, so `run_arm` would have quarantined a dead embedding key one question at a time.
    #
    # ⛔ Through `_probe`, not bare `getattr`, and mirroring `_is_transient` here is not cosmetic.
    # `getattr(x, name, None)` swallows only `AttributeError`, and these are exceptions from
    # arbitrary libraries whose attribute may be a property free to raise anything. EVERY caller of
    # this function is inside an `except Exception as exc:` block, so a classifier that raises does
    # not misclassify — it replaces the provider's error and leaves the handler, taking the
    # quarantine, the `RunAborted` wrapping and the failure floor with it.
    status = _probe(exc, "status_code")
    for spelling in ("status", "http_status"):
        if status is None:
            status = _probe(exc, spelling)
    if isinstance(status, int):
        return status in TERMINAL_STATUS_CODES
    return any(marker in text for marker in _AMBIGUOUS_MARKERS)


def _shape_note(content: object, answer: str) -> str:
    """A fragment naming what actually arrived, when that is what an operator needs to see.

    Takes the READING as well as the content, because the two failures it distinguishes look
    identical from the content alone: blocks this reader could not read (a gateway encoding fault)
    and blocks that were read and held only whitespace (a model fault). Naming the first for both
    is the same misdiagnosis this note was added to fix, one shape further out.

    ⛔ No count, and no other number. This message is the one `EmptyCompletion` deliberately keeps
    free of digits: `is_terminal` and `_is_transient` both substring-match rendered exceptions on
    the bare markers "401" and "402", and a list of 402 blocks would render one. The type-based
    short-circuit in `PERMANENT_ERRORS` currently stops that reaching either classifier, but a
    message that is safe only because of a guard elsewhere is the arrangement this file has twice
    been bitten by.

    Never raises: it is evaluated while building an exception, so a raise here would replace the
    typed `EmptyCompletion` with an untyped error and lose the classification that keeps it out of
    the retry loop. `len()` and `type().__name__` can both raise on a hostile object, which is the
    same reason `is_terminal` guards `str(exc)`.
    """
    try:
        # Inside the guard with everything else: `isinstance` consults `__class__` when the type
        # check misses, and that can raise. This line sat outside and made the "never raises"
        # claim below false.
        if content is None or isinstance(content, str):
            return ""
        if isinstance(content, list):
            # `not content` calls `__bool__`, which calls `__len__` — so even the emptiness test
            # runs code the provider supplied. That is why the whole body is inside the guard
            # rather than just the parts that obviously compute something.
            if not content:
                return ""
            if answer:
                return " content came back as text blocks carrying only whitespace;"
            return " content came back as text blocks carrying no readable text;"
        return f" content came back as {type(content).__name__}, not a string;"
    except Exception:  # noqa: BLE001 - a diagnostic must never beat the error it describes
        return " content came back in a shape this reader could not describe;"


def _safe_reason(reason: object) -> str:
    """`reason` rendered for a message that another module substring-matches.

    A known value is passed through, because it is the field that tells a filter refusal apart from
    a tool-call stub and the operator needs it. Anything else is reported as unrecognised, with the
    real value left on `EmptyCompletion.finish_reason` rather than discarded.
    """
    # Guarded like `_shape_note`, and for the same reason: this is evaluated while an
    # `EmptyCompletion` is being built, so a raise replaces the typed error with an untyped one and
    # loses the classification that keeps it out of the retry loop. `isinstance` consults
    # `__class__` and `in` calls `__hash__`, so both halves can run provider-supplied code.
    # `str.__str__` rather than the object itself, so a `str` subclass cannot smuggle a hostile
    # `__format__` into the f-string that interpolates this.
    try:
        if isinstance(reason, str) and reason in KNOWN_FINISH_REASONS:
            return str.__str__(reason)
    except Exception:  # noqa: BLE001 - a renderer must never beat the error it renders
        pass
    return "unrecognised"


def _classify(exc: Exception) -> bool:
    """Is `exc` worth another attempt? An OVERRIDE of the shared heuristic, not a replacement.

    Only this module's own types are decided here, because only their raise sites know something
    the rendered text cannot express: whether repeating the call reproduces the failure at full
    price. Everything else — 429, 5xx, connection resets — still goes to `_is_transient`, which is
    the one definition of "transient" this repo has, and dropping that delegation would mean a
    single rate limit ends a 1,986-question run.

    `_is_transient` is private to `recall.embeddings` and imported anyway: its own docstring names
    `benchmarks/llm.py` as a caller it reasons about, so the two are already coupled by design, and
    reimplementing the heuristic to avoid an underscore would give this repo two of them.
    """
    if isinstance(exc, PERMANENT_ERRORS):
        return False
    if isinstance(exc, TRANSIENT_ERRORS):
        return True
    return _is_transient(exc)


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OpenRouterLLM:
    """OpenAI-compatible chat client pointed at OpenRouter. Lazily imports the `openai` SDK so the
    module imports without the `bench` extra installed (construction is what tests exercise).

    Every call costs money, and a benchmark run makes thousands of them back to back — which is
    exactly the traffic shape a provider rate-limits. Without a retry, ONE 429 on question 900 of
    1,986 propagates out of `main` and the final `.json` is never written: the incremental sidecar
    survives, but the run has to be finished by hand. `complete` therefore retries transient
    failures (429/5xx/network) with exponential backoff and full jitter.

    The backoff is `recall.embeddings.retry_with_backoff`, the one this repo already ships and
    tests — reused rather than reimplemented, so there is one retry policy in the codebase and one
    definition of "transient" (`_is_transient`, which fails fast on a non-transient error such as
    a 401). `sleep` is injectable so the retry path is exercised offline, at no wall-clock cost.

    **`max_tokens` is set, and that is not a micro-optimisation.** Omitting it reserves the model's
    maximum — 65,536 on gpt-5 — and providers bill availability against that RESERVATION, not
    against what the call returns. A BEAM run died mid-arm on
    ``402 … You requested up to 65536 tokens, but can only afford 64714`` while its answers were
    measuring ~850 completion tokens: the request was refused over a ceiling it would never have
    approached, with roughly 75x the headroom it needed.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.0,
        max_tokens: int | None = DEFAULT_MAX_TOKENS,
        max_attempts: int = 4,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self._api_key = api_key
        self._sleep = sleep
        self._client: object | None = None
        #: Guards the lazy build in `_complete_once`, for the same reason `_usage_lock` guards the
        #: counters below: ONE instance is driven by an 8-thread pool, and `if x is None: x = ...`
        #: is a check-then-set. Measured before this existed: eight threads released from a
        #: barrier onto one instance built eight `OpenAI` clients, seven of them overwritten on
        #: the next assignment and dropped while still owning an `httpx.Client` whose connection
        #: pool is never closed.
        #:
        #: A SEPARATE lock rather than a reuse of `_usage_lock`. The two protect unrelated state,
        #: and `threading.Lock` is not reentrant, so folding them together would leave any later
        #: edit that builds a client while holding the usage lock deadlocked rather than slow.
        self._client_lock = threading.Lock()
        #: The benchmark's OWN generator+judge usage (this instance drives both). Recorded as the
        #: `harness` baseline so the memory layer's cost can be isolated as total - harness.
        #:
        #: Lock-guarded, because ONE instance is driven concurrently by `benchmarks.beam.run`'s
        #: worker pool (8 threads by default) and `+=` on a dict value is a read-modify-write.
        #: The process-wide meter this gets SUBTRACTED FROM (`benchmarks.usage`) is already
        #: locked, so lost updates here made `harness < total` and published a spuriously
        #: positive `memory_layer` — the number whose entire job is to show that RE-call's
        #: retrieval path spends no tokens. An undercount in the subtrahend invents cost that
        #: was never incurred, in the one field that is supposed to prove the opposite.
        self._usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._latency_ms = 0
        self._monetary_cost_usd: float | None = None
        self._model_revision: str | None = None
        self._usage_lock = threading.Lock()

    def usage(self) -> dict[str, int]:
        with self._usage_lock:
            return dict(self._usage)

    def provider_metadata(self) -> ProviderMetadata:
        with self._usage_lock:
            prompt = self._usage["prompt_tokens"]
            completion = self._usage["completion_tokens"]
            return ProviderMetadata(
                provider_id="openrouter",
                model_id=self.model,
                model_revision=self._model_revision,
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
                latency_ms=self._latency_ms,
                monetary_cost_usd=self._monetary_cost_usd,
            )

    def complete(self, system: str, user: str) -> str:
        def _once() -> str:
            return self._complete_once(system, user)

        return retry_with_backoff(
            _once, attempts=self.max_attempts, sleep=self._sleep, is_transient=_classify
        )

    def _complete_once(self, system: str, user: str) -> str:
        from openai import OpenAI  # lazy: only needed at real run time

        # The lock is taken unconditionally rather than as the second half of a double-check. The
        # unlocked fast path would buy an uncontended `Lock.acquire` (tens of nanoseconds) off the
        # front of an HTTP round trip measured in hundreds of milliseconds, and would pay for it
        # with a publication argument that has no settled answer on a free-threaded build, which
        # this package supports (Python 3.14 is in the tested matrix).
        #
        # Still lazy, and that is not an oversight this could tidy away by building in `__init__`.
        # `openai` is in the `bench` extra, `dev` excludes it deliberately, and CI installs
        # `.[dev]`, so the SDK is absent wherever the tests run; the class is documented as
        # constructible without it and `test_bench_llm.py` constructs one with no `openai` in
        # `sys.modules` at all. Building eagerly would move the ImportError from the first
        # request onto every construction, green on a developer machine that happens to have the
        # extra and red in CI.
        #
        # The request below then reads `self._client` OUTSIDE this block, deliberately, and
        # `VoyageReranker._voyage_client` makes the opposite choice for its own read. Both are
        # sound and the difference is only what each method costs: holding a lock across the
        # network call here would serialise the whole pool, whereas the reranker's helper returns
        # immediately and holds nothing. The unlocked read is safe because the slot is write-once
        # (assigned only here and in `__init__`, never reset) and every reader has already been
        # through this acquire/release pair, which supplies the ordering a free-threaded build
        # would otherwise need. `tests/lazy_client_lock_helpers.py` enforces that premise rather
        # than trusting this paragraph.
        with self._client_lock:
            if self._client is None:
                # `max_retries=0` because `retry_with_backoff` in `complete` owns the retry
                # policy. The SDK default is 2 retries, so leaving it on multiplies the two
                # layers: one 429 costs 4 x 3 = 12 requests rather than the 4 `max_attempts` asks
                # for, and the outer FULL-jitter backoff (which exists so a fleet does not remarch
                # onto the provider in lockstep) ends up wrapping an inner loop that smears its
                # own doubling schedule by only `1 - 0.25 * random()` — a 25% jitter, not a draw
                # across the interval, so it separates a fleet far less than the layer wrapping it.
                self._client = OpenAI(api_key=self._api_key, base_url=self.base_url, max_retries=0)
        extra = {} if self.max_tokens is None else {"max_tokens": self.max_tokens}
        started = time.perf_counter()
        resp = self._client.chat.completions.create(  # type: ignore[union-attr]
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **extra,
        )
        elapsed_ms = max(0, int((time.perf_counter() - started) * 1000))
        resp_usage = getattr(resp, "usage", None)
        with self._usage_lock:
            self._latency_ms += elapsed_ms
            revision = getattr(resp, "model", None)
            if isinstance(revision, str) and revision:
                self._model_revision = revision
            cost = _usage_cost_usd(resp_usage)
            if cost is not None:
                self._monetary_cost_usd = (self._monetary_cost_usd or 0.0) + cost
            if resp_usage is not None:
                self._usage["calls"] += 1
                self._usage["prompt_tokens"] += int(getattr(resp_usage, "prompt_tokens", 0) or 0)
                self._usage["completion_tokens"] += int(
                    getattr(resp_usage, "completion_tokens", 0) or 0
                )
        # Read the choice ONCE, and defensively. A 200 can carry an empty `choices` list when the
        # upstream OpenRouter routed to faults, and indexing it blind raised `IndexError`, which
        # named a Python operation instead of the provider fault it actually was.
        #
        # Everything from here down sits AFTER the usage block on purpose: the tokens were spent
        # whatever the body turned out to carry, and `benchmarks.usage` subtracts this figure to
        # publish `memory_layer`, so an undercount invents cost that was never incurred.
        choices = resp.choices
        if not choices:
            raise NoCompletionChoices(
                # "empty OR ABSENT": `not choices` also takes this branch for `None`, which is what
                # the SDK surfaces when the body omits `choices` entirely. Kept word-for-word in
                # step with `benchmarks/mtrag/generation.py`, which raises the same type.
                "the provider returned a 200 with an empty or absent `choices` list, so there is "
                "no completion to read. This is an upstream fault on the provider's side, not a "
                "property of the request."
            )
        choice = choices[0]
        # A response that stopped for `length` is truncated. Fail loudly: scoring a half-written
        # answer would charge our own ceiling to the system under test.
        reason = getattr(choice, "finish_reason", None)
        if reason == "length":
            raise CompletionTruncated(
                f"completion hit max_tokens={self.max_tokens} and was cut off. Raise max_tokens "
                f"(benchmarks.llm.DEFAULT_MAX_TOKENS) — do NOT score this answer."
            )
        # `object`, not `str | None`. The SDK types this `Optional[str]` but builds its response
        # models WITHOUT validating, so a non-conforming body (a list of content parts, say) passes
        # straight through, and annotating the narrower type would be a lie that `.strip()` then
        # pays for with `AttributeError: 'list' object has no attribute 'strip'` — a message naming
        # a Python operation rather than the provider, which is what `NoCompletionChoices` exists
        # to abolish. Narrowing by `isinstance` below is also what makes the `return` a genuine
        # `str`, rather than the `Any` a bare read leaks out of a signature declaring `str`.
        #
        # Read through a SENTINEL, matching `benchmarks/mtrag/generation.py`. `choice.message` was
        # dereferenced blind, so a 200 whose first choice carries `message: null` escaped this seam
        # as `AttributeError: 'NoneType' object has no attribute 'content'` — an untyped Python
        # error crossing into the BEAM answerer and the judge, where every other unusable shape
        # here raises a named one.
        #
        # The sentinel is what keeps the two cases apart, and they are priced differently: a
        # MISSING `message` is a malformed body (the provider's fault, transient, `NoCompletionChoices`)
        # while a `content` that arrived and is null is an empty ANSWER (permanent,
        # `EmptyCompletion`). `getattr(..., None)` would merge them and charge one as the other.
        content: object = getattr(getattr(choice, "message", None), "content", _NO_CONTENT_FIELD)
        if content is _NO_CONTENT_FIELD:
            raise NoCompletionChoices(
                "the provider returned a 200 whose first choice carries no `message.content` "
                "field at all, so there is no completion to read. This is an upstream fault on "
                "the provider's side, not a property of the request."
            )
        # Emptiness is decided on the STRIPPED text but the value handed back is untouched.
        # `generate_one` returns `.strip()`ed text because a stripped string is what it writes to a
        # submission; this is the `Completer` seam feeding arbitrary consumers, so stripping here
        # would silently change what every existing caller receives. Whitespace-only content is
        # still as unscorable as `None`, and `content or ""` returned both as `""`.
        #
        # The finish reason is named because it is the only field telling a filter refusal apart
        # from a tool-call stub or a provider bug, and the three want different responses — but
        # only through `_safe_reason`, because the raw value is the provider's text and this
        # message is read by another substring classifier. The raw value rides the exception.
        answer = assistant_text(content)
        if not answer.strip():
            shape = _shape_note(content, answer)
            raise EmptyCompletion(
                f"the provider returned a completion with no usable text "
                f"(finish_reason={_safe_reason(reason)}).{shape} There is no answer. Returning it "
                f"would put an unscorable empty string where a judge cannot tell it apart from a "
                f"system that had nothing to say.",
                finish_reason=reason,
            )
        return answer


def _usage_cost_usd(usage: object | None) -> float | None:
    if usage is None:
        return None
    for name in ("cost", "total_cost", "total_cost_usd"):
        raw = getattr(usage, name, None)
        if raw is not None and not isinstance(raw, dict):
            return float(raw)
    details = getattr(usage, "cost", None)
    if isinstance(details, dict):
        if "usd" in details:
            return float(details["usd"])
        if "total_usd" in details:
            return float(details["total_usd"])
    return None
