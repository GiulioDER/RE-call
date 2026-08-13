from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Protocol

from recall.embeddings import _is_transient, retry_with_backoff
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


class CompletionTruncated(RuntimeError):
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


class EmptyCompletion(RuntimeError):
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
    """A 200 whose `choices` list is empty, so there is no completion to read at all.

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
PERMANENT_ERRORS: tuple[type[Exception], ...] = (CompletionTruncated, EmptyCompletion)

#: ⚠️ Stated explicitly because the default here is the OPPOSITE of the one in
#: `benchmarks/mtrag/generation.py`, which retries anything not named permanent. `_is_transient`
#: returns False for an exception carrying no numeric status and no marker word, which is exactly
#: what `NoCompletionChoices` is, so omitting it would silently make it fail fast.
TRANSIENT_ERRORS: tuple[type[Exception], ...] = (NoCompletionChoices,)

#: The values the OpenAI wire format actually defines for `finish_reason`. Anything else is text
#: the provider chose, and only these are safe to render into an exception message: see
#: `EmptyCompletion`, and `benchmarks/beam/run.py:_is_terminal`, which substring-matches that
#: message against markers where a hit aborts an entire run.
KNOWN_FINISH_REASONS = frozenset(
    {"stop", "length", "content_filter", "tool_calls", "function_call"}
)


def _safe_reason(reason: object) -> str:
    """`reason` rendered for a message that another module substring-matches.

    A known value is passed through, because it is the field that tells a filter refusal apart from
    a tool-call stub and the operator needs it. Anything else is reported as unrecognised, with the
    real value left on `EmptyCompletion.finish_reason` rather than discarded.
    """
    return reason if isinstance(reason, str) and reason in KNOWN_FINISH_REASONS else "unrecognised"


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

        if self._client is None:
            # `max_retries=0` because `retry_with_backoff` in `complete` owns the retry policy.
            # The SDK default is 2 retries, so leaving it on multiplies the two layers: one 429
            # costs 4 x 3 = 12 requests rather than the 4 `max_attempts` asks for, and the outer
            # FULL-jitter backoff (which exists so a fleet does not remarch onto the provider in
            # lockstep) ends up wrapping an inner loop that smears its own doubling schedule by
            # only `1 - 0.25 * random()` — a 25% jitter, not a draw across the interval, so it
            # separates a fleet far less than the layer wrapping it.
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
                "the provider returned a 200 with an empty `choices` list, so there is no "
                "completion to read. This is an upstream fault on the provider's side, not a "
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
        content: object = choice.message.content
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
        if not isinstance(content, str) or not content.strip():
            shape = "" if content is None or isinstance(content, str) else (
                f" content came back as {type(content).__name__}, not a string;"
            )
            raise EmptyCompletion(
                f"the provider returned a completion with no usable text "
                f"(finish_reason={_safe_reason(reason)}).{shape} There is no answer. Returning it "
                f"would put an unscorable empty string where a judge cannot tell it apart from a "
                f"system that had nothing to say.",
                finish_reason=reason,
            )
        return content


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
