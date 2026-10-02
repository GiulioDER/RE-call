"""What every embedder shares: the protocols, profile identity, artifact pinning and the retry
policy for hosted calls.

Split out of ``recall.embeddings`` so a provider module can depend on these without importing
the factory that depends on every provider. Every name here is re-exported from
``recall.embeddings``, the public home.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import random
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal, Protocol, TypeVar, cast, runtime_checkable

from recall.errors import RecallError

#: Return type of the callable `retry_with_backoff` wraps — it hands back whatever `fn` returns,
#: so the retry is transparent to the caller's type rather than widening it to `object`.
_R = TypeVar("_R")


#: The text fallback's whole vocabulary, hoisted out of `_is_transient` so a test can WALK it.
#: Inline, each marker could be deleted with the suite staying green, which mattered because this
#: tuple is the only evidence some callers ever get: an error carrying no numeric status reaches
#: nothing else.
#:
#: ⚠️ These are substrings, not words, and that has bitten twice. `"429"` matches inside any
#: number containing it — a 400 reading `"…resulted in 10429 tokens"` was classified a rate limit,
#: and `benchmarks/llm.py`'s `CompletionTruncated` interpolates `max_tokens`, so a ceiling of
#: 4290, 429 or 1429 read as one too — until that caller put the type in `PERMANENT_ERRORS` and
#: classified it ahead of this function. That is the lesson, not the leftover: prefer classifying
#: by TYPE or status where you can. This is the last resort, for errors that offer nothing else.
_TRANSIENT_MARKERS = (
    "429", " 500", " 502", " 503", " 504", "rate limit", "too many requests",
    "timeout", "timed out", "temporarily", "connection", "reset by peer", "unavailable",
)


_log = logging.getLogger("recall.embeddings")


class NonTransientError(RecallError):
    """Marker: ``retry_with_backoff`` must never retry this, whatever the message happens to say.

    ``_is_transient`` classifies by heuristic, and its last resort is substring-matching the
    exception's rendered text. That text is written for a human, so any wording that happens to
    contain a marker is read as retryable and the caller silently pays ``attempts`` times for a
    failure guaranteed to repeat. Measured cases: a ceiling of 4,290 tokens contains "429"; a path
    containing "timeout"; a host named "connection-broker".

    ``benchmarks/llm.py:CompletionTruncated`` is what this was found on. Measured against THIS
    FUNCTION, the property held only for ceilings spelled without a marker: 16,384 classifies
    permanent, while 4,290 and 429 and 1,429 all classify transient.

    ⚠️ What that did NOT cost, stated precisely because an earlier draft of this docstring
    overclaimed it: no bill was ever paid for it. BOTH callers that raise this type were already
    protected, by two different mechanisms, and each billed ONE request rather than four.
    ``OpenRouterLLM.complete`` passes ``is_transient=_classify``, which short-circuits on its own
    ``PERMANENT_ERRORS`` tuple; ``benchmarks/mtrag/generation.py`` runs its own retry loop and
    re-raises this type out of it directly. The 4x is what a caller using the DEFAULT classifier
    would pay — a hazard for the next such caller, not a measured historical loss.

    ⚠️ Fixed HERE rather than in the wording. Rewording relocates the coincidence instead of
    removing it, and leaves the same fragility for every other caller of ``retry_with_backoff``:
    the two sites in this module and ``recall/truth_extraction/_openai_engine.py``, none of which
    passes a custom classifier. ``benchmarks/llm.py`` had already protected its own path with a
    ``PERMANENT_ERRORS`` tuple. Both callers had a fix, by two different mechanisms, which is why
    this went unnoticed: the hazard lives in the DEFAULT classifier, which neither of them uses.

    Inherited ALONGSIDE the exception's own base, never instead of it, so existing
    ``except RuntimeError`` still catches.
    """


def _probe(exc: Exception, name: str) -> object | None:
    """Read ``name`` off an arbitrary exception, refusing to raise while doing it.

    ``getattr(x, name, None)`` swallows only ``AttributeError``, and the thing being probed is an
    exception from an arbitrary library where the attribute may be a property free to raise
    anything — a deprecated alias under ``-W error``, a lazy parse of a malformed response. That
    matters more here than it looks: ``_is_transient`` is called from inside
    ``retry_with_backoff``'s ``except Exception`` block, so a classifier that raises does not
    merely misclassify, it REPLACES the provider's error with its own and kills the run with the
    wrong exception. ``tests/test_bench_systems.py`` reaches for the same guard for the same
    reason.
    """
    try:
        return getattr(exc, name, None)
    except Exception:  # noqa: BLE001 - a probe must never beat the error it is probing  # BROAD-CATCH: error-translation
        return None


def _is_transient(exc: Exception) -> bool:
    """Heuristic: is this exception worth retrying?

    Covers request-timeout (408), rate-limit (429), server (5xx) and network/timeout errors
    WITHOUT importing any provider-specific exception type (voyageai is an optional dependency).
    A non-transient error (e.g. 401 auth) returns False so it fails fast.

    A numeric ``status_code``/``status`` is DECISIVE: when the transport has stated the status,
    that answer is returned and the text markers below are never consulted. They used to be, and
    they could overturn a correct verdict — the marker ``"429"`` is a substring of any number
    containing it, so ``"…your messages resulted in 10429 tokens"`` made a permanent HTTP 400
    context-length overflow look like a rate limit. That is the worst case to be wrong on:
    ``retry_with_backoff`` resends the entire payload, so a caller whose payload is a prompt with
    a whole document body inside it pays for the same refused request on every attempt (three by
    default, four from ``benchmarks/llm.py``), and no retry can make an over-long prompt fit.
    ``recall/truth_extraction/_openai_engine.py`` is the case it was found on and the sharpest
    example: its prompt embeds a whole memo body, which makes a context-length overflow a normal
    failure of that engine rather than an exotic one. ``benchmarks/llm.py`` is the same shape,
    sending a retrieved context per question across thousands of calls.

    Three spellings are read, because the SDKs do not agree: ``status_code`` (openai),
    ``status``, and ``http_status`` (voyageai). The third is not decoration. Until it was added,
    NO Voyage error reached the numeric branch, so a real ``ServerError`` on an HTTP 500 was not
    retried at all, and a real ``RateLimitError`` was retried only when the provider's wording
    happened to hit one of the markers below ("rate limit", "too many requests", "429") — a
    corpus index dying on the first 500, from a path whose whole point is surviving them.

    The markers remain as a fallback for errors that carry no status at all —
    ``openai.APIConnectionError``/``APITimeoutError`` carry none — which is the only evidence
    available there.

    408 is in the numeric branch so that "network/timeout" does not depend on how a provider
    words its body. A CLIENT-side timeout arrives as an ``APITimeoutError`` carrying no status
    and is caught by the markers; a server-declared 408 was only ever caught when the body
    happened to spell "timeout", which is not a contract so much as a coincidence.

    409 is deliberately NOT here, and it is the one status openai's own client retries that
    nothing retries now. The SDK treats it as a lock timeout, a semantic of its STATEFUL
    endpoints (vector stores, assistant runs); ``/v1/embeddings`` and ``/v1/chat/completions``
    are stateless POSTs with no resource to lock, so a 409 from an OpenAI-compatible proxy is a
    real conflict that resending cannot resolve. Being wrong in that direction costs one request
    instead of three.

    Every caller now builds its SDK client with ``max_retries=0``, so this function is the single
    owner of the policy and its numeric contract IS what reaches the provider. That is what makes
    408 this function's business: with the SDK retrying underneath, a 408 was being retried twice
    regardless of what was decided here.
    """
    # 🔑 Ahead of EVERY heuristic below, including the numeric one. A status describes what the
    # transport saw; the marker describes what the RAISER knows, and only the raiser can know that
    # resending reproduces the failure at full price. A wrapper carrying an upstream 500 alongside
    # its own "do not retry" verdict must be believed about its own verdict.
    #
    # ⛔ `issubclass(type(exc), ...)`, NOT `isinstance`. `isinstance` falls back to reading
    # `exc.__class__` whenever the fast type check misses — which is every exception that is not a
    # marker, i.e. all of them today — and `__class__` is free to raise. This function is called
    # from inside `retry_with_backoff`'s `except Exception`, so a raise here does not misclassify:
    # it REPLACES the provider's error. `type()` cannot be intercepted, and
    # `type.__subclasscheck__` on a plain metaclass runs no user code. The first draft of this
    # line used `isinstance` and re-opened, as the FIRST statement of the function, the exact hole
    # that `_probe` and the guarded `str(exc)` below exist to close.
    if issubclass(type(exc), NonTransientError):
        return False
    status = _probe(exc, "status_code")
    if status is None:
        status = _probe(exc, "status")
    if status is None:
        status = _probe(exc, "http_status")
    # `issubclass(type(status), int)`, not `isinstance`. `_probe` guards READING the attribute;
    # the value it hands back is still arbitrary provider data, and `isinstance` reads ITS
    # `__class__`, which can raise — the same argument as the marker check above, one
    # indirection in, and the door that stayed open when that one was closed. `issubclass`
    # on `type(...)` also keeps int-SUBCLASS semantics (an `IntEnum` status), which
    # `type(status) is int` would silently drop.
    # The second `isinstance` is for the TYPE CHECKER, not the runtime, and it is safe: it runs
    # only once `issubclass` has proved `type(status)` is an int subclass, so CPython's
    # `PyType_IsSubtype` fast path answers it without ever consulting `__class__`. Written the
    # other way round it would be the unguarded read again.
    if issubclass(type(status), int) and isinstance(status, int):
        return status in (408, 429) or 500 <= status < 600
    try:
        text = f"{type(exc).__name__} {exc}".lower()
    except Exception:  # noqa: BLE001 - see `_probe`: a hostile __str__ must not beat the error  # BROAD-CATCH: error-translation
        # `_probe` closes the attribute door and this closes the other one. Formatting an
        # arbitrary exception runs ITS ``__str__``, which is free to raise — and a body that was
        # never decoded is a realistic way for that to happen. The class name alone still gives
        # the markers something to match on.
        try:
            text = type(exc).__name__.lower()
        except Exception:  # noqa: BLE001 - nested, because the fallback can raise too  # BROAD-CATCH: fail-open
            # ``__name__`` resolves through the METACLASS, where a `@property` is a data
            # descriptor that beats ``type.__name__``. Unnested, this line sits inside the
            # handler and its exception escapes `_is_transient` — the very outcome the outer
            # guard exists to stop, one line further down. Empty text classifies as permanent,
            # which fails fast rather than resending, and is the safe direction for an object
            # this hostile.
            text = ""
    return any(m in text for m in _TRANSIENT_MARKERS)


#: Ceiling on a provider's ``Retry-After``, matching the one the openai SDK applies. Past it the
#: header is ignored rather than obeyed: a proxy answering "3600" would otherwise park a corpus
#: indexing run for an hour inside what the caller believes is a bounded retry.
_MAX_RETRY_AFTER_S = 60.0


def _retry_after_seconds(exc: Exception, *, cap: float | None = _MAX_RETRY_AFTER_S) -> float | None:
    """How long the provider asked us to wait, or None if it did not ask for anything usable.

    Read off the exception by duck-typing, on the same rule as ``_is_transient``: no
    provider-specific exception type is imported. Two shapes, because the two SDKs this repo
    retries for do not agree — ``openai`` raises an error carrying an httpx ``response``, while
    ``voyageai`` hangs ``headers`` straight off the exception and has no ``response`` attribute
    at all. Walking only the first shape would silently cover one of the two cloud embedders
    while claiming both.

    That ``headers`` is ``requests.Response.headers``, a CASE-INSENSITIVE mapping, which is what
    makes the lowercase lookups below safe on that path: a provider sending the RFC's canonical
    ``Retry-After`` is found anyway. Do not "simplify" it to a plain dict in a test and conclude
    the casing does not matter — under a plain dict it would not be found.

    Both spellings are read. ``retry-after-ms`` is what OpenAI and most of its compatible proxies
    actually send; ``Retry-After`` is the RFC one and is defined as EITHER a delay in seconds or
    an HTTP-date, and real providers send both forms. Reading only the integer would leave the
    pacing gap open for whichever half of them chose the other.

    Anything unparseable, negative, or above the cap returns None, which puts the caller back on
    its own jittered backoff — the behaviour that was there before, and bounded. That promise is
    kept with a blanket ``except`` rather than a named tuple on purpose: the header is attacker-
    adjacent input reached through a mapping this function does not control, and a plain dict can
    hold a value of any type. ``parsedate_to_datetime`` calls ``.split()`` before it validates, so
    a non-string raises ``AttributeError`` — which, inside ``retry_with_backoff``'s ``except
    Exception`` block, would replace the provider's error and kill an indexing run with a stdlib
    string-method failure instead of retrying it.
    """
    try:
        return _read_retry_after(exc, cap)
    except Exception:  # noqa: BLE001 - a bad header must never beat the error it arrived on  # BROAD-CATCH: error-translation
        return None


def _read_retry_after(exc: Exception, cap: float | None = _MAX_RETRY_AFTER_S) -> float | None:
    """The parsing half of ``_retry_after_seconds``, free to raise. See its docstring."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    get = getattr(headers, "get", None)
    if not callable(get):
        return None

    raw_ms = get("retry-after-ms")
    if raw_ms is not None:
        try:
            return _capped(float(raw_ms) / 1000.0, cap)
        except (TypeError, ValueError):
            # Fall through rather than return: a junk millisecond header must not discard a
            # perfectly good `Retry-After` sitting beside it, which is what dropping out here
            # did — straight back onto the 1.5s unpaced budget this function exists to replace.
            pass

    raw = get("retry-after")
    if raw is None:
        return None
    try:
        return _capped(float(raw), cap)
    except (TypeError, ValueError):
        pass
    when = parsedate_to_datetime(raw)
    # A date with no zone is UTC by RFC 9110; without this, subtracting an aware `now` raises.
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return _capped((when - datetime.now(timezone.utc)).total_seconds(), cap)


def _capped(seconds: float, cap: float | None = _MAX_RETRY_AFTER_S) -> float | None:
    """The wait if it is positive and within ``cap`` (any positive wait when ``cap`` is None)."""
    return seconds if 0.0 < seconds and (cap is None or seconds <= cap) else None


def retry_with_backoff(
    fn: Callable[[], _R],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    is_transient: Callable[[Exception], bool] = _is_transient,
    sleep: Callable[[float], None] = time.sleep,
) -> _R:
    """Call ``fn()`` with exponential backoff, retrying only transient failures.

    Re-raises immediately for a non-transient error, and re-raises the last error after
    ``attempts`` tries. ``sleep`` is injectable so tests can exercise the retry path without
    real delays.

    Delay for retry i is FULL JITTER over ``min(max_delay, base_delay * 2**i)`` — a uniform
    draw in [0, cap], not the cap itself. A rate-limit or 5xx typically hits every client at
    once, so a deterministic schedule marches the whole fleet back onto the provider in
    lockstep at each step; jitter spreads the retries out instead of reconverging them.

    A ``Retry-After`` the provider actually sent overrides that draw. The unpaced schedule spends
    every attempt within 1.5s at the defaults, so against a per-minute rate limit all three
    requests land inside the same closed window and the call fails where it would have recovered.
    Every caller of this function now builds its SDK client with ``max_retries=0``, which is what
    makes this the layer that must honour the header: the transport layer it replaced did, and
    removing that layer without picking the header up here would have traded a cost problem for
    an availability one. The jitter is added ON TOP of what the provider asked for rather than drawn
    within it — waiting less than the stated time is what the header exists to prevent, while a
    fleet handed the same number still needs spreading out.

    ⚠️ A caller that has NOT switched its SDK's retries off would pay this pacing twice, once here
    and once inside the transport, since ``openai``'s own ``_calculate_retry_timeout`` applies the
    same 60s rule. No caller here is in that state; keep it that way when adding one.
    """
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # BROAD-CATCH: fail-closed
            last = exc
            if i == attempts - 1 or not is_transient(exc):
                raise
            jitter = random.uniform(0.0, min(max_delay, base_delay * (2 ** i)))
            asked = _retry_after_seconds(exc)
            sleep(jitter if asked is None else asked + jitter)
    assert last is not None  # unreachable: loop either returns or raises
    raise last


def batched_embed(
    texts: list[str],
    embed_batch: Callable[[list[str]], list[list[float]]],
    *,
    batch_size: int = 128,
    max_batch_chars: int | None = None,
    max_workers: int = 1,
) -> list[list[float]]:
    """Embed ``texts`` in provider-safe batches, concatenating results in input order.

    ``embed_batch`` embeds a single batch. Batches are cut on ``batch_size`` (count) and, when
    ``max_batch_chars`` is set, also on a cumulative character budget — a guard against a batch
    that is few in count but huge in tokens. A single text over the char budget still goes out
    alone (never dropped). Order is preserved: batch results are appended in sequence.

    ``max_workers`` above 1 sends the same batches concurrently and still concatenates them
    in input order, so the vectors are the ones a sequential run returns; only the wall
    time changes. The default of 1 is the original sequential loop.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive int")
    if max_workers <= 0:
        raise ValueError("max_workers must be a positive int")
    if max_workers > 1:
        return _batched_embed_concurrently(
            texts, embed_batch, batch_size=batch_size, max_batch_chars=max_batch_chars,
            max_workers=max_workers,
        )
    out: list[list[float]] = []
    batch: list[str] = []
    chars = 0
    for t in texts:
        if batch and (
            len(batch) >= batch_size
            or (max_batch_chars is not None and chars + len(t) > max_batch_chars)
        ):
            out.extend(_checked(embed_batch, batch))
            batch, chars = [], 0
        batch.append(t)
        chars += len(t)
    if batch:
        out.extend(_checked(embed_batch, batch))
    return out


def _batched_embed_concurrently(
    texts: list[str],
    embed_batch: Callable[[list[str]], list[list[float]]],
    *,
    batch_size: int,
    max_batch_chars: int | None,
    max_workers: int,
) -> list[list[float]]:
    batches: list[list[str]] = []
    batch: list[str] = []
    chars = 0
    for t in texts:
        if batch and (
            len(batch) >= batch_size
            or (max_batch_chars is not None and chars + len(t) > max_batch_chars)
        ):
            batches.append(batch)
            batch, chars = [], 0
        batch.append(t)
        chars += len(t)
    if batch:
        batches.append(batch)
    if len(batches) <= 1:
        return [vector for one in batches for vector in _checked(embed_batch, one)]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as pool:
        results = list(pool.map(lambda one: _checked(embed_batch, one), batches))
    return [vector for result in results for vector in result]


def _checked(
    embed_batch: Callable[[list[str]], list[list[float]]], batch: list[str]
) -> list[list[float]]:
    """Embed one batch, refusing a response that does not line up with its input.

    Positional pairing is the whole contract (chunk i <-> vector i). A short batch would shift
    every later chunk onto its neighbour's vector — silently, because the only downstream check
    is the TOTAL count, which a compensating batch satisfies.
    """
    vecs = embed_batch(batch)
    if len(vecs) != len(batch):
        raise RuntimeError(
            f"embedder returned {len(vecs)} embeddings for {len(batch)} texts — refusing to "
            f"index misaligned vectors"
        )
    return vecs


@runtime_checkable
class Embedder(Protocol):
    """Turns text into dense vectors. Implementations must be deterministic and
    order-preserving: `embed(texts)` returns one vector per input text, in input order,
    each of length `dim`. `name` identifies the backend (used in logging / evals).
    """

    @property
    def dim(self) -> int: ...

    @property
    def name(self) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


EmbeddingPurpose = Literal["query", "passage", "legacy"]


@runtime_checkable
class AsymmetricEmbedder(Embedder, Protocol):
    """Optional extension for models with distinct retrieval encoders."""

    def embed_query(self, text: str) -> list[float]: ...

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class GroupedDocumentEmbedder(Embedder, Protocol):
    """Optional passage encoder whose vectors depend on document group membership."""

    def embed_document_groups(self, groups: list[list[str]]) -> list[list[list[float]]]: ...


@dataclass(frozen=True)
class EmbeddingProfile:
    """Immutable identity for every input that can change stored vectors."""

    profile_id: str
    model_name: str
    artifact_digest: str
    dimension: int
    query_mode: str
    passage_mode: str
    normalization: str = "l2"
    instruction_version: str = "none"
    chunker_version: str = "chunk-text-v1"
    context_version: str = "raw-v1"
    dependencies: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.profile_id or not self.model_name or not self.artifact_digest:
            raise ValueError("embedding profile identity fields must be non-empty")
        if self.dimension < 1:
            raise ValueError("embedding profile dimension must be positive")

    def fingerprint(self) -> str:
        """SHA256 over the COMPLETE identity, as durable cache and provenance key material.

        The encoding, which the pinned test transcribes independently rather than reading back
        off this method: a domain tag, then every field in declaration order, then each
        dependency as name followed by version, each item UTF-8 encoded and terminated by a NUL.
        The terminators are what make the concatenation unambiguous; without them
        ``("ab", "c")`` and ``("a", "bc")`` hash alike.

        Every field is included, including the four that nothing else reads
        (``normalization``, ``instruction_version``, ``chunker_version``, ``dependencies``).
        That is the answer to what those fields are for: they are not documentation, they are
        key material, and a change in any of them re-partitions the cache rather than silently
        serving vectors produced under the old value. ``dependencies`` carries the inference
        library version, so a fastembed upgrade costs a re-embed, deliberately, because ONNX
        runtime changes are free to move the last bits of a vector and a cache cannot tell.

        Stability is the contract. Cached vectors outlive the process that wrote them, so
        changing this encoding invalidates every cache in existence at once; if that is ever
        wanted, bump the domain tag so the change is legible instead of mysterious.
        """
        digest = hashlib.sha256()
        parts = [
            "embedding-profile-fingerprint-v1",
            self.profile_id,
            self.model_name,
            self.artifact_digest,
            str(self.dimension),
            self.query_mode,
            self.passage_mode,
            self.normalization,
            self.instruction_version,
            self.chunker_version,
            self.context_version,
        ]
        for name, pinned_version in self.dependencies:
            parts.extend((name, pinned_version))
        for part in parts:
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


#: The digest value a profile carries when its weights are provisioned by the operator and
#: nothing verified them. Pre-dates the registry; every legacy embedder mints it.
LEGACY_UNVERIFIED_DIGEST = "legacy-unverified"


#: The digest value a REGISTERED HOSTED profile carries. A hosted provider serves weights it can
#: replace behind a stable model name, so there is no artifact to hash and no revision to pin: the
#: honest value is a marker saying so, not a digest that would be invented.
#:
#: Deliberately DISTINCT from `LEGACY_UNVERIFIED_DIGEST`, and the distinction is load-bearing in
#: two places that ask different questions of it:
#:
#: * `recall.readiness` asks "is the artifact immutably pinned?". Both answer no, so that site
#:   compares against `UNVERIFIED_ARTIFACT_DIGESTS` rather than either literal.
#: * `recall.index` asks "does this profile make a real claim about its context?". A legacy
#:   profile does not (its `context_version` is a default nobody chose) and is exempt; a
#:   registered hosted profile DOES, so it must stay subject to the check. That site therefore
#:   compares against the LEGACY literal alone, on purpose. A shared "is unverified" predicate
#:   there would exempt hosted profiles from a check they should pass, which is the defect that
#:   sank an earlier attempt at this feature.
HOSTED_UNVERIFIED_DIGEST = "hosted-unverifiable"


#: Every digest value that is a marker rather than a pinned artifact. One named set so that adding
#: a third kind cannot silently pass a gate that enumerates the other two.
UNVERIFIED_ARTIFACT_DIGESTS = frozenset({LEGACY_UNVERIFIED_DIGEST, HOSTED_UNVERIFIED_DIGEST})


def artifact_is_pinned(profile: EmbeddingProfile) -> bool:
    """Whether this profile names an artifact whose bytes something actually verified."""
    return profile.artifact_digest not in UNVERIFIED_ARTIFACT_DIGESTS


def _check_declared_width(identity: EmbeddingProfile | None, actual_dim: int, what: str) -> None:
    """Refuse an identity whose declared width the live encoder does not produce.

    The declared dimension is a CLAIM, and until this existed nothing checked it on the hosted
    path. `check_enterprise_readiness` looks like it would, since it compares
    `profile.dimension != embedder.dim`, but on an embedder with no identity that profile comes
    from `legacy_embedding_profile`, which sets `dimension` FROM `embedder.dim`, so the comparison
    is vacuously true. Measured 2026-08-18 before this guard: a stub returning 512-wide vectors
    under a profile declaring 1024 built cleanly and passed the readiness gate.

    Raising here rather than at the gate is deliberate. A provider that changes the width behind a
    model name has changed the model, and the cheapest moment to say so is before a single vector
    is written into a store built at the other width. `FastEmbedEmbedder` already refuses this for
    local artifacts; this is the same refusal for a hosted one.
    """
    if identity is not None and actual_dim != identity.dimension:
        raise ValueError(
            f"profile {identity.profile_id!r} declares dimension {identity.dimension} but "
            f"{what} embeds at {actual_dim}; this endpoint is not that profile"
        )


def legacy_embedding_profile(embedder: Embedder) -> EmbeddingProfile:
    """Describe a legacy embedder without changing its public protocol."""
    name = getattr(embedder, "name", type(embedder).__name__)
    dim = int(getattr(embedder, "dim"))
    return EmbeddingProfile(
        profile_id=str(name),
        model_name=str(name),
        artifact_digest=LEGACY_UNVERIFIED_DIGEST,
        dimension=dim,
        query_mode="legacy",
        passage_mode="legacy",
        normalization="embedder-defined",
    )


def embedding_profile(embedder: Embedder) -> EmbeddingProfile:
    profile = getattr(embedder, "profile", None)
    return profile if isinstance(profile, EmbeddingProfile) else legacy_embedding_profile(embedder)


def embedding_profile_id(embedder: Embedder) -> str:
    profile = getattr(embedder, "profile", None)
    if isinstance(profile, EmbeddingProfile):
        return profile.profile_id
    name = getattr(embedder, "name", None)
    return name if isinstance(name, str) else type(embedder).__name__


def resolve_registered_embedder(
    profile_id: str,
    env: Mapping[str, str] | None = None,
    *,
    shadow: bool = False,
) -> Embedder:
    """Build one registered profile from its operator supplied artifact settings.

    The registry owns the profile identity and the profile class owns artifact construction. This
    small environment adapter is shared by the CLI and MCP boundaries so a profile cannot resolve
    differently merely because the caller is local or remote. Shadow builds may use the explicitly
    mapped shadow artifact variables, while retaining the same profile ID and context policy.
    """
    from recall.embedding_registry import registered_profile, registered_profile_ids

    values = dict(os.environ if env is None else env)
    if shadow:
        for source, target in (
            ("RECALL_SHADOW_MODEL_CACHE", "RECALL_MODEL_CACHE"),
            ("RECALL_SHADOW_MODEL_SHA256", "RECALL_MODEL_SHA256"),
            ("RECALL_SHADOW_QWEN_MODEL_PATH", "RECALL_QWEN_MODEL_PATH"),
        ):
            if source in values:
                values[target] = values[source]
    try:
        entry = registered_profile(profile_id)
    except ValueError:
        raise ValueError(
            f"unknown RECALL_EMBED_PROFILE: {profile_id!r} "
            f"(registered: {', '.join(registered_profile_ids())})"
        ) from None
    if entry.rejected:
        record = entry.rejection
        assert record is not None
        _log.warning(
            "embedding profile %s was REJECTED on %s (%s) and is being loaded anyway; "
            "the measured reason was %s",
            entry.profile_id,
            record.decided_on,
            record.reason,
            ", ".join(f"{key}={value}" for key, value in record.measurements),
        )
    artifact_digest = values.get("RECALL_MODEL_SHA256", "")
    artifact_path = values.get(entry.artifact_path_env, "")
    if entry.hosted:
        return entry.build(api_key=values.get(entry.api_key_env) or None, env=values)
    if not artifact_path or not artifact_digest:
        raise ValueError(
            f"profile {profile_id!r} requires {entry.artifact_path_env} and "
            "RECALL_MODEL_SHA256"
        )
    return entry.build(artifact_path=artifact_path, artifact_digest=artifact_digest, env=values)


def embed_query(embedder: Embedder, text: str) -> list[float]:
    """Encode one query, falling back to the legacy symmetric interface."""
    method = getattr(embedder, "embed_query", None)
    if callable(method):
        return [float(x) for x in method(text)]
    return [float(x) for x in embedder.embed([text])[0]]


def embed_passages(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    """Encode passages, falling back to the legacy symmetric interface."""
    method = getattr(embedder, "embed_passages", None)
    raw = method(texts) if callable(method) else embedder.embed(texts)
    return [[float(x) for x in vector] for vector in raw]


def embed_document_groups(
    embedder: Embedder, groups: list[list[str]]
) -> list[list[list[float]]]:
    """Embed ordered document groups without flattening a context boundary."""
    method = getattr(embedder, "embed_document_groups", None)
    if callable(method):
        result = method(groups)
        if len(result) != len(groups):
            raise RuntimeError(
                f"embedder {embedder.name!r} returned {len(result)} document groups for "
                f"{len(groups)} inputs"
            )
        for group, vectors in zip(groups, result, strict=True):
            if len(vectors) != len(group):
                raise RuntimeError(
                    f"embedder {embedder.name!r} returned {len(vectors)} vectors for "
                    f"a document group containing {len(group)} chunks"
                )
        return cast(list[list[list[float]]], result)
    return [embed_passages(embedder, group) for group in groups]


def artifact_tree_sha256(path: str | Path, *, follow_file_symlinks: bool = False) -> str:
    """Hash a provisioned file or directory without following directory symlinks.

    ⛔ **`follow_file_symlinks` defaults to False, and that default is deliberate.** The strict
    behaviour is what `verify_artifact` compares against a pinned, declared SHA, where a file
    symlinked in from outside the tree would let unpinned bytes into a verified digest. Nothing that
    checks against a declared expectation should follow links, so the default does not.

    ⚠️ **It is True for provenance, because the strict rule made the digest unobtainable on Linux.**
    `huggingface_hub` stores weights once under `<cache>/models--org--repo/blobs/<etag>` and makes
    `snapshots/<rev>/<file>` a SYMLINK to it. `blobs/` is a sibling of `snapshots/`, so every weight
    file resolves outside the snapshot root and the escape check refuses the entire tree — meaning
    `embedder_artifact_digest` returned None, the identity stayed unverified, and a production
    upload was refused exactly as before. Three auditors reached this independently; the generated
    stack is `python:3.13-slim` and CI is Linux, so that is the deploy target, not an edge case.

    It went unnoticed here because Windows without developer privileges cannot create symlinks at
    all (`WinError 1314`), so the hub copies real files and the strict path succeeds. The measured
    "5 files, 67 MB, 0.95s" in `embedder_artifact_digest` is a Windows measurement.

    Following a FILE symlink is safe for provenance and is in fact the point: the bytes behind the
    link are the bytes the model loaded, and the digest still changes when they change. DIRECTORY
    symlinks are not followed in either mode, because `rglob` does not descend them.

    The recorded name comes from the UNRESOLVED path in this mode, so the digest describes the
    snapshot's own layout rather than the blob store's content-addressed filenames — otherwise two
    identical trees laid out differently would hash differently.
    """
    root = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"model artifact has no files: {root}")
    for file in files:
        resolved = file.resolve(strict=True)
        escapes = root.is_dir() and not resolved.is_relative_to(root)
        if escapes and not follow_file_symlinks:
            raise ValueError(f"model artifact symlink escapes its root: {file}")
        if root.is_file():
            relative = resolved.name
        elif escapes:
            # Named by where it sits in the tree, not by the blob it points at.
            relative = file.relative_to(root).as_posix()
        else:
            relative = resolved.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        with resolved.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


#: Digests already computed, keyed by resolved artifact directory, each stored WITH the cheap
#: directory signature it was computed from. Hashing a model costs ~1s for a 67 MB snapshot, which
#: is cheap once and wasteful per upload.
#:
#: ⛔ **Keyed by path alone, this cached a claim about bytes that may have changed.** The digest
#: exists to say "these are the weights that produced this index"; a cache with no invalidation
#: says it about whatever was there the first time the process looked. A re-download, a partial
#: write, or an edited model file kept the old answer for the life of the process, and a
#: long-running MCP server is exactly the process this matters in.
_ARTIFACT_DIGESTS: dict[str, tuple[tuple[int, int, int], str]] = {}


#: Cleared wholesale past this many entries. A process sees a handful of model directories at most,
#: so this is a runaway guard rather than a policy; clearing costs one re-hash and bounds nothing
#: that matters.
_ARTIFACT_DIGEST_LIMIT = 32


def _artifact_signature(path: Path) -> tuple[int, int, int] | None:
    """File count, total size and newest mtime of an artifact directory. `None` if unreadable.

    Follows symlinks, because `artifact_tree_sha256(..., follow_file_symlinks=True)` does: a
    HuggingFace snapshot is a farm of links into a sibling `blobs/`, and a signature that stopped at
    the link would not see the bytes the digest actually covers.

    ⚠️ **This detects staleness, not tampering, and the difference is worth stating.** Someone able
    to write into the model directory can also set mtimes, so a same-size same-mtime replacement
    keeps the cached digest. The defence against that is recomputing the digest, which is what a
    fresh process does; this only stops the cache from confidently reporting a value it can no
    longer justify. `None` is returned rather than a partial signature when anything cannot be
    stat'd, and an unsignable directory is never cached — recomputing is the safe answer when the
    question "has this changed?" cannot be answered.
    """
    count = 0
    total = 0
    newest = 0
    try:
        for file in sorted(path.rglob("*")):
            if not file.is_file():
                continue
            stat = file.stat()
            count += 1
            total += stat.st_size
            newest = max(newest, stat.st_mtime_ns)
    except OSError:
        return None
    return (count, total, newest)


def embedder_artifact_path(embedder: object) -> Path | None:
    """The directory holding the weights this embedder actually loaded, or None.

    ⚠️ **The model's OWN snapshot directory, never the shared cache.** Measured on this machine:
    `cache_dir` held 45 files and 1.5 GB across several models, so its digest would change whenever
    an unrelated model was downloaded and would not identify anything. `_model_dir` is 5 files and
    67 MB — `model_optimized.onnx`, the tokenizer and the configs — and its directory name is the
    upstream revision hash.

    Returns None rather than guessing when the path cannot be recovered. This reaches into
    fastembed's internals, which are free to change between versions, and a wrong answer here would
    be worse than no answer: it feeds an identity that claims to be verified.
    """
    model = getattr(embedder, "_model", None)
    inner = getattr(model, "model", None)
    raw = getattr(inner, "_model_dir", None)
    if raw is None:
        return None
    try:
        path = Path(str(raw)).resolve(strict=True)
    except (OSError, ValueError):
        return None
    return path if path.is_dir() else None


def embedder_artifact_digest(embedder: object) -> str | None:
    """A SHA256 over the weights this embedder loaded, or None when they cannot be located.

    ⛔ **None is a real answer and must stay one.** `HashingEmbedder` has no artifacts at all: it is
    defined by code, not weights, and there is nothing on disk to hash. Manufacturing a digest for
    it — over the model name, say — would turn an honest "unverified" into a claim of provenance
    that no bytes back, which is worse than the refusal it would bypass.
    """
    path = embedder_artifact_path(embedder)
    if path is None:
        return None
    key = str(path)
    signature = _artifact_signature(path)
    cached = _ARTIFACT_DIGESTS.get(key)
    if cached is not None and signature is not None and cached[0] == signature:
        return cached[1]
    try:
        # `follow_file_symlinks=True`: a HuggingFace snapshot is a farm of symlinks into a
        # sibling `blobs/` directory, and the strict rule refuses the whole tree. See
        # `artifact_tree_sha256`. Without this the digest is None on every Linux install, which
        # is the deploy target.
        digest = artifact_tree_sha256(path, follow_file_symlinks=True)
    except (OSError, ValueError):
        return None
    if signature is not None:
        # Not cached when the directory could not be signed: without a signature there is no way to
        # notice the next change, and a value that cannot be invalidated should not be stored.
        if len(_ARTIFACT_DIGESTS) >= _ARTIFACT_DIGEST_LIMIT:
            _ARTIFACT_DIGESTS.clear()
        _ARTIFACT_DIGESTS[key] = (signature, digest)
    return digest


def verify_artifact(path: str | Path, expected_sha256: str) -> Path:
    """Resolve and checksum a local model artifact before a runtime loads it."""
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in expected_sha256):
        raise ValueError("model artifact SHA256 must be 64 hexadecimal characters")
    resolved = Path(path).resolve(strict=True)
    actual = artifact_tree_sha256(resolved)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"model artifact checksum mismatch: expected {expected_sha256.lower()}, got {actual}"
        )
    return resolved


class HashingEmbedder:
    """Deterministic, dependency-free embedder for tests and offline demos.

    Hashes whitespace tokens into a fixed-width bag-of-words vector, then
    L2-normalizes. Not semantic, but stable and fast — good enough to exercise
    plumbing and to keep the test suite offline.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"hashing-{self._dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode("utf-8"), usedforsecurity=False).hexdigest(), 16)
            vec[h % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
