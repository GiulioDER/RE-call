from __future__ import annotations

import hashlib
import math
import os
import random
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from recall.observability import get_logger
from typing import Literal, Protocol, TypeVar, runtime_checkable

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


class NonTransientError(Exception):
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
    except Exception:  # noqa: BLE001 - a probe must never beat the error it is probing
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
    except Exception:  # noqa: BLE001 - see `_probe`: a hostile __str__ must not beat the error
        # `_probe` closes the attribute door and this closes the other one. Formatting an
        # arbitrary exception runs ITS ``__str__``, which is free to raise — and a body that was
        # never decoded is a realistic way for that to happen. The class name alone still gives
        # the markers something to match on.
        try:
            text = type(exc).__name__.lower()
        except Exception:  # noqa: BLE001 - nested, because the fallback can raise too
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


def _retry_after_seconds(exc: Exception) -> float | None:
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
        return _read_retry_after(exc)
    except Exception:  # noqa: BLE001 - a bad header must never beat the error it arrived on
        return None


def _read_retry_after(exc: Exception) -> float | None:
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
            return _capped(float(raw_ms) / 1000.0)
        except (TypeError, ValueError):
            # Fall through rather than return: a junk millisecond header must not discard a
            # perfectly good `Retry-After` sitting beside it, which is what dropping out here
            # did — straight back onto the 1.5s unpaced budget this function exists to replace.
            pass

    raw = get("retry-after")
    if raw is None:
        return None
    try:
        return _capped(float(raw))
    except (TypeError, ValueError):
        pass
    when = parsedate_to_datetime(raw)
    # A date with no zone is UTC by RFC 9110; without this, subtracting an aware `now` raises.
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return _capped((when - datetime.now(timezone.utc)).total_seconds())


def _capped(seconds: float) -> float | None:
    """The wait if it is both positive and within the cap, else None."""
    return seconds if 0.0 < seconds <= _MAX_RETRY_AFTER_S else None


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
        except Exception as exc:
            last = exc
            if i == attempts - 1 or not is_transient(exc):
                raise
            jitter = random.uniform(0.0, min(max_delay, base_delay * (2 ** i)))
            asked = _retry_after_seconds(exc)
            sleep(jitter if asked is None else asked + jitter)
    assert last is not None  # unreachable: loop either returns or raises
    raise last


def _batches(seq: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def batched_embed(
    texts: list[str],
    embed_batch: Callable[[list[str]], list[list[float]]],
    *,
    batch_size: int = 128,
    max_batch_chars: int | None = None,
) -> list[list[float]]:
    """Embed ``texts`` in provider-safe batches, concatenating results in input order.

    ``embed_batch`` embeds a single batch. Batches are cut on ``batch_size`` (count) and, when
    ``max_batch_chars`` is set, also on a cumulative character budget — a guard against a batch
    that is few in count but huge in tokens. A single text over the char budget still goes out
    alone (never dropped). Order is preserved: batch results are appended in sequence.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive int")
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
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def resolve_thread_budget(
    env: dict[str, str] | None = None, cpu_count: int | None = None
) -> int | None:
    """Threads for the local embedder, or None to leave fastembed's default alone.

    Exists for one situation: several embedding processes sharing a CPU budget. In an unprivileged
    container `os.cpu_count()` reports the HOST's cores while a cgroup quota caps real runtime, and
    fastembed sizes its pool from the former — so N workers each request N x (host cores). Measured
    on a box showing 256 CPUs against a ~61-CPU quota, seven workers spawned ~945 threads and
    aggregate throughput fell to roughly what a single process managed alone.

    Returns None when unset, deliberately. One process is FASTER with the default: measured
    2.2 / 5.7 / 9.0 / 10.3 docs/s at 1 / 8 / 32 / default threads, monotonically increasing. A cap
    must therefore never be the default — it fixes a multi-process regime and pessimises every
    other one.

    Junk is ignored rather than raised: a typo'd variable must not kill hour eight of a long run.
    """
    import os as _os

    raw = (env if env is not None else _os.environ).get("RECALL_EMBED_THREADS")
    if not raw:
        return None
    try:
        want = int(raw)
    except ValueError:
        return None
    if want < 1:
        return None
    ceiling = cpu_count if cpu_count is not None else (_os.cpu_count() or want)
    return min(want, ceiling)


#: What `_session_providers` returns when fastembed's internals do not expose a session. It is a
#: THIRD state, distinct from both a known CPU run and a known GPU run, and it must never be
#: allowed to read as either.
PROVIDERS_UNKNOWN = "<session not reachable>"


def _provider_dependencies(providers: Iterable[str]) -> tuple[tuple[str, str], ...]:
    """The execution provider as fingerprint key material, plus whether it is actually KNOWN.

    Two entries, not one. The provider list alone cannot distinguish "this ran on CPU" from "I
    could not tell what this ran on", and collapsing those two into one key is the same
    negative-guard mistake as reporting an unrecorded session as a match: `_session_providers` is
    documented as observational and must never fail a run, so its failure sentinel reaches here
    on perfectly healthy deployments. Recording the SOURCE alongside the value keeps an
    introspection failure legible in the profile instead of silently minting a new identity that
    looks like a real provider change.
    """
    known = [p for p in providers]
    unknown = not known or known == [PROVIDERS_UNKNOWN]
    return (
        ("onnx-providers-source", "unavailable" if unknown else "session"),
        ("onnx-providers", "unavailable" if unknown else ",".join(known)),
    )


def _with_provider_dependency(
    identity: EmbeddingProfile, providers: Iterable[str]
) -> EmbeddingProfile:
    """Attach the provider entries to a REGISTERED profile without mutating the registry's copy.

    `EmbeddingProfile` is frozen, so this replaces rather than edits. Idempotent: a profile that
    already carries the entries is returned unchanged, so re-wrapping cannot double them and
    change the fingerprint a second time.
    """
    from dataclasses import replace

    if any(k == "onnx-providers" for k, _ in identity.dependencies):
        return identity
    return replace(
        identity, dependencies=identity.dependencies + _provider_dependencies(providers)
    )


def _session_providers(model: object) -> list[str]:
    """The execution providers of fastembed's live ONNX session, or a marker if unreachable.

    fastembed does not expose the session on a stable public attribute, so this walks the couple
    of places it has lived. It returns a marker rather than raising: failing to introspect is not
    a reason to fail an embedding run, but it must not be reported as a known-CPU result either.
    """
    # Wrapped whole. This runs on the CONSTRUCTION path of every FastEmbedEmbedder, including
    # registered enterprise profiles, and it is purely observational — so no change in fastembed's
    # or onnxruntime's internals may ever be the reason a deployment cannot build its embedder.
    # The docstring promised that; nothing enforced it.
    try:
        for outer in ("model", "_model"):
            inner = getattr(model, outer, None)
            if inner is None:
                continue
            for attr in ("model", "session", "_session"):
                getter = getattr(getattr(inner, attr, None), "get_providers", None)
                if callable(getter):
                    return [str(p) for p in getter()]
            inner_getter = getattr(inner, "get_providers", None)
            if callable(inner_getter):
                return [str(p) for p in inner_getter()]
    except Exception:  # pragma: no cover - defensive; fastembed internals are not a contract
        return [PROVIDERS_UNKNOWN]
    # The COMMON path — fastembed simply exposing no session — must return the constant too. It
    # was left as a bare literal, equal by value today, so `_provider_dependencies` still matched
    # it. Change the constant's text and this path would have started recording the sentinel as a
    # provider NAME: exactly the collapse the constant exists to prevent.
    return [PROVIDERS_UNKNOWN]


#: The identifiers the no-identity path has always minted, keyed by ``asymmetric``.
#:
#: They name ONE model at ONE width. Kept as literals rather than derived so that the default
#: embedder's id is stable by construction: it is the key every shipped calibration file, every
#: recorded promotion decision under `results/`, and every corpus indexed by `FastEmbedEmbedder()`
#: is already written under, and re-deriving it would re-partition all of them for no defect.
_LEGACY_FALLBACK_PROFILE_IDS = {
    False: "bge-small-symmetric-v1",
    True: "bge-small-asymmetric-v1",
}


def _fallback_profile_id(model_name: str, dimension: int, asymmetric: bool) -> str:
    """The profile id for an embedder built with no registered identity and no explicit id.

    The legacy literal is returned ONLY when this embedder is the model that literal names, at
    the width the registry declares for it. Anything else gets an id derived from what actually
    varies, because the literal was previously unconditional and a `profile_id` is a CLAIM about
    which model wrote a vector, not a label. Measured 2026-08-18 before this guard existed: a
    `fastembed:BAAI/bge-large-en-v1.5` embedder reported ``dim=1024`` under
    ``profile_id='bge-small-symmetric-v1'``, an id whose registry entry is 384-dimensional, and a
    production corpus of 8,716 chunks had stored that pairing in its chunk metadata.

    Why the id and not just the fingerprint. `EmbeddingProfile.fingerprint` already covers
    `model_name` and `dimension`, so the embedding cache (`recall/cache.py`) never confused the
    two models. `recall.index._index_fingerprint` does not: it hashes `embedding_profile_id`
    alone, so under the unconditional literal a bge-small corpus and a bge-large corpus produced
    the SAME index fingerprint for the same file, and the incremental skip guard treated a model
    swap as a no-op. Verified by execution: the 384-dimension and 1024-dimension fingerprints were
    equal.

    The comparison covers `model_name` and `dimension` only. The encoder modes cannot disagree
    here: the legacy id is looked up BY ``asymmetric``, and both the registry pair and this
    fallback derive their modes from that same flag, so a mode check could never fail and would
    be a guard that cannot fire.
    """
    # Function-local: `recall.embedding_registry` imports this module at module level, so a
    # top-level import here would be a cycle. Safe at call time because the registry only builds
    # a `FastEmbedEmbedder` inside `RegisteredProfile.build`, never while it is being imported.
    from recall.embedding_registry import find_registered_profile

    # `bool(...)` because the expression this replaced was a conditional on TRUTHINESS, and a dict
    # lookup is not. A library caller passing `asymmetric=2` (or a string out of a config file)
    # used to get the asymmetric branch and would now get a KeyError from a public constructor.
    # Narrowing a public signature is not part of this fix.
    asymmetric = bool(asymmetric)
    legacy = _LEGACY_FALLBACK_PROFILE_IDS[asymmetric]
    entry = find_registered_profile(legacy)
    if entry is not None and entry.model_name == model_name and entry.dimension == dimension:
        return legacy
    # Namespaced so the id is self-describing in a chunk's metadata and cannot be mistaken for a
    # registered profile. Injective in exactly the three inputs that reach it, which is the
    # property `_index_fingerprint` needs and the regression test pins.
    #
    # ⚠️ FILENAME-SAFE, which is a constraint and not a style choice. A profile id is not only
    # compared: `recall.eval.promotion.run.ArmConfig.key` interpolates it into a result FILENAME,
    # and its own docstring says so ("a generation id can be long and a filename cannot"). A raw
    # HuggingFace name would put a `/` in that path and a `:` that Windows refuses outright, so
    # the separator is `__` and the org separator goes the same way. `SparseProfile` already
    # resolved this identically (`model_name.replace("/", "__")`); this follows that precedent
    # rather than inventing a second convention. Injectivity survives it for every real model
    # name, none of which contain `__`.
    kind = "asymmetric" if asymmetric else "symmetric"
    return f"unregistered__{model_name.replace('/', '__')}__{dimension}__{kind}"


_log = get_logger("embeddings")


#: Bad `RECALL_FASTEMBED_BATCH` values already reported. Deduplicated by VALUE, not by a single
#: flag: a variable corrected mid-process should warn again for the new bad value.
_WARNED_BATCH_VALUES: set[str] = set()


def _warn_once(raw: str, problem: str) -> None:
    """Report a bad batch setting the FIRST time it is seen, and not on every batch after.

    `_batch_size_from_env` runs once per batch, so warning unconditionally produced hundreds of
    identical lines during a single index and buried anything real. The docstring below claimed
    "warns once" before the code did.
    """
    if raw in _WARNED_BATCH_VALUES:
        return
    _WARNED_BATCH_VALUES.add(raw)
    _log.warning("RECALL_FASTEMBED_BATCH=%r %s; using the backend default", raw, problem)


def _batch_size_from_env() -> int | None:
    """`RECALL_FASTEMBED_BATCH` as a positive int, or `None` for the backend's own default.

    ⚠️ **An unreadable value degrades rather than raising.** `int()` on a mistyped variable used to
    raise `ValueError` out of the middle of an index run, after several projects had already been
    written. A guard against running out of memory that instead kills the job on a typo is worse
    than no guard. It warns once so the setting is not silently ignored either.
    """
    raw = os.environ.get("RECALL_FASTEMBED_BATCH")
    if not raw:
        return None
    try:
        size = int(raw)
    except ValueError:
        _warn_once(raw, "is not an integer")
        return None
    if size < 1:
        _warn_once(raw, "is not positive")
        return None
    return size


class FastEmbedEmbedder:
    """Real local embeddings (no API key). Requires `pip install "recall-rag[fastembed]"`."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        asymmetric: bool = False,
        profile_id: str | None = None,
        cache_dir: str | Path | None = None,
        artifact_sha256: str | None = None,
        require_local: bool = False,
        context_version: str = "raw-v1",
        identity: EmbeddingProfile | None = None,
        providers: list[str] | None = None,
    ) -> None:
        """Load a local fastembed model, optionally under a supplied immutable identity.

        ``identity`` is how a registered profile is built: `RegisteredProfile.build` passes the
        one `EmbeddingProfile` the registry constructed, and the encoder methods named in it
        (``query_mode`` / ``passage_mode``) are the ones actually called. Without it the class
        keeps its previous behaviour and derives a profile from ``asymmetric``, the legacy
        default path, where no artifact is pinned and nothing enterprise depends on the result.

        ``providers`` is an ONNX Runtime execution-provider REQUEST forwarded to fastembed. It is
        not a guarantee: asking for ``CUDAExecutionProvider`` against a wheel built for a
        different CUDA major falls back to CPU with only a ``RuntimeWarning``. Read
        ``self.session_providers`` for what the session actually resolved — never
        ``onnxruntime.get_available_providers()``, which reports what the wheel was compiled with
        and stays true while the session sits on CPU.
        """
        # Artifact first, backend second. A deployment whose weights are missing or tampered
        # with gets that error whether or not the optional extra happens to be installed, and
        # nothing loads before the tree has been checksummed.
        if require_local and cache_dir is None:
            raise ValueError("offline embedding profiles require a provisioned cache_dir")
        if require_local and artifact_sha256 is None:
            raise ValueError("offline embedding profiles require an artifact_sha256")
        local_cache = None
        if cache_dir is not None:
            local_cache = str(
                verify_artifact(cache_dir, artifact_sha256)
                if artifact_sha256 is not None
                else Path(cache_dir).resolve(strict=True)
            )
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "FastEmbedEmbedder requires the fastembed extra: "
                'pip install "recall-rag[fastembed]"'
            ) from exc
        threads = resolve_thread_budget()
        kwargs: dict[str, object] = {"model_name": identity.model_name if identity else model_name}
        if threads is not None:
            kwargs["threads"] = threads
        if local_cache is not None:
            kwargs["cache_dir"] = local_cache
        if require_local:
            kwargs["local_files_only"] = True
        if providers is not None:
            kwargs["providers"] = list(providers)
        self._model = (
            TextEmbedding(**kwargs)
        )
        #: The providers the ONNX session ACTUALLY resolved, read back from the live session.
        #:
        #: NOT `onnxruntime.get_available_providers()`, which reports what the wheel was compiled
        #: with and stays true even when the session ran on CPU. Measured on an RTX 5090 host:
        #: requesting `CUDAExecutionProvider` against a CUDA-13 wheel on a CUDA-12.8 box falls
        #: back to CPU with only a `RuntimeWarning`, so anything trusting availability would
        #: record a GPU run that never happened. Exposed because an index built on one provider
        #: and served from another is a provenance fact a benchmark has to be able to state.
        self.session_providers = _session_providers(self._model)
        self._name = identity.model_name if identity else model_name
        self._query_mode = identity.query_mode if identity else (
            "query_embed" if asymmetric else "embed"
        )
        self._passage_mode = identity.passage_mode if identity else (
            "passage_embed" if asymmetric else "embed"
        )
        # Both encoders are resolved HERE, at construction. Resolving the query encoder lazily
        # would let a deployment index an entire corpus and only discover a missing encoder on
        # its first query, which is the worst possible moment to find out.
        self._encoder(self._query_mode)
        # Dimension discovery goes through the PASSAGE encoder: the stored vectors are passages,
        # so the width the store is built at must be the width the passage encoder produces.
        probe = next(iter(self._encoder(self._passage_mode)(["probe"])))
        self._dim = len(list(probe))
        if identity is not None and self._dim != identity.dimension:
            raise ValueError(
                f"profile {identity.profile_id!r} declares dimension {identity.dimension} but "
                f"the provisioned artifact embeds at {self._dim}; this artifact is not that "
                f"profile"
            )
        # Applied to BOTH branches. `identity or EmbeddingProfile(...)` short-circuits, so building
        # the provider pair only inside the right-hand side left every REGISTERED profile carrying
        # fastembed's version and nothing else. A fingerprint fix that skips the registered path
        # is not a fix.
        #
        # ⚠️ SCOPE, stated precisely because an earlier version of this comment overstated it:
        # what this reaches is the v1 profile-fingerprint binding (`calibration.load_for_profile`)
        # and the embedding cache key (`recall/cache.py`). It does NOT reach the CERTIFIED v2
        # binding, which stores `recall.lineage.EmbedderIdentity` — provider, model, dimension,
        # revision, artifact_digest — and has no `dependencies` field at all. A CPU-fit certified
        # calibration therefore still binds cleanly to a CUDA-served pipeline. Closing that needs
        # the providers added to `EmbedderIdentity`, which is a separate change.
        self._profile = (
            _with_provider_dependency(identity, self.session_providers)
            if identity
            else EmbeddingProfile(
                profile_id=profile_id or _fallback_profile_id(
                    model_name, self._dim, asymmetric
                ),
                model_name=model_name,
                artifact_digest=artifact_sha256 or LEGACY_UNVERIFIED_DIGEST,
                dimension=self._dim,
                query_mode=self._query_mode,
                passage_mode=self._passage_mode,
                context_version=context_version,
                dependencies=(
                    ("fastembed", _package_version("fastembed")),
                    # The ONNX execution provider is KEY MATERIAL, not metadata. This class's own
                    # fingerprint docstring gives the reason — "ONNX runtime changes are free to
                    # move the last bits of a vector and a cache cannot tell" — and a provider
                    # swap is exactly such a change. Measured on an RTX 5090: CPU and CUDA
                    # sessions over the same weights moved top-45 SET membership on 2 of 64
                    # queries. Without this the two provenances share one cache key
                    # (recall/cache.py) and one calibration binding (recall/calibration.py), so
                    # CPU vectors would be served for a GPU-configured embedder. fastembed also
                    # reaches CUDA on its own via `cuda=Device.AUTO` whenever onnxruntime-gpu is
                    # importable, so this fires without anyone passing `providers=`.
                    *_provider_dependencies(self.session_providers),
                ),
            )
        )

    def _encoder(self, mode: str) -> Callable[[list[str]], Iterable[Iterable[float]]]:
        """Resolve the encoder a profile NAMES, refusing a mode this backend does not have.

        Refusing matters: a missing `query_embed` that silently fell back to `embed` would give
        an asymmetric profile passage vectors for its queries, which is invisible downstream.
        """
        method = getattr(self._model, mode, None)
        if not callable(method):
            raise ValueError(f"fastembed model has no encoder named {mode!r}")
        encoder: Callable[[list[str]], Iterable[Iterable[float]]] = method
        return encoder

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in next(iter(self._encoder(self._query_mode)([text])))]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        # ⚠️ **`RECALL_FASTEMBED_BATCH` bounds the batch fastembed builds internally.** Unset, this
        # behaves exactly as before. The default of 256 is too large for a 1024-dim model on long
        # passages: bge-large asked onnxruntime for a single 1.24 GB buffer and the arena refused,
        # which fails an index part-way through rather than merely running slowly.
        #
        # Passed as fastembed's OWN parameter rather than by slicing the list here. An earlier
        # version sliced and called the encoder once per slice, and at size 16 that wedged: 152 of
        # 208 files in, eight cores pinned, no database writes for three minutes, and every server
        # connection idle in `ClientRead`, so the stall was in this process rather than on I/O.
        encoder = self._encoder(self._passage_mode)
        size = _batch_size_from_env()
        if size is not None:
            try:
                return [
                    [float(x) for x in vec]
                    for vec in encoder(texts, batch_size=size)  # type: ignore[call-arg]
                ]
            except TypeError:
                # This backend's encoder does not take the argument. Fall through rather than fail:
                # the variable is a memory guard, not a contract.
                pass
        return [[float(x) for x in vec] for vec in encoder(texts)]


SFR_CODE_EMBEDDER_MODEL = "Salesforce/SFR-Embedding-Code-2B_R"
SFR_CODE_EMBEDDER_REVISION = "c73d8631a005876ed5abde34db514b1fb6566973"
REMOTE_MODEL_CODE_OPT_IN = "RECALL_ACCEPT_REMOTE_MODEL_CODE"


def _truthy_env(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _require_research_model_opt_in(source: Mapping[str, str], model: str) -> None:
    if not _truthy_env(source.get("RECALL_ACCEPT_RESEARCH_MODEL_LICENSE")):
        raise ValueError(
            f"{model} is a research/Gemma-terms model, not a default RE-call shipping model. "
            "Set RECALL_ACCEPT_RESEARCH_MODEL_LICENSE=1 to use the named research alias, or pass "
            "the full st:<model> spelling if you are deliberately managing the licence outside "
            "RE-call."
        )


def _require_remote_model_code_opt_in(source: Mapping[str, str], model: str) -> None:
    if not _truthy_env(source.get(REMOTE_MODEL_CODE_OPT_IN)):
        raise ValueError(
            f"{model} requires Hugging Face remote model code. Set {REMOTE_MODEL_CODE_OPT_IN}=1 "
            "only after reviewing the pinned model revision and accepting that the model repository "
            "can execute Python code during load."
        )


class SentenceTransformerEmbedder:
    """Any `sentence-transformers` model by name or local path — including one fine-tuned here.

    `finetune/train.py` writes a model to disk; without a way to LOAD it back into the retrieval
    stack, a fine-tuning result can only ever be measured by the trainer's own evaluator, on its
    own split. Pointing the real harness at the saved directory is what makes the lift comparable
    to every other embedder measured in this repo:

        python -m recall.eval.labelled --embedder st:finetune/model ...

    Requires `pip install "recall-rag[rerank]"` (or `[entail]`) — both pull sentence-transformers.
    """

    def __init__(
        self,
        model: str,
        batch_size: int = 64,
        *,
        trust_remote_code: bool = False,
        revision: str | None = None,
        name: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                'SentenceTransformerEmbedder requires: pip install "recall-rag[rerank]"'
            ) from exc
        kwargs: dict[str, object] = {"trust_remote_code": trust_remote_code}
        if revision is not None:
            kwargs["revision"] = revision
        self._model = SentenceTransformer(model, **kwargs)
        self._name = name or f"st:{model}"
        self._batch_size = batch_size
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    def embed(self, texts: list[str]) -> list[list[float]]:
        # normalize_embeddings: the store scores with cosine distance, and an unnormalised
        # vector would make those scores incomparable with every other embedder here.
        vecs = self._model.encode(
            texts, batch_size=self._batch_size, normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in v] for v in vecs]


QWEN3_RETRIEVAL_INSTRUCTION_V1 = (
    "Given a user question, retrieve passages that answer the question"
)


class Qwen3EmbeddingEmbedder:
    """Offline, instruction-aware Qwen3 0.6B experiment truncated to 384 dimensions."""

    def __init__(
        self,
        model_path: str | Path,
        artifact_sha256: str,
        *,
        dimension: int = 384,
        context_version: str = "raw-v1",
        batch_size: int = 32,
        identity: EmbeddingProfile | None = None,
    ) -> None:
        """Load the offline Qwen3 artifact under the identity the registry built for it.

        See `recall.embedding_registry` for the recorded rejection: this profile was measured on
        CPU and refused on latency. The class is retained so the negative result stays
        reproducible, not because the profile is a candidate.
        """
        if identity is not None:
            dimension = identity.dimension
        if dimension != 384:
            raise ValueError("the registered Qwen3 experiment is fixed at 384 dimensions")
        local = verify_artifact(model_path, artifact_sha256)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'Qwen3EmbeddingEmbedder requires: pip install "recall-rag[rerank]"'
            ) from exc
        threads = resolve_thread_budget()
        if threads is not None:
            import torch

            torch.set_num_threads(threads)
        self._model = SentenceTransformer(
            str(local), local_files_only=True, truncate_dim=dimension
        )
        self._dim = dimension
        self._batch_size = batch_size
        self._profile = identity or EmbeddingProfile(
            profile_id="qwen3-embedding-0.6b-384-v1",
            model_name="Qwen/Qwen3-Embedding-0.6B",
            artifact_digest=artifact_sha256,
            dimension=dimension,
            query_mode="instruction-v1",
            passage_mode="document",
            instruction_version="retrieval-v1",
            context_version=context_version,
            dependencies=(("sentence-transformers", _package_version("sentence-transformers")),),
        )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._profile.model_name

    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    def _encode(self, texts: list[str], prompt: str | None = None) -> list[list[float]]:
        kwargs: dict[str, object] = {
            "batch_size": self._batch_size,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
        if prompt is not None:
            kwargs["prompt"] = prompt
        vectors = self._model.encode(texts, **kwargs)
        normalized: list[list[float]] = []
        for vector in vectors:
            values = [float(x) for x in vector]
            # Sentence Transformers normalizes before truncate_dim is applied for this
            # model, so the returned 384-wide prefix is no longer unit length.  Normalize
            # the final representation explicitly, which is the vector stored and scored.
            norm = math.sqrt(sum(value * value for value in values))
            if norm == 0.0:
                raise RuntimeError("Qwen3 embedding produced a zero vector")
            normalized.append([value / norm for value in values])
        return normalized

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_passages(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], prompt=QWEN3_RETRIEVAL_INSTRUCTION_V1)[0]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)


class VoyageEmbedder:
    """Voyage cloud embeddings. Requires `pip install "recall-rag[voyage]"` and VOYAGE_API_KEY."""

    def __init__(
        self,
        model: str = "voyage-3",
        api_key: str | None = None,
        batch_size: int = 128,
        max_retries: int = 3,
        identity: EmbeddingProfile | None = None,
    ) -> None:
        """Build a Voyage client, optionally under a registered profile's immutable identity.

        ``identity`` is how a registered hosted profile is built: `RegisteredProfile.build` passes
        the identity it declared, and this class then carries THAT object rather than minting a
        second one. Without it every consumer falls back to `legacy_embedding_profile`, so the
        profile id a generation and a calibration were registered under could never match the
        runtime and the corpus could not be served. When it is supplied, the identity's model name
        is what gets sent to the provider, so the registry decides the request rather than
        describing it.
        """
        key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise RuntimeError("VoyageEmbedder needs VOYAGE_API_KEY (env) or an explicit api_key")
        try:
            import voyageai
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError('VoyageEmbedder requires: pip install "recall-rag[voyage]"') from exc
        # Stated rather than inherited: voyageai already defaults `max_retries` to 0, so this
        # changes nothing today. It pins the same single-owner policy `OpenAICompatEmbedder`
        # needs explicitly, so that an SDK release which starts retrying cannot quietly
        # reintroduce the multiplication with `retry_with_backoff` in `embed` below.
        self._client = voyageai.Client(api_key=key, max_retries=0)
        self._model = identity.model_name if identity is not None else model
        self._name = f"voyage:{self._model}"
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._dim = len(self._client.embed(["probe"], model=self._model).embeddings[0])
        _check_declared_width(identity, self._dim, "the Voyage endpoint")
        self._profile = identity

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    @property
    def profile(self) -> EmbeddingProfile | None:
        """The registered identity, or None for a legacy construction.

        `embedding_profile` reads this attribute and falls back to `legacy_embedding_profile`
        when it is not an `EmbeddingProfile`, so returning None preserves every existing caller.
        """
        return self._profile

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed in provider-safe batches with exponential-backoff retry per batch.

        A single request sending every chunk at once will exceed the API's per-request limit on
        a real corpus and has no tolerance for a transient 429/5xx; batching + retry make bulk
        indexing survivable. Results are concatenated in input order (see ``batched_embed``).
        """
        def _embed_batch(batch: list[str]) -> list[list[float]]:
            result = retry_with_backoff(
                lambda: self._client.embed(batch, model=self._model),
                attempts=self._max_retries,
            )
            return [[float(x) for x in v] for v in result.embeddings]

        return batched_embed(texts, _embed_batch, batch_size=self._batch_size)


class OpenAICompatEmbedder:
    """OpenAI-compatible cloud embeddings via any ``base_url`` (OpenRouter, OpenAI, Azure, vLLM).

    Defaults to OpenRouter so the exact ``openai/text-embedding-3-small`` model is reachable on the
    same key the benchmark already uses for its generator and judge — no separate OpenAI billing,
    which is the whole reason this backend exists. The request/response shape is OpenAI's
    ``/v1/embeddings``, so the stock ``openai`` SDK works unchanged against OpenRouter's endpoint.

    ``dimensions`` is optional rather than implicit: ``text-embedding-3-small`` returns its native
    1536-wide vector when the field is omitted, while newer OpenRouter embedding models such as
    Gemini Embedding 2 expose selectable output widths.
    """

    def __init__(
        self,
        model: str = "openai/text-embedding-3-small",
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        batch_size: int = 128,
        max_retries: int = 3,
        dimensions: int | None = None,
        name_prefix: str = "openai",
        identity: EmbeddingProfile | None = None,
    ) -> None:
        """Build an OpenAI-compatible client, optionally under a registered profile's identity.

        See `VoyageEmbedder.__init__` for why ``identity`` matters; the reasoning is identical.
        Note that ``base_url`` and ``dimensions`` are NOT derivable from the identity and must be
        passed alongside it: the same model name answers at different widths depending on the
        ``dimensions`` field, so a registered profile has to supply both and `_check_declared_width`
        then holds them to it.
        """
        key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OpenAICompatEmbedder needs an API key (OPENROUTER_API_KEY or OPENAI_API_KEY in "
                "the environment, or an explicit api_key)"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the SDK
            raise ImportError(
                'OpenAICompatEmbedder requires: pip install "recall-rag[openai]"'
            ) from exc
        # `max_retries=0` because `retry_with_backoff` in `_embed_one_batch` owns the retry
        # policy. The SDK default is 2 retries, which does not replace ours but multiplies with
        # it: one 429 costs 3 x 3 = 9 requests rather than the 3 the policy asks for, and the
        # outer FULL-jitter backoff (which exists so a fleet does not remarch onto the provider
        # in lockstep) ends up wrapping an inner loop whose own jitter is only `1 - 0.25 *
        # random()` on its doubling schedule — a 25% smear, not a spread across the interval, so
        # the fleet it is meant to separate stays largely in step. This is the
        # corpus indexing path, so the multiplication lands batch after batch on a provider that
        # has just said it is overloaded.
        self._client = OpenAI(api_key=key, base_url=base_url, max_retries=0)
        self._model = identity.model_name if identity is not None else model
        self._name = f"{name_prefix}:{self._model}"
        self._batch_size = batch_size
        self._max_retries = max_retries
        if dimensions is not None and dimensions < 1:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions
        # Probe the width once, the same way the other cloud embedder does, so a store can be built
        # at the matching ``dim`` before the first real batch is embedded.
        self._dim = len(self._embed_one_batch(["probe"])[0])
        _check_declared_width(identity, self._dim, f"{base_url} model {self._model!r}")
        self._profile = identity

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    @property
    def profile(self) -> EmbeddingProfile | None:
        """The registered identity, or None for a legacy construction."""
        return self._profile

    def _embed_one_batch(self, batch: list[str]) -> list[list[float]]:
        request: dict[str, object] = {
            "model": self._model,
            "input": batch,
            "encoding_format": "float",
        }
        if self._dimensions is not None:
            request["dimensions"] = self._dimensions
        result = retry_with_backoff(
            lambda: self._client.embeddings.create(**request),
            attempts=self._max_retries,
        )
        return [[float(x) for x in item.embedding] for item in result.data]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed in provider-safe batches with exponential-backoff retry per batch — the same
        contract as ``VoyageEmbedder.embed``, so this is a drop-in cloud embedder on the RE-call
        arm."""
        return batched_embed(texts, self._embed_one_batch, batch_size=self._batch_size)


def _optional_dimensions(source: Mapping[str, str]) -> int | None:
    raw = source.get("RECALL_EMBED_DIMENSIONS", "").strip()
    if not raw:
        return None
    try:
        dimensions = int(raw)
    except ValueError as exc:
        raise ValueError("RECALL_EMBED_DIMENSIONS must be a positive integer") from exc
    if dimensions < 1:
        raise ValueError("RECALL_EMBED_DIMENSIONS must be a positive integer")
    return dimensions


def resolve_embedder(name: str, env: dict[str, str] | None = None) -> Embedder:
    """Build an embedder from a short config string.

    Supported spellings:
    ``hashing``, ``fastembed``, ``fastembed:<model>``, ``st:<model>``,
    ``sfr-code``, ``voyage``, ``voyage:<model>``, ``openai``, ``openai:<model>``,
    ``openrouter`` and ``openrouter:<model>``.
    """
    if name == "hashing" or name.startswith("hashing-") or name.startswith("hashing:"):
        return HashingEmbedder(dim=64)
    if name == "fastembed":
        return FastEmbedEmbedder()
    if name.startswith("fastembed:"):
        return FastEmbedEmbedder(model_name=name[len("fastembed:"):])
    source = os.environ if env is None else env
    if name.startswith("st:"):
        return SentenceTransformerEmbedder(name[3:])
    if name == "sfr-code":
        _require_research_model_opt_in(source, SFR_CODE_EMBEDDER_MODEL)
        _require_remote_model_code_opt_in(source, SFR_CODE_EMBEDDER_MODEL)
        return SentenceTransformerEmbedder(
            SFR_CODE_EMBEDDER_MODEL,
            trust_remote_code=True,
            revision=SFR_CODE_EMBEDDER_REVISION,
            name=f"sfr-code:{SFR_CODE_EMBEDDER_MODEL}",
        )
    if name == "voyage":
        return VoyageEmbedder(api_key=source.get("VOYAGE_API_KEY"))
    if name.startswith("voyage:"):
        return VoyageEmbedder(
            model=name[len("voyage:"):], api_key=source.get("VOYAGE_API_KEY")
        )
    if name == "openai":
        return OpenAICompatEmbedder(
            api_key=source.get("OPENROUTER_API_KEY") or source.get("OPENAI_API_KEY"),
            dimensions=_optional_dimensions(source),
        )
    if name.startswith("openai:"):
        return OpenAICompatEmbedder(
            model=name[len("openai:"):],
            api_key=source.get("OPENROUTER_API_KEY") or source.get("OPENAI_API_KEY"),
            dimensions=_optional_dimensions(source),
        )
    if name == "openrouter":
        return OpenAICompatEmbedder(
            model="google/gemini-embedding-2",
            api_key=source.get("OPENROUTER_API_KEY") or source.get("OPENAI_API_KEY"),
            dimensions=_optional_dimensions(source),
            name_prefix="openrouter",
        )
    if name == "gemini-embedding-2":
        return OpenAICompatEmbedder(
            model="google/gemini-embedding-2",
            api_key=source.get("OPENROUTER_API_KEY") or source.get("OPENAI_API_KEY"),
            dimensions=_optional_dimensions(source),
            name_prefix="openrouter",
        )
    if name.startswith("openrouter:"):
        return OpenAICompatEmbedder(
            model=name[len("openrouter:"):],
            api_key=source.get("OPENROUTER_API_KEY") or source.get("OPENAI_API_KEY"),
            dimensions=_optional_dimensions(source),
            name_prefix="openrouter",
        )
    raise ValueError(
        f"unknown embedder: {name!r} (use hashing, fastembed, fastembed:<model>, "
        "st:<model>, sfr-code, voyage, voyage:<model>, openai, openai:<model>, "
        "openrouter, or openrouter:<model>)"
    )
