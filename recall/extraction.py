"""Model-backed extraction of declared relations from memo prose. OFF by default.

**The prior is a measured failure, not a hunch.** `fix.py:10` states it: "Measured on that
corpus, it proposes ZERO edges. Four survived the mechanical rules and all four were wrong on
review." The four were reported speech, superseding a claim *inside* the target twice, and a
hedge whose author, asked directly, said *augments*. Every one of those is a semantic
distinction — narrating versus declaring, part versus whole, qualifying versus committing — and
no pattern over the text can see any of them. That is the entire argument for a model here, and
it is also the reason the refusal rules are restated in the prompt rather than assumed.

**Less of a departure than it looks.** The library already ships an OpenAI-compatible HTTP client
defaulting to the OpenRouter base URL (`embeddings.py:806`), and `openai>=1.0` is already an extra
under `bench`. This is the first *generative* call, not the first network model call.

**What the model is trusted with, and what it is not.** It supplies semantics: which relation, to
which document, on what evidence. It supplies no identity — proposal ids are content hashes the
library computes, and `_providers.py:86` recomputes and raises on mismatch. It supplies no
authority — nothing here reaches corpus metadata; that requires a named human going through
`promotion.py`, and `rewrite.py` will not accept anything else.

**Determinism, and the honest caveat.** Temperature 0, a pinned revision, and a cache keyed on
``(sha256(human_body), model_id, model_revision, prompt_version)``. Temperature 0 is not a
guarantee from any hosted provider, so `recheck` exists to *measure* whether it holds here rather
than assume it: it deliberately re-calls the model on cached keys and reports the mismatch rate.
A non-zero rate means the cache, not the sampler, is what makes runs reproducible — which is
worth knowing before a cache eviction silently renumbers every proposal id derived from it.

**Frontmatter is not in the prompt and not in the cache key.** Suppressing already-declared
values belongs to phase 2, which sees frontmatter through chunk metadata. Keeping it out of the
key means a frontmatter edit — including one this system just made through `rewrite.py` — never
re-invokes the model, so applying an accepted proposal does not invalidate the corpus it came
from.

Requires ``pip install recall[extract]``.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from recall.provider_metadata import ProviderMetadata

#: Default generative model, reached over the OpenRouter base URL the library already uses.
#: A DATED slug, not a floating alias: an alias moves under you and every cached claim then
#: belongs to weights nobody can name. Override with `RECALL_EXTRACT_MODEL`.
DEFAULT_EXTRACT_MODEL = "openai/gpt-4o-mini-2024-07-18"
#: The revision pin, recorded in the cache key and in every artifact. It is a separate field from
#: the model id for the same reason `entailment.py` pins a Hub commit beside a model name: the
#: name is what you asked for, the revision is what answered.
DEFAULT_EXTRACT_REVISION = "2024-07-18"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

PROMPT_VERSION = "extract-v1"

#: Upper bound on claims from one memo. A memo declaring more relations than this is not a
#: precision win, it is a runaway generation, and the message wording below is load-bearing:
#: `_providers.py:227` keys `wrong_cardinality` off the substring "maximum is", so reusing it
#: reaches that mapping without adding a second failure-handling path.
MAX_CLAIMS = 8

#: Only relations `rewrite.py` has a destination for. `references` is deliberately absent: the
#: write path refuses it, and an extractor that emits claims the writer cannot place would
#: reproduce the `fix.py:62` failure of output a human must themselves filter.
ExtractedRelation = Literal["supersedes", "contradicts", "same_entity"]
_RELATIONS: frozenset[str] = frozenset({"supersedes", "contradicts", "same_entity"})

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})

#: FROZEN. Changing a single byte changes `PROMPT_SHA256`, which is recorded on every claim and
#: in every artifact. The precision prediction for this extractor is conditional on this text;
#: an unhashed prompt would make that condition unverifiable after the fact, which turns the
#: prediction into a claim that cannot be checked.
#:
#: The scope rule is stated explicitly because it is the refusal the model is predicted to need:
#: two of `fix.py`'s four wrong proposals were a claim or a scope *inside* the target.
PROMPT_TEMPLATE = """\
You read one memo from a personal knowledge corpus and report only the relations its author \
actually DECLARES. You are given the memo's prose and the list of document names that exist in \
the corpus. You may name a document only if it appears in that list.

Refuse, and return no claim, in each of these cases:

* Reported speech. If the subject of the sentence is some other document rather than the memo \
you are reading, the memo is NARRATING a relation, not declaring its own. Attributing it to this \
memo invents a second claimant for an edge another document already states correctly.
* Partial scope. A supersession demotes the entire predecessor. When only part of the target is \
replaced, as in "supersedes the estimate in X" or "supersedes the scope of X", the whole of X is \
not superseded and you must refuse. Everything else X holds is still current.
* Hedging. When the author qualified the claim, as in "largely supersedes" or \
"supersedes/augments", they declined to commit. Resolving that for them is a guess, and an \
augmenting memo does not replace its predecessor.
* No named target. A bare closure marker with no document named in the same sentence is never to \
be guessed at.

An absent relation costs nothing. A wrong one demotes a memo that is still current, which is the \
exact failure the surrounding system exists to prevent.

Reply with JSON and nothing else, shaped as {"claims": [...]}. Each claim is an object with:
  "relation"   one of: supersedes, contradicts, same_entity
  "object"     a name copied exactly from the corpus list
  "evidence"   a span copied VERBATIM from the memo prose, long enough to judge the claim
  "confidence" a number between zero and one

Reply with {"claims": []} when the memo declares nothing. That is the expected answer for most \
memos.

CORPUS DOCUMENT NAMES:
{corpus_names}

MEMO PROSE:
{human_body}
"""

PROMPT_SHA256 = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


@runtime_checkable
class ChatClient(Protocol):
    """The narrow slice of an OpenAI-compatible client this module uses.

    A protocol rather than the SDK type so the contract tests can capture the bytes that leave
    the process. The leakage guard has to run against what the provider would actually receive;
    asserting against the template instead would pass for a caller that appends frontmatter on
    the way out, which is the defect worth catching.
    """

    def complete(
        self, *, model: str, messages: list[dict[str, str]], temperature: float
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class ExtractedClaim:
    """One relation the model says the author declared, with the span that says it."""

    relation: ExtractedRelation
    subject: str
    object: str
    evidence: str
    confidence: float
    prompt_version: str = PROMPT_VERSION
    prompt_sha256: str = PROMPT_SHA256


@dataclass(frozen=True)
class RecheckReport:
    """The result of deliberately re-calling the model on cached keys."""

    rechecked: int
    mismatches: int

    @property
    def mismatch_rate(self) -> float:
        return self.mismatches / self.rechecked if self.rechecked else 0.0


def claim_cache_key(
    human_body: str, *, model_id: str, model_revision: str, prompt_version: str
) -> str:
    """Content-address one extraction by everything that can change its answer.

    Mirrors `cache.py:58`, including its reason: a key that misses part of the identity serves a
    result computed under different conditions and nothing downstream can tell. Frontmatter is
    deliberately NOT part of this — see the module docstring.
    """
    digest = hashlib.sha256()
    for part in (
        hashlib.sha256(human_body.encode("utf-8")).hexdigest(),
        model_id,
        model_revision,
        prompt_version,
    ):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def build_prompt(human_body: str, corpus_names: frozenset[str] | set[str]) -> list[dict[str, str]]:
    """The exact messages sent to the provider. Pure, and sorted so it cannot vary by set order.

    A set's iteration order is stable within a process and not across them, so an unsorted name
    list would make the request — and any hash taken over it — differ between runs while every
    input was identical.
    """
    names = "\n".join(sorted(corpus_names))
    content = PROMPT_TEMPLATE.replace("{corpus_names}", names).replace(
        "{human_body}", human_body
    )
    return [{"role": "user", "content": content}]


class ClaimCache:
    """SQLite sidecar mapping a claim cache key to the provider's raw response text.

    ``put`` REFUSES to overwrite an existing key with different content and raises. That is the
    unusual choice and the deliberate one: proposal ids are content hashes over these claims, so
    an entry silently changing under a stable key would renumber ids that other artifacts already
    reference, and the renumbering would look like new findings rather than a cache event.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if self._path.parent and not self._path.parent.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS claims (key TEXT PRIMARY KEY, response TEXT NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT response FROM claims WHERE key = ?", (key,)
        ).fetchone()
        return str(row[0]) if row else None

    def put(self, key: str, response: str) -> None:
        existing = self.get(key)
        if existing is not None:
            if existing == response:
                return  # idempotent: a repeat of the same answer is not a conflict
            raise ValueError(
                f"claim cache key {key[:12]}… already holds different content; refusing to "
                "overwrite, because proposal ids are hashes over this response"
            )
        self._conn.execute(
            "INSERT INTO claims (key, response) VALUES (?, ?)", (key, response)
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ClaimCache":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _parse_claims(
    raw: str, *, corpus_names: frozenset[str] | set[str], subject: str, human_body: str
) -> tuple[ExtractedClaim, ...]:
    """Validate the provider's response, or raise a ValueError naming the offending field.

    Every raise here lands on `_providers.py`'s `except ValueError` branch, which maps it to
    ``malformed_output`` — except the cardinality message, which that same branch routes to
    ``wrong_cardinality`` by its wording. So both failure classes are reached without this module
    adding any failure-handling code of its own.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"provider response is not json: {exc.msg}") from exc
    if not isinstance(payload, dict) or "claims" not in payload:
        raise ValueError("provider response has no 'claims' key")
    items = payload["claims"]
    if not isinstance(items, list):
        raise ValueError("provider response 'claims' is not a list")
    if len(items) > MAX_CLAIMS:
        raise ValueError(f"provider returned {len(items)} claims, maximum is {MAX_CLAIMS}")

    claims: list[ExtractedClaim] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("claim is not an object")
        relation = item.get("relation")
        if relation not in _RELATIONS:
            raise ValueError(
                f"claim relation {relation!r} is not one of {sorted(_RELATIONS)}"
            )
        obj = item.get("object")
        if not isinstance(obj, str) or not obj.strip():
            raise ValueError("claim object is missing")
        if obj not in corpus_names:
            raise ValueError(f"claim object {obj!r} is not a document in the corpus")
        if obj == subject:
            raise ValueError(f"claim object {obj!r} is the memo itself; a self-relation is not one")
        evidence = item.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("claim evidence is missing")
        if evidence not in human_body:
            # A paraphrase puts the model's words where the author's are supposed to be, and the
            # span is the reviewer's entire basis for judging the claim.
            raise ValueError("claim evidence is not a verbatim span of the memo prose")
        confidence = item.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise ValueError("claim confidence is missing or not a number")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"claim confidence {confidence!r} is outside [0, 1]")
        claims.append(
            ExtractedClaim(
                relation=relation,
                subject=subject,
                object=obj,
                evidence=evidence,
                confidence=float(confidence),
            )
        )
    return tuple(claims)


class ProseClaimExtractor:
    """Turns one memo's prose into structured claims, deterministically and off the query path.

    **This never runs on a query.** It is an ingest-side tool. The planner's `max_model_calls`
    defaults to 0 and nothing here changes that; a retrieval path that could make a generative
    call would put an unbounded, unpinned cost and a nondeterministic answer inside the operation
    the trust layer is supposed to make predictable.
    """

    def __init__(
        self,
        *,
        client: ChatClient | None = None,
        cache: ClaimCache | None = None,
        model: str = DEFAULT_EXTRACT_MODEL,
        revision: str = DEFAULT_EXTRACT_REVISION,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
    ) -> None:
        self.provider_id = "recall.extraction"
        self.model_id = model
        self.provider_revision = revision
        self.max_proposals = MAX_CLAIMS
        self._cache = cache
        self._client = client if client is not None else _OpenAICompatChatClient(
            base_url=base_url, api_key=api_key
        )
        self.last_provider_metadata: ProviderMetadata | None = None

    def _call(self, human_body: str, corpus_names: frozenset[str] | set[str]) -> str:
        response = self._client.complete(
            model=self.model_id,
            messages=build_prompt(human_body, corpus_names),
            temperature=0,
        )
        cost = response.get("monetary_cost_usd")
        if cost is None:
            # Refused rather than defaulted to zero. A null cost recorded as 0.0 would make an
            # unmeasured spend indistinguishable from a free one, which is the same encoding
            # mistake `promotion.py:31` refuses for an unmeasured safety rate.
            raise ValueError(
                "provider response omitted monetary_cost_usd; an unauditable cost is refused"
            )
        self.last_provider_metadata = ProviderMetadata(
            provider_id=self.provider_id,
            model_id=self.model_id,
            model_revision=self.provider_revision,
            prompt_tokens=response.get("prompt_tokens"),
            completion_tokens=response.get("completion_tokens"),
            total_tokens=response.get("total_tokens"),
            latency_ms=response.get("latency_ms"),
            monetary_cost_usd=float(cost),
        )
        return str(response["text"])

    def _key(self, human_body: str) -> str:
        return claim_cache_key(
            human_body,
            model_id=self.model_id,
            model_revision=self.provider_revision,
            prompt_version=PROMPT_VERSION,
        )

    def extract(
        self, human_body: str, corpus_names: frozenset[str] | set[str], *, subject: str
    ) -> tuple[ExtractedClaim, ...]:
        """The claims this memo declares. Served from cache when the key is already known."""
        key = self._key(human_body)
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return _parse_claims(
                    cached, corpus_names=corpus_names, subject=subject, human_body=human_body
                )
        raw = self._call(human_body, corpus_names)
        claims = _parse_claims(
            raw, corpus_names=corpus_names, subject=subject, human_body=human_body
        )
        if self._cache is not None:
            self._cache.put(key, raw)
        return claims

    def recheck(
        self, human_body: str, corpus_names: frozenset[str] | set[str], *, subject: str
    ) -> RecheckReport:
        """Re-call the model on an already-cached key and report whether the answer changed.

        Exists because temperature 0 is a request, not a guarantee any hosted provider makes. If
        the mismatch rate here is non-zero then this cache — not the sampler — is the only thing
        making runs reproducible, and that has to be known before an eviction silently renumbers
        every proposal id computed from these claims.

        The cached entry is never replaced by what this finds. Overwriting it would destroy the
        content every existing proposal id was hashed from, in the middle of the measurement
        whose entire purpose is to tell you those ids are load-bearing.
        """
        if self._cache is None:
            raise ValueError("recheck needs a cache; there is nothing to compare against")
        key = self._key(human_body)
        cached = self._cache.get(key)
        if cached is None:
            raise ValueError("recheck found no cached entry for this memo")

        before = _parse_claims(
            cached, corpus_names=corpus_names, subject=subject, human_body=human_body
        )
        raw = self._call(human_body, corpus_names)
        try:
            after = _parse_claims(
                raw, corpus_names=corpus_names, subject=subject, human_body=human_body
            )
        except ValueError:
            # A re-call that no longer validates is the strongest possible mismatch, and this is
            # a measurement, not a gate: it reports rather than raises.
            return RecheckReport(rechecked=1, mismatches=1)
        return RecheckReport(rechecked=1, mismatches=int(before != after))


class _OpenAICompatChatClient:
    """The real client. Imported lazily so the library loads without the `extract` extra."""

    def __init__(self, *, base_url: str = DEFAULT_BASE_URL, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "the prose extractor needs an API key (OPENROUTER_API_KEY or OPENAI_API_KEY in "
                "the environment, or an explicit api_key)"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "ProseClaimExtractor requires: pip install recall[extract]"
            ) from exc
        self._client = OpenAI(api_key=key, base_url=base_url)

    def complete(
        self, *, model: str, messages: list[dict[str, str]], temperature: float
    ) -> dict[str, Any]:
        completion = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        usage = getattr(completion, "usage", None)
        cost = getattr(usage, "cost", None) if usage is not None else None
        return {
            "text": completion.choices[0].message.content or "",
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
            "monetary_cost_usd": cost,
        }


def resolve_claim_extractor(
    env: dict[str, str] | None = None, *, cache: ClaimCache | None = None
) -> ProseClaimExtractor | None:
    """Build the extractor from env settings, or return None when it is off.

    Copies `entailment.py:75` exactly, including the part that matters most: unset means None,
    not a default-on network call. A generative call that happens because someone installed the
    package is a cost and a data-egress decision nobody made.
    """
    source = env if env is not None else dict(os.environ)
    raw = source.get("RECALL_EXTRACT", "").strip().lower()
    if raw in _FALSE:
        return None
    if raw not in _TRUE:
        raise ValueError(
            f"RECALL_EXTRACT={raw!r} is not a boolean. Use one of {sorted(_TRUE)} to enable "
            "or leave it unset."
        )
    return ProseClaimExtractor(
        cache=cache,
        model=source.get("RECALL_EXTRACT_MODEL", DEFAULT_EXTRACT_MODEL),
        revision=source.get("RECALL_EXTRACT_REVISION", DEFAULT_EXTRACT_REVISION),
        base_url=source.get("RECALL_EXTRACT_BASE_URL", DEFAULT_BASE_URL),
    )


__all__ = [
    "DEFAULT_EXTRACT_MODEL",
    "DEFAULT_EXTRACT_REVISION",
    "MAX_CLAIMS",
    "PROMPT_SHA256",
    "PROMPT_TEMPLATE",
    "PROMPT_VERSION",
    "ChatClient",
    "ClaimCache",
    "ExtractedClaim",
    "ProseClaimExtractor",
    "RecheckReport",
    "build_prompt",
    "claim_cache_key",
    "resolve_claim_extractor",
]
