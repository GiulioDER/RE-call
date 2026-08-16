"""The provider/key page, the live model catalogue, and the model-authored query generator.

The offline generator in `recall.wizard.queryset` needs neither key nor network and is the fallback
whenever one is unavailable. This is the other arm: a model reads real chunks and writes questions
about them. Whether that is better is a measured question, registered in
`docs/preregistrations/2026-08-16-generated-calibration-query-sets.md`.

Three decisions are load-bearing.

**The security answer gates the page.** Query generation sends sampled chunks of the corpus to
whatever endpoint is chosen. When the user answered that data security is required, cloud providers
are withheld entirely rather than shown as unavailable, matching
`recall.setup.reasoning_provider_choices`: an uninstalled package is something a reader can go and
fix, whereas this is a decision they already made, and re-offering it invites undoing it by
accident.

**The catalogue is fetched, not hardcoded.** `recall/setup.py:625` already carries the reason as a
comment — "a static list inside a released artifact, and the provider's roster is not ours" — and
was proved right by its own `gpt-4o-mini` default going two generations stale. The pinned list here
is a fallback for when the fetch fails, not the source of truth, and manual entry stays last on
every provider.

**Keys go to `.env` only.** Nothing at serving time needs one, and `.mcp.json` is a file users
commit.
"""

from __future__ import annotations

import json
import random
import urllib.error
import urllib.request
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Protocol

from recall.observability import get_logger
from recall.setup import (
    LOCAL_BASE_URL_DEFAULT,
    LOCAL_PROVIDER,
    MANUAL_MODEL,
    OPENAI_BASE_URL,
    OPENROUTER_BASE_URL,
    Choice,
    probe_reasoning_model,
)
from recall.wizard.queryset import (
    DEFAULT_PER_CLASS,
    QuerySetError,
    prepare_for_calibration,
)

__all__ = [
    "LOCAL_BASE_URL_DEFAULT",
    "LOCAL_PROVIDER",
    "MANUAL_MODEL",
    "OPENAI_BASE_URL",
    "OPENROUTER_BASE_URL",
    "CatalogueModel",
    "generate_llm",
    "model_choices",
    "openai_catalogue",
    "openrouter_catalogue",
    "provider_choices",
    "validate_credentials",
]

_log = get_logger("wizard.llm")

OPENROUTER_CATALOGUE_URL = "https://openrouter.ai/api/v1/models"
CATALOGUE_TIMEOUT_SECONDS = 8.0

#: Read from the live OpenRouter catalogue on 2026-08-16. `gpt-5.6-luna` is the cheapest full
#: general model there ($0.10/$0.60 per 1M, 1M context) and supports structured outputs.
DEFAULT_OPENROUTER_MODEL = "openai/gpt-5.6-luna"

#: The same model addressed directly, so switching provider changes the base URL and nothing else.
DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"

#: How many chunks the model is shown. Enough to write `per_class` distinct grounded questions with
#: margin, small enough that the prompt stays a few thousand tokens: at the default that is roughly
#: 8k in and 3k out, which is fractions of a cent on any model in the menu.
CHUNKS_PER_PROMPT_MULTIPLIER = 3


@dataclass(frozen=True)
class CatalogueModel:
    id: str
    context_length: int | None = None
    prompt_price: float | None = None
    completion_price: float | None = None


#: Fallback only. Ordered with the default first, because `recall.setup._choose` returns
#: `choices[0]` when a reader picks something this machine cannot run.
PINNED_OPENROUTER_MODELS: tuple[CatalogueModel, ...] = (
    CatalogueModel(DEFAULT_OPENROUTER_MODEL),
    CatalogueModel("openai/gpt-5.6-terra"),
    CatalogueModel("deepseek/deepseek-v4-flash-0731"),
    CatalogueModel("qwen/qwen3.7-flash"),
    CatalogueModel("anthropic/claude-haiku-4.5"),
)

PINNED_OPENAI_MODELS: tuple[CatalogueModel, ...] = (
    CatalogueModel(DEFAULT_OPENAI_MODEL),
    CatalogueModel("gpt-5.6-terra"),
    CatalogueModel("gpt-5.6-sol"),
)


class LLMClient(Protocol):
    """Anything that can be asked for JSON matching a schema.

    A protocol rather than the concrete client so the generator is testable without a network, a
    key, or the `openai` package, and so a local endpoint and a cloud one are one code path.
    """

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        ...


# --------------------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------------------


def provider_choices(*, security_required: bool, internet: bool) -> list[Choice]:
    """Providers for query generation, local first.

    Local leads because `_choose` refuses a menu whose first entry is unavailable, and a local
    endpoint is the only option needing neither a key nor a network, so it is the only entry that
    can be offered unconditionally.
    """
    choices = [
        Choice(
            label="local endpoint",
            value=LOCAL_PROVIDER,
            description="An OpenAI compatible server you run yourself, such as Ollama or vLLM",
        )
    ]
    if security_required:
        # Withheld, not marked unavailable. See the module docstring.
        return choices

    note = "no internet connection was detected" if not internet else ""
    for label, value, description in (
        ("openrouter", OPENROUTER_BASE_URL, "Many providers behind one key"),
        ("openai", OPENAI_BASE_URL, "OpenAI directly, for an existing OpenAI key"),
    ):
        choices.append(
            Choice(
                label=label,
                value=value,
                description=description,
                available=internet,
                unavailable_note=note,
            )
        )
    return choices


def validate_credentials(*, base_url: str, api_key: str, model: str) -> str | None:
    """`None` when one real call succeeded, else what went wrong.

    Delegates to `recall.setup.probe_reasoning_model`, which already sends one minimal completion
    on a timeout-bounded daemon thread and never raises. A wrong key, a retired model id and a
    local server that is not running are the three likely failures, and each is far cheaper to
    find here than midway through generating a corpus's query set.
    """
    return probe_reasoning_model(base_url=base_url, api_key=api_key, model=model)


# --------------------------------------------------------------------------------------
# The catalogue
# --------------------------------------------------------------------------------------


def _fetch(url: str, *, headers: dict[str, str] | None = None) -> bytes:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=CATALOGUE_TIMEOUT_SECONDS) as response:  # noqa: S310
        return bytes(response.read())


def _as_price(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def openrouter_catalogue() -> list[CatalogueModel]:
    """The live OpenRouter roster, or the pinned fallback. Never raises.

    Public and unauthenticated, so this runs before a key is entered and the menu can show real
    ids at real prices. A model whose completion price is zero does not generate and cannot author
    questions, so it is filtered out rather than offered.
    """
    try:
        payload = json.loads(_fetch(OPENROUTER_CATALOGUE_URL))
        entries = payload["data"]
    except Exception as exc:
        _log.warning("falling back to the pinned model list: %s", exc)
        return list(PINNED_OPENROUTER_MODELS)

    models: list[CatalogueModel] = []
    for entry in entries:
        try:
            model_id = str(entry["id"])
            pricing = entry.get("pricing") or {}
            completion = _as_price(pricing.get("completion"))
            # Strictly positive, not merely truthy. Zero means the model does not generate (an
            # embedding model) and cannot author questions. NEGATIVE is the live roster's sentinel
            # for a variable price, carried by the router pseudo-models `openrouter/auto`,
            # `openrouter/fusion` and friends — which pick a different underlying model per call,
            # so a calibration measured through one would be bound to a model nobody can name.
            # `if not completion` kept every one of them, because -1.0 is truthy.
            if completion is None or completion <= 0:
                continue
            models.append(
                CatalogueModel(
                    id=model_id,
                    context_length=entry.get("context_length"),
                    prompt_price=_as_price(pricing.get("prompt")),
                    completion_price=completion,
                )
            )
        except Exception:  # one malformed entry must not lose the roster
            continue
    return models or list(PINNED_OPENROUTER_MODELS)


def openai_catalogue(api_key: str) -> list[CatalogueModel]:
    """OpenAI's roster for this key, or the pinned fallback. Never raises."""
    try:
        payload = json.loads(
            _fetch(
                f"{OPENAI_BASE_URL}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        )
        ids = [str(item["id"]) for item in payload["data"]]
    except Exception as exc:
        _log.warning("falling back to the pinned model list: %s", exc)
        return list(PINNED_OPENAI_MODELS)
    return [CatalogueModel(i) for i in ids] or list(PINNED_OPENAI_MODELS)


def model_choices(base_url: str, *, api_key: str = "") -> list[Choice]:
    """A menu for `base_url`: the default first, live ids after it, manual entry last.

    No price appears in a label. `recall/setup.py:631` states the convention and the reason: a
    number in a shipped menu is a measurement nothing re-checks. The live catalogue carries prices
    for a caller that wants to show them next to the id it just fetched.
    """
    if base_url == OPENAI_BASE_URL:
        default, catalogue = DEFAULT_OPENAI_MODEL, openai_catalogue(api_key)
    else:
        default, catalogue = DEFAULT_OPENROUTER_MODEL, openrouter_catalogue()

    ordered = [default] + [m.id for m in catalogue if m.id != default]
    choices = [
        Choice(
            label=model_id,
            value=model_id,
            description="Writes the calibration questions for your corpus",
        )
        for model_id in ordered
    ]
    choices.append(
        Choice(
            label="enter a model id",
            value=MANUAL_MODEL,
            description="Type an id yourself, for anything not listed",
        )
    )
    return choices


# --------------------------------------------------------------------------------------
# The generator
# --------------------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You write labelled evaluation questions for a retrieval system.

You are given excerpts from one document collection. Produce two lists.

1. `answerable`: questions a reader could answer USING THESE EXCERPTS. Each must be grounded in a
   specific excerpt, phrased as somebody familiar with the subject would actually ask it. Do not
   quote an excerpt verbatim; ask about what it says.

2. `unanswerable`: questions from a COMPLETELY DIFFERENT SUBJECT AREA that this collection does not
   cover at all. These must NOT be near-misses, variations, or harder versions of the answerable
   questions, and must NOT reuse the collection's vocabulary or subject matter. Pick unrelated
   everyday or scientific topics instead.

The second instruction is the one that matters and the easy one to get wrong. A plausible
near-miss is indistinguishable from an answerable question to a retrieval system, which makes the
whole measurement useless. Genuinely unrelated is what is wanted.

Return only the two lists, each with exactly the requested number of entries.
"""

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answerable": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        "unanswerable": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answerable", "unanswerable"],
    "additionalProperties": False,
}


def _corpus_words(chunks: Sequence[str]) -> set[str]:
    from recall.eval.vocab import word_tokens

    return set(word_tokens(list(chunks)))


def _reuses_corpus_vocabulary(query: str, corpus_words: set[str], *, threshold: int = 2) -> bool:
    """Whether `query` leans on the corpus's own words.

    A gap question is allowed ordinary English — `synthetic.py` measured its own templates at
    median top cosine 0.570, so shared function words are demonstrably harmless. What is not
    allowed is the subject matter, so this counts only longer content words and needs more than
    one before rejecting, which keeps a stray coincidence from emptying the class.
    """
    from recall.eval.vocab import word_tokens

    content = {w for w in word_tokens([query]) if len(w) > 4}
    return len(content & corpus_words) >= threshold


def generate_llm(
    chunks: Sequence[str],
    *,
    client: LLMClient,
    per_class: int = DEFAULT_PER_CLASS,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """A labelled set authored by `client` from real excerpts of `chunks`.

    Chunk selection is seeded, so the prompt does not drift between runs of the registered
    measurement. What the model returns is of course not deterministic, which is why the
    comparison reports the fitted threshold rather than treating either arm as reproducible
    byte for byte.

    The model is asked for extra of each class, because some of what it returns is discarded:
    duplicates, and gap questions that ignored the instruction and reused the corpus's subject
    matter. Asking for exactly `per_class` and discarding any would leave the set short.
    """
    if per_class < 1:
        raise QuerySetError("per_class must be at least 1")
    if not chunks:
        raise QuerySetError("no chunks to generate questions from")

    rng = random.Random(seed)
    sample_size = min(len(chunks), per_class * CHUNKS_PER_PROMPT_MULTIPLIER)
    excerpts = [chunks[i] for i in sorted(rng.sample(range(len(chunks)), sample_size))]

    asked = per_class + max(5, per_class // 2)
    user = (
        f"Write {asked} answerable and {asked} unanswerable questions.\n\n"
        "Excerpts from the collection:\n\n"
        + "\n\n---\n\n".join(excerpts)
    )

    payload = client.complete_json(system=_SYSTEM_PROMPT, user=user, schema=_SCHEMA)

    corpus_words = _corpus_words(chunks)
    entries: list[dict[str, Any]] = []
    for item in payload.get("answerable") or []:
        query = (item or {}).get("query")
        if isinstance(query, str) and query.strip():
            entries.append({"query": query.strip(), "answerable": True})

    dropped = 0
    for item in payload.get("unanswerable") or []:
        query = (item or {}).get("query")
        if not isinstance(query, str) or not query.strip():
            continue
        if _reuses_corpus_vocabulary(query, corpus_words):
            # Enforced rather than trusted. The instruction says a different subject area and a
            # model will sometimes write a near-miss anyway, which is the case measured as not
            # separable at all.
            dropped += 1
            continue
        entries.append({"query": query.strip(), "answerable": False})

    if dropped:
        _log.warning("dropped %d gap question(s) that reused the corpus vocabulary", dropped)

    # Trimmed per class AFTER canonicalization, so duplicates the model produced do not consume a
    # slot and leave the set short.
    canonical = prepare_for_calibration(entries, min_per_class=per_class)
    answerable = [e for e in canonical if e["answerable"]][:per_class]
    unanswerable = [e for e in canonical if not e["answerable"]][:per_class]
    return answerable + unanswerable
