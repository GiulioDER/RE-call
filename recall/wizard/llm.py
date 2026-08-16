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

#: How many chunks the model is shown, before the character budget below trims it.
CHUNKS_PER_PROMPT_MULTIPLIER = 3

#: The real bound on prompt size. Counting chunks alone sent 120 of them at the default, which
#: measured 75 000 characters (~18 800 tokens) on this repository's `docs/` — not the "roughly 8k
#: in" the docstring claimed, and more than an 8k or 16k context local endpoint can take. The
#: local endpoint is offered first and unconditionally, so the budget has to suit it.
MAX_PROMPT_CHARS = 24_000

#: How many entries the model menu shows. See `model_choices`.
MENU_LIMIT = 12


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


#: The live roster is around 1 MB. This bounds what a captive portal, a proxy error page or a
#: misbehaving endpoint can make `recall setup` buffer into memory.
MAX_CATALOGUE_BYTES = 8 * 1024 * 1024


def _fetch(url: str, *, headers: dict[str, str] | None = None) -> bytes:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=CATALOGUE_TIMEOUT_SECONDS) as response:  # noqa: S310
        body = response.read(MAX_CATALOGUE_BYTES + 1)
    if len(body) > MAX_CATALOGUE_BYTES:
        raise ValueError(f"catalogue response exceeded {MAX_CATALOGUE_BYTES} bytes")
    return bytes(body)


def _safe_reason(exc: BaseException) -> str:
    """A log-safe description of `exc`, never its message.

    `http.client` raises `ValueError("Invalid header value %r" % value)` where `value` is the whole
    `Bearer <key>`, which happens whenever a key carries a trailing newline — the ordinary result
    of pasting one out of a terminal. Logging `%s` of the exception therefore wrote the user's API
    key into a warning line. Only the class name is ever logged now.
    """
    return type(exc).__name__


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
    models: list[CatalogueModel] = []
    try:
        payload = json.loads(_fetch(OPENROUTER_CATALOGUE_URL))
        entries = payload["data"]
        if not isinstance(entries, list):
            # The loop used to sit outside this try, so `{"data": null}` — an ordinary gateway
            # error shape — escaped as a TypeError and broke the "never raises" contract.
            raise TypeError(f"'data' is {type(entries).__name__}, not a list")
    except Exception as exc:
        _log.warning("falling back to the pinned model list: %s", _safe_reason(exc))
        return list(PINNED_OPENROUTER_MODELS)

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


#: `/v1/models` returns every model on the account, including ones that cannot answer a chat
#: completion at all. Offering `text-embedding-3-small` under the label "writes the calibration
#: questions for your corpus" is the same defect the OpenRouter path was fixed for.
_NON_CHAT_PREFIXES = (
    "text-embedding",
    "text-moderation",
    "omni-moderation",
    "whisper",
    "tts-",
    "dall-e",
    "gpt-image",
    "sora",
    "babbage",
    "davinci",
)


def openai_catalogue(api_key: str) -> list[CatalogueModel]:
    """OpenAI's chat-capable roster for this key, or the pinned fallback. Never raises."""
    try:
        payload = json.loads(
            _fetch(
                f"{OPENAI_BASE_URL}/models",
                # Stripped: a key carrying the trailing newline you get from pasting one out of a
                # terminal makes `http.client` raise a ValueError whose message is the whole
                # header, key included.
                headers={"Authorization": f"Bearer {api_key.strip()}"},
            )
        )
        entries = payload["data"]
        if not isinstance(entries, list):
            raise TypeError(f"'data' is {type(entries).__name__}, not a list")
    except Exception as exc:
        _log.warning("falling back to the pinned model list: %s", _safe_reason(exc))
        return list(PINNED_OPENAI_MODELS)

    models: list[CatalogueModel] = []
    for item in entries:
        try:
            # Per entry, matching the OpenRouter loop: one malformed record must not lose the
            # roster, which is what a single comprehension inside the try did.
            model_id = str(item["id"])
        except Exception:
            continue
        if model_id.startswith(_NON_CHAT_PREFIXES):
            continue
        models.append(CatalogueModel(model_id))
    return models or list(PINNED_OPENAI_MODELS)


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

    # Capped. `recall.setup._choose` prints every entry as a numbered line before prompting, and
    # the live OpenRouter roster is ~390 models: a 390-line scroll is not a page anybody can read,
    # and this is the page being added. The cheapest capable models are the useful ones here,
    # since the workload is a few thousand tokens once per corpus. Manual entry, kept last, is
    # what covers everything not shown.
    rest = [m for m in catalogue if m.id != default]
    rest.sort(key=lambda m: (m.completion_price if m.completion_price is not None else 1e9, m.id))
    ordered = [default] + [m.id for m in rest[: MENU_LIMIT - 1]]
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


#: A corpus term counts as "subject matter" only if it is not everywhere in the corpus. Above this
#: share of chunks a word is the corpus's connective tissue ("should", "before", "value"), not what
#: it is about, and matching on it rejects any question written in English.
#:
#: 0.05, and not by taste. Measured against this repository's `docs/`, where
#: `offtopic_subjects_absent_from` independently finds 13 of 25 subjects disjoint and so defines
#: the answer as 60 of 125 pool questions rejected:
#:
#:     stopwords only, threshold 2   86 rejects   26 disagreements
#:     stopwords only, threshold 3   42 rejects   18 disagreements
#:     df<=5%,  threshold 2          60 rejects    0 disagreements   <-- exact
#:     df<=5%,  threshold 3          30 rejects   30 disagreements
#:     df<=10%, threshold 2          73 rejects   13 disagreements
#:     df<=25%, threshold 2          86 rejects   26 disagreements
#:
#: Only one configuration agrees exactly, and it also catches 4 of 4 genuinely on-topic questions.
#: A larger fraction admits MORE words as subject matter and therefore rejects more, which is the
#: opposite of the intuition that a looser ceiling is safer.
_UBIQUITOUS_DF_FRACTION = 0.05

#: Floor for the ceiling above, so a small corpus does not filter itself into silence. At 5% with
#: no floor a 30-chunk corpus admitted only words appearing in ONE chunk, which on any corpus whose
#: chunks resemble each other left the subject set EMPTY — and an empty set makes
#: `_reuses_corpus_vocabulary` return False for everything, so the guard stops guarding while every
#: test that feeds it a large corpus still passes.
_MIN_DF_CEILING = 3


def _corpus_subject_words(chunks: Sequence[str]) -> set[str]:
    """The corpus's SUBJECT vocabulary: prose words that are not stopwords and not ubiquitous.

    The first version intersected against every token in the corpus, filtered only by
    `len(w) > 4`. Measured on this repository's `docs/` (1815 chunks, 8853 word types), that
    rejected 83 of the 125 questions in the project's own certified-disjoint off-topic pool,
    because "should", "before", "which" and "engine" all appear in a corpus of any size. At the
    default that raised `QuerySetError` on every real corpus — after the paid model call had
    already been made. A three-chunk test fixture with 24 word types cannot show this.
    """
    from collections import Counter

    from recall.eval.vocab import word_tokens
    from recall.wizard.queryset import _is_prose, _STOPWORDS

    total = max(1, len(chunks))
    df: Counter[str] = Counter()
    for chunk in chunks:
        df.update({w for w in word_tokens([chunk]) if _is_prose(w) and w not in _STOPWORDS})
    ceiling = max(_MIN_DF_CEILING, int(total * _UBIQUITOUS_DF_FRACTION))
    subject_words = {word for word, count in df.items() if count <= ceiling}
    if not subject_words and df:
        # Every term looked ubiquitous, which happens on a corpus whose chunks are near-copies of
        # each other. Returning the empty set would silently disable the filter, so fall back to
        # every prose term: over-strict is recoverable (the model is asked for extra), while a
        # guard that quietly stops guarding is not.
        _log.warning("every corpus term looked ubiquitous; using the unfiltered vocabulary")
        return set(df)
    return subject_words


def _reuses_corpus_vocabulary(query: str, subject_words: set[str], *, threshold: int = 2) -> bool:
    """Whether `query` leans on the corpus's SUBJECT MATTER.

    A gap question is allowed ordinary English: `synthetic.py` measured its own templates at
    median top cosine 0.570 with 78% below the answerable floor, so shared function words are
    demonstrably harmless. What is not allowed is the topic, so both sides are reduced to
    non-stopword prose terms and the corpus side to terms that are not ubiquitous in it. Two
    matches are required, so one coincidence cannot cost a legitimate question.
    """
    from recall.eval.vocab import word_tokens
    from recall.wizard.queryset import _is_prose, _STOPWORDS

    content = {w for w in word_tokens([query]) if _is_prose(w) and w not in _STOPWORDS}
    return len(content & subject_words) >= threshold


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

    # Trimmed by characters, not by count: a chunk is up to 800 characters, so a count-based bound
    # says nothing about how large the prompt gets.
    #
    # Filled in SAMPLED order with `continue`, not sorted order with `break`. This is the second
    # time this exact mistake was made in this package — see the same lesson at
    # `recall/wizard/queryset.py`'s sampling loop — and it produces the same result: trimming a
    # sorted sample from the tail keeps only its lowest-indexed part. Measured on this
    # repository's `docs/` at the default, sorted-and-break showed the model 39 chunks drawn from
    # 3 of 51 files; sampled-and-continue fills the same budget with 40 chunks from 21 of 51. A
    # question set written from three documents is not a question set about the corpus.
    #
    # `continue` also matters on its own: one oversized chunk near the front should skip itself,
    # not end the selection.
    kept: list[int] = []
    budget = MAX_PROMPT_CHARS
    for index in rng.sample(range(len(chunks)), sample_size):
        if len(chunks[index]) <= budget:
            budget -= len(chunks[index])
            kept.append(index)
    # Sorted only for presentation, so excerpts reach the model in corpus order.
    excerpts = [chunks[i] for i in sorted(kept)]
    if not excerpts:
        raise QuerySetError(
            f"every sampled chunk is longer than the {MAX_PROMPT_CHARS} character prompt budget"
        )

    # Generous margin. Duplicates and gap questions that reused the corpus's subject matter are
    # both discarded, and asking for barely more than `per_class` left the set short whenever the
    # model ignored the disjointness instruction more than a couple of times.
    asked = per_class * 2
    user = (
        f"Write {asked} answerable and {asked} unanswerable questions.\n\n"
        "Excerpts from the collection:\n\n"
        + "\n\n---\n\n".join(excerpts)
    )

    payload = client.complete_json(system=_SYSTEM_PROMPT, user=user, schema=_SCHEMA)

    corpus_words = _corpus_subject_words(chunks)
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
