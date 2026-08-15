"""Prompt construction. Pure: builds a string, never sends one.

Keeping this module model free is what lets the prompt be diffed, reviewed, and version
pinned like any other artifact. `PROMPT_REVISION` is part of the cache key, so editing the
wording here invalidates cached extractions instead of silently mixing two prompts' output
in one corpus.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from recall.truth_extraction.types import (
    MAX_CLAIMS_PER_FILE,
    STATUS_VOCABULARY,
    VALIDITY_CLAIM_KEYS,
    coerce_status_vocabulary,
)

#: Bump on ANY wording change below. Cached extractions are keyed on it.
#:
#: v2 states the DIRECTION of `superseded`. v1 showed the shape and the corpus rule and
#: never said which end of the relation the field holds — that semantic lived only in
#: `SupersessionClaim`'s docstring, which the model never sees. Measured on `python/peps`:
#: given "This PEP was superseded by :pep:`345`", the model filled the slot with the
#: SUBJECT, so pep-0262 declared itself its own predecessor.
PROMPT_REVISION = "truth-extraction-prompt-v2"

_SYSTEM = (
    "You extract structured claims from a single document. You do not summarise, infer, or "
    "reconcile documents against each other. Every claim you emit must be supported by a "
    "quote copied verbatim from the document body. Return JSON and nothing else."
)

_SHAPES = """\
Return exactly this JSON object:

  {"claims": [ ... ]}

Each entry is one of exactly four shapes:

  {"kind": "supersession", "superseded": "<the OLD document THIS ONE replaces>", \
"quote": "<verbatim>"}
  {"kind": "validity", "key": "valid_from"|"valid_until", "date": "YYYY-MM-DD", \
"quote": "<verbatim>"}
  {"kind": "status", "value": "<one of the status vocabulary>", "quote": "<verbatim>"}
  {"kind": "identity", "entity": "<canonical name>", "alias": "<other name>", \
"quote": "<verbatim>"}
"""


@dataclass(frozen=True)
class ExtractionPrompt:
    """A rendered prompt plus the inputs the ladder will check the answer against."""

    file: str
    human_body: str
    corpus_names: tuple[str, ...]
    system: str
    user: str
    revision: str = PROMPT_REVISION
    #: Carried, not just rendered, because `extraction_cache_key` hashes this object's fields
    #: rather than the rendered text. A vocabulary that changed the prompt without changing the
    #: key would let two runs share a cache entry produced under a DIFFERENT prompt.
    status_vocabulary: tuple[str, ...] = STATUS_VOCABULARY


def build_extraction_prompt(
    *,
    file: str,
    human_body: str,
    corpus_names: Sequence[str],
    status_vocabulary: Sequence[str] = STATUS_VOCABULARY,
) -> ExtractionPrompt:
    """Render the extraction prompt for one document.

    `human_body` must already have the frontmatter block removed (`human_body_of`). The model
    is shown the body it must quote from and the corpus names a supersession target must
    resolve to, because a target it cannot see is a target it will invent.

    `status_vocabulary` is the closed set of lifecycle values, and it is a PARAMETER because the
    shipped set is memo-shaped. Measured on `python/peps`: the model reads `Status: Final` and
    emits `final`, which is outside the shipped set, and since `claim_shape` is a BATCH rung that
    refused 12 of 30 documents outright, taking their supersession claims down with them. The
    vocabulary was the mismatch; refusing was correct behaviour against the wrong list.

    ⚠️ Widening it here does NOT widen what may be written into a corpus. `recall.rewrite` keeps
    validating against the shipped `STATUS_VOCABULARY`, because a value the trust layer does not
    understand must not reach a user's memo just because a research run named it. Configurable for
    MEASUREMENT, closed for WRITING.
    """
    names = tuple(corpus_names)
    # One coercion, shared with the ladder. The emptiness check used to live here alone, so `()`
    # raised at render time and silently refused every status claim at a batch rung when it went
    # straight to `normalize_extraction` instead.
    vocabulary = coerce_status_vocabulary(status_vocabulary)
    user = "\n".join(
        (
            f"Document: {file}",
            "",
            _SHAPES,
            "Rules, each of which is enforced after you answer:",
            f"- At most {MAX_CLAIMS_PER_FILE} claims. Emit fewer rather than pad.",
            "- Every `quote` must be a VERBATIM substring of the document body below. A "
            "paraphrase is rejected.",
            "- A `superseded` target must be one of the corpus documents listed below. A name "
            "that is not in that list is rejected, and so is one that matches more than one "
            "document.",
            "- DIRECTION. `superseded` names the OLDER document that THIS document replaces. "
            "Never name this document itself, and never name the newer one.",
            "- If the body says this document was superseded BY another, emit NO supersession "
            "claim. There is no field for that direction, and the edge belongs to the other "
            "document. Getting this backwards would mark the live document stale and rank it "
            "beneath the one it replaced.",
            "- A `date` must appear literally in the document body. Do not compute or infer a "
            "date.",
            f"- `key` is one of {list(VALIDITY_CLAIM_KEYS)}.",
            f"- `value` is one of {list(vocabulary)}.",
            "- Emit nothing for a relation the body only mentions, reports, or hedges. "
            "Reported speech and 'supersedes/augments' style hedging are not claims.",
            "- Return an empty `claims` array when the body states none of these.",
            "",
            "Corpus documents:",
            *(f"- {name}" for name in names),
            "",
            "Document body:",
            human_body,
        )
    )
    return ExtractionPrompt(
        file=file,
        human_body=human_body,
        corpus_names=names,
        status_vocabulary=vocabulary,
        system=_SYSTEM,
        user=user,
        revision=PROMPT_REVISION,
    )


__all__ = ["PROMPT_REVISION", "ExtractionPrompt", "build_extraction_prompt"]
