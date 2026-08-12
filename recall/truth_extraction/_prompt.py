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
)

#: Bump on ANY wording change below. Cached extractions are keyed on it.
PROMPT_REVISION = "truth-extraction-prompt-v1"

_SYSTEM = (
    "You extract structured claims from a single document. You do not summarise, infer, or "
    "reconcile documents against each other. Every claim you emit must be supported by a "
    "quote copied verbatim from the document body. Return JSON and nothing else."
)

_SHAPES = """\
Return exactly this JSON object:

  {"claims": [ ... ]}

Each entry is one of exactly four shapes:

  {"kind": "supersession", "superseded": "<corpus document name>", "quote": "<verbatim>"}
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


def build_extraction_prompt(
    *, file: str, human_body: str, corpus_names: Sequence[str]
) -> ExtractionPrompt:
    """Render the extraction prompt for one document.

    `human_body` must already have the frontmatter block removed (`human_body_of`). The model
    is shown the body it must quote from and the corpus names a supersession target must
    resolve to, because a target it cannot see is a target it will invent.
    """
    names = tuple(corpus_names)
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
            "- A `date` must appear literally in the document body. Do not compute or infer a "
            "date.",
            f"- `key` is one of {list(VALIDITY_CLAIM_KEYS)}.",
            f"- `value` is one of {list(STATUS_VOCABULARY)}.",
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
        system=_SYSTEM,
        user=user,
        revision=PROMPT_REVISION,
    )


__all__ = ["PROMPT_REVISION", "ExtractionPrompt", "build_extraction_prompt"]
