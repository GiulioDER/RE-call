"""The engine port, and a deterministic reference implementation of it.

An engine takes a rendered prompt and returns raw text. That is the whole contract: whatever
it returns still goes through the full validation ladder, so a model engine and the rules
engine below are held to identical standards. Nothing in this package calls a model, and
`resolve_extraction_engine` ships OFF, matching `resolve_entailment_judge`.

The deterministic reference exists so the whole pipeline is testable end to end with no model
and no network, and so a model engine can be measured against a fixed floor rather than
against nothing. Two deliberate limits:

- **Active voice only.** *"this supersedes X"* is extractable; *"this is superseded by X"* is
  not, because a claim is scoped to the file it was read from and the four output shapes give
  no way to attribute a claim to the OTHER document. Inverting it here would produce exactly
  the backwards edge `recall/fix.py` documents: the live memo declared stale and demoted
  beneath the one it replaced. Refusing is the correct floor.
- **No cap of its own.** When a body carries more markers than `MAX_CLAIMS_PER_FILE`, the
  engine emits all of them and the ladder refuses the file. Truncating would hand a reviewer
  a partial batch that looks complete.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from recall.truth_extraction._prompt import ExtractionPrompt

DETERMINISTIC_EXTRACTION_ENGINE_ID = "recall.truth_extraction.deterministic"
DETERMINISTIC_EXTRACTION_MODEL_ID = "rules"
DETERMINISTIC_EXTRACTION_REVISION = "truth-extraction-rules-v1"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})

#: The same verb set `recall.reasoning_proposals._deterministic._DIRECT_REF_RE` uses, so this
#: engine cannot find strictly less than the rules it is meant to replace. Looser verbs are
#: contained by the ladder: the target must still resolve to exactly one corpus file.
_ACTIVE_SUPERSESSION_RE = re.compile(
    r"\b(?:replaces|supersedes|supercedes|deprecates|updates|revises)\s+"
    r"`?(?P<target>[\w./\-[\]]+)`?",
    re.IGNORECASE,
)
_VALID_FROM_RE = re.compile(
    r"\b(?:valid[ _]from|effective(?:\s+from)?)\b[^\n]{0,40}?(?P<date>\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)
_VALID_UNTIL_RE = re.compile(
    r"\b(?:valid[ _]until|expires?(?:\s+on)?)\b[^\n]{0,40}?(?P<date>\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)
#: Not anchored to the line start: memos write `Status: deprecated` mid sentence as often as
#: on its own line. The closed vocabulary does the filtering an anchor would have done, and it
#: does it on the value rather than on the layout.
_STATUS_RE = re.compile(r"\bstatus\s*[:=]\s*(?P<value>[A-Za-z]+)", re.IGNORECASE)
_IDENTITY_RE = re.compile(
    r"(?P<entity>[A-Z][\w .'\-]{2,60}?)\s*\("
    r"(?:formerly|previously|f/k/a|also known as|aka)\s+(?:the\s+)?(?P<alias>[^)\n]{2,60})\)",
    re.IGNORECASE,
)
_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


class ExtractionEngine(Protocol):
    """Port for whatever answers an extraction prompt. May call a model; need not."""

    engine_id: str
    model_id: str
    revision: str

    def run(self, prompt: ExtractionPrompt) -> str:
        ...


class DeterministicExtractionEngine:
    """Rules only. No model, no network, no clock. Stable output for a stable body."""

    engine_id = DETERMINISTIC_EXTRACTION_ENGINE_ID
    model_id = DETERMINISTIC_EXTRACTION_MODEL_ID
    revision = DETERMINISTIC_EXTRACTION_REVISION

    def run(self, prompt: ExtractionPrompt) -> str:
        body = prompt.human_body
        claims: list[Mapping[str, Any]] = []
        for match in _ACTIVE_SUPERSESSION_RE.finditer(body):
            claims.append(
                {
                    "kind": "supersession",
                    "superseded": match.group("target").rstrip(".,;:)"),
                    "quote": _line_of(body, match.start()),
                }
            )
        for key, pattern in (("valid_from", _VALID_FROM_RE), ("valid_until", _VALID_UNTIL_RE)):
            for match in pattern.finditer(body):
                claims.append(
                    {
                        "kind": "validity",
                        "key": key,
                        "date": match.group("date"),
                        "quote": _line_of(body, match.start()),
                    }
                )
        # `prompt.status_vocabulary`, not the module constant. Reading the constant meant this
        # engine emitted the five shipped memo words whatever the caller asked for, so under a
        # PEP vocabulary a document with a status line produced a claim the ladder then refused
        # at `claim_shape` — a BATCH rung, which takes the file's supersession claims down with
        # it. The two arms would also have been measured against different vocabularies while
        # the report recorded one.
        vocabulary = {str(word).casefold(): str(word) for word in prompt.status_vocabulary}
        for match in _STATUS_RE.finditer(body):
            value = vocabulary.get(match.group("value").casefold())
            if value is not None:
                claims.append(
                    {"kind": "status", "value": value, "quote": _line_of(body, match.start())}
                )
        for match in _IDENTITY_RE.finditer(body):
            claims.append(
                {
                    "kind": "identity",
                    "entity": _LEADING_ARTICLE_RE.sub("", match.group("entity").strip()),
                    "alias": match.group("alias").strip(),
                    "quote": _line_of(body, match.start()),
                }
            )
        return json.dumps({"claims": claims})


def _line_of(body: str, offset: int) -> str:
    """The line containing `offset`, sliced out of `body` so the quote is verbatim by
    construction rather than by reassembly."""
    start = body.rfind("\n", 0, offset) + 1
    end = body.find("\n", offset)
    return body[start:] if end == -1 else body[start:end]


def _openai_factory(source: Mapping[str, str]) -> ExtractionEngine:
    from recall.truth_extraction._openai_engine import openai_engine_from_env

    return openai_engine_from_env(source)


#: Name to factory, not name to class, and each factory TAKES the resolved settings mapping.
#:
#: Deferred construction is what keeps the extra optional: importing this package never imports
#: an engine's optional dependency, and only naming `openai` pulls `openai` in.
#:
#: The `source` argument is what keeps configuration honest. `resolve_extraction_engine` accepts
#: an explicit mapping, and a nullary factory could only reach `os.environ`, so a caller passing
#: a complete env would be refused for a key it had supplied, or silently answered by whatever
#: model the ambient environment named. That wrong name then reaches the cache key and the audit
#: record, which is the failure the refuse-don't-downgrade rule below exists to prevent.
_ENGINES: Mapping[str, Callable[[Mapping[str, str]], ExtractionEngine]] = {
    "deterministic": lambda _source: DeterministicExtractionEngine(),
    "openai": _openai_factory,
}


def resolve_extraction_engine(env: Mapping[str, str] | None = None) -> ExtractionEngine | None:
    """Build the extraction engine from env settings, or return None when off.

    Extraction is OFF unless `RECALL_TRUTH_EXTRACTION` is explicitly enabled, which is the
    default shipped behavior. `RECALL_TRUTH_EXTRACTION_ENGINE` selects the implementation and
    defaults to the deterministic reference. An unrecognised name is refused rather than
    quietly downgraded to rules, because silently running a different engine than the one
    named would make an audit record wrong about how a claim was produced.
    """
    source = env if env is not None else os.environ
    raw = source.get("RECALL_TRUTH_EXTRACTION", "").strip().lower()
    if raw in _FALSE:
        return None
    if raw not in _TRUE:
        raise ValueError(
            f"RECALL_TRUTH_EXTRACTION={raw!r} is not a boolean. Use one of {sorted(_TRUE)} to "
            "enable or leave it unset."
        )
    name = source.get("RECALL_TRUTH_EXTRACTION_ENGINE", "deterministic").strip().lower()
    if name not in _ENGINES:
        raise ValueError(
            f"RECALL_TRUTH_EXTRACTION_ENGINE={name!r} is not a known engine. Known engines: "
            f"{sorted(_ENGINES)}."
        )
    engine: ExtractionEngine = _ENGINES[name](source)
    return engine


__all__ = [
    "DETERMINISTIC_EXTRACTION_ENGINE_ID",
    "DETERMINISTIC_EXTRACTION_MODEL_ID",
    "DETERMINISTIC_EXTRACTION_REVISION",
    "DeterministicExtractionEngine",
    "ExtractionEngine",
    "resolve_extraction_engine",
]
