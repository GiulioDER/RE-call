"""The validation ladder. Pure: no model, no network, no filesystem, no clock.

The ladder is ordered, and the order is the design. Rungs 1 to 4 refuse the file's WHOLE
output, because each of them means the model ignored the contract rather than got one claim
wrong, and a batch that ignored the contract cannot be partially trusted. Rungs 5 to 7 refuse
a single claim and keep the rest.

Rung 5, the verbatim quote, is the strongest guard here. A claim whose quote is a real
substring of the memo body can be checked by a human in one glance; a paraphrase cannot, and
a paraphrase is exactly what a model produces when it is inferring rather than reading. Two of
the four false positives the rule based attempt produced (`recall/fix.py`) were the model of
that failure in miniature: text that *mentioned* a supersession without *making* one.

Rung 6 is the refusal `recall/fix.py` already makes: a target that does not resolve to exactly
one corpus file is refused, never guessed at. Ambiguity is a refusal, not a coin flip.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from recall.frontmatter import parse_frontmatter, supersedes_key
from recall.truth_extraction.types import (
    MAX_CLAIMS_PER_FILE,
    STATUS_VOCABULARY,
    VALIDITY_CLAIM_KEYS,
    ClaimRejection,
    ExtractedClaim,
    ExtractionBatchRejected,
    IdentityClaim,
    StatusClaim,
    SupersessionClaim,
    ValidityClaim,
)

#: Field names each kind must carry, exactly. Extra fields are a shape failure: the model
#: supplies semantics only, and a field nobody declared is a field nobody validates.
_REQUIRED_FIELDS: Mapping[str, frozenset[str]] = {
    "supersession": frozenset({"kind", "superseded", "quote"}),
    "validity": frozenset({"kind", "key", "date", "quote"}),
    "status": frozenset({"kind", "value", "quote"}),
    "identity": frozenset({"kind", "entity", "alias", "quote"}),
}


def human_body_of(text: str) -> str:
    """Return the authored body with the frontmatter block removed.

    Quotes are checked against THIS, not the raw document. A model that could quote the
    frontmatter would be able to justify `supersedes: X` with the string `supersedes: X`,
    which proves only that the block it is supposed to be inferring already exists.
    """
    _meta, body = parse_frontmatter(text)
    return body


def refuse_unclosed_frontmatter(text: str) -> None:
    """Refuse a document that opens a frontmatter block and never closes it.

    `parse_frontmatter` documents its behaviour here: an unclosed block is returned as body.
    That is the right call for retrieval, and the wrong one for extraction, because it hands
    the metadata lines to the ladder as prose. The verbatim quote guard would then accept
    `supersedes: X` as evidence for a supersession claim, which is exactly the thing the guard
    exists to prevent. A malformed document is refused rather than half read.
    """
    lines = text.split("\n")
    if not lines or lines[0].lstrip("﻿").strip() != "---":
        return
    if any(line.strip() == "---" for line in lines[1:]):
        return
    raise ExtractionBatchRejected(
        "unclosed_frontmatter",
        "the document opens a frontmatter block that is never closed, so its metadata cannot "
        "be separated from its prose",
    )


def normalize_extraction(
    raw: str,
    *,
    file: str,
    human_body: str,
    corpus_names: Sequence[str],
) -> tuple[tuple[ExtractedClaim, ...], tuple[ClaimRejection, ...]]:
    """Apply the ladder to one file's raw engine output.

    Returns `(survivors, rejections)`. Raises `ExtractionBatchRejected` when a batch level
    rung refuses the whole output.
    """
    payloads = _batch_rungs(raw)
    by_key = _corpus_index(corpus_names)
    survivors: list[ExtractedClaim] = []
    rejections: list[ClaimRejection] = []
    for index, payload in enumerate(payloads):
        outcome = _claim_rungs(
            payload, index=index, file=file, human_body=human_body, by_key=by_key
        )
        if isinstance(outcome, ClaimRejection):
            rejections.append(outcome)
        else:
            survivors.append(outcome)
    return tuple(survivors), tuple(rejections)


def _batch_rungs(raw: str) -> tuple[Mapping[str, Any], ...]:
    """Rungs 1 to 4. Any failure refuses the file's whole output."""
    try:
        decoded = json.loads(raw)
    except (ValueError, TypeError, RecursionError) as exc:
        # RecursionError is not a ValueError. Output nested past the interpreter's limit is
        # still just malformed output, and it must refuse like any other malformed output
        # rather than unwind through the caller's corpus loop.
        raise ExtractionBatchRejected("json", f"output is not JSON: {exc!r}") from exc
    if not isinstance(decoded, Mapping):
        raise ExtractionBatchRejected("top_level_shape", "output is not a JSON object")
    claims = decoded.get("claims")
    if not isinstance(claims, list):
        raise ExtractionBatchRejected("top_level_shape", "output has no 'claims' array")
    if len(claims) > MAX_CLAIMS_PER_FILE:
        raise ExtractionBatchRejected(
            "max_claims", f"{len(claims)} claims exceeds the maximum of {MAX_CLAIMS_PER_FILE}"
        )
    return tuple(_shape(index, claim) for index, claim in enumerate(claims))


def _shape(index: int, claim: Any) -> Mapping[str, Any]:
    """Rung 4. One malformed claim refuses the batch it arrived in."""
    if not isinstance(claim, Mapping):
        raise ExtractionBatchRejected("claim_shape", f"claim {index} is not an object")
    kind = claim.get("kind")
    # `isinstance` first: `kind` arrives unvalidated from model JSON, and an unhashable value
    # such as a list or an object would raise TypeError out of the membership test below.
    if not isinstance(kind, str) or kind not in _REQUIRED_FIELDS:
        raise ExtractionBatchRejected("claim_shape", f"claim {index} has unknown kind {kind!r}")
    expected = _REQUIRED_FIELDS[str(kind)]
    present = frozenset(str(key) for key in claim)
    if present != expected:
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        raise ExtractionBatchRejected(
            "claim_shape",
            f"claim {index} of kind {kind!r} has missing fields {missing} "
            f"and unexpected fields {extra}",
        )
    for field in expected - {"kind"}:
        if not isinstance(claim[field], str) or not claim[field].strip():
            raise ExtractionBatchRejected(
                "claim_shape", f"claim {index} field {field!r} is not a non-empty string"
            )
    if kind == "validity":
        _shaped_validity(index, claim)
    if kind == "status" and claim["value"] not in STATUS_VOCABULARY:
        raise ExtractionBatchRejected(
            "claim_shape",
            f"claim {index} status {claim['value']!r} is outside {list(STATUS_VOCABULARY)}",
        )
    return claim


def _shaped_validity(index: int, claim: Mapping[str, Any]) -> None:
    if claim["key"] not in VALIDITY_CLAIM_KEYS:
        raise ExtractionBatchRejected(
            "claim_shape",
            f"claim {index} validity key {claim['key']!r} is not one of "
            f"{list(VALIDITY_CLAIM_KEYS)}",
        )
    try:
        datetime.strptime(claim["date"], "%Y-%m-%d")
    except ValueError as exc:
        raise ExtractionBatchRejected(
            "claim_shape", f"claim {index} date {claim['date']!r} is not YYYY-MM-DD"
        ) from exc


def _corpus_index(corpus_names: Sequence[str]) -> Mapping[str, tuple[str, ...]]:
    index: dict[str, list[str]] = {}
    for name in corpus_names:
        names = index.setdefault(supersedes_key(name), [])
        # Deduplicate. A caller passing the same file name twice is not two files, and
        # counting it as two would turn a resolvable target into a false ambiguity refusal.
        if name not in names:
            names.append(name)
    return {key: tuple(names) for key, names in index.items()}


def _claim_rungs(
    payload: Mapping[str, Any],
    *,
    index: int,
    file: str,
    human_body: str,
    by_key: Mapping[str, tuple[str, ...]],
) -> ExtractedClaim | ClaimRejection:
    """Rungs 5 to 7, in order. The earliest failing rung is the one reported."""
    kind = str(payload["kind"])
    quote = str(payload["quote"])
    if quote not in human_body:
        return ClaimRejection(
            index=index,
            kind=kind,
            rung="quote_not_verbatim",
            reason=f"quote {_clipped(quote)!r} is not a verbatim substring of the body",
        )
    if kind == "supersession":
        return _supersession(payload, index=index, file=file, by_key=by_key, quote=quote)
    if kind == "validity":
        return _validity(payload, index=index, human_body=human_body, quote=quote)
    if kind == "status":
        return StatusClaim(value=str(payload["value"]), quote=quote)
    return IdentityClaim(
        entity=str(payload["entity"]).strip(), alias=str(payload["alias"]).strip(), quote=quote
    )


def _supersession(
    payload: Mapping[str, Any],
    *,
    index: int,
    file: str,
    by_key: Mapping[str, tuple[str, ...]],
    quote: str,
) -> ExtractedClaim | ClaimRejection:
    """Rung 6. Resolve the target to exactly one corpus file, or refuse."""
    target = str(payload["superseded"]).strip()
    candidates = by_key.get(supersedes_key(target), ())
    if len(candidates) != 1:
        reason = (
            f"names {target!r}, which matches {len(candidates)} files in the corpus"
            if candidates
            else f"names {target!r}, which is not a file in the corpus"
        )
        return ClaimRejection(index=index, kind="supersession", rung="target_not_in_corpus",
                              reason=reason)
    resolved = candidates[0]
    if supersedes_key(resolved) == supersedes_key(file):
        return ClaimRejection(
            index=index,
            kind="supersession",
            rung="target_not_in_corpus",
            reason=f"names itself ({target!r}); a document cannot supersede itself",
        )
    return SupersessionClaim(superseded=resolved, quote=quote)


def _validity(
    payload: Mapping[str, Any], *, index: int, human_body: str, quote: str
) -> ExtractedClaim | ClaimRejection:
    """Rung 7. A date the body does not literally contain was computed, not read."""
    date = str(payload["date"])
    if date not in human_body:
        return ClaimRejection(
            index=index,
            kind="validity",
            rung="date_not_in_body",
            reason=f"date {date!r} does not appear literally in the body",
        )
    return ValidityClaim(key=payload["key"], date=date, quote=quote)


def _clipped(value: str, limit: int = 60) -> str:
    return value if len(value) <= limit else f"{value[:limit]}..."


__all__ = ["human_body_of", "normalize_extraction", "refuse_unclosed_frontmatter"]
