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
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from recall.document import parse_document
from recall.frontmatter import VALIDITY_KEYS, supersedes_key
from recall.truth_extraction.types import (
    MAX_CLAIMS_PER_FILE,
    VALIDITY_CLAIM_KEYS,
    ClaimRejection,
    ExtractedClaim,
    ExtractionBatchRejected,
    IdentityClaim,
    StatusClaim,
    SupersessionClaim,
    ValidityClaim,
    coerce_status_vocabulary,
)

#: A frontmatter key line. `parse_frontmatter` strips a block only when it opens on line 0 and
#: closes; every other shape leaves `key: value` lines in the body, where they are verbatim
#: substrings and pass rung 5. Built from VALIDITY_KEYS so the two cannot drift, and MULTILINE
#: so a key line anywhere in a multi line region counts.
_FRONTMATTER_QUOTE_RE = re.compile(
    r"^[ \t]*(?:" + "|".join(re.escape(key) for key in VALIDITY_KEYS) + r")[ \t]*:",
    re.MULTILINE,
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
    return parse_document(text).human_body


def normalize_extraction(
    raw: str,
    *,
    file: str,
    human_body: str,
    corpus_names: Sequence[str],
    status_vocabulary: Sequence[str] | None = None,
) -> tuple[tuple[ExtractedClaim, ...], tuple[ClaimRejection, ...]]:
    """Apply the ladder to one file's raw engine output.

    Returns `(survivors, rejections)`. Raises `ExtractionBatchRejected` when a batch level
    rung refuses the whole output.

    `status_vocabulary=None` means the shipped default, matching `extract_file_claims`. Both are
    public, and they used to disagree: None was "use the default" in one and `TypeError` out of
    the other, escaping the ladder as a crash rather than a refusal and discarding every
    extraction a corpus run had already built. `()` disagreed the same way one step later, and
    is now refused here as it is at render time.
    """
    vocabulary = coerce_status_vocabulary(status_vocabulary)
    payloads = _batch_rungs(raw, vocabulary)
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


_FENCE_MARK = "```"

#: An opening fence's info string: `json`, `JSON`, `Json`, or nothing. Applied to one short line
#: with `fullmatch`, and it has no ambiguous quantifier, so it cannot backtrack.
_INFO_STRING = re.compile(r"[A-Za-z0-9_+-]*")


def _unfence(raw: str) -> str:
    """Unwrap output that is exactly one markdown code fence, and nothing looser.

    Chat models wrap JSON in ```json fences by habit, and the system prompt saying "Return JSON
    and nothing else" does not stop them: the default extraction model does it on a request as
    small as `Return exactly {"claims": []}`. Unstripped, `json.loads` fails at character 0 and
    the whole file is refused at the `json` rung, so a correct answer is discarded and the run
    reports ZERO claims. Measured: 30 of 30 documents, every claim kind, on the shipped default
    engine — a feature that looked like a careful model refusing and was a parser rejecting.

    ⚠️ **Deliberately not a regular expression.** Two successive regex versions each turned a
    degenerate reply into a denial of service on the INGEST path, where `extract_file_claims`'s
    broad guard does not reach (it wraps `engine.run` alone) and the HTTP timeout is long since
    satisfied. The first was ambiguous between two whitespace classes and the newline between
    them: `` ```json `` plus 20k newlines consumed **3,389 seconds of CPU without returning**.
    The second removed that and left the lazy body ambiguous with the closing fence's indent,
    which is cleanly quadratic on spaces and tabs: 11.1s at 64k spaces, extrapolating to roughly
    45 minutes at 1MB. Both were "fixed" and both shipped a new degenerate character. The
    scanning below is linear in the length of `raw` by construction, so there is no third
    character class to find.

    Deliberately narrow, unchanged. This accepts a fence around the WHOLE payload and nothing
    else: prose before or after it, or a second fence, still fails the rung, since the unwrapped
    text has to parse as JSON and a second fence does not. Being lenient about a near-universal
    wrapper is not the same as accepting arbitrary text, and only the first is safe here.

    ⚠️ **The type guard below is load-bearing.** `raw` is `str` by the engine port's contract,
    but a third-party engine that returns `None`, an `int` or a list is exactly the kind of
    contract violation this rung exists to absorb, and every operation here is a string method.
    Without the guard, `raw.strip()` raised `AttributeError` OUT of the ladder: `extract.py`
    guards `engine.run` alone, and the two `except` clauses downstream catch only
    `ExtractionBatchRejected`, so one such reply aborted `extract_corpus_claims` and discarded
    every extraction already built. Refusing at the `json` rung records it per file instead, and
    that is what the code did before the fence work: the regression came in with this function.
    """
    if not isinstance(raw, str):
        # Straight through to `json.loads`, which parses `bytes` and refuses everything else at
        # the `json` rung. Not unwrapped: a fence around bytes is not a shape any engine sends,
        # and inventing a decode here would be guessing at an encoding.
        return raw
    text = raw.strip()
    if len(text) < 2 * len(_FENCE_MARK) or not text.startswith(_FENCE_MARK):
        return raw
    if not text.endswith(_FENCE_MARK):
        return raw
    opening_end = text.find("\n")
    if opening_end == -1:
        # A single line that opens and closes a fence carries no payload line to unwrap.
        return raw
    if not _INFO_STRING.fullmatch(text[len(_FENCE_MARK) : opening_end].strip()):
        return raw
    body = text[opening_end + 1 : -len(_FENCE_MARK)].rstrip("\r\n \t")
    return body or raw


def _batch_rungs(raw: str, status_vocabulary: Sequence[str]) -> tuple[Mapping[str, Any], ...]:
    """Rungs 1 to 4. Any failure refuses the file's whole output."""
    try:
        decoded = json.loads(_unfence(raw))
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
    return tuple(_shape(index, claim, status_vocabulary) for index, claim in enumerate(claims))


def _shape(index: int, claim: Any, status_vocabulary: Sequence[str]) -> Mapping[str, Any]:
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
    if kind == "status":
        # Case-insensitive, and the claim is rewritten to the VOCABULARY's spelling. The bug this
        # rung produced was a casing mismatch, not a vocabulary mismatch: PEP bodies write
        # `Status: Final` and the model answered `final`, and an exact-membership test against a
        # caller-supplied `Final,Rejected,...` refuses that at a BATCH rung, taking the file's
        # supersession claims down with a status the reader would have accepted. Passing the
        # vocabulary in its own casing is the natural thing for a caller to do, so a comparison
        # that only works in one casing moves the defect rather than fixing it. Canonicalising
        # means one spelling reaches the store however the model wrote it.
        canonical = {str(word).casefold(): str(word) for word in status_vocabulary}
        folded = str(claim["value"]).strip().casefold()
        if folded not in canonical:
            raise ExtractionBatchRejected(
                "claim_shape",
                f"claim {index} status {claim['value']!r} is outside {list(status_vocabulary)}",
            )
        return {**claim, "value": canonical[folded]}
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
    if _quote_sits_in_frontmatter(human_body, quote):
        # Verbatim, and still not evidence. Justifying `supersedes: X` with the string
        # `supersedes: X` proves only that the line exists, which is what the claim was
        # supposed to establish. Refuse the CLAIM, not the document: the document's real
        # prose claims are unaffected, and no boundary has to be guessed at.
        return ClaimRejection(
            index=index,
            kind=kind,
            rung="quote_is_frontmatter",
            reason=f"quote {_clipped(quote)!r} is a frontmatter key line, not prose",
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


def _base_name(name: str) -> str:
    """`supersedes_key` with one further dotted segment removed, by pure string operations.

    `Path(...).stem` was used here and is wrong twice over: it splits on `\\` and on a drive
    prefix on Windows and not on POSIX, which made the ladder's verdict OS-dependent inside a
    module whose first line promises no filesystem; and it disagrees with `supersedes_key`,
    which splits on `/` alone. One rule, spelled out.
    """
    tail = supersedes_key(name)
    stem, dot, _suffix = tail.rpartition(".")
    return stem if dot else tail


def _own_entries(file: str, by_key: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    """The corpus entries `file` itself denotes.

    Self-supersession is "the target resolved to THIS document", so the document is resolved the
    same way the target was. Four rules have been wrong here, each fixing one name shape and
    breaking another, so the shapes are enumerated rather than argued about:

    - `supersedes_key(resolved) == supersedes_key(file)` was inert outside markdown, because
      `supersedes_key` strips `.md` and nothing else, so `pep-0262.rst` never equalled the corpus
      key `pep-0262`. A real run had `pep-0262` declare itself its own predecessor.
    - Stemming the `file` side alone fixed that and broke every corpus whose names carry the
      extension, which is most of them: `corpus_names` is usually just the list of file names.
    - Stemming BOTH sides fixed those and refused genuine edges between DIFFERENT documents
      sharing a base name: `guide.rst` superseding `guide.txt` was refused as "names itself".
    - Falling back to the file's stem alone fixed THAT and refused `v1.2` superseding `v1`, and
      it still could not see a stem-shaped `file` against an extensioned corpus, which is the
      mirror of the asymmetry it was written for.

    - Trying the two fallbacks IN ORDER let one shadow the other. A single `guide.rst.orig`
      beside `guide` made the first fallback answer with the backup file, so the second never
      ran and `guide.rst` was free to declare `guide` its predecessor: an ACCEPTED self-edge,
      which is the worst outcome this guard exists to prevent, produced by adding one unrelated
      name to a corpus.

    The exact match runs first and alone, so a corpus holding `guide.rst` AND `guide.txt` never
    reaches a base-name comparison. Below it the two fallbacks are UNIONED rather than ordered,
    because neither is more specific than the other and whichever runs second can be shadowed.

    ⚠️ **A trade this cannot avoid, stated rather than hidden.** Two corpora are structurally
    identical and want opposite answers:

    - `file="guide.rst"`, corpus `("guide", "guide.rst.orig")`, target `guide` — the same
      document, must be REFUSED.
    - `file="v1.2"`, corpus `("v1.2.rst", "v1")`, target `v1` — a different document, would
      ideally be accepted.

    In both, the corpus holds one name that is the file's key plus a suffix and one that is its
    base. Nothing in the names separates them; only knowing that `.rst` is an extension and `.2`
    is not would, and that is a guess about the corpus rather than a fact in it. So the union
    refuses both, and the asymmetry of the stakes is the reason: a wrongly ACCEPTED self-edge
    marks a live document stale beneath itself, which is the exact failure the trust layer
    exists to prevent, while a wrongly refused edge costs one proposal a reviewer never sees.
    `_supersession` says which of the two happened rather than asserting "names itself" for both.
    """
    key = supersedes_key(file)
    exact = by_key.get(key)
    if exact:
        return exact
    # The corpus carries a suffix the file does not: `file="pep-0262"`, corpus `pep-0262.rst`.
    # Read out of `by_key`'s values rather than taking `corpus_names` as a second argument, so
    # there is one corpus here and no way for the two to be passed out of step.
    carried = tuple(
        name for entries in by_key.values() for name in entries if _base_name(name) == key
    )
    # The file carries a suffix the corpus does not: `file="pep-0262.rst"`, corpus `pep-0262`.
    # This is how the PEPs arm is configured.
    return carried + by_key.get(_base_name(file), ())


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
    if resolved in _own_entries(file, by_key):
        # Two different findings, and the message says which. The exact case is a fact: the
        # target and the file are one corpus entry. The other is an INFERENCE from how this
        # corpus names its files, and `_own_entries` documents why it refuses rather than
        # guessing. Reporting both as "names itself" would state as fact something the code
        # concluded from a suffix.
        itself = resolved in by_key.get(supersedes_key(file), ())
        return ClaimRejection(
            index=index,
            kind="supersession",
            rung="target_not_in_corpus",
            reason=(
                f"names itself ({target!r}); a document cannot supersede itself"
                if itself
                else f"names {target!r}, which this corpus's naming resolves to {file!r} itself; "
                f"a document cannot supersede itself"
            ),
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


def _quote_sits_in_frontmatter(human_body: str, quote: str) -> bool:
    """True when the quote is part of a metadata line rather than of prose.

    Judged by WHERE the quote sits, not by the quote text alone: quoting just the value half of
    a `key: value` line would otherwise carry the declaration through untouched. So the check
    widens to the whole lines the quote occupies and asks whether any of them is a key line.

    EVERY occurrence is checked, not the first. When the same text appears both in metadata and
    in prose, judging only the first would make the verdict depend on document order rather than
    on what the model read, and would flip in both directions.

    Deliberately NOT reconstructed: which key a continuation line belongs to. An earlier version
    walked upward from an indented line to find its owning key, which meant guessing where a
    YAML block ends — the same undecidable boundary that sank the document level version of this
    guard, and it refused ordinary markdown bullet lists. It also defended nothing:
    `parse_frontmatter` reads the text after the first colon, so `supersedes:` with a list below
    it declares the EMPTY string, which the trust layer ignores. With no declaration there is
    nothing for a quote to be circular with.
    """
    trimmed = quote.strip("\n")
    if not trimmed:  # pragma: no cover - `_shape` has already rejected a blank quote
        return False
    start = human_body.find(trimmed)
    # Occurrences that fall entirely inside the region just scanned share that region, so they
    # need no work. Without this a quote recurring inside one long line — a pasted config, a
    # minified blob — re-slices and re-scans the whole line once per occurrence, which is
    # quadratic in the body. The containment test must check the END too: an occurrence that
    # starts inside the region but runs past it spans more lines and needs its own scan.
    line_start, line_end = 0, -1
    while start != -1:
        end = start + len(trimmed)
        if not (line_start <= start and end <= line_end):
            line_start = human_body.rfind("\n", 0, start) + 1
            newline = human_body.find("\n", end)
            line_end = len(human_body) if newline == -1 else newline
            if _FRONTMATTER_QUOTE_RE.search(human_body[line_start:line_end]):
                return True
        start = human_body.find(trimmed, start + 1)
    return False


def _clipped(value: str, limit: int = 60) -> str:
    return value if len(value) <= limit else f"{value[:limit]}..."


__all__ = ["human_body_of", "normalize_extraction"]
