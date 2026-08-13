"""Exact round trip between `FileExtraction` and JSON text.

A persisted extraction cache is only safe if reloading an entry gives back exactly what was
stored. `recheck_cached_extractions` compares `tuple(claims)` between a fresh engine answer and
a cached one and reports any difference as ENGINE NONDETERMINISM, so a round trip that reorders
claims, or that turns an `int` index into a `bool`, does not surface as a serialization bug. It
surfaces as a number telling the user their model is unreliable.

So the guarantee here is total equality, with no carve outs a later reader has to remember:

    extraction_from_json(extraction_to_json(x)) == x

`cached` is stored along with everything else for the same reason. This module's job is
fidelity, not judgment, and `extract.py` already sets `cached=True` at its own read site.

The awkward part is `claims`. It is a tuple over four frozen dataclasses discriminated by a
`kind` ClassVar, and a ClassVar is not a field: `dataclasses.asdict` drops the discriminator and
the four shapes become indistinguishable. So `kind` is written out explicitly and resolved
through `_CLAIM_TYPES` on the way back. Order is preserved by storing claims as a JSON list.

Nothing here touches disk, SQLite or an engine. That is deliberate: the round trip is the
property most likely to break silently under a later refactor of `types.py`, and a pure function
is one a test can pin with nothing standing between the assertion and the defect.
"""

from __future__ import annotations

import json
from typing import Any

from recall.truth_extraction.types import (
    ClaimRejection,
    ExtractedClaim,
    FileExtraction,
    IdentityClaim,
    StatusClaim,
    SupersessionClaim,
    ValidityClaim,
)


class ExtractionPayloadInvalid(ValueError):
    """A stored payload is not something this module wrote.

    Raised for bad JSON, an unknown claim kind, a field set that is not exactly right, or a
    value of the wrong type. Callers treat it as a cache MISS: the entry is re-paid from the
    engine. It is never allowed to escape a cache `get`, because a corruption the user did not
    cause must not abort an ingest partway through a corpus.
    """


#: Claim kind, as written on each shape's `kind` ClassVar, to the shape it names. A contract
#: test pins this against `CLAIM_KINDS` and against each class's own ClassVar, so a fifth claim
#: kind added to `types.py` fails a test here rather than quietly making every extraction that
#: contains one uncacheable.
_CLAIM_TYPES: dict[str, type[ExtractedClaim]] = {
    "supersession": SupersessionClaim,
    "validity": ValidityClaim,
    "status": StatusClaim,
    "identity": IdentityClaim,
}

#: Exact field name to Python type for every shape stored here. Written out rather than read
#: off `dataclasses.fields`, whose `.type` is a STRING under `from __future__ import
#: annotations` and would have to be re-parsed to be worth anything. A contract test pins this
#: table against the dataclasses, so drift fails rather than silently loosening the check.
#:
#: `ClaimRejection` has a field literally named `kind`. It does not collide with the claim
#: discriminator: rejections are stored without one, because their type is known from position.
_FIELDS: dict[type, dict[str, type]] = {
    SupersessionClaim: {"superseded": str, "quote": str},
    ValidityClaim: {"key": str, "date": str, "quote": str},
    StatusClaim: {"value": str, "quote": str},
    IdentityClaim: {"entity": str, "alias": str, "quote": str},
    ClaimRejection: {"index": int, "kind": str, "rung": str, "reason": str},
}

#: Top level field name to Python type, mirroring `FileExtraction`. `claims`, `rejections` and
#: `batch_rejection` are checked separately because their types are not plain scalars.
_TOP_LEVEL_SCALARS: dict[str, type] = {
    "file": str,
    "engine_id": str,
    "model_id": str,
    "revision": str,
    "prompt_revision": str,
    "cached": bool,
}

_TOP_LEVEL_KEYS = frozenset(_TOP_LEVEL_SCALARS) | {"claims", "rejections", "batch_rejection"}


def _checked(cls: type, obj: object, *, what: str) -> dict[str, Any]:
    """Validate one stored object against `cls`'s field table and return its values.

    The field set must match EXACTLY, neither a superset nor a subset. An extra key means the
    payload was written by something that is not this module; a missing one means it was
    truncated. Both are corruption, and both are better answered with a re-paid engine call
    than with a `FileExtraction` that is quietly missing a quote.
    """
    if type(obj) is not dict:
        raise ExtractionPayloadInvalid(f"{what} is {type(obj).__name__}, not an object")
    expected = _FIELDS[cls]
    if set(obj) != set(expected):
        raise ExtractionPayloadInvalid(
            f"{what} has fields {sorted(obj)}, expected {sorted(expected)}"
        )
    for name, want in expected.items():
        value = obj[name]
        # `type(...) is`, NOT `isinstance`. `bool` subclasses `int`, so `isinstance(True, int)`
        # is true and a `ClaimRejection` would round trip its `index` back as `True`, which
        # compares unequal to the `1` that was stored and reads downstream as engine drift.
        if type(value) is not want:
            raise ExtractionPayloadInvalid(
                f"{what} field {name!r} is {type(value).__name__}, expected {want.__name__}"
            )
    return dict(obj)


def _claim_to_object(claim: ExtractedClaim) -> dict[str, Any]:
    payload: dict[str, Any] = {"kind": claim.kind}
    payload.update({name: getattr(claim, name) for name in _FIELDS[type(claim)]})
    return payload


def _claim_from_object(obj: object, *, index: int) -> ExtractedClaim:
    if type(obj) is not dict:
        raise ExtractionPayloadInvalid(f"claim {index} is {type(obj).__name__}, not an object")
    body = dict(obj)
    kind = body.pop("kind", None)
    # The type check comes FIRST, and it is not decoration. `kind not in _CLAIM_TYPES` HASHES
    # the key, so a stored `"kind"` that is a JSON array or object raised
    # `TypeError: unhashable type` straight out of this function, breaking the one contract this
    # module has: that a bad payload arrives as `ExtractionPayloadInvalid`. `get` catches
    # everything, so it degraded to a miss there rather than crashing an ingest, but
    # `extraction_from_json` is exported and its callers were promised otherwise.
    if type(kind) is not str or kind not in _CLAIM_TYPES:
        raise ExtractionPayloadInvalid(
            f"claim {index} has kind {kind!r}, not one of {sorted(_CLAIM_TYPES)}"
        )
    # No `cast` here any more: the `type(kind) is not str` guard above narrows `kind` for mypy,
    # so the cast the untyped membership test used to need is now redundant and rejected by
    # `warn_redundant_casts`. A pleasant side effect of guarding for the right reason.
    cls = _CLAIM_TYPES[kind]
    # No cast on the result: `_CLAIM_TYPES` values are already the four claim classes, so mypy
    # infers the union and a cast here is redundant, which `warn_unused_ignores` reports as an
    # error rather than ignoring.
    return cls(**_checked(cls, body, what=f"claim {index}"))


def _rejection_to_object(rejection: ClaimRejection) -> dict[str, Any]:
    return {name: getattr(rejection, name) for name in _FIELDS[ClaimRejection]}


def _rejection_from_object(obj: object, *, what: str) -> ClaimRejection:
    return ClaimRejection(**_checked(ClaimRejection, obj, what=what))


def extraction_to_json(extraction: FileExtraction) -> str:
    """Serialize one extraction. `sort_keys` orders KEYS; claims keep their list order."""
    return json.dumps(
        {
            "file": extraction.file,
            "claims": [_claim_to_object(claim) for claim in extraction.claims],
            "rejections": [_rejection_to_object(item) for item in extraction.rejections],
            "engine_id": extraction.engine_id,
            "model_id": extraction.model_id,
            "revision": extraction.revision,
            "prompt_revision": extraction.prompt_revision,
            "batch_rejection": (
                None
                if extraction.batch_rejection is None
                else _rejection_to_object(extraction.batch_rejection)
            ),
            "cached": extraction.cached,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def extraction_from_json(text: str) -> FileExtraction:
    """Rebuild one extraction, or raise `ExtractionPayloadInvalid`.

    Every list becomes a tuple again here, at the boundary. Leaving them as lists would make a
    reloaded `FileExtraction` compare unequal to the one that was stored, which `recheck` would
    report as the engine disagreeing with itself.
    """
    try:
        raw = json.loads(text)
    except ValueError as exc:
        raise ExtractionPayloadInvalid(f"payload is not JSON: {exc}") from exc
    if type(raw) is not dict:
        raise ExtractionPayloadInvalid(f"payload is {type(raw).__name__}, not an object")
    if set(raw) != _TOP_LEVEL_KEYS:
        raise ExtractionPayloadInvalid(
            f"payload has fields {sorted(raw)}, expected {sorted(_TOP_LEVEL_KEYS)}"
        )
    for name, want in _TOP_LEVEL_SCALARS.items():
        if type(raw[name]) is not want:
            raise ExtractionPayloadInvalid(
                f"payload field {name!r} is {type(raw[name]).__name__}, "
                f"expected {want.__name__}"
            )
    for name in ("claims", "rejections"):
        if type(raw[name]) is not list:
            raise ExtractionPayloadInvalid(
                f"payload field {name!r} is {type(raw[name]).__name__}, expected list"
            )
    batch = raw["batch_rejection"]
    return FileExtraction(
        file=raw["file"],
        claims=tuple(
            _claim_from_object(item, index=index) for index, item in enumerate(raw["claims"])
        ),
        rejections=tuple(
            _rejection_from_object(item, what=f"rejection {index}")
            for index, item in enumerate(raw["rejections"])
        ),
        engine_id=raw["engine_id"],
        model_id=raw["model_id"],
        revision=raw["revision"],
        prompt_revision=raw["prompt_revision"],
        batch_rejection=(
            None if batch is None else _rejection_from_object(batch, what="batch rejection")
        ),
        cached=raw["cached"],
    )


__all__ = [
    "ExtractionPayloadInvalid",
    "extraction_from_json",
    "extraction_to_json",
]
