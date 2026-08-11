"""Structured claim shapes for model backed truth extraction.

RE-call's model of truth is authored frontmatter, and exactly three keys are recognised
(`recall/frontmatter.py`): `supersedes`, `valid_from`, `valid_until`. Prose that states the
same relation is retrieved but never interpreted, so a memo whose body says "superseded by X"
is still served with `verdict == "ok"`.

These four shapes are the interface between a model and that trust layer. Each one carries a
`quote`, and the quote must be a verbatim substring of the memo body — that is what makes a
claim checkable by a human rather than a summary they have to take on faith.

Nothing here proposes, promotes, or writes. A claim is an observation about text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

#: Per file ceiling. A model returning more than this has stopped extracting and started
#: narrating; the whole file's output is refused rather than truncated, because a truncated
#: batch is indistinguishable from a complete one downstream.
MAX_CLAIMS_PER_FILE = 12

ClaimKind = Literal["supersession", "validity", "status", "identity"]
CLAIM_KINDS: tuple[ClaimKind, ...] = ("supersession", "validity", "status", "identity")

ValidityKey = Literal["valid_from", "valid_until"]
#: The only validity keys `recall.frontmatter` recognises. `supersedes` is the third
#: recognised key but it has its own claim kind, because it names another document.
VALIDITY_CLAIM_KEYS: tuple[ValidityKey, ...] = ("valid_from", "valid_until")

#: Closed vocabulary. An open status field would let a model invent a taxonomy per memo, and
#: a taxonomy nobody declared cannot be acted on. Off vocabulary values reject the batch.
STATUS_VOCABULARY: tuple[str, ...] = (
    "active",
    "draft",
    "deprecated",
    "superseded",
    "withdrawn",
)

#: Rungs of the validation ladder, in the order they are applied. The batch rungs reject
#: the file's entire output; the claim rungs reject one claim and keep the rest.
BATCH_RUNGS: tuple[str, ...] = (
    "json",
    "top_level_shape",
    "max_claims",
    "claim_shape",
)
CLAIM_RUNGS: tuple[str, ...] = (
    "quote_not_verbatim",
    "quote_is_frontmatter",
    "target_not_in_corpus",
    "date_not_in_body",
)


@dataclass(frozen=True)
class SupersessionClaim:
    """`superseded` is the document this file replaces, canonicalised to a corpus name."""

    superseded: str
    quote: str

    kind: ClassVar[ClaimKind] = "supersession"


@dataclass(frozen=True)
class ValidityClaim:
    """A `valid_from` / `valid_until` date the body states in prose but never declares."""

    key: ValidityKey
    date: str
    quote: str

    kind: ClassVar[ClaimKind] = "validity"


@dataclass(frozen=True)
class StatusClaim:
    """A lifecycle status drawn from `STATUS_VOCABULARY`."""

    value: str
    quote: str

    kind: ClassVar[ClaimKind] = "status"


@dataclass(frozen=True)
class IdentityClaim:
    """Two names the body treats as one entity."""

    entity: str
    alias: str
    quote: str

    kind: ClassVar[ClaimKind] = "identity"


ExtractedClaim = SupersessionClaim | ValidityClaim | StatusClaim | IdentityClaim


@dataclass(frozen=True)
class ClaimRejection:
    """One claim refused, recorded so a reviewer can see WHY rather than see nothing.

    `index` is the claim's position in the model's output, which is the only handle a
    reviewer has on a claim that never became an object.
    """

    index: int
    kind: str
    rung: str
    reason: str


@dataclass(frozen=True)
class FileExtraction:
    """Everything one file's extraction produced, survivors and refusals alike."""

    file: str
    claims: tuple[ExtractedClaim, ...]
    rejections: tuple[ClaimRejection, ...]
    engine_id: str
    model_id: str
    revision: str
    #: The prompt revision that produced this result. Part of the audit identity: the same
    #: engine under a reworded prompt is not the same extractor.
    prompt_revision: str
    #: Set when a batch level rung refused the file's whole output. Its `index` is -1,
    #: because the refusal is scoped to the file rather than to any one claim. Recorded
    #: rather than raised: a refusal nobody sees is a refusal nobody reviews.
    batch_rejection: ClaimRejection | None = None
    #: True when this result came back from the cache rather than from the engine.
    cached: bool = False


class ExtractionBatchRejected(ValueError):
    """The file's whole output is refused. Carries the ladder rung that refused it."""

    def __init__(self, rung: str, reason: str) -> None:
        super().__init__(f"{rung}: {reason}")
        self.rung = rung
        self.reason = reason


__all__ = [
    "BATCH_RUNGS",
    "CLAIM_KINDS",
    "CLAIM_RUNGS",
    "ClaimKind",
    "ClaimRejection",
    "ExtractedClaim",
    "ExtractionBatchRejected",
    "FileExtraction",
    "IdentityClaim",
    "MAX_CLAIMS_PER_FILE",
    "STATUS_VOCABULARY",
    "StatusClaim",
    "SupersessionClaim",
    "VALIDITY_CLAIM_KEYS",
    "ValidityClaim",
    "ValidityKey",
]
