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

from collections.abc import Sequence
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

#: Consecutive engine failures after which the engine is treated as unavailable for the rest of
#: a run. Consecutive rather than total: a corpus with a few individually awkward memos should
#: not stop, while an endpoint that is simply down should stop being asked once per file.
#: Lives here, the leaf, because both `extract` and `_cache` need it and `extract` imports
#: `_cache`, so defining it in either would be a cycle.
CONSECUTIVE_ENGINE_FAILURE_LIMIT = 3

#: Rungs of the validation ladder, in the order they are applied. The batch rungs reject
#: the file's entire output; the claim rungs reject one claim and keep the rest.
BATCH_RUNGS: tuple[str, ...] = (
    "engine_error",
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


def coerce_status_vocabulary(status_vocabulary: Sequence[str] | None) -> tuple[str, ...]:
    """Resolve a caller's vocabulary, or refuse it. ONE definition, used by both entry points.

    `build_extraction_prompt` and `normalize_extraction` are both public and both take this
    argument, and they disagreed about every degenerate value: `None` meant "the default" in one
    and `TypeError` in the other; `()` raised at render time and silently refused every status
    claim at a BATCH rung in the ladder; a bare `str` passed both and exploded into single
    characters, so `--status-vocabulary final` rendered `['f', 'i', 'n', 'a', 'l']` into the
    prompt. A shared coercion is what stops the next value being decided twice.
    """
    if status_vocabulary is None:
        return tuple(STATUS_VOCABULARY)
    if isinstance(status_vocabulary, str):
        # `Sequence[str]` accepts `str`, and mypy will not flag it. A single word is the natural
        # thing to hand a `--status-vocabulary` flag, and it is never what the caller meant.
        raise ValueError(
            f"status_vocabulary must be a sequence of words, not the single string "
            f"{status_vocabulary!r}, which would be read one character at a time"
        )
    given = tuple(status_vocabulary)  # materialised once: an iterator must not be walked twice
    wrong_type = sorted({type(word).__name__ for word in given if not isinstance(word, str)})
    if wrong_type:
        # Refused, not stringified. `str(b"Final")` is `"b'Final'"`, which renders into the
        # prompt and lands in the audit record looking like a word the caller chose.
        raise ValueError(f"status_vocabulary entries must be strings, not {wrong_type}")
    # STRIPPED, because `_shape` strips the model's value before comparing. Unstripped, a
    # `" Final"` from a comma split that forgot to strip renders into the prompt, never matches
    # the answer, and refuses the whole file at a BATCH rung — which is the exact failure this
    # parameter exists to remove, reintroduced by the validation meant to prevent it.
    vocabulary = tuple(word.strip() for word in given)
    if not vocabulary:
        raise ValueError(
            "status_vocabulary is empty: every status claim would be refused, and at a BATCH "
            "rung, so each file carrying one would lose its other claims too"
        )
    blank = [raw for raw, word in zip(given, vocabulary) if not word]
    if blank:
        raise ValueError(
            f"status_vocabulary has entries that are blank once stripped: {blank}. A blank word "
            f"is rendered into the prompt and can never match an answer"
        )
    # One pass, not n². The detector was O(n²) with n² `casefold()` calls and ran twice per
    # document (once at render, once in the ladder): 2.8 s per call at 4000 words, so a
    # 792-document corpus would have spent over an hour in argument validation.
    #
    # `previous != word` is also what makes an exact repeat harmless: it maps to the same fold
    # AND the same spelling, so nothing is reported. The old form counted casefold-equal entries
    # instead, which counts a word against its own duplicate, and aborted the run claiming two
    # identical words "differ only by case".
    folded: dict[str, str] = {}
    collisions: list[str] = []
    for word in vocabulary:
        previous = folded.setdefault(word.casefold(), word)
        if previous != word:
            collisions.append(word)
    if collisions:
        # Matching is case-insensitive, so these are one word to the ladder and two to the
        # model. Which spelling reaches the store would then depend on list order.
        raise ValueError(
            f"status_vocabulary has entries that differ only by case: {sorted(collisions)}. "
            f"Matching is case-insensitive, so one of them would silently never be stored"
        )
    return vocabulary


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
    #: The status vocabulary the model was shown. Also part of the audit identity, by exactly the
    #: argument above: the vocabulary is rendered into the prompt, so changing it IS rewording
    #: the prompt. It reached the cache key and stopped there, so two runs under different
    #: vocabularies produced byte-identical records and a reviewer could not tell which list the
    #: model had been given.
    status_vocabulary: tuple[str, ...] = STATUS_VOCABULARY
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
