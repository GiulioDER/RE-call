"""Frozen input manifests: question ids and input hashes, fixed before any candidate exists.

This is the ladder manifest pattern (`benchmarks/ladder/manifest.py`) applied to promotion
evidence, and it is deliberately the same shape rather than a second one: sorted canonical
rendering, a digest over the body AND the provenance header, the digest excluded from what it
covers, and a reader that recomputes and refuses a mismatch instead of repairing it.

Why a promotion harness needs it at all. The gate in `recall/promotion.py` compares a baseline
arm against a candidate arm question by question. If the question set can move between the two
runs — a question dropped because it errored, a label corrected after seeing a bad result, a
query reworded — then the paired test is no longer paired, and every number it produces is a
statement about a comparison that was never made. Freezing ids and input hashes BEFORE either arm
runs is what makes "the same questions" checkable rather than assumed.

`input_hash` covers the question's INPUT SIDE only: its id, its corpus, the query text, and its
expected relevance labels. Two things fall out of that, and both are the point:

- A source corpus that drifts under a frozen manifest is caught at run time, per question, by
  `FrozenQuestion.verify`. The manifest digest cannot catch that on its own: it proves the
  manifest was not edited, not that the corpus still says what the manifest was built from.
- A label edited after seeing a candidate's results changes `input_hash`, which changes the body,
  which changes the manifest digest. The edit is therefore loud, which is the only defence
  against the failure this file exists for.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

#: Bumping this invalidates every frozen manifest at once, so it is versioned rather than
#: implicit. A frozen artifact whose digest silently changed meaning is the failure mode the
#: whole module is built against.
MANIFEST_VERSION = "1.0"

_DOMAIN = b"recall-promotion-frozen-manifest-v1\n"

_FIELDS = ("question_id", "corpus", "input_hash", "expected_relevance_labels")


def question_input_hash(
    *, question_id: str, corpus: str, query: str, expected_relevance_labels: Sequence[str]
) -> str:
    """SHA-256 over the complete input identity of one question.

    NUL-terminated field encoding, the same reason `EmbeddingProfile.fingerprint` uses it: without
    terminators, ``("ab", "c")`` and ``("a", "bc")`` hash alike, and two different questions would
    share an input hash. Labels are sorted, so a manifest built from a set-ordered source and one
    built from a list-ordered source agree.
    """
    digest = hashlib.sha256()
    digest.update(_DOMAIN)
    for field in (question_id, corpus, query):
        digest.update(field.encode("utf-8"))
        digest.update(b"\x00")
    for label in sorted(expected_relevance_labels):
        digest.update(label.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenQuestion:
    """One question, frozen: what it is called, and what its inputs hashed to."""

    question_id: str
    corpus: str
    input_hash: str
    #: The chunk / document ids a correct retrieval must surface. **Empty means unanswerable**,
    #: and that is a label, not missing data: an unanswerable question carries no hit@5 (there is
    #: nothing to hit) and is scored on the safety axis instead. `aggregate.py` splits on exactly
    #: this, so an adapter that emits `()` for "I could not find the labels" would move a question
    #: from the quality set into the safety set without anything raising.
    expected_relevance_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.question_id or not self.corpus or not self.input_hash:
            raise ValueError("question_id, corpus and input_hash must all be non-empty")
        if not isinstance(self.expected_relevance_labels, tuple):
            raise TypeError("expected_relevance_labels must be a tuple — it is hashed and frozen")

    @property
    def answerable(self) -> bool:
        return bool(self.expected_relevance_labels)

    def verify(self, query: str) -> None:
        """Refuse a query text that does not hash to the frozen `input_hash`.

        Called by the runner with the text it is about to send, so a corpus that changed under a
        frozen manifest fails on the question it changed rather than producing a plausible number
        for a question that is no longer the one that was frozen.
        """
        actual = question_input_hash(
            question_id=self.question_id,
            corpus=self.corpus,
            query=query,
            expected_relevance_labels=self.expected_relevance_labels,
        )
        if actual != self.input_hash:
            raise ValueError(
                f"{self.corpus}/{self.question_id}: the query text and labels being run hash to "
                f"{actual}, but the frozen manifest recorded {self.input_hash}. The source corpus "
                f"or its labels changed after the manifest was frozen; re-freeze deliberately "
                f"rather than scoring a question that is no longer the frozen one."
            )


def question_to_dict(question: FrozenQuestion) -> dict:
    return {
        "question_id": question.question_id,
        "corpus": question.corpus,
        "input_hash": question.input_hash,
        "expected_relevance_labels": list(question.expected_relevance_labels),
    }


def question_from_dict(payload: Mapping) -> FrozenQuestion:
    missing = [field for field in _FIELDS if field not in payload]
    if missing:
        raise ValueError(f"frozen question is missing fields: {missing}")
    return FrozenQuestion(
        question_id=payload["question_id"],
        corpus=payload["corpus"],
        input_hash=payload["input_hash"],
        expected_relevance_labels=tuple(payload["expected_relevance_labels"]),
    )


def _canonical(question: FrozenQuestion) -> str:
    return json.dumps(question_to_dict(question), sort_keys=True, ensure_ascii=False)


def manifest_digest(
    questions: Sequence[FrozenQuestion], *, corpus_hashes: Mapping[str, str]
) -> str:
    """SHA-256 over every frozen question AND the provenance saying what they were built from.

    `corpus_hashes` is covered, not merely stored beside the body. It is the field a tamperer
    edits to make a manifest claim a provenance it does not have, and a digest over question
    bodies alone accepts that edit in silence.

    A digest cannot cover itself, so the header's `digest` field is excluded by construction.
    Questions are sorted, so the result does not depend on the order an adapter emitted them.
    """
    digest = hashlib.sha256()
    digest.update(_DOMAIN)
    for line in sorted(_canonical(question) for question in questions):
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    digest.update(b"--provenance--\n")
    digest.update(
        json.dumps({"corpus_hashes": dict(corpus_hashes)}, sort_keys=True, ensure_ascii=False)
        .encode("utf-8")
    )
    digest.update(b"\n")
    return digest.hexdigest()


def write_manifest(
    path: Path,
    questions: Sequence[FrozenQuestion],
    *,
    corpus_hashes: Mapping[str, str],
) -> str:
    """Write header + one JSON object per question. Returns the digest.

    Refuses a duplicate `(corpus, question_id)`. The gate's `_groups` refuses one too, but by then
    a whole arm has been run; catching it at freeze time costs nothing and catches it before any
    embedding is paid for.
    """
    seen: set[tuple[str, str]] = set()
    for question in questions:
        identity = (question.corpus, question.question_id)
        if identity in seen:
            raise ValueError(
                f"duplicate frozen question {question.corpus}/{question.question_id}"
            )
        seen.add(identity)
    if not questions:
        raise ValueError("refusing to freeze an empty manifest — a gate over no questions passes")
    digest = manifest_digest(questions, corpus_hashes=corpus_hashes)
    header = {
        "manifest_version": MANIFEST_VERSION,
        "digest": digest,
        "n_questions": len(questions),
        "corpus_hashes": dict(corpus_hashes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": text-mode translation would write CRLF on Windows while the digest above is
    # computed over "\n"-joined lines — same content, different bytes. A frozen artifact's bytes
    # must not depend on the OS that froze it.
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(header, sort_keys=True, ensure_ascii=False) + "\n")
        for question in sorted(questions, key=lambda q: (q.corpus, q.question_id)):
            handle.write(_canonical(question) + "\n")
    return digest


def read_manifest(path: Path) -> tuple[list[FrozenQuestion], dict]:
    """Read and VERIFY. A body that does not match its header digest is refused, not repaired."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path} is empty")
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}: could not parse the header line — the file is truncated or not a manifest."
        ) from exc
    version = header.get("manifest_version")
    if version != MANIFEST_VERSION:
        raise ValueError(
            f"{path}: manifest_version {version!r} is not {MANIFEST_VERSION!r}. "
            f"Refusing to guess how to read it."
        )
    try:
        questions = [
            question_from_dict(json.loads(line)) for line in lines[1:] if line.strip()
        ]
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise ValueError(
            f"{path}: could not parse a question line — the file is truncated or not a manifest."
        ) from exc
    actual = manifest_digest(questions, corpus_hashes=header.get("corpus_hashes", {}))
    if actual != header.get("digest"):
        raise ValueError(
            f"{path}: body digest {actual} does not match header digest {header.get('digest')}. "
            f"The manifest has been edited since it was frozen; re-freeze it deliberately rather "
            f"than trusting it."
        )
    return questions, header


def file_digest(path: Path) -> str:
    """SHA-256 of a source file, for the `corpus_hashes` provenance block."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
