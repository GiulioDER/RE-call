"""The released artifact: instances, frozen excision doc-id lists, and a digest over both.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

The manifest is the benchmark. Everything else in this package is replaceable plumbing, and a
reader who distrusts the builder must be able to verify this file and read it without running our
code. That imposes two properties the tests pin:

- The digest is over a **canonical** rendering (sorted keys, sorted instances), so it does not
  depend on the order the builder happened to emit.
- `read_manifest` **recomputes** the digest and refuses a body that does not match its header. A
  manifest that silently tolerated an edited excision list would be the same failure shape as an
  artifact with no provenance: a plausible answer with no signal that it is wrong.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

MANIFEST_VERSION = "1.0"

LABEL_ANSWERABLE = "answerable"
LABEL_UNANSWERABLE = "unanswerable"
_LABELS = frozenset({LABEL_ANSWERABLE, LABEL_UNANSWERABLE})

#: Ring sentinel for "excise the whole cluster". Not a width, so it cannot be confused with one.
RING_MAX = -1

#: Ring sentinel for the answerable original, which excises nothing. Distinct from RING_MAX so the
#: scorer can never group an original into a rung.
RING_ORIGINAL = -2

_FIELDS = (
    "instance_id",
    "corpus",
    "source_question_id",
    "question",
    "label",
    "ring",
    "excised_doc_ids",
    "gold_doc_ids",
    "pair_id",
)


@dataclass(frozen=True)
class Instance:
    """One question at one excision distance, paired to its own answerable original."""

    instance_id: str
    corpus: str
    source_question_id: str
    question: str
    label: str
    ring: int
    excised_doc_ids: tuple[str, ...]
    gold_doc_ids: tuple[str, ...]
    pair_id: str

    def __post_init__(self) -> None:
        if self.label not in _LABELS:
            raise ValueError(f"label must be one of {sorted(_LABELS)}, got {self.label!r}")
        if not isinstance(self.excised_doc_ids, tuple) or not isinstance(self.gold_doc_ids, tuple):
            raise TypeError("doc-id collections must be tuples — they are hashed and frozen")


def instance_to_dict(inst: Instance) -> dict:
    return {
        "instance_id": inst.instance_id,
        "corpus": inst.corpus,
        "source_question_id": inst.source_question_id,
        "question": inst.question,
        "label": inst.label,
        "ring": inst.ring,
        "excised_doc_ids": list(inst.excised_doc_ids),
        "gold_doc_ids": list(inst.gold_doc_ids),
        "pair_id": inst.pair_id,
    }


def instance_from_dict(d: Mapping) -> Instance:
    missing = [f for f in _FIELDS if f not in d]
    if missing:
        raise ValueError(f"instance is missing fields: {missing}")
    return Instance(
        instance_id=d["instance_id"],
        corpus=d["corpus"],
        source_question_id=d["source_question_id"],
        question=d["question"],
        label=d["label"],
        ring=int(d["ring"]),
        excised_doc_ids=tuple(d["excised_doc_ids"]),
        gold_doc_ids=tuple(d["gold_doc_ids"]),
        pair_id=d["pair_id"],
    )


def _canonical(inst: Instance) -> str:
    return json.dumps(instance_to_dict(inst), sort_keys=True, ensure_ascii=False)


def manifest_digest(instances: Sequence[Instance]) -> str:
    """SHA-256 over the canonical rendering of every instance, order-independent."""
    h = hashlib.sha256()
    for line in sorted(_canonical(i) for i in instances):
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def write_manifest(
    path: Path,
    instances: Sequence[Instance],
    *,
    ring_widths: Sequence[int],
    corpus_hashes: Mapping[str, str],
) -> str:
    """Write header line + one JSON object per instance. Returns the digest."""
    digest = manifest_digest(instances)
    header = {
        "manifest_version": MANIFEST_VERSION,
        "digest": digest,
        "n_instances": len(instances),
        "ring_widths": list(ring_widths),
        "corpus_hashes": dict(corpus_hashes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, sort_keys=True, ensure_ascii=False) + "\n")
        for inst in instances:
            fh.write(_canonical(inst) + "\n")
    return digest


def read_manifest(path: Path) -> tuple[list[Instance], dict]:
    """Read and VERIFY. A body that does not match its header digest is refused, not repaired."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path} is empty")
    header = json.loads(lines[0])
    instances = [instance_from_dict(json.loads(line)) for line in lines[1:] if line.strip()]
    actual = manifest_digest(instances)
    if actual != header.get("digest"):
        raise ValueError(
            f"{path}: body digest {actual} does not match header digest {header.get('digest')}. "
            f"The manifest has been edited since it was written; rebuild it rather than trusting it."
        )
    return instances, header
