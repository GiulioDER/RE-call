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

MANIFEST_VERSION = "2.0"

#: Versions `read_manifest` accepts. v1's frozen manifest (`results/ladder/manifest.jsonl`) has
#: no `scope_cluster_ids` on any instance and must keep reading forever — that's the point of
#: `instance_from_dict` tolerating the key's absence. v2 adds `scope_cluster_ids` but otherwise
#: reuses the same digest machinery, so both versions are accepted by the same reader.
_SUPPORTED_MANIFEST_VERSIONS = frozenset({"1.0", "2.0"})

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
    #: v2: the cluster ids the ingested slice was drawn from (the question's own conversation
    #: plus its distractors). Last field, defaulted, so every v1 call site keeps working
    #: unchanged. `()` keeps v1's meaning: "inferred from the gold id's own cluster" — v2 states
    #: it explicitly only when the ingest scope is wider than that.
    scope_cluster_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.label not in _LABELS:
            raise ValueError(f"label must be one of {sorted(_LABELS)}, got {self.label!r}")
        if (
            not isinstance(self.excised_doc_ids, tuple)
            or not isinstance(self.gold_doc_ids, tuple)
            or not isinstance(self.scope_cluster_ids, tuple)
        ):
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
        "scope_cluster_ids": list(inst.scope_cluster_ids),
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
        # v1 files predate this field entirely — tolerate its absence rather than requiring it,
        # so the frozen v1 manifest keeps reading under v2 code.
        scope_cluster_ids=tuple(d.get("scope_cluster_ids", ())),
    )


def _canonical(inst: Instance) -> str:
    """Canonical rendering used for both the digest and the persisted body line.

    `scope_cluster_ids` is omitted here when empty (v1's meaning: "inferred from the gold id's
    own cluster"), rather than always rendered as `[]`. That is not cosmetic: every v1 instance
    has an empty `scope_cluster_ids`, so if this dict always included the key, EVERY v1 canonical
    line would gain a `"scope_cluster_ids": []` it never had, and `manifest_digest` would compute
    a different digest for the already-published `results/ladder/manifest.jsonl` than the one in
    its own header — silently invalidating a frozen, released artifact. Omitting the key when
    empty makes a v2-code canonical rendering of a v1-shaped instance byte-identical to what v1
    itself produced, so the frozen manifest's digest is provably unchanged (see
    `tests/test_ladder_manifest.py::test_the_frozen_v1_manifest_still_reads_with_its_digest_intact`).
    """
    d = instance_to_dict(inst)
    if not d["scope_cluster_ids"]:
        del d["scope_cluster_ids"]
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


def manifest_digest(
    instances: Sequence[Instance],
    *,
    ring_widths: Sequence[int],
    corpus_hashes: Mapping[str, str],
) -> str:
    """SHA-256 over every instance AND the provenance saying what corpus they came from.

    The header is covered, not only the bodies. `corpus_hashes` identifies WHICH corpus this
    manifest was built from and `ring_widths` says how its x-axis was constructed — precisely the
    fields a tamperer edits to make a manifest claim a provenance it does not have. A digest over
    instance bodies alone accepts that edit silently; that was measured on this file, not assumed.

    A digest cannot cover itself, so the header's own `digest` field is excluded by construction.
    Instances are sorted, so the result does not depend on the order the builder emitted them.
    """
    h = hashlib.sha256()
    h.update(b"ladder-manifest-v1\n")
    for line in sorted(_canonical(i) for i in instances):
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    h.update(b"--provenance--\n")
    h.update(
        json.dumps(
            {"ring_widths": list(ring_widths), "corpus_hashes": dict(corpus_hashes)},
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    )
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
    digest = manifest_digest(instances, ring_widths=ring_widths, corpus_hashes=corpus_hashes)
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
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}: could not parse the header line — the file is truncated or not a manifest."
        ) from exc
    version = header.get("manifest_version")
    if version is not None and version not in _SUPPORTED_MANIFEST_VERSIONS:
        raise ValueError(
            f"{path}: manifest_version {version!r} is not one of "
            f"{sorted(_SUPPORTED_MANIFEST_VERSIONS)}. Refusing to guess how to read it."
        )
    try:
        instances = [instance_from_dict(json.loads(line)) for line in lines[1:] if line.strip()]
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise ValueError(
            f"{path}: could not parse an instance line — the file is truncated or not a manifest."
        ) from exc
    actual = manifest_digest(
        instances,
        ring_widths=header.get("ring_widths", []),
        corpus_hashes=header.get("corpus_hashes", {}),
    )
    if actual != header.get("digest"):
        raise ValueError(
            f"{path}: body digest {actual} does not match header digest {header.get('digest')}. "
            f"The manifest has been edited since it was written; rebuild it rather than trusting it."
        )
    return instances, header
