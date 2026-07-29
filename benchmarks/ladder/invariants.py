"""Assertions on the ARTEFACT, not on the process. Exit code 0 is not a measurement.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

Every check here exists because its failure mode produces a plausible number rather than an error,
and a plausible number with no signal that it is wrong is the failure this repo keeps paying for.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from benchmarks.ladder.manifest import LABEL_ANSWERABLE, Instance, manifest_digest


class InvariantViolation(RuntimeError):
    """A measured artefact contradicts something the design guarantees."""


def assert_excised_absent(instance: Instance, indexed: frozenset[str]) -> None:
    """Invariant 1: the system really dropped what this rung excises.

    A system that cached across rings would pass every rung, and the curve would look like a
    strong result instead of a broken harness.
    """
    leaked = sorted(set(instance.excised_doc_ids) & indexed)
    if leaked:
        raise InvariantViolation(
            f"{instance.instance_id}: {len(leaked)} excised documents are still indexed "
            f"({leaked[:5]}). The system retained state across rings; its ingest must replace, "
            f"not merge."
        )


def assert_ring_zero_has_survivors(
    instance: Instance, indexed: frozenset[str], cluster: Sequence[str]
) -> None:
    """Invariant 2: at d=0 the topic must survive, or d=0 is d=max wearing a different number."""
    if instance.ring != 0:
        return
    survivors = (set(cluster) - set(instance.excised_doc_ids)) & indexed
    if not survivors:
        raise InvariantViolation(
            f"{instance.instance_id}: nothing from its cluster survived d=0, so this rung is "
            f"d=max under another name. Drop the instance rather than scoring it."
        )


def assert_originals_were_answered(
    answered: Mapping[str, bool], instances: Sequence[Instance]
) -> None:
    """Invariant 3: a question no system answers with its gold present is broken, not hard."""
    originals = [i for i in instances if i.label == LABEL_ANSWERABLE]
    if not originals:
        return
    unanswered = [i.instance_id for i in originals if not answered.get(i.instance_id, False)]
    if len(unanswered) == len(originals):
        raise InvariantViolation(
            f"all {len(originals)} answerable originals were abstained on. Those questions are "
            f"broken, not hard, and cannot anchor a pair. Check ingest before reading any curve."
        )


def assert_manifest_digest(instances: Sequence[Instance], header: Mapping) -> None:
    """Invariant 4: the instances being scored are the instances that were published.

    Takes the whole header, not just its digest string, because the digest covers PROVENANCE as
    well as bodies — `corpus_hashes` says which corpus these instances came from, and a review
    demonstrated that a digest over bodies alone accepts a forged one silently.
    """
    expected = header.get("digest")
    actual = manifest_digest(
        instances,
        ring_widths=header.get("ring_widths", []),
        corpus_hashes=header.get("corpus_hashes", {}),
    )
    if actual != expected:
        raise InvariantViolation(
            f"manifest digest {actual} != expected {expected}. The manifest changed between build "
            f"and scoring; results computed against it are not results for the published benchmark."
        )
