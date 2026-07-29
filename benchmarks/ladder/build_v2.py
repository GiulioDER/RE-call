"""Build the v2 manifest: fractional rungs + distractor conversations in the ingested slice.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

A NEW module, not an edit to `build.py`. v1's builder is frozen alongside a published manifest
(`results/ladder/manifest.jsonl`, digest `6bfe2d2b…`) and a published verdict
(`results/ladder/H1_VERDICT.md`) — see `PREREGISTRATION-ladder-v2.md` §0 for why editing it in
place would be indistinguishable from rewriting history to make v1 look like it asked the right
question.

Every question still becomes a FAMILY sharing one `pair_id`, as in v1: one answerable original
with nothing excised, and one instance per fraction (`FractionSpec`, `rings.build_fractional_rings`
— see that module for why fractions are basis points and not floats). What is new is
`scope_cluster_ids`: every instance in the family — the original included — carries the sorted
union of the question's own conversation plus its 2 distractor conversations
(`select_distractors`). `run.py` reads this field to decide what the INGESTED slice is; this
module never ingests anything itself, it only decides and records the scope.

The BM25 index that orders excision is still built over the WHOLE corpus, exactly as in v1's
`build_instances` — ring order must not depend on which questions were sampled or which
distractors a question drew, or the sample would silently reshape the x-axis.

Usage::

    python -m benchmarks.ladder.build_v2 --locomo locomo10.json --out results/ladder/manifest_v2.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import random
from collections.abc import Sequence
from pathlib import Path

from benchmarks.ladder.build import sample_questions
from benchmarks.ladder.manifest import (
    LABEL_ANSWERABLE,
    LABEL_UNANSWERABLE,
    RING_ORIGINAL,
    Instance,
    write_manifest,
)
from benchmarks.ladder.rings import FractionSpec, build_fractional_rings, fraction_to_ring
from benchmarks.ladder.sources.locomo import SourceCorpus, load_locomo
from recall.eval.bm25 import BM25Index


def _distractor_seed_key(cluster_id: str, seed: int) -> int:
    """A `random.Random` seed derived deterministically from `cluster_id` and `seed`.

    `hash()` is salted per process (`PYTHONHASHSEED`), which would make `select_distractors`
    disagree between two runs of the SAME builder on the SAME corpus — a manifest that is not
    reproducible from itself. This is exactly the trap the v1 builder was tested against; sha256
    has no such salt, so the key below is identical in every process, every time.
    """
    digest = hashlib.sha256(f"{seed}:{cluster_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def select_distractors(
    cluster_id: str,
    all_cluster_ids: Sequence[str],
    *,
    n: int = 2,
    seed: int = 0,
) -> tuple[str, ...]:
    """Pick `n` conversations that are NOT `cluster_id`, reproducibly from the manifest alone.

    Keyed on `cluster_id` (not on any individual question), so every question in a conversation
    draws the SAME pair — the pre-registered rule
    (`PREREGISTRATION-ladder-v2.md` §2: "selected by `random.Random(0)` keyed on the question's
    `cluster_id` so every question in a conversation gets the same pair"). The candidate list is
    sorted before sampling, and the result is sorted again, so the choice depends only on
    `cluster_id`, `all_cluster_ids`'s SET (not its order) and `seed` — never on iteration order or
    on `PYTHONHASHSEED` (see `_distractor_seed_key`).
    """
    candidates = sorted(c for c in all_cluster_ids if c != cluster_id)
    if len(candidates) < n:
        raise ValueError(
            f"only {len(candidates)} candidate conversation(s) available to draw {n} "
            f"distractor(s) for {cluster_id!r} — the corpus is too small for this many distractors."
        )
    rng = random.Random(_distractor_seed_key(cluster_id, seed))
    chosen = rng.sample(candidates, n)
    return tuple(sorted(chosen))


def build_v2_instances(
    corpus: SourceCorpus,
    spec: FractionSpec,
    *,
    corpus_name: str,
    sample: int | None,
    sample_seed: int = 0,
    distractors: int = 2,
    distractor_seed: int = 0,
) -> list[Instance]:
    """One family per question: the answerable original, plus one instance per fraction.

    Every instance in a family carries the same `scope_cluster_ids` — sorted(own cluster ∪
    distractor clusters) — including the original, so a caller never has to special-case which
    ring the field applies to (Change A, `PREREGISTRATION-ladder-v2.md` §1). Only the question's
    OWN cluster is ever excised: `build_fractional_rings` is called with `cluster =
    corpus.cluster_members[question.cluster_id]`, exactly as v1's `build_instances` does, so
    distractor conversations are never touched by excision at any fraction — `r=1.00` removes the
    whole OWN cluster and nothing else, which is what makes it a real far gap rather than an empty
    index (v1's defect, `PREREGISTRATION-ladder-v2.md` §0).

    The BM25 index spans the WHOLE corpus, as in v1, so ring order does not depend on the sample
    or on which distractors a question happened to draw.
    """
    index = BM25Index(corpus.documents)
    all_cluster_ids = sorted(corpus.cluster_members)
    instances: list[Instance] = []

    for question in sample_questions(corpus.questions, sample, sample_seed):
        pair_id = f"{corpus_name}/{question.question_id}"
        cluster = corpus.cluster_members.get(question.cluster_id, ())
        distractor_ids = select_distractors(
            question.cluster_id, all_cluster_ids, n=distractors, seed=distractor_seed
        )
        scope_cluster_ids = tuple(sorted({question.cluster_id, *distractor_ids}))

        instances.append(
            Instance(
                instance_id=f"{pair_id}#original",
                corpus=corpus_name,
                source_question_id=question.question_id,
                question=question.question,
                label=LABEL_ANSWERABLE,
                ring=RING_ORIGINAL,
                excised_doc_ids=(),
                gold_doc_ids=question.gold_doc_ids,
                pair_id=pair_id,
                scope_cluster_ids=scope_cluster_ids,
            )
        )

        rings = build_fractional_rings(
            index, question.question, question.gold_doc_ids, cluster, spec
        )

        for f in spec.fractions:
            rung_label = f"r{f:.2f}"  # matches rings._fraction_label's format, pinned by its tests
            instances.append(
                Instance(
                    instance_id=f"{pair_id}#{rung_label}",
                    corpus=corpus_name,
                    source_question_id=question.question_id,
                    question=question.question,
                    label=LABEL_UNANSWERABLE,
                    ring=fraction_to_ring(f),
                    excised_doc_ids=rings[rung_label],
                    gold_doc_ids=question.gold_doc_ids,
                    pair_id=pair_id,
                    scope_cluster_ids=scope_cluster_ids,
                )
            )

    return instances


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the v2 Answerability Ladder manifest (fractional rungs + distractors)."
    )
    parser.add_argument("--locomo", type=Path, required=True, help="path to locomo10.json")
    parser.add_argument("--out", type=Path, required=True, help="manifest output path (.jsonl)")
    parser.add_argument(
        "--fractions",
        default="0.00,0.25,0.50,0.75,1.00",
        help="comma-separated rung fractions, fixed in PREREGISTRATION-ladder-v2.md",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=200,
        help="pre-registered question sample size; 0 means all",
    )
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--distractors",
        type=int,
        default=2,
        help="number of distractor conversations per question, fixed in the pre-registration",
    )
    parser.add_argument("--distractor-seed", type=int, default=0)
    args = parser.parse_args(argv)

    spec = FractionSpec(fractions=tuple(float(f) for f in args.fractions.split(",")))
    corpus = load_locomo(args.locomo)
    instances = build_v2_instances(
        corpus,
        spec,
        corpus_name="locomo",
        sample=args.sample or None,
        sample_seed=args.sample_seed,
        distractors=args.distractors,
        distractor_seed=args.distractor_seed,
    )
    digest = write_manifest(
        args.out,
        instances,
        ring_widths=[fraction_to_ring(f) for f in spec.fractions],
        corpus_hashes={"locomo": corpus.content_hash},
    )
    print(f"wrote {len(instances)} instances to {args.out}")
    print(f"digest {digest}")
    print(f"corpus locomo {corpus.content_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
