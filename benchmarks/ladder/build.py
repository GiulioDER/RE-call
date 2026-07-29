"""Build the frozen manifest: source corpus + ring spec -> paired instances.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

Every question becomes a FAMILY: one answerable original with nothing excised, one instance per
ring width, and one at RING_MAX. They share a `pair_id`, which is what lets the scorer pair each
unanswerable instance against its own original — the design that defended the Mem0 head-to-head
and that differences out LOCOMO's shared annotation error.

Determinism is not a nicety here. This manifest is released and cited; two builds that disagree
mean there is no benchmark, only a run. Everything is sorted, and the only randomness is the
explicitly seeded P4 arm.

Usage::

    python -m benchmarks.ladder.build --locomo locomo10.json --out results/ladder/manifest.jsonl
"""
from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from pathlib import Path

from benchmarks.ladder.manifest import (
    LABEL_ANSWERABLE,
    LABEL_UNANSWERABLE,
    RING_MAX,
    RING_ORIGINAL,
    Instance,
    write_manifest,
)
from benchmarks.ladder.rings import RingSpec, build_rings, random_rings
from benchmarks.ladder.sources.locomo import SourceCorpus, SourceQuestion, load_locomo
from recall.eval.bm25 import BM25Index


def sample_questions(
    questions: Sequence[SourceQuestion], sample: int | None, seed: int
) -> list[SourceQuestion]:
    """A deterministic subset, or all of them when `sample` is None.

    Sorted by `question_id` BEFORE sampling, so the subset depends on the seed and not on the
    order `load_locomo` happened to walk the file. Sorted again afterwards so two builds emit
    identical manifests, not merely equal ones.
    """
    ordered = sorted(questions, key=lambda q: q.question_id)
    if sample is None or sample >= len(ordered):
        return ordered
    chosen = random.Random(seed).sample(ordered, sample)
    return sorted(chosen, key=lambda q: q.question_id)


def build_instances(
    corpus: SourceCorpus,
    spec: RingSpec,
    *,
    corpus_name: str,
    random_seed: int | None = None,
    sample: int | None = None,
    sample_seed: int = 0,
) -> list[Instance]:
    """One family per question: the answerable original, each rung, and RING_MAX.

    `sample` is the pre-registered question subset (300, seed 0). Its reason is a cost the
    ADOPTER pays, not only us: the full set is ~7 680 corpus states, each of which must be indexed
    before it can be queried.

    The BM25 index is built over the WHOLE corpus even when sampling, because ring order must not
    depend on which questions were drawn — otherwise the sample silently reshapes the x-axis.
    """
    index = BM25Index(corpus.documents)
    instances: list[Instance] = []

    for question in sample_questions(corpus.questions, sample, sample_seed):
        pair_id = f"{corpus_name}/{question.question_id}"
        cluster = corpus.cluster_members.get(question.cluster_id, ())

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
            )
        )

        if random_seed is None:
            rings = build_rings(index, question.question, question.gold_doc_ids, cluster, spec)
        else:
            rings = random_rings(
                question.question, question.gold_doc_ids, cluster, spec, seed=random_seed
            )

        for level in sorted(rings, key=lambda r: (r == RING_MAX, r)):
            instances.append(
                Instance(
                    instance_id=f"{pair_id}#d{level}",
                    corpus=corpus_name,
                    source_question_id=question.question_id,
                    question=question.question,
                    label=LABEL_UNANSWERABLE,
                    ring=level,
                    excised_doc_ids=rings[level],
                    gold_doc_ids=question.gold_doc_ids,
                    pair_id=pair_id,
                )
            )

    return instances


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Answerability Ladder manifest.")
    parser.add_argument("--locomo", type=Path, required=True, help="path to locomo10.json")
    parser.add_argument("--out", type=Path, required=True, help="manifest output path (.jsonl)")
    parser.add_argument(
        "--widths",
        default="0,4,16,64",
        help="comma-separated ring widths, fixed in PREREGISTRATION-ladder.md",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="build the P4 robustness arm with random-within-cluster neighbours instead of BM25",
    )
    parser.add_argument(
        "--sample-questions",
        type=int,
        default=300,
        help="pre-registered question sample size; 0 means all",
    )
    parser.add_argument("--sample-seed", type=int, default=0)
    args = parser.parse_args(argv)

    spec = RingSpec(widths=tuple(int(w) for w in args.widths.split(",")))
    corpus = load_locomo(args.locomo)
    instances = build_instances(
        corpus,
        spec,
        corpus_name="locomo",
        random_seed=args.random_seed,
        sample=args.sample_questions or None,
        sample_seed=args.sample_seed,
    )
    digest = write_manifest(
        args.out,
        instances,
        ring_widths=list(spec.widths),
        corpus_hashes={"locomo": corpus.content_hash},
    )
    print(f"wrote {len(instances)} instances to {args.out}")
    print(f"digest {digest}")
    print(f"corpus locomo {corpus.content_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
