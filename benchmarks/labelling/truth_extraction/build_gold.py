"""Freeze the gold positives and the transplanted negatives into one manifest.

The positives cost no labelling: every one is an edge a PEP author declared in a machine-readable
header, years ago, with no knowledge of this project. That is what removes labelling bias from
the recall denominator — the denominator was not chosen by anyone measuring against it.

Freezing reuses `recall/eval/promotion/manifest.py` unchanged. Its guarantee is the one this set
needs: a label edited after seeing an extractor's results changes `input_hash`, which changes the
body, which changes the digest, so `read_manifest` refuses the file rather than repairing it.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from benchmarks.labelling.truth_extraction.census import compute_census
from benchmarks.labelling.truth_extraction.peps_header import split_header
from recall.eval.promotion.manifest import (
    FrozenQuestion,
    file_digest,
    question_input_hash,
    write_manifest,
)

CORPUS = "peps"
#: The four transplanted private failures live in their own corpus namespace: they are not PEPs,
#: and an aggregate that mixed them into the PEPs arm would report a precision over two corpora.
FIXTURE_CORPUS = "fix-transplant"
#: Anchored to this module, not the cwd — the freeze must produce the same manifest wherever
#: it is invoked from.
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def build_gold_questions(peps_dir: Path) -> list[FrozenQuestion]:
    """One FrozenQuestion per header edge, plus one per transplanted negative."""
    census = compute_census(peps_dir)
    questions: list[FrozenQuestion] = []

    for edge in census.edges:
        source = peps_dir / f"{edge.superseded}.rst"
        # The extractor's INPUT is the superseded PEP's prose, so that is what the hash covers.
        # The body is hashed, never stored: question_to_dict writes only id, corpus, hash, labels.
        _, body = _split(source)
        labels = (f"{edge.successor}.rst",)
        questions.append(FrozenQuestion(
            question_id=f"{edge.superseded}->{edge.successor}",
            corpus=CORPUS,
            input_hash=question_input_hash(
                question_id=f"{edge.superseded}->{edge.successor}",
                corpus=CORPUS,
                query=body,
                expected_relevance_labels=labels,
            ),
            expected_relevance_labels=labels,
        ))

    for path in sorted(FIXTURES.glob("*.md")):
        body = path.read_text(encoding="utf-8")
        # Empty labels: manifest.py:71-76 reads () as UNANSWERABLE and scores it on the safety
        # axis. For a labelled negative that is correct — there is no edge to hit, and scoring it
        # as a retrieval miss would charge a true refusal as a failure.
        questions.append(FrozenQuestion(
            question_id=path.stem,
            corpus=FIXTURE_CORPUS,
            input_hash=question_input_hash(
                question_id=path.stem,
                corpus=FIXTURE_CORPUS,
                query=body,
                expected_relevance_labels=(),
            ),
            expected_relevance_labels=(),
        ))
    return questions


def _split(path: Path) -> tuple[str, str]:
    return split_header(path.read_text(encoding="utf-8", errors="replace"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peps-dir", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path,
        default=Path("benchmarks/labelling/truth_extraction/gold.manifest.jsonl"),
    )
    args = parser.parse_args()

    questions = build_gold_questions(args.peps_dir)
    corpus_hashes = {
        "peps_sha": subprocess.run(
            ["git", "-C", str(args.peps_dir.parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip(),
        "census": file_digest(Path("results/truth_extraction/census.json")),
    }
    digest = write_manifest(args.out, questions, corpus_hashes=corpus_hashes)
    positives = sum(1 for q in questions if q.expected_relevance_labels)
    print(f"{args.out}\n  {positives} positives + {len(questions) - positives} negatives")
    print(f"  digest {digest}")


if __name__ == "__main__":
    main()
