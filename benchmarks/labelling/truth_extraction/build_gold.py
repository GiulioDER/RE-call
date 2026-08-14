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
import json
import re
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


#: Repo root, from this module's location. Every path below is anchored to it, so the freeze
#: produces the same manifest regardless of the directory it is invoked from.
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: A `git rev-parse HEAD` output: 40 lowercase hex characters. Same format `census.py` enforces
#: on its own `--peps-sha`.
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _check_peps_sha(peps_sha: str, census_path: Path) -> None:
    """Refuse a `--peps-sha` that is malformed, or that disagrees with `census.json`'s recorded one.

    `--peps-sha` is taken unvalidated and is the SOLE corpus provenance in the frozen gold
    manifest, so a typo here becomes unfalsifiable truth: nothing downstream can tell the
    manifest names the wrong corpus. `census.json`'s own `_provenance.peps_sha` is an independent
    record of the same corpus snapshot, already on disk by the time the gold manifest is frozen,
    so comparing against it catches the typo instead of freezing it.
    """
    if not _SHA_RE.fullmatch(peps_sha):
        raise ValueError(f"--peps-sha {peps_sha!r} is not a 40-character lowercase hex git SHA")
    # Every failure reading the cross-check's OTHER side is folded into the same ValueError the
    # caller already handles, with the path named, rather than left to surface as a bare
    # FileNotFoundError/JSONDecodeError/KeyError traceback that names no file.
    try:
        census_sha = json.loads(census_path.read_text(encoding="utf-8"))["_provenance"]["peps_sha"]
    except FileNotFoundError as exc:
        raise ValueError(f"{census_path} does not exist: run census.py before build_gold.py") from exc
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(
            f"{census_path} could not be read as a census artifact with _provenance.peps_sha: {exc}"
        ) from exc
    if peps_sha != census_sha:
        raise ValueError(
            f"--peps-sha {peps_sha} does not match {census_path}'s recorded peps_sha "
            f"{census_sha}: the gold manifest and the census must describe the same corpus "
            f"snapshot. Regenerate one to match the other rather than freezing a mismatch."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peps-dir", type=Path, required=True)
    # Taken explicitly, exactly as `census.py` takes it, and NOT derived by running
    # `git -C <peps_dir>/.. rev-parse HEAD`. Git discovers repositories upwards: if the clone is
    # ever laid out differently from the assumed nested `peps/peps`, that command succeeds
    # against some enclosing repository and records an unrelated commit as the corpus
    # provenance. A frozen artifact that silently names the wrong corpus is the precise failure
    # the freeze exists to prevent, and it would be undetectable after the fact.
    parser.add_argument("--peps-sha", required=True, help="git rev-parse HEAD of that clone")
    parser.add_argument(
        "--out", type=Path,
        default=_REPO_ROOT / "benchmarks" / "labelling" / "truth_extraction" / "gold.manifest.jsonl",
    )
    args = parser.parse_args()

    census_path = _REPO_ROOT / "results" / "truth_extraction" / "census.json"
    try:
        _check_peps_sha(args.peps_sha, census_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    questions = build_gold_questions(args.peps_dir)
    corpus_hashes = {
        "peps_sha": args.peps_sha,
        "census": file_digest(census_path),
    }
    digest = write_manifest(args.out, questions, corpus_hashes=corpus_hashes)
    positives = sum(1 for q in questions if q.expected_relevance_labels)
    print(f"{args.out}\n  {positives} positives + {len(questions) - positives} negatives")
    print(f"  digest {digest}")


if __name__ == "__main__":
    main()
