"""Print the ladder: 2x2 per rung, lambda costs, and the H1 verdict — PASS or FAIL.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

Exits 1 on FAIL, so a scheduled run cannot report success merely by finishing. A FAIL here is not
a bug: it is the pre-registered kill condition, and it means no paid comparative arm runs.

AMENDMENT (2026-07-29, written before the verdict was computed): ingest scope is one conversation,
and RING_MAX excises the whole cluster — so at `d=max` the ingested corpus is EMPTY. A system that
abstains there has not recognised an unanswerable question; it has nothing to retrieve. That makes
the pre-registered `d=0` vs `d=max` contrast CONFOUNDED: it can report PASS for a system whose only
abstention trigger is an empty index.

The pre-registered verdict below is still computed and printed exactly as pre-registered — not
edited, softened, or replaced, and the 0.15 threshold does not move. What this module adds is the
evidence that qualifies it: `n` on every comparison, surviving-document counts per rung (so the
`d=max` confound is visible in the output itself, not only in prose), every adjacent pairwise
contrast plus explicitly the widest contrast whose corpus is not empty, a qualification line
derived from those numbers rather than hardcoded, and a disclosure of the uncalibrated bge-small
abstention floor this arm actually ran with.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from benchmarks.ladder.manifest import RING_MAX, RING_ORIGINAL, Instance, read_manifest
from benchmarks.ladder.score import (
    confusion_by_ring,
    correct_abstain_rate,
    h1_verdict,
    lambda_cost,
    paired_difference_ci,
)

LAMBDAS = (1.0, 3.0, 10.0)

#: The abstention threshold this arm actually ran with: no calibration exists for `bge-small`, so
#: abstention used this untuned cosine floor. That is the CORRECT configuration under the suite's
#: shipped-defaults rule — and it is also the exact constant already measured as not comparable
#: across embedders ([[project-recall-threshold-embedder-fragile-2026-07-28]]).
UNCALIBRATED_BGE_SMALL_FLOOR = 0.50

_QUALIFICATION_LINE = 'the axis as built prices "is anything indexed at all", not answerability'


def _ring_label(ring: int) -> str:
    if ring == RING_MAX:
        return "d=max"
    if ring == RING_ORIGINAL:
        return "original"
    return f"d={ring}"


def _ring_sort_key(ring: int) -> tuple[int, int]:
    """original first, then ascending widths, then d=max last."""
    if ring == RING_ORIGINAL:
        return (0, 0)
    if ring == RING_MAX:
        return (2, 0)
    return (1, ring)


def _cluster_id(instance: Instance) -> str:
    """Recover the LOCOMO `sample_id` a question came from.

    `sources/locomo.py` builds `source_question_id` as `f"{sample_id}/qa{i}"`; splitting on the
    LAST "/" recovers `sample_id` without re-parsing the corpus question by question. `sample_id`
    values seen in practice (e.g. `conv-26`) never contain a "/" themselves.
    """
    return instance.source_question_id.rsplit("/", 1)[0]


def _surviving_doc_counts(instances: list[Instance], corpus_path: Path) -> dict[int, float]:
    """Median surviving documents per rung, from the manifest's excised ids and the source corpus.

    Surviving = the size of the question's own conversation cluster minus the docs THIS instance
    excised. The answerable original excises nothing, so its count is the whole cluster; RING_MAX
    excises the whole cluster, so its count is always 0 — the confound this report exists to
    surface, made visible in the table itself rather than only in prose.
    """
    from benchmarks.ladder.sources.locomo import load_locomo

    corpus = load_locomo(corpus_path)
    per_ring: dict[int, list[int]] = {}
    for inst in instances:
        cluster = corpus.cluster_members.get(_cluster_id(inst), ())
        surviving = len(cluster) - len(inst.excised_doc_ids)
        per_ring.setdefault(inst.ring, []).append(surviving)
    return {ring: statistics.median(vals) for ring, vals in per_ring.items()}


def _paired_n(instances: list[Instance], abstained: dict[str, bool], low: int, high: int) -> int:
    """How many questions appear at BOTH rungs and have a recorded response at both.

    Mirrors `score._paired_flags`'s own membership test without reaching into its private name —
    this is printed even for a pair whose difference-CI raised, so it cannot depend on that call
    having succeeded.
    """
    lo_pairs = {i.pair_id for i in instances if i.ring == low and i.instance_id in abstained}
    hi_pairs = {i.pair_id for i in instances if i.ring == high and i.instance_id in abstained}
    return len(lo_pairs & hi_pairs)


def _widest_finite_ring(rings: set[int]) -> int | None:
    """The widest excision width present that is neither the original nor RING_MAX — i.e. the
    widest rung whose corpus is not (by construction) empty. `None` when no such rung exists."""
    finite = [r for r in rings if r not in (RING_MAX, RING_ORIGINAL) and r > 0]
    return max(finite) if finite else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report the Answerability Ladder.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help=(
            "path to the source corpus (e.g. locomo10.json). When given, the report prints "
            "median surviving documents per rung, computed from the manifest's excised ids "
            "against this corpus's cluster sizes."
        ),
    )
    args = parser.parse_args(argv)

    instances, header = read_manifest(args.manifest)
    abstained: dict[str, bool] = {}
    for line in args.responses.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            abstained[row["instance_id"]] = bool(row["abstained"])

    cells = confusion_by_ring(instances, abstained)
    ring_order = sorted(cells, key=_ring_sort_key)
    surviving = _surviving_doc_counts(instances, args.corpus) if args.corpus is not None else {}

    print(f"manifest digest {header['digest']}  n_instances={header['n_instances']}")
    print(f"scored responses: {len(abstained)}")
    print()

    header_row = (
        f"{'rung':<10}{'n':>6}{'corr-abst':>11}{'false-ans':>11}{'false-abst':>12}"
        + "".join(f"{'L=' + str(int(lam)):>8}" for lam in LAMBDAS)
    )
    if surviving:
        header_row += f"{'surv-docs':>11}"
    print(header_row)

    for ring in ring_order:
        cell = cells[ring]
        n = (
            cell.correct_abstain
            + cell.false_answer
            + cell.false_abstain
            + cell.answered_answerable
        )
        row = (
            f"{_ring_label(ring):<10}{n:>6}{correct_abstain_rate(cell):>11.3f}"
            f"{cell.false_answer:>11}{cell.false_abstain:>12}"
            + "".join(f"{lambda_cost(cell, lam):>8.1f}" for lam in LAMBDAS)
        )
        if surviving:
            row += f"{surviving.get(ring, float('nan')):>11.0f}"
        print(row)

    if args.corpus is None:
        print()
        print(
            "surviving-document counts per rung: SKIPPED (pass --corpus to compute them from "
            "the source corpus)"
        )

    # ---- The pre-registered H1 headline: d=0 vs d=max, computed and printed unedited. ----
    # No try/except here: a verdict with no paired data must not be printable, so a ValueError
    # from an empty overlap propagates out of main() rather than being swallowed into a fake PASS
    # or FAIL.
    print()
    diff, low, high = paired_difference_ci(instances, abstained, 0, RING_MAX)
    n_headline = _paired_n(instances, abstained, 0, RING_MAX)
    verdict = h1_verdict(diff, low, high)
    print(
        f"H1 (pre-registered) paired delta(correct-abstain), d=max - d=0: {diff:+.3f} "
        f"[{low:+.3f}, {high:+.3f}]  n={n_headline}"
    )
    print(f"H1: {verdict}")

    # ---- Supplementary, diagnostic-only contrasts: every adjacent pair, plus the widest ----
    # ---- contrast whose corpus is not empty. Never the headline verdict.                 ----
    print()
    print("Supplementary pairwise contrasts (diagnostic - not the pre-registered verdict):")
    widest = _widest_finite_ring(set(ring_order))
    pairs = list(zip(ring_order, ring_order[1:]))
    if widest is not None and (0, widest) not in pairs and (widest, 0) not in pairs:
        pairs.append((0, widest))

    widest_verdict: str | None = None
    for lo, hi in pairs:
        label = f"{_ring_label(lo)} vs {_ring_label(hi)}"
        try:
            d, l, h = paired_difference_ci(instances, abstained, lo, hi)
        except ValueError:
            print(f"  {label:<24} n=0 (no paired data)")
            continue
        n = _paired_n(instances, abstained, lo, hi)
        v = h1_verdict(d, l, h)
        print(f"  {label:<24} delta={d:+.3f} [{l:+.3f}, {h:+.3f}]  n={n}  {v}")
        if widest is not None and {lo, hi} == {0, widest}:
            widest_verdict = v

    # ---- The machine-checked qualification: derived from the two verdicts above, never    ----
    # ---- hardcoded. Fires only when the pre-registered contrast PASSES while the widest    ----
    # ---- non-empty contrast does not — the mirror-image of the failure this axis was built ----
    # ---- to catch.                                                                         ----
    if widest is not None and verdict == "PASS" and widest_verdict == "FAIL":
        print()
        print(_QUALIFICATION_LINE)

    # ---- Disclosure: the abstention threshold this arm actually ran with. ----
    print()
    print(
        "Disclosure: this arm ran with NO calibration for bge-small, so abstention used the "
        f"untuned {UNCALIBRATED_BGE_SMALL_FLOOR:.2f} cosine floor. That is the correct "
        "configuration under the suite's shipped-defaults rule, and it is also the exact "
        "constant already measured as embedder-fragile "
        "([[project-recall-threshold-embedder-fragile-2026-07-28]])."
    )

    if verdict == "FAIL":
        print(
            "\nThis is the pre-registered kill condition, not a disappointing result. A flat curve "
            "means excision distance is not the axis this benchmark claimed it was. Do NOT run the "
            "Mem0 arm, and publish this."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
