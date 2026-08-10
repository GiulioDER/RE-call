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

v1 vs v2 — the confound above is a v1-ONLY defect. v2 (`benchmarks/archive/preregistrations/PREREGISTRATION-ladder-v2.md`
§0) widens the ingested slice to the question's own conversation plus its distractor conversations
(`Instance.scope_cluster_ids`), so even v2's top rung (fraction 1.00, stored as basis points 10000)
still holds roughly 1,200 distractor documents — an abstention there is not explained by "the index
has nothing in it". This module scores both manifest versions through the same machinery and reads
which one it is from the manifest header (`manifest_version`), never from a module-level default,
so a v2 run is labelled with `r=<fraction>` rather than v1's `d=<count>` and the empty-corpus
warning above must not be assumed to apply to whichever manifest produced the run in front of you —
the surviving-document column (`--corpus`) is what settles that per rung, not the version alone.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from benchmarks.ladder.manifest import RING_MAX, RING_ORIGINAL, Instance, read_manifest
from benchmarks.ladder.rings import ring_to_fraction
from benchmarks.ladder.score import (
    confusion_by_ring,
    correct_abstain_rate,
    h1_verdict,
    lambda_cost,
    paired_difference_ci,
)

#: The version string a v2 manifest declares in its own header.
#:
#: Deliberately NOT imported from `manifest.py`. Labelling must follow what a *given manifest file*
#: declares, not whatever the writing module's constant happens to equal at import time — `main()`
#: always decides `is_v2` from the header it just read. Importing the writer's constant would
#: couple how a manifest is READ to how new ones are WRITTEN, so bumping the writer would silently
#: relabel already-frozen artifacts.
#:
#: The cost of that decoupling is a second place the literal "2.0" is spelled out, and a drift
#: between the two would relabel every v2 rung with v1's `d=` notation — exactly the defect the
#: derived-headline fix removed, reintroduced silently. So the two are pinned equal by
#: `test_ladder_report.py::test_the_v2_version_literal_matches_what_the_writer_stamps`, which
#: fails loudly if the writer's value ever moves without this one.
_V2_MANIFEST_VERSION = "2.0"

LAMBDAS = (1.0, 3.0, 10.0)

#: The abstention threshold this arm actually ran with: no calibration exists for `bge-small`, so
#: abstention used this untuned cosine floor. That is the CORRECT configuration under the suite's
#: shipped-defaults rule — and it is also the exact constant already measured as not comparable
#: across embedders ([[project-recall-threshold-embedder-fragile-2026-07-28]]).
UNCALIBRATED_BGE_SMALL_FLOOR = 0.50

_QUALIFICATION_LINE = 'the axis as built prices "is anything indexed at all", not answerability'


def _ring_label(ring: int, *, is_v2: bool = False) -> str:
    """Render a rung. v1's sentinels (`d=max`, `original`) are shared by both versions; everything
    else is version-specific — v1 rungs are counts (`d=<n>`), v2 rungs are basis points that MUST
    render as the fraction they encode (`r=<f>`, via `rings.ring_to_fraction`), never reused under
    v1's `d=` notation (`rings.py`'s own docstring: the two schemes must never be conflated in one
    file's output). `is_v2` is always threaded down from the manifest header `main()` just read —
    see `_V2_MANIFEST_VERSION` above for why it is never inferred from the module constant.
    """
    if ring == RING_MAX:
        return "d=max"
    if ring == RING_ORIGINAL:
        return "original"
    if is_v2:
        return f"r={ring_to_fraction(ring):.2f}"
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

    Surviving = the size of the **whole ingested slice** minus the docs THIS instance excised.

    The slice is `scope_cluster_ids` when the manifest states one (v2, where it is the question's
    own conversation plus distractors), and the question's own cluster when it does not (v1).
    Getting this wrong is not cosmetic: counting only the own cluster reports 0 survivors for a v2
    top rung whose index actually holds ~1 200 distractor documents, which is precisely the
    empty-corpus confound this column exists to expose — reproduced as a false positive. A column
    that cries confound where there is none is as useless as one that hides a real one.
    """
    from benchmarks.ladder.sources.locomo import load_locomo

    corpus = load_locomo(corpus_path)
    per_ring: dict[int, list[int]] = {}
    for inst in instances:
        if inst.scope_cluster_ids:
            slice_ids: set[str] = set()
            for cluster_id in inst.scope_cluster_ids:
                slice_ids |= set(corpus.cluster_members.get(cluster_id, ()))
        else:
            slice_ids = set(corpus.cluster_members.get(_cluster_id(inst), ()))
        surviving = len(slice_ids - set(inst.excised_doc_ids))
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
    parser.add_argument(
        "--low",
        type=int,
        default=0,
        help="near rung of the pre-registered contrast. Default 0 (v1 d=0 and v2 r=0.00 share it).",
    )
    parser.add_argument(
        "--high",
        type=int,
        default=RING_MAX,
        help=(
            "far rung of the pre-registered contrast. Default RING_MAX (-1), which is v1's. v2 "
            "uses basis points, so its far rung is 10000 (r=1.00). Stated explicitly rather than "
            "guessed from the manifest: which contrast is the headline is a pre-registration "
            "decision, and a report that infers it could silently change the verdict's meaning "
            "when a manifest gains a rung."
        ),
    )
    args = parser.parse_args(argv)

    instances, header = read_manifest(args.manifest)
    # is_v2 drives EVERY _ring_label call below (table, headline, supplementary contrasts) and the
    # FIX-ENV4 mismatch check just past the table. Read directly off THIS manifest's own header,
    # never off manifest.py's MANIFEST_VERSION constant (see _V2_MANIFEST_VERSION above for why).
    # A header with no manifest_version key at all predates the field entirely — fall back to v1
    # labelling for it rather than guessing.
    manifest_version = header.get("manifest_version")
    is_v2 = manifest_version == _V2_MANIFEST_VERSION
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
            f"{_ring_label(ring, is_v2=is_v2):<10}{n:>6}{correct_abstain_rate(cell):>11.3f}"
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

    # ---- FIX-ENV4: --high defaults to RING_MAX (v1's sentinel). A v2 manifest never has a
    # ---- RING_MAX rung, so running with the bare default produced a generic, uninformative
    # ---- ValueError ("no question appears at BOTH rung 0 and rung -1") that never named the
    # ---- v1/v2 mismatch. Detected and named explicitly, BEFORE the generic error has a chance to
    # ---- fire — and deliberately NOT auto-selected: which contrast is the headline is a
    # ---- pre-registration decision (see the --high help text above), so this raises rather than
    # ---- silently picking 10000 for the caller.
    if is_v2 and args.high == RING_MAX:
        raise ValueError(
            f"--high was left at its default, RING_MAX ({RING_MAX}) — v1's sentinel for 'excise "
            f"the whole cluster'. This manifest is v2-shaped (manifest_version={manifest_version!r})"
            f", whose rungs are basis points 0-10000, so RING_MAX never appears in it. Pass "
            f"--high 10000 explicitly for the v2 headline contrast (r=0.00 vs r=1.00) — inferring "
            f"it here would make the report choose the pre-registered contrast silently."
        )

    # ---- The pre-registered H1 headline: derived from the rungs actually passed in (FIX-A) —
    # ---- never hardcoded as "d=max - d=0", which is only ever true for v1's default flags and
    # ---- was printed unconditionally even when --low/--high named a different contrast entirely
    # ---- (v2 basis points, or a custom v1 pair). Computed and printed unedited otherwise. No
    # ---- try/except: a verdict with no paired data must not be printable, so a ValueError from an
    # ---- empty overlap propagates out of main() rather than being swallowed into a fake PASS/FAIL.
    print()
    diff, ci_low, ci_high = paired_difference_ci(instances, abstained, args.low, args.high)
    n_headline = _paired_n(instances, abstained, args.low, args.high)
    verdict = h1_verdict(diff, ci_low, ci_high)
    headline_contrast = (
        f"{_ring_label(args.high, is_v2=is_v2)} - {_ring_label(args.low, is_v2=is_v2)}"
    )
    print(
        f"H1 (pre-registered) paired delta(correct-abstain), {headline_contrast}: {diff:+.3f} "
        f"[{ci_low:+.3f}, {ci_high:+.3f}]  n={n_headline}"
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
        label = f"{_ring_label(lo, is_v2=is_v2)} vs {_ring_label(hi, is_v2=is_v2)}"
        try:
            d, lo_ci, hi_ci = paired_difference_ci(instances, abstained, lo, hi)
        except ValueError:
            print(f"  {label:<24} n=0 (no paired data)")
            continue
        n = _paired_n(instances, abstained, lo, hi)
        v = h1_verdict(d, lo_ci, hi_ci)
        print(f"  {label:<24} delta={d:+.3f} [{lo_ci:+.3f}, {hi_ci:+.3f}]  n={n}  {v}")
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
